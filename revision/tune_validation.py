#!/usr/bin/env python3
"""Phase-one NODE-LAC tuning using only the fixed training/validation split."""
import argparse,csv,hashlib,json,math,os,statistics,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_standardized_revision as frozen

PROTOCOL="aij-r2-validation-tuning-v1"
ROOT=frozen.NAS/"checkpoints/node_lac_aij"/PROTOCOL
CACHE=frozen.NAS/"datasets/node_lac_aij"/PROTOCOL/"train_only"
REPO=Path(__file__).resolve().parents[1]
LEGACY=frozen.DEFAULT_OUTPUT
RUNNER_HASH="36e82eeff846395e12edfc167f32a85f472e1ed5a5353738318430c1bf8c0b0a"
CANDIDATES=[(.1,.05),(.05,.05),(.1,.01),(.01,.01)]
RHOS=[0,.05,.1,.15,.2,.25,.3,.4,.5,.75,1,1.5,2]
COARSE=[0,.1,.2,.5,1,1.5,2]
SEEDS=[42,123,456]
sha=frozen.hash_file
write=frozen.write_json
SOURCE_HASH=sha(__file__)
CONFIG={"protocol":PROTOCOL,"systems":frozen.SYSTEMS,"seeds":SEEDS,"candidates":CANDIDATES,
        "rho_grid":RHOS,"original_rho_grid":COARSE,"epochs":100,"split_seed":20260906,
        "fit_count":102,"validation_count":26,"training_rho":.3,"primal_lr":.006,
        "lr_final":.0001,"dual_lr":.01,"dual_target":.01,"mu_bounds":[.01,1],
        "batch_size":64,"weight_decay":.0001,"gradient_clip":1,
        "selection":"Per-seed validation raw-MSE minimum rho; mean over three seeds chooses candidate; candidate-order ties.",
        "training_function":"frozen.train_lac","frozen_runner_sha256":RUNNER_HASH}
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()

def read(p):return json.loads(Path(p).read_text())
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def candidate_dir(system,index,seed):return ROOT/system/f"candidate{index}"/f"seed{seed}"
def options(index):return dict(zip(["mu_init","effort_weight"],CANDIDATES[index]))
def check_frozen():
    if sha(Path(frozen.__file__))!=RUNNER_HASH:raise ValueError("Frozen training runner changed")

def prepare():
    """One-time whitelist extraction. Search workers never open the source dataset."""
    check_frozen();ROOT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    artifact=read(REPO/"revision/legacy_artifacts.json")
    caches={}
    for system in frozen.SYSTEMS:
        source=frozen.DATA/f"{system}_data.pt";expected=artifact[f"results/{system}_data.pt"]["sha256"]
        if sha(source)!=expected:raise ValueError("Original dataset provenance differs")
        target=CACHE/f"{system}.pt";meta=target.with_suffix(".json")
        if not target.exists():
            original=torch.load(source,map_location="cpu",weights_only=False)
            payload={"train_states":original["train_states"],"times":original["times"]}
            if original.get("k_max") is not None:payload["k_max"]=original["k_max"]
            torch.save(payload,target)
            write(meta,{"source_dataset_sha256":expected,"source_path":str(source),
                        "train_tensor_sha256":tensor_sha(payload["train_states"]),
                        "times_tensor_sha256":tensor_sha(payload["times"]),
                        "cache_sha256":sha(target),"allowed_keys":sorted(payload)})
            del original,payload
        info=read(meta)
        if info["source_dataset_sha256"]!=expected or info["cache_sha256"]!=sha(target):raise ValueError("Train-only cache changed")
        caches[system]=info
    reuse={}
    for system in frozen.SYSTEMS:
        for index in range(4):
            if index and system!="fitzhugh_nagumo":continue
            mu,effort=CANDIDATES[index]
            variant="main" if index==0 else f"hyper_mu{mu}_effort{effort}"
            for seed in SEEDS:
                directory=LEGACY/system/variant/"NODE-LAC"/f"seed{seed}"
                raw=read(directory/"result.json")
                # Copy only identity/training/validation metadata, never evaluation metrics.
                metadata={key:raw[key] for key in ["protocol","system","method","variant","seed","epochs","source_sha256","dataset_sha256","selection"]}
                del raw
                if (metadata["protocol"],metadata["system"],metadata["method"],metadata["seed"],metadata["epochs"])!=(frozen.PROTOCOL,system,"NODE-LAC",seed,100):raise ValueError("Reuse identity differs")
                if metadata["source_sha256"]!=RUNNER_HASH or metadata["dataset_sha256"]!=caches[system]["source_dataset_sha256"]:raise ValueError("Reuse provenance differs")
                checkpoint=torch.load(directory/"model.pt",map_location="cpu",weights_only=True)
                actual={"mu_init":checkpoint["options"].get("mu_init",.1),"effort_weight":checkpoint["options"].get("effort_weight",.05)}
                if actual!=options(index) or checkpoint["training_scale"]!=.3:raise ValueError("Reuse configuration differs")
                if set(checkpoint["options"])-{"mu_init","effort_weight"}:raise ValueError("Unexpected checkpoint options")
                provenance=read(directory/"provenance.json")
                keep={key:provenance[key] for key in ["fit_indices","normalizer_fit_indices","validation_indices","mean","std","times","constraint_coordinates","constraint_energy_threshold","constraint_amplitude_threshold","data_fraction","normalized_noise_sigma"]}
                if keep["data_fraction"]!=1 or keep["normalized_noise_sigma"]!=0:raise ValueError("Reuse training data differs")
                reuse[f"{system}/{index}/{seed}"]={"checkpoint_path":str(directory/"model.pt"),
                   "checkpoint_sha256":sha(directory/"model.pt"),"source_result_sha256":sha(directory/"result.json"),
                   "metadata":metadata,"preprocessing":keep}
    if len(reuse)!=21:raise ValueError("Expected 21 authorized reusable checkpoints")
    write(ROOT/"prepared.json",{"protocol":PROTOCOL,"config":CONFIG,"config_sha256":CONFIG_HASH,
          "cache":caches,"reuse":reuse,"prepared_source_sha256":SOURCE_HASH})
    print(json.dumps({"prepared":True,"reused_checkpoints":21,"new_training_runs":27}),flush=True)

def prepared():
    info=read(ROOT/"prepared.json")
    if info["config_sha256"]!=CONFIG_HASH:raise ValueError("Prepared protocol differs")
    check_frozen()
    for system in frozen.SYSTEMS:
        if sha(CACHE/f"{system}.pt")!=info["cache"][system]["cache_sha256"]:raise ValueError("Prepared train-only cache changed")
    return info

def load_train_validation(system,device,info):
    target=CACHE/f"{system}.pt"
    if sha(target)!=info["cache"][system]["cache_sha256"]:raise ValueError("Training cache hash differs")
    payload=torch.load(target,map_location=device,weights_only=True)
    if set(payload)-{"train_states","times","k_max"}:raise ValueError("Unapproved cache key")
    states=payload["train_states"];times=payload["times"]
    if states.shape[:2]!=(128,100):raise ValueError("Original training shape differs")
    split=torch.randperm(128,generator=torch.Generator().manual_seed(20260906))
    fit_ids=split[:102];val_ids=split[102:]
    fit=states[fit_ids];mean=fit.reshape(-1,fit.shape[-1]).mean(0)
    std=fit.reshape(-1,fit.shape[-1]).std(0).clamp_min(1e-6)
    tr=(fit-mean)/std;val=(states[val_ids]-mean)/std
    classes=dict(zip(frozen.SYSTEMS,[frozen.FitzHughNagumo,frozen.LotkaVolterra,frozen.ShallowWater,frozen.FrankaRobot]))
    system_object=classes[system]()
    if "k_max" in payload:system_object.k_max=payload["k_max"]
    constraint=frozen.StandardizedConstraint(system_object.get_manifold(),mean,std)
    provenance={"fit_indices":fit_ids.tolist(),"normalizer_fit_indices":fit_ids.tolist(),
       "validation_indices":val_ids.tolist(),"mean":mean.tolist(),"std":std.tolist(),"times":times.tolist(),
       "constraint_coordinates":"z=(raw_state-mean)/std",
       "constraint_energy_threshold":constraint.physical.e_threshold,"constraint_amplitude_threshold":2.,
       "data_fraction":1.,"normalized_noise_sigma":0.}
    return tr,val,times,constraint,provenance

def verify_preprocessing(actual,expected):
    for name in ["fit_indices","normalizer_fit_indices","validation_indices","constraint_coordinates",
                 "constraint_energy_threshold","constraint_amplitude_threshold","data_fraction","normalized_noise_sigma"]:
        if actual[name]!=expected[name]:raise ValueError("Preprocessing differs: "+name)
    for name in ["mean","std","times"]:
        torch.testing.assert_close(torch.tensor(actual[name],dtype=torch.float64),
                                   torch.tensor(expected[name],dtype=torch.float64),rtol=1e-12,atol=1e-14)

def model_from_checkpoint(path,dim,constraint,device,index):
    ck=torch.load(path,map_location=device,weights_only=True)
    actual={"mu_init":ck["options"].get("mu_init",.1),"effort_weight":ck["options"].get("effort_weight",.05)}
    if actual!=options(index) or ck["training_scale"]!=.3:raise ValueError("Model configuration differs")
    if set(ck["options"])-{"mu_init","effort_weight"}:raise ValueError("Ablation checkpoint forbidden")
    node=frozen.NeuralODE(dim,(256,256),solver="euler").to(device=device,dtype=frozen.DTYPE)
    gain=frozen.GainNet(dim,(128,128)).to(device=device,dtype=frozen.DTYPE)
    node.load_state_dict(ck["node"]);gain.load_state_dict(ck["gain"]);node.eval();gain.eval()
    return node,gain,ck

def score_rhos(node,gain,val,times,constraint,rhos):
    rows=[]
    with torch.no_grad():
        for rho in rhos:
            dyn=frozen.ClosedLoopDynamics(node.f,gain,constraint,correction_scale=rho)
            pred=frozen.euler_integrate(dyn,val[:,0],times).permute(1,0,2)
            delta=(pred-val)*constraint.std
            k=constraint.k(pred)
            metrics={"MSE":float(delta.square().mean()),"MAE":float(delta.abs().mean()),
              "TCE":float(delta.diff(dim=1).square().mean()),
              "CE":float(k.square().sum(-1).mean()),
              "feasible_fraction":float((k.amax(-1)<=1e-6).double().mean())}
            nonfinite=[name for name,value in metrics.items() if not math.isfinite(value)]
            rows.append({"rho":rho,"validation":{name:value if math.isfinite(value) else None for name,value in metrics.items()},
                         "finite_mse":math.isfinite(metrics["MSE"]),"nonfinite_metrics":nonfinite})
    return rows

def choose(rows,grid):
    eligible=[row for row in rows if row["rho"] in grid and row["finite_mse"]]
    if not eligible:raise FloatingPointError("No finite validation MSE")
    return min(eligible,key=lambda row:(row["validation"]["MSE"],grid.index(row["rho"])))

def train_new(tr,val,times,constraint,device,path,index):
    # The released revision trainer is called directly; no orchestration/evaluation entrypoint.
    return frozen.train_lac(tr,val,times,constraint,device,100,path,options(index))

def worker(job):
    system,index,seed=job;path=candidate_dir(system,index,seed);path.mkdir(parents=True,exist_ok=True)
    start=time.time()
    try:
        info=prepared();torch.set_num_threads(2)
        if (path/"candidate.json").exists():
            old=read(path/"candidate.json")
            if old["config_sha256"]!=CONFIG_HASH or old["tuning_source_sha256"]!=SOURCE_HASH:raise ValueError("Cached candidate belongs to another source/config")
            if sha(Path(old["checkpoint"]["path"]))!=old["checkpoint"]["sha256"]:raise ValueError("Cached checkpoint changed")
            return {"job":job,"status":"cached"}
        tr,val,times,constraint,provenance=load_train_validation(system,"cuda:1",info)
        inherited=info["reuse"].get(f"{system}/{index}/{seed}")
        if inherited:
            verify_preprocessing(provenance,inherited["preprocessing"])
            checkpoint=Path(inherited["checkpoint_path"])
            if sha(checkpoint)!=inherited["checkpoint_sha256"]:raise ValueError("Frozen reusable checkpoint changed")
            origin="reused_frozen_revision"
        else:
            checkpoint=path/"model.pt";origin="new_100_epoch_training"
            request={"system":system,"candidate_index":index,"seed":seed,"epochs":100,
                     "options":options(index),"config_sha256":CONFIG_HASH,"source_sha256":SOURCE_HASH,
                     "cache_sha256":info["cache"][system]["cache_sha256"]}
            request_path=path/"training_request.json"
            if request_path.exists() and read(request_path)!=request:raise ValueError("Resume training configuration differs")
            if not request_path.exists():
                if checkpoint.exists():raise ValueError("Checkpoint lacks its training request")
                write(request_path,request)
            completion_path=path/"training_record.json"
            valid_training=False
            if checkpoint.exists() and completion_path.exists() and (path/"history.json").exists():
                completed=read(completion_path)
                valid_training=(completed.get("request")==request and
                    completed.get("checkpoint_sha256")==sha(checkpoint) and
                    completed.get("history_sha256")==sha(path/"history.json"))
            if not valid_training:
                orphan=path/"orphans"/str(time.time_ns())
                for filename in ["model.pt","history.json","training_record.json"]:
                    existing=path/filename
                    if existing.exists():
                        orphan.mkdir(parents=True,exist_ok=True)
                        existing.replace(orphan/filename)
                frozen.seed_everything(seed)
                train_new(tr,val,times,constraint,"cuda:1",path,index)
                if [entry["epoch"] for entry in read(path/"history.json")]!=list(range(1,101)):
                    raise ValueError("Training did not record all 100 epochs")
                write(completion_path,{"request":request,"checkpoint_sha256":sha(checkpoint),
                                      "history_sha256":sha(path/"history.json")})
        node,gain,ck=model_from_checkpoint(checkpoint,tr.shape[-1],constraint,"cuda:1",index)
        rows=score_rhos(node,gain,val,times,constraint,RHOS)
        refined=choose(rows,RHOS);coarse=choose(rows,COARSE)
        with torch.no_grad():
            gains=gain(val.reshape(-1,val.shape[-1]))
            gain_summary={"scope":"validation_reference_states","mean":float(gains.mean()),"min":float(gains.min()),"max":float(gains.max())}
        record={"protocol":PROTOCOL,"config_sha256":CONFIG_HASH,"tuning_source_sha256":SOURCE_HASH,
          "frozen_runner_sha256":RUNNER_HASH,"system":system,"candidate_index":index,"seed":seed,
          "options":options(index),"epochs":100,"training_rho":.3,"origin":origin,
          "source_dataset_sha256":info["cache"][system]["source_dataset_sha256"],
          "train_only_cache_sha256":info["cache"][system]["cache_sha256"],
          "checkpoint":{"path":str(checkpoint),"sha256":sha(checkpoint)},
          "preprocessing":provenance,"rho_scores":rows,"selected":refined,"original_grid_selected":coarse,
          "checkpoint_original_selected_rho":ck["selected_scale"],
          "mu_final":float(ck["log_mu"].exp().clamp(.01,1.)),
          "gain":gain_summary,"seconds":time.time()-start}
        if inherited:record["reuse_source_result_sha256"]=inherited["source_result_sha256"]
        write(path/"candidate.json",record)
        print(json.dumps({"job":job,"status":"complete","origin":origin,"rho":refined["rho"],"validation_MSE":refined["validation"]["MSE"],"seconds":record["seconds"]}),flush=True)
        return {"job":job,"status":"complete"}
    except Exception as error:
        failure={"job":job,"status":"failed","error":str(error),"traceback":traceback.format_exc()}
        write(path/"failure.json",failure);print(json.dumps(failure),flush=True);return failure

def lock_selection():
    info=prepared()
    records=[];inventory=[]
    for system in frozen.SYSTEMS:
        for index in range(4):
            for seed in SEEDS:
                path=candidate_dir(system,index,seed)/"candidate.json"
                row=read(path)
                if row["config_sha256"]!=CONFIG_HASH or row["tuning_source_sha256"]!=SOURCE_HASH:raise ValueError("Mixed candidate provenance")
                if (row["system"],row["candidate_index"],row["seed"])!=(system,index,seed):raise ValueError("Candidate identity differs")
                if row["train_only_cache_sha256"]!=info["cache"][system]["cache_sha256"]:raise ValueError("Candidate cache provenance differs")
                if sha(Path(row["checkpoint"]["path"]))!=row["checkpoint"]["sha256"]:raise ValueError("Checkpoint changed before lock")
                if [item["rho"] for item in row["rho_scores"]]!=RHOS:raise ValueError("Rho grid is incomplete or duplicated")
                if row["selected"]!=choose(row["rho_scores"],RHOS):raise ValueError("Fine-grid selection differs from recorded scores")
                if row["original_grid_selected"]!=choose(row["rho_scores"],COARSE):raise ValueError("Original-grid selection differs")
                records.append(row);inventory.append({"path":str(path),"sha256":sha(path)})
    selections={};groups=[];csv_rows=[]
    for system in frozen.SYSTEMS:
        scores=[]
        for index in range(4):
            subset=[row for row in records if row["system"]==system and row["candidate_index"]==index]
            values=[row["selected"]["validation"]["MSE"] for row in subset]
            coarse=[row["original_grid_selected"]["validation"]["MSE"] for row in subset]
            group={"system":system,"candidate_index":index,"options":options(index),"seed_ids":SEEDS,
                "validation_MSE_mean":statistics.mean(values),"validation_MSE_sample_sd":statistics.stdev(values),
                "original_grid_validation_MSE_mean":statistics.mean(coarse),
                "selected_rhos":[row["selected"]["rho"] for row in subset],
                "selected_validation_by_seed":[row["selected"]["validation"] for row in subset],
                "checkpoint_by_seed":[row["checkpoint"] for row in subset]}
            groups.append(group);scores.append(group)
            csv_rows.append({key:json.dumps(value) if isinstance(value,(list,dict)) else value for key,value in group.items()})
        winner=min(scores,key=lambda row:(row["validation_MSE_mean"],row["candidate_index"]))
        selections[system]={"selected":winner,"default_original_grid_MSE_mean":scores[0]["original_grid_validation_MSE_mean"],
                            "default_refined_grid_MSE_mean":scores[0]["validation_MSE_mean"]}
    report={"protocol":PROTOCOL,"locked":True,"selection_split":"validation",
      "config":CONFIG,"config_sha256":CONFIG_HASH,"tuning_source_sha256":SOURCE_HASH,
      "record_count":48,"reused_checkpoint_count":21,"new_training_count":27,
      "selection_by_system":selections,"all_candidate_groups":groups,"candidate_inventory":inventory}
    lock=ROOT/"selection_lock.json"
    if lock.exists() and read(lock)!=report:raise ValueError("Existing immutable selection lock differs")
    if not lock.exists():write(lock,report)
    with (ROOT/"candidate_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({"selection_lock":str(lock),"record_count":48,"selected_candidates":{s:r["selected"]["candidate_index"] for s,r in selections.items()}}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");parser.add_argument("--lock-only",action="store_true")
    args=parser.parse_args()
    if args.prepare:prepare();return
    if args.lock_only:lock_selection();return
    prepared()
    jobs=[(system,index,seed) for system in frozen.SYSTEMS for index in range(4) for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context("spawn")) as pool:outcomes=list(pool.map(worker,jobs))
    write(ROOT/"batch_outcomes.json",outcomes)
    if any(row["status"]=="failed" for row in outcomes):raise SystemExit(1)
    lock_selection()
if __name__=="__main__":main()

