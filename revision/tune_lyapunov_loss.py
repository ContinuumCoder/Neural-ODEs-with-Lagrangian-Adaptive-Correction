#!/usr/bin/env python3
"""Validation-only Lyapunov-loss extension; prior experiments remain immutable."""
import argparse,csv,hashlib,json,math,statistics,sys,time,traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
import numpy as np
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tune_validation as prior
frozen=prior.frozen
PROTOCOL="aij-r2-lyapunov-loss-v1"
ROOT=frozen.NAS/"checkpoints/node_lac_aij"/PROTOCOL
REPO=Path(__file__).resolve().parents[1]
SEEDS=[42,123,456]
LAMBDAS=[0.,.001,.01,.1]
RHOS=prior.RHOS
BASE_INDEX={system:3 if system=="fitzhugh_nagumo" else 1 for system in frozen.SYSTEMS}
PARENT_SOURCE_HASH="3ae1b5ef2d0fcaafd8960110665eec6f76b7f7eec2622a02a9c48a6b2f8318fb"
PARENT_LOCK_HASH="c48134c467c85c587f03ed93525977022a657afe29f3d752c8ac36383762c9e2"
sha=frozen.hash_file
write=frozen.write_json
read=prior.read
SOURCE_HASH=sha(__file__)
CONFIG={"protocol":PROTOCOL,"systems":frozen.SYSTEMS,"seeds":SEEDS,"lambda_L":LAMBDAS,
  "base_options":{s:prior.options(i) for s,i in BASE_INDEX.items()},"rho_grid":RHOS,
  "epochs":100,"training_rho":.3,"decay_rate":.1,"reference_batch_samples":512,
  "rollout_batch_samples":512,"sampling":"flatten all 100 times; float64 linspace(0,N-1,min(512,N)).long(); detach; no RNG",
  "training_loss":"0.5 mean_ref (relu(dotV+0.1V)/(1+V))^2 + 0.5 mean_roll (relu(dotV+0.1V)/(1+V))^2",
  "validation_S":"0.5 mean_active_reference normalized_residual_sq + 0.5 mean_all_rollout normalized_residual_sq",
  "reference_active_threshold":1e-12,"mean_MSE_cap":1.05,"per_seed_MSE_cap":1.10,
  "rho_selection":"finite raw validation MSE minimum; grid-order tie",
  "lambda_selection":"minimum three-seed mean S among complete finite-S candidates within MSE caps; smaller lambda tie; no reference active states retains zero",
  "parent_config_sha256":prior.CONFIG_HASH,"parent_source_sha256":PARENT_SOURCE_HASH,
  "parent_selection_lock_sha256":PARENT_LOCK_HASH,"frozen_runner_sha256":prior.RUNNER_HASH}
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
NeuralODE,GainNet,FixedGain,ClosedLoopDynamics=(frozen.NeuralODE,frozen.GainNet,frozen.FixedGain,frozen.ClosedLoopDynamics)
CosineSchedule,euler_integrate,DTYPE,write_json=(frozen.CosineSchedule,frozen.euler_integrate,frozen.DTYPE,frozen.write_json)

def candidate_dir(system,index,seed):return ROOT/system/f"candidate{index}"/f"seed{seed}"
def options(system):return prior.options(BASE_INDEX[system])

def check_parent():
    prior.check_frozen()
    if sha(prior.__file__)!=PARENT_SOURCE_HASH:raise ValueError("Parent tuner changed")
    if sha(prior.ROOT/"selection_lock.json")!=PARENT_LOCK_HASH:raise ValueError("Parent selection lock changed")

def prepare():
    check_parent();p=prior.prepared();lock=read(prior.ROOT/"selection_lock.json");reuse={}
    inventory={item["path"]:item["sha256"] for item in lock["candidate_inventory"]}
    ROOT.mkdir(parents=True,exist_ok=True)
    for system in frozen.SYSTEMS:
        selected=lock["selection_by_system"][system]["selected"]
        if selected["candidate_index"]!=BASE_INDEX[system] or selected["options"]!=options(system):
            raise ValueError("Prespecified base configuration differs from parent lock")
        for seed in SEEDS:
            record_path=prior.candidate_dir(system,BASE_INDEX[system],seed)/"candidate.json"
            if str(record_path) not in inventory or sha(record_path)!=inventory[str(record_path)]:
                raise ValueError("Base candidate differs from parent immutable inventory")
            row=read(record_path);checkpoint=row["checkpoint"]
            if (row["system"],row["candidate_index"],row["seed"])!=(system,BASE_INDEX[system],seed):
                raise ValueError("Base candidate identity differs")
            if checkpoint!=selected["checkpoint_by_seed"][selected["seed_ids"].index(seed)]:
                raise ValueError("Base checkpoint differs from parent locked selection")
            if sha(checkpoint["path"])!=checkpoint["sha256"]:raise ValueError("Base checkpoint changed")
            if row["tuning_source_sha256"]!=PARENT_SOURCE_HASH or row["config_sha256"]!=prior.CONFIG_HASH:
                raise ValueError("Base record provenance differs")
            reuse[f"{system}/{seed}"]={"record_path":str(record_path),"record_sha256":sha(record_path),
                "checkpoint":checkpoint,"preprocessing":row["preprocessing"],"rho_scores":row["rho_scores"],
                "selected":row["selected"]}
    info={"protocol":PROTOCOL,"config":CONFIG,"config_sha256":CONFIG_HASH,"tuning_source_sha256":SOURCE_HASH,
          "cache":p["cache"],"reuse":reuse}
    target=ROOT/"prepared.json"
    if target.exists() and read(target)!=info:raise ValueError("Existing preparation differs")
    if not target.exists():write(target,info)
    print(json.dumps({"prepared":True,"reuse":12,"new_training":36}),flush=True)

def prepared():
    check_parent();p=prior.prepared();info=read(ROOT/"prepared.json")
    if info["config_sha256"]!=CONFIG_HASH or info["tuning_source_sha256"]!=SOURCE_HASH:
        raise ValueError("Prepared source/config differs")
    if info["cache"]!=p["cache"]:raise ValueError("Train-only caches differ")
    for entry in info["reuse"].values():
        if sha(entry["record_path"])!=entry["record_sha256"] or sha(entry["checkpoint"]["path"])!=entry["checkpoint"]["sha256"]:
            raise ValueError("Base record/checkpoint changed")
    return info

def load_train_validation(system,device,info):
    return prior.load_train_validation(system,device,info)

def model_from_checkpoint(path,dim,constraint,device,system):
    return prior.model_from_checkpoint(path,dim,constraint,device,BASE_INDEX[system])

def sample_states(states):
    flat=states.reshape(-1,states.shape[-1])
    indices=torch.linspace(0,len(flat)-1,steps=min(512,len(flat)),dtype=torch.float64,device=flat.device).long()
    return flat[indices].detach()

def residual_values(node,gain,states,constraint,rho,require_param_grad=False):
    """Actual clipped field; true unclipped grad V; independent state sample."""
    with torch.enable_grad():
        x=states.reshape(-1,states.shape[-1]).detach().requires_grad_(True)
        k=constraint.k(x);v=.5*k.square().sum(-1)
        u=torch.autograd.grad(v.sum(),x,create_graph=False)[0].detach()
    v=v.detach();x=x.detach()
    # Gate and grad V are sample features. Only f and gain carry new parameter gradients.
    gate=torch.tanh((2*v).sqrt()).detach()
    with torch.set_grad_enabled(require_param_grad):
        field=node.f(x)-rho*gain(x)*gate.unsqueeze(-1)*(2*u).clamp(-5,5)
        dot=(u*field).sum(-1)
        signed=dot+.1*v
        raw=torch.relu(signed)
        norm=(raw/(1+v)).square()
    return {"V":v,"u":u,"field":field,"dotV":dot,"signed_residual":signed,
            "raw_R":raw,"normalized_residual_sq":norm,"active":v>1e-12}

def loss_residual(node,gain,reference,rollout,constraint,rho):
    a=residual_values(node,gain,sample_states(reference),constraint,rho,True)
    b=residual_values(node,gain,sample_states(rollout),constraint,rho,True)
    return .5*a["normalized_residual_sq"].mean()+.5*b["normalized_residual_sq"].mean()

def finite_float(value):
    number=float(value)
    return number if math.isfinite(number) else None

def diagnostic_summary(values):
    v,dot,raw,norm,active=(values[k].detach() for k in ["V","dotV","raw_R","normalized_residual_sq","active"])
    satisfied=values["signed_residual"].detach()<=1e-10
    count=int(active.sum());result={
        "sample_count":len(v),"active_count":count,"active_fraction":float(active.double().mean()),
        "V_mean":finite_float(v.mean()),"dotV_mean":finite_float(dot.mean()),
        "dotV_min":finite_float(dot.min()),"dotV_max":finite_float(dot.max()),
        "raw_R_mean":finite_float(raw.mean()),"raw_R_max":finite_float(raw.max()),
        "normalized_residual_sq_mean":finite_float(norm.mean()),"normalized_residual_sq_max":finite_float(norm.max()),
        "residual_satisfied_fraction":float(satisfied.double().mean()),
        "active_normalized_residual_sq_mean":finite_float(norm[active].mean()) if count else None,
        "active_raw_R_mean":finite_float(raw[active].mean()) if count else None,
        "active_raw_R_max":finite_float(raw[active].max()) if count else None,
        "active_residual_satisfied_fraction":float(satisfied[active].double().mean()) if count else None}
    return result

def score_rhos(node,gain,val,times,constraint,rhos=RHOS):
    rows=[]
    with torch.no_grad():
        for rho in rhos:
            dyn=ClosedLoopDynamics(node.f,gain,constraint,correction_scale=rho)
            pred=euler_integrate(dyn,val[:,0],times).permute(1,0,2)
            delta=(pred-val)*constraint.std;k=constraint.k(pred)
            metrics={"MSE":float(delta.square().mean()),"MAE":float(delta.abs().mean()),
              "TCE":float(delta.diff(dim=1).square().mean()),"CE":float(k.square().sum(-1).mean()),
              "feasible_fraction":float((k.amax(-1)<=1e-6).double().mean())}
            reference=diagnostic_summary(residual_values(node,gain,val,constraint,rho))
            rollout=diagnostic_summary(residual_values(node,gain,pred,constraint,rho))
            a,b=reference["active_normalized_residual_sq_mean"],rollout["normalized_residual_sq_mean"]
            S=.5*a+.5*b if a is not None and b is not None else None
            rows.append({"rho":rho,"validation":{k:finite_float(v) for k,v in metrics.items()},
              "finite_mse":math.isfinite(metrics["MSE"]),
              "nonfinite_metrics":[k for k,v in metrics.items() if not math.isfinite(v)],
              "S":finite_float(S) if S is not None else None,"diagnostics":{"reference":reference,"rollout":rollout}})
    return rows

def choose(rows):
    if [r["rho"] for r in rows]!=RHOS:raise ValueError("Rho grid incomplete or duplicated")
    return prior.choose(rows,RHOS)

def train_lac(tr,val,times,constraint,device,epochs,path,options,lambda_L):
    dim=tr.shape[-1]
    node=NeuralODE(dim,(256,256),solver="euler").to(device=device,dtype=DTYPE)
    gain=(FixedGain() if options.get("fixed_gain") else GainNet(dim,(128,128))).to(device=device,dtype=DTYPE)
    training_scale=0. if options.get("no_correction") else .3
    effort_weight=options.get("effort_weight",.05)
    constraint_weight=0. if options.get("no_constraint_loss") else 1.
    params=list(node.parameters())+list(gain.parameters())
    primal_opt=torch.optim.Adam(params,lr=6e-3,weight_decay=1e-4)
    primal_schedule=CosineSchedule(primal_opt,6e-3,1e-4,epochs)
    log_mu=torch.tensor(math.log(options.get("mu_init",.1)),dtype=DTYPE,device=device,requires_grad=True)
    mu_opt=torch.optim.Adam([log_mu],lr=1e-2)
    history=[]
    for epoch in range(epochs):
        node.train();gain.train();primal_schedule.step(epoch)
        permutation=torch.randperm(len(tr),device=device)
        rows=[]
        for offset in range(0,len(tr),64):
            batch=tr[permutation[offset:offset+64]]
            mu=log_mu.exp().clamp(.01,1.)
            dyn=ClosedLoopDynamics(node.f,gain,constraint,correction_scale=training_scale)
            pred=euler_integrate(dyn,batch[:,0],times).permute(1,0,2)
            trajectory_loss=(pred-batch).square().mean()
            full_jc=constraint.distance(pred.detach()).mean()
            xa=batch[:,:-1].reshape(-1,dim).detach()
            dta=times.diff().unsqueeze(0).expand(len(batch),-1).reshape(-1,1)
            with torch.no_grad():fx=node.f(xa)
            with torch.enable_grad():
                xd=xa.detach().requires_grad_(True);kx=constraint.k(xd)
                gradient=torch.autograd.grad(kx.square().sum(),xd)[0].detach().clamp(-5,5)
            g=gain(xa)
            correction=training_scale*g*torch.tanh(kx.detach().square().sum(-1,keepdim=True).sqrt())*gradient
            next_x=xa+dta*(fx-correction)
            one_step_jc=constraint.distance(next_x).mean()
            effort=g.square().mean()
            # Inherited block gradients: theta only trajectory; omega only lookahead+effort.
            primal_loss=trajectory_loss+constraint_weight*mu.detach()*one_step_jc+effort_weight*effort
            if lambda_L>0:
                residual_loss=loss_residual(node,gain,batch,pred,constraint,training_scale)
                if not torch.isfinite(residual_loss):raise FloatingPointError("Non-finite Lyapunov training loss")
                primal_loss=primal_loss+lambda_L*residual_loss
            primal_opt.zero_grad();primal_loss.backward()
            nn.utils.clip_grad_norm_(params,1.);primal_opt.step()
            # Existing log-space dual update, driven by sampled full-rollout violation.
            dual_loss=-log_mu*(full_jc.detach()-.01)
            mu_opt.zero_grad();dual_loss.backward();mu_opt.step()
            values=[float(trajectory_loss.detach()),float(full_jc),float(one_step_jc.detach()),float(effort.detach())]
            if not all(math.isfinite(x) for x in values):raise FloatingPointError("Non-finite NODE-LAC loss")
            if lambda_L>0:values.append(float(residual_loss.detach()))
            rows.append(values)
        averages=np.mean(rows,axis=0)
        history.append({"epoch":epoch+1,"trajectory_MSE":float(averages[0]),
                        "rollout_Jc":float(averages[1]),"lookahead_Jc":float(averages[2]),
                        "gain_effort":float(averages[3]),"mu":float(log_mu.exp().clamp(.01,1.).detach())})
        if lambda_L>0:history[-1]["Lyapunov_loss"]=float(averages[4])
    node.eval();gain.eval()
    scale_scores={}
    with torch.no_grad():
        for scale in ([0.] if options.get("no_correction") else [0,.1,.2,.5,1.,1.5,2.]):
            dyn=ClosedLoopDynamics(node.f,gain,constraint,correction_scale=scale)
            p=euler_integrate(dyn,val[:,0],times).permute(1,0,2)
            value=float(((p-val)*constraint.std).square().mean())
            if math.isfinite(value):scale_scores[str(scale)]=value
    if not scale_scores:raise FloatingPointError("No finite validation scale")
    selected=float(min(scale_scores,key=scale_scores.get))
    torch.save({"node":node.state_dict(),"gain":gain.state_dict(),"log_mu":log_mu.detach(),
                "selected_scale":selected,"training_scale":training_scale,"options":options,"lambda_L":lambda_L},path/"model.pt")
    write_json(path/"history.json",history)
    return ClosedLoopDynamics(node.f,gain,constraint,correction_scale=selected),{
        "scale":selected,"validation_scale_scores":scale_scores,"effective_mu":history[-1]["mu"],
        "final_trajectory_loss":history[-1]["trajectory_MSE"],"options":options}


def worker(job):
    system,index,seed=job;path=candidate_dir(system,index,seed);path.mkdir(parents=True,exist_ok=True)
    start=time.time();info=prepared()
    identity={"protocol":PROTOCOL,"tuning_source_sha256":SOURCE_HASH,"config_sha256":CONFIG_HASH,
      "system":system,"candidate_index":index,"lambda_L":LAMBDAS[index],"seed":seed,"options":options(system),
      "source_dataset_sha256":info["cache"][system]["source_dataset_sha256"],
      "train_only_cache_sha256":info["cache"][system]["cache_sha256"]}
    try:
        if (path/"candidate.json").exists():
            old=read(path/"candidate.json")
            if any(old.get(k)!=v for k,v in identity.items()):raise ValueError("Cached candidate provenance differs")
            if old["status"]=="complete" and sha(old["checkpoint"]["path"])!=old["checkpoint"]["sha256"]:
                raise ValueError("Cached checkpoint changed")
            return {"job":job,"status":"cached","candidate_status":old["status"]}
        torch.set_num_threads(2)
        tr,val,times,constraint,provenance=load_train_validation(system,"cuda:1",info)
        base=info["reuse"][f"{system}/{seed}"]
        prior.verify_preprocessing(provenance,base["preprocessing"])
        if index==0:
            checkpoint=Path(base["checkpoint"]["path"]);origin="reused_selected_parent"
        else:
            checkpoint=path/"model.pt";origin="new_100_epoch_training"
            request={**identity,"epochs":100}
            rp=path/"training_request.json"
            if rp.exists() and read(rp)!=request:raise ValueError("Training request differs")
            if not rp.exists():write(rp,request)
            completion=path/"training_record.json";history=path/"history.json"
            valid=(checkpoint.exists() and completion.exists() and history.exists())
            if valid:
                record=read(completion)
                valid=(record.get("request")==request and record.get("checkpoint_sha256")==sha(checkpoint)
                       and record.get("history_sha256")==sha(history)
                       and [x["epoch"] for x in read(history)]==list(range(1,101)))
            if not valid:
                orphan=path/"orphans"/str(time.time_ns())
                for file in [checkpoint,completion,history]:
                    if file.exists():orphan.mkdir(parents=True,exist_ok=True);file.replace(orphan/file.name)
                frozen.seed_everything(seed)
                train_lac(tr,val,times,constraint,"cuda:1",100,path,options(system),LAMBDAS[index])
                if [x["epoch"] for x in read(history)]!=list(range(1,101)):raise ValueError("Incomplete training")
                write(completion,{"request":request,"checkpoint_sha256":sha(checkpoint),"history_sha256":sha(history)})
        node,gain,ck=model_from_checkpoint(checkpoint,tr.shape[-1],constraint,"cuda:1",system)
        if index and ck.get("lambda_L")!=LAMBDAS[index]:raise ValueError("Checkpoint residual weight differs")
        rows=score_rhos(node,gain,val,times,constraint);selected=choose(rows)
        if index==0:
            for actual,expected in zip(rows,base["rho_scores"]):
                if actual["rho"]!=expected["rho"] or actual["validation"]!=expected["validation"]:
                    raise ValueError("Zero-weight validation replay differs")
        with torch.no_grad():
            g=gain(val.reshape(-1,val.shape[-1]))
            gain_summary={"scope":"validation_reference_states","mean":float(g.mean()),"min":float(g.min()),"max":float(g.max())}
        record={**identity,"status":"complete","epochs":100,"training_rho":.3,"origin":origin,
          "checkpoint":{"path":str(checkpoint),"sha256":sha(checkpoint)},"preprocessing":provenance,
          "rho_scores":rows,"selected":selected,"gain":gain_summary,
          "mu_final":float(ck["log_mu"].exp().clamp(.01,1)),"seconds":time.time()-start,
          "parent_base_record_sha256":base["record_sha256"]}
        write(path/"candidate.json",record)
        result={"job":job,"status":"complete","rho":selected["rho"],"MSE":selected["validation"]["MSE"],"S":selected["S"],"seconds":record["seconds"]}
    except Exception as error:
        result={"job":job,"status":"failed","error":str(error),"traceback":traceback.format_exc()}
        write(path/"candidate.json",{**identity,**result,"seconds":time.time()-start})
    print(json.dumps(result),flush=True);return result

def select_system(records,system):
    ordered=lambda index:sorted([r for r in records if r["candidate_index"]==index],key=lambda r:SEEDS.index(r["seed"]))
    base=ordered(0)
    if len(base)!=3 or any(r["status"]!="complete" for r in base):raise ValueError("Incomplete zero-weight baseline")
    base_mse=[r["selected"]["validation"]["MSE"] for r in base]
    if any(v is None or not math.isfinite(v) for v in base_mse):raise ValueError("Invalid baseline MSE")
    active_counts=[r["selected"]["diagnostics"]["reference"]["active_count"] for r in base]
    if len(set(active_counts))!=1:raise ValueError("Reference active set must be identical for all seeds")
    empty=active_counts[0]==0
    groups=[]
    for index,value in enumerate(LAMBDAS):
        subset=ordered(index)
        if len(subset)!=3 or [r["seed"] for r in subset]!=SEEDS:raise ValueError("Missing or duplicate seed")
        complete=all(r["status"]=="complete" for r in subset)
        mses=[r["selected"]["validation"]["MSE"] for r in subset] if complete else []
        ss=[r["selected"]["S"] for r in subset] if complete else []
        finite=complete and all(x is not None and math.isfinite(x) for x in mses+ss)
        mean_gate=finite and statistics.mean(mses)<=1.05*statistics.mean(base_mse)
        seed_gate=finite and all(x<=1.10*b for x,b in zip(mses,base_mse))
        eligible=bool(finite and mean_gate and seed_gate and not empty)
        if empty:eligible=index==0
        group={"system":system,"candidate_index":index,"lambda_L":value,"options":options(system),"seed_ids":SEEDS,
          "complete":complete,"finite_selected_S":bool(finite),"mean_MSE_gate":bool(mean_gate),"per_seed_MSE_gate":bool(seed_gate),
          "eligible":eligible,"reference_active_count":active_counts[0],
          "validation_MSE_mean":statistics.mean(mses) if complete else None,
          "validation_MSE_sample_sd":statistics.stdev(mses) if complete else None,
          "validation_S_mean":statistics.mean(ss) if finite else None,
          "selected_rhos":[r["selected"]["rho"] for r in subset] if complete else [],
          "selected_validation_by_seed":[r["selected"]["validation"] for r in subset] if complete else [],
          "selected_S_by_seed":ss,"checkpoint_by_seed":[r["checkpoint"] for r in subset] if complete else [],
          "failure_seeds":[r["seed"] for r in subset if r["status"]!="complete"]}
        groups.append(group)
    if not groups[0]["eligible"]:raise ValueError("Baseline score must be eligible")
    winner=groups[0] if empty else min([g for g in groups if g["eligible"]],key=lambda g:(g["validation_S_mean"],g["lambda_L"]))
    return {"selected":winner,"zero":groups[0],"reference_active_empty":empty,"candidate_groups":groups}

def lock_selection():
    info=prepared();records=[];inventory=[]
    for system in frozen.SYSTEMS:
        for index in range(4):
            for seed in SEEDS:
                p=candidate_dir(system,index,seed)/"candidate.json";row=read(p)
                if (row["system"],row["candidate_index"],row["seed"],row["lambda_L"])!=(system,index,seed,LAMBDAS[index]):
                    raise ValueError("Candidate identity mismatch")
                if row["config_sha256"]!=CONFIG_HASH or row["tuning_source_sha256"]!=SOURCE_HASH:raise ValueError("Mixed source/config")
                if row["train_only_cache_sha256"]!=info["cache"][system]["cache_sha256"]:raise ValueError("Cache mismatch")
                if row["status"]=="complete":
                    if sha(row["checkpoint"]["path"])!=row["checkpoint"]["sha256"]:raise ValueError("Checkpoint changed")
                    if row["selected"]!=choose(row["rho_scores"]):raise ValueError("Recorded rho selection differs")
                    for score in row["rho_scores"]:
                        for scope in ["reference","rollout"]:
                            if score["diagnostics"][scope]["sample_count"]!=2600:raise ValueError("Incomplete validation diagnostic scope")
                        d=score["diagnostics"]
                        a,b=d["reference"]["active_normalized_residual_sq_mean"],d["rollout"]["normalized_residual_sq_mean"]
                        expected=.5*a+.5*b if a is not None and b is not None else None
                        if score["S"]!=expected:raise ValueError("Recorded S differs")
                elif row["status"]!="failed":raise ValueError("Invalid candidate status")
                records.append(row);inventory.append({"path":str(p),"sha256":sha(p)})
    selections={system:select_system([r for r in records if r["system"]==system],system) for system in frozen.SYSTEMS}
    report={"protocol":PROTOCOL,"locked":True,"selection_split":"validation","config":CONFIG,
      "config_sha256":CONFIG_HASH,"tuning_source_sha256":SOURCE_HASH,"record_count":48,
      "parent_selection_lock_sha256":PARENT_LOCK_HASH,"reused_checkpoint_count":12,
      "requested_new_training_count":36,"complete_count":sum(r["status"]=="complete" for r in records),
      "failed_count":sum(r["status"]=="failed" for r in records),
      "selection_by_system":selections,"candidate_inventory":inventory}
    p=ROOT/"selection_lock.json"
    if p.exists() and read(p)!=report:raise ValueError("Existing immutable lock differs")
    if not p.exists():write(p,report)
    rows=[g for s in selections.values() for g in s["candidate_groups"]]
    with (ROOT/"candidate_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    print(json.dumps({"selection_lock":str(p),"selected":{s:r["selected"]["lambda_L"] for s,r in selections.items()}}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--lock-only",action="store_true");args=p.parse_args()
    if args.prepare:prepare();return
    if args.lock_only:lock_selection();return
    prepared()
    jobs=[(system,index,seed) for system in frozen.SYSTEMS for index in range(4) for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context("spawn")) as pool:
        outcomes=list(pool.map(worker,jobs))
    write(ROOT/"batch_outcomes.json",outcomes)
    lock_selection()
if __name__=="__main__":main()
