#!/usr/bin/env python3
"""Minimal AIJ revision: archived data, original baseline roster and network sizes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from nodesac.core.neural_ode import NeuralODE, euler_integrate
from nodesac.core.nodesac import GainNet, ClosedLoopDynamics
from nodesac.utils.training import CosineSchedule
from nodesac.utils import seed_everything
from nodesac.baselines import NODE,SNDE,ConCerNet,SymODEN,HNN,CLNN,PortHJNN,PNODE,CPNODE
from nodesac.systems import FitzHughNagumo,LotkaVolterra,ShallowWater,FrankaRobot

PROTOCOL="aij-r2-standardized-v2"
NAS=Path("/media/diskstation/dzheng")
DATA=NAS/"datasets/node_lac_aij/legacy_6bd8aa1"
DEFAULT_OUTPUT=NAS/"checkpoints/node_lac_aij"/PROTOCOL
SYSTEMS=["fitzhugh_nagumo","lotka_volterra","shallow_water","franka_robot"]
METHODS=["NODE-LAC","NODE","SNDE","ConCerNet","SymODEN","HNN","CLNN","PORT-HJNN","PNODE","CPNODE"]
SEEDS=[42,123,456]
DTYPE=torch.float64

def write_json(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)

def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class StandardizedConstraint:
    """Original standardized-state envelope k(z); physical map is metadata only."""
    def __init__(self,physical,mean,std):
        self.physical,self.mean,self.std=physical,mean,std
    def k(self,z):
        return self.physical.k(z)
    def distance(self,z):
        return self.k(z).square().sum(-1)

def load_data(system,device,fraction=1.,noise=0.,seed=42):
    path=DATA/(system+"_data.pt")
    data=torch.load(path,map_location=device,weights_only=False)
    # Original test partition remains untouched; validation is from original train only.
    split=torch.randperm(128,generator=torch.Generator().manual_seed(20260906))
    fit_ids=split[:102];val_ids=split[102:]
    original_fit=data["train_states"][fit_ids]
    mean=original_fit.reshape(-1,original_fit.shape[-1]).mean(0)
    std=original_fit.reshape(-1,original_fit.shape[-1]).std(0).clamp_min(1e-6)
    tr=(original_fit-mean)/std
    val=(data["train_states"][val_ids]-mean)/std
    te=(data["test_states"]-mean)/std
    raw=data["test_states"]
    times=data["times"]
    # Preserve the inherited reviewer protocol: noise before prefix subsampling.
    if noise:
        seed_everything(seed)
        tr=tr+noise*torch.randn_like(tr)
    used=max(2,int(len(tr)*fraction))
    tr=tr[:used]
    classes=dict(zip(SYSTEMS,[FitzHughNagumo,LotkaVolterra,ShallowWater,FrankaRobot]))
    physical=classes[system]()
    if data.get("k_max") is not None:physical.k_max=data["k_max"]
    physical=physical.get_manifold()
    constraint=StandardizedConstraint(physical,mean,std)
    provenance={"source_sha256":hash_file(path),"source":str(path),
                "fit_indices":fit_ids[:used].tolist(),"normalizer_fit_indices":fit_ids.tolist(),
                "validation_indices":val_ids.tolist(),"test_indices":list(range(128)),
                "test_partition":"unchanged archived test_states",
                "mean":mean.tolist(),"std":std.tolist(),
                "constraint_coordinates":"z=(raw_state-mean)/std",
                "raw_equivalent_memory_E_upper_bound":float(mean[-1]+std[-1]*physical.e_threshold),
                "raw_equivalent_amplitude_bound":"mean(((u-mu_u)/sigma_u)^2+((v-mu_v)/sigma_v)^2)<=2",
                "constraint_energy_threshold":physical.e_threshold,
                "constraint_amplitude_threshold":2.,
                "cached_e_threshold_unused":data.get("e_threshold"),
                "data_fraction":fraction,"normalized_noise_sigma":noise,
                "states_per_trajectory":len(times),"times":times.tolist()}
    return tr,val,te,raw,times,mean,std,physical,constraint,provenance

def build_baseline(name,dim,constraint):
    even=dim+dim%2
    constructors={
      "NODE":lambda:NODE(dim),"SNDE":lambda:SNDE(dim,constraint_fn=constraint.k),
      "ConCerNet":lambda:ConCerNet(dim),"SymODEN":lambda:SymODEN(even),
      "HNN":lambda:HNN(even),"CLNN":lambda:CLNN(even),
      "PORT-HJNN":lambda:PortHJNN(even,hidden_dim=64),
      "PNODE":lambda:PNODE(dim,constraint_fn=constraint.k),
      "CPNODE":lambda:CPNODE(dim,constraint_fn=constraint.k)}
    return constructors[name](),name in ["SymODEN","HNN","CLNN","PORT-HJNN"] and dim%2==1

def pad(x,enabled):
    return torch.cat([x,torch.zeros_like(x[...,:1])],-1) if enabled else x

def metrics(pred,true,physical,pred_z,true_z):
    k=physical.k(pred_z);truth_k=physical.k(true_z)
    return {"MSE":float((pred-true).square().mean()),
            "MAE":float((pred-true).abs().mean()),
            "TCE":float((pred.diff(dim=1)-true.diff(dim=1)).square().mean()),
            "CE":float(k.square().sum(-1).mean()),
            "CE_max":float(k.square().sum(-1).max()),
            "reference_CE":float(truth_k.square().sum(-1).mean()),
            "Stability":float((k.square().sum(-1)<.1).double().mean()),
            "feasible_fraction":float((k.amax(-1)<=1e-6).double().mean()),
            "memory_E_MSE":float((pred[...,-1]-true[...,-1]).square().mean())}

def diagnostics(dynamics,states,constraint,rate=.1):
    z=states.reshape(-1,states.shape[-1])
    z=z[::max(1,math.ceil(len(z)/4096))]
    with torch.enable_grad():
        z=z.detach().requires_grad_(True)
        k=constraint.k(z);v=.5*k.square().sum(-1)
        grad=torch.autograd.grad(v.sum(),z)[0]
    with torch.no_grad():
        field=dynamics(torch.tensor(0.,device=z.device,dtype=z.dtype),z)
        dv=(grad*field).sum(-1)
        res=dv+rate*v.detach()
        active=v.detach()>1e-12
    return {"sample_count":len(z),"active_count":int(active.sum()),
            "V_mean":float(v.mean().detach()),"dVdt_mean":float(dv.mean()),
            "dVdt_max":float(dv.max()),
            "positive_residual_mean":float(res.clamp_min(0).mean()),
            "positive_residual_max":float(res.clamp_min(0).max()),
            "residual_satisfied_fraction":float((res<=1e-10).double().mean()),
            "active_residual_satisfied_fraction":float((res[active]<=1e-10).double().mean()) if active.any() else None,
            "rate":rate,"definition":"V=.5||k||²; residual=dV/dt+rate*V; gradients in normalized model coordinates"}

class FixedGain(nn.Module):
    def forward(self,x):
        return torch.ones_like(x[...,:1])

def train_lac(tr,val,times,constraint,device,epochs,path,options):
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
            primal_opt.zero_grad();primal_loss.backward()
            nn.utils.clip_grad_norm_(params,1.);primal_opt.step()
            # Existing log-space dual update, driven by sampled full-rollout violation.
            dual_loss=-log_mu*(full_jc.detach()-.01)
            mu_opt.zero_grad();dual_loss.backward();mu_opt.step()
            values=[float(trajectory_loss.detach()),float(full_jc),float(one_step_jc.detach()),float(effort.detach())]
            if not all(math.isfinite(x) for x in values):raise FloatingPointError("Non-finite NODE-LAC loss")
            rows.append(values)
        averages=np.mean(rows,axis=0)
        history.append({"epoch":epoch+1,"trajectory_MSE":float(averages[0]),
                        "rollout_Jc":float(averages[1]),"lookahead_Jc":float(averages[2]),
                        "gain_effort":float(averages[3]),"mu":float(log_mu.exp().clamp(.01,1.).detach())})
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
                "selected_scale":selected,"training_scale":training_scale,"options":options},path/"model.pt")
    write_json(path/"history.json",history)
    return ClosedLoopDynamics(node.f,gain,constraint,correction_scale=selected),{
        "scale":selected,"validation_scale_scores":scale_scores,"effective_mu":history[-1]["mu"],
        "final_trajectory_loss":history[-1]["trajectory_MSE"],"options":options}

def train_baseline(name,tr,times,constraint,device,epochs,path):
    model,padded=build_baseline(name,tr.shape[-1],constraint)
    model=model.to(device=device,dtype=DTYPE)
    opt=torch.optim.Adam(model.parameters(),lr=6e-3,weight_decay=1e-4)
    schedule=CosineSchedule(opt,6e-3,1e-4,epochs)
    history=[]
    for epoch in range(epochs):
        model.train();schedule.step(epoch)
        permutation=torch.randperm(len(tr),device=device)
        losses=[];trajectory_losses=[]
        for offset in range(0,len(tr),64):
            batch=pad(tr[permutation[offset:offset+64]],padded)
            if name=="CPNODE":
                pred,pen=model.predict_with_penalty(batch[:,0],times)
                loss=model.compute_loss(pred.permute(1,0,2),batch,penalty_final=pen)
            else:
                pred=model.predict(batch[:,0],times)
                loss=model.compute_loss(pred.permute(1,0,2),batch)
            trajectory_losses.append(float((pred.permute(1,0,2)[...,:tr.shape[-1]]-batch[...,:tr.shape[-1]]).square().mean().detach()))
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite baseline loss")
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
        history.append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),
                        "trajectory_MSE":float(np.mean(trajectory_losses))})
    model.eval()
    torch.save(model.state_dict(),path/"model.pt")
    write_json(path/"history.json",history)
    return model,padded

def worker(job):
    system,method,seed,variant,fraction,noise,epochs,device,output=job[:9]
    options=job[9] if len(job)>9 else {}
    torch.set_num_threads(2)
    key=f"{system}/{variant}/{method}/seed{seed}"
    path=Path(output)/key;path.mkdir(parents=True,exist_ok=True)
    if (path/"result.json").exists():return {"key":key,"status":"cached"}
    started=time.time()
    try:
        seed_everything(seed)
        tr,val,te,raw,times,mean,std,physical,constraint,provenance=load_data(system,device,fraction,noise,seed)
        seed_everything(seed)
        write_json(path/"provenance.json",provenance)
        extra={};diag={}
        if method=="NODE-LAC":
            dynamics,extra=train_lac(tr,val,times,constraint,device,epochs,path,options)
            with torch.no_grad():
                pred=euler_integrate(dynamics,te[:,0],times).permute(1,0,2)
            diag={"test_reference":diagnostics(dynamics,te,constraint),
                  "test_rollout":diagnostics(dynamics,pred,constraint)}
        else:
            model,padded=train_baseline(method,tr,times,constraint,device,epochs,path)
            with torch.no_grad():
                pred=model.predict(pad(te[:,0],padded),times).permute(1,0,2)
                pred=pred[...,:tr.shape[-1]]
            if method=="NODE":
                diag={"test_reference":diagnostics(model,te,constraint),
                      "test_rollout":diagnostics(model,pred,constraint)}
        with torch.no_grad():
            pred_raw=pred*std+mean
            if not torch.isfinite(pred_raw).all():raise FloatingPointError("Non-finite test rollout")
            values=metrics(pred_raw,raw,physical,pred,te)
            curves={"times":times.tolist(),"MSE":(pred_raw-raw).square().mean((0,2)).tolist(),
                    "CE":physical.k(pred).square().sum(-1).mean(0).tolist()}
        result={"protocol":PROTOCOL,"system":system,"method":method,"seed":seed,
                "variant":variant,"epochs":epochs,"metrics":values,"diagnostics":diag,
                "selection":extra,"seconds":time.time()-started,
                "source_sha256":hash_file(__file__),"dataset_sha256":provenance["source_sha256"],
                "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,
                               "device":device,"gpu":torch.cuda.get_device_name(device)}}
        write_json(path/"curves.json",curves)
        write_json(path/"result.json",result)
        print(json.dumps({"key":key,"status":"complete","seconds":result["seconds"],"metrics":values}),flush=True)
        return {"key":key,"status":"complete","seconds":result["seconds"]}
    except Exception as error:
        failure={"key":key,"status":"failed","error":str(error),"traceback":traceback.format_exc(),
                 "seconds":time.time()-started}
        write_json(path/"failure.json",failure)
        print(json.dumps(failure),flush=True)
        return failure

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",choices=["main","reviewer","ablation","hyperparam","all"],default="main")
    parser.add_argument("--systems",nargs="+",choices=SYSTEMS,default=SYSTEMS)
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS)
    parser.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if NAS not in args.output_root.parents:raise ValueError("Outputs must remain on NAS")
    jobs=[]
    for system in args.systems:
        if args.suite in ["main","all"]:
            for method in args.methods:
                for seed in args.seeds:
                    jobs.append((system,method,seed,"main",1.,0.,args.epochs,args.device,str(args.output_root)))
        if args.suite in ["reviewer","all"] and system!="franka_robot":
            for method in args.methods:
                if method in ["NODE-LAC","NODE","SNDE"]:
                    for fraction in [.25,.5,.75]:
                        for seed in args.seeds:jobs.append((system,method,seed,f"data_eff_{fraction}",fraction,0.,args.epochs,args.device,str(args.output_root)))
                if method in ["NODE-LAC","NODE"]:
                    for noise in [.05,.1,.2]:
                        for seed in args.seeds:jobs.append((system,method,seed,f"noise_{noise}",1.,noise,args.epochs,args.device,str(args.output_root)))
    if args.suite in ["ablation","all"]:
        for system in [x for x in args.systems if x!="franka_robot"]:
            for name,options in [("NoGainNet",{"fixed_gain":True}),
                                 ("NoConstraintLoss",{"no_constraint_loss":True}),
                                 ("NoCorrection",{"no_correction":True})]:
                for seed in args.seeds:
                    jobs.append((system,"NODE-LAC",seed,"ablation_"+name,1.,0.,args.epochs,args.device,str(args.output_root),options))
    if args.suite in ["hyperparam","all"] and "fitzhugh_nagumo" in args.systems:
        for mu_init,effort in [(.1,.05),(.05,.05),(.1,.01),(.01,.01)]:
            for seed in args.seeds:
                jobs.append(("fitzhugh_nagumo","NODE-LAC",seed,f"hyper_mu{mu_init}_effort{effort}",1.,0.,args.epochs,args.device,str(args.output_root),{"mu_init":mu_init,"effort_weight":effort}))
    print(json.dumps({"jobs":len(jobs),"epochs":args.epochs,"workers":args.workers,
                      "output":str(args.output_root),"source_sha256":hash_file(__file__)}),flush=True)
    if args.workers==1:
        results=[worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as executor:
            results=list(executor.map(worker,jobs))
    write_json(args.output_root/f"batch_{args.suite}_{int(time.time())}.json",results)
    if any(r["status"]=="failed" for r in results):sys.exit(1)

if __name__=="__main__":main()
