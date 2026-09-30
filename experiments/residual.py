"""Lyapunov-residual loss and NODE-LAC training with residual supervision."""
import math

import numpy as np
import torch
from torch import nn

from experiments import training

NeuralODE = training.NeuralODE
GainNet = training.GainNet
FixedGain = training.FixedGain
ClosedLoopDynamics = training.ClosedLoopDynamics
CosineSchedule = training.CosineSchedule
euler_integrate = training.euler_integrate
DTYPE = training.DTYPE
write_json = training.write_json

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
            # The prediction and constraint terms retain their separate block gradients.
            primal_loss=trajectory_loss+constraint_weight*mu.detach()*one_step_jc+effort_weight*effort
            if lambda_L>0:
                residual_loss=loss_residual(node,gain,batch,pred,constraint,training_scale)
                if not torch.isfinite(residual_loss):raise FloatingPointError("Non-finite Lyapunov training loss")
                primal_loss=primal_loss+lambda_L*residual_loss
            primal_opt.zero_grad();primal_loss.backward()
            nn.utils.clip_grad_norm_(params,1.);primal_opt.step()
            # Update the log multiplier using the detached full-rollout violation.
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
