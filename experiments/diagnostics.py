"""Full-grid Lyapunov derivative and residual diagnostics."""
import torch


def full_diagnostics(dyn, states, constraint):
    """Independent reference calculation through the actual closed-loop field."""
    flat = states.detach().reshape(-1,states.shape[-1])
    values = []
    derivatives = []
    for part in flat.split(1024):
        with torch.enable_grad():
            z = part.detach().requires_grad_(True)
            v = .5*constraint.k(z).square().sum(-1)
            u = torch.autograd.grad(v.sum(),z)[0]
        with torch.no_grad():
            f = dyn(torch.zeros((),device=z.device,dtype=z.dtype),z)
            derivatives.append((u*f).sum(-1).detach())
            values.append(v.detach())
    v = torch.cat(values)
    dv = torch.cat(derivatives)
    signed = dv + .1*v
    raw = signed.clamp_min(0)
    norm = (raw/(1+v)).square()
    active = v > 1e-12
    if not torch.isfinite(torch.stack([v,dv,raw,norm])).all():
        raise FloatingPointError('Nonfinite full-grid diagnostic')
    out = {'sample_count':len(v),'active_count':int(active.sum()),'active_fraction':float(active.double().mean()),
           'V_mean':float(v.mean()),'dotV_mean':float(dv.mean()),'dotV_max':float(dv.max()),'dotV_min':float(dv.min()),
           'raw_R_mean':float(raw.mean()),'raw_R_max':float(raw.max()),
           'normalized_residual_sq_mean':float(norm.mean()),'normalized_residual_sq_max':float(norm.max()),
           'residual_satisfied_fraction':float((signed <= 1e-10).double().mean()),
           'active_normalized_residual_sq_mean':float(norm[active].mean()) if active.any() else None,
           'active_raw_R_mean':float(raw[active].mean()) if active.any() else None,
           'active_raw_R_max':float(raw[active].max()) if active.any() else None,
           'active_residual_satisfied_fraction':float((signed[active] <= 1e-10).double().mean()) if active.any() else None}
    return out

def score(diag):
    a = diag['reference']['active_normalized_residual_sq_mean']
    return None if a is None else .5*a+.5*diag['rollout']['normalized_residual_sq_mean']
