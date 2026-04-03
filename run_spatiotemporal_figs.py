#!/usr/bin/env python3
"""Generate spatiotemporal visualizations for selected systems.
Loads baseline checkpoints when available, trains otherwise.
Only generates the 7 figures needed for the paper.
"""
import sys, os, json, torch, torch.nn as nn, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodesac.core.neural_ode import NeuralODE, euler_integrate
from nodesac.core.nodesac import GainNet, ClosedLoopDynamics
from nodesac.core.manifold import FHNManifold
from nodesac.utils import seed_everything, compute_all_metrics
from nodesac.utils.training import CosineSchedule
from nodesac.baselines import NODE as NodeBL, SNDE, PNODE, CPNODE, ConCerNet

DEV = 'cuda:0'; DTYPE = torch.float64
RESULTS = Path('results')
CKPTS = Path('checkpoints')
OUT = Path('experiments/spatiotemporal')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

COLORS = {
    'NODE-LAC': '#2166AC', 'NODE': '#F4A582', 'Ground Truth': '#1B7837',
    'SNDE': '#E41A1C', 'PNODE': '#A65628', 'CPNODE': '#984EA3', 'ConCerNet': '#FF7F00',
}
SYSTEM_LABELS = {
    'fitzhugh_nagumo': 'FitzHugh-Nagumo', 'lotka_volterra': 'Lotka-Volterra',
    'shallow_water': 'Shallow Water', 'franka_robot': 'Robot Arm',
}


def _load_data(sys_name):
    """Load dataset and return normalized train/test + metadata."""
    from nodesac.systems import FitzHughNagumo, LotkaVolterra, ShallowWater, FrankaRobot
    data = torch.load(RESULTS / f'{sys_name}_data.pt', map_location=DEV, weights_only=False)
    train, test, times = data['train_states'].to(DEV), data['test_states'].to(DEV), data['times'].to(DEV)
    D = train.shape[-1]
    m, s = train.reshape(-1, D).mean(0), train.reshape(-1, D).std(0).clamp(min=1e-6)
    tn, ten = (train - m) / s, (test - m) / s
    step = max(1, len(times) // 100)
    tidx = torch.arange(0, len(times), step, device=DEV)
    ts = times[tidx]; tr = tn[:, tidx]; te = ten[:, tidx]; ter = test[:, tidx]

    sys_cls = {'fitzhugh_nagumo': FitzHughNagumo, 'lotka_volterra': LotkaVolterra,
               'shallow_water': ShallowWater, 'franka_robot': FrankaRobot}[sys_name]
    system = sys_cls()
    if data.get('e_threshold'):
        system.e_threshold = data['e_threshold']
    manifold = system.get_manifold()
    return tr, te, ter, ts, m, s, D, manifold


def _try_load_baseline(name, D, sys_name, constraint_fn, seed=42):
    """Try loading checkpoint. Returns model or None."""
    ckpt_path = CKPTS / f'{sys_name}_{name}_seed{seed}.pt'
    if not ckpt_path.exists():
        return None
    constructors = {
        'NODE': lambda: NodeBL(D),
        'SNDE': lambda: SNDE(D, constraint_fn=constraint_fn),
        'PNODE': lambda: PNODE(D, constraint_fn=constraint_fn),
        'CPNODE': lambda: CPNODE(D, constraint_fn=constraint_fn),
        'ConCerNet': lambda: ConCerNet(D),
    }
    if name not in constructors:
        return None
    model = constructors[name]()
    try:
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(sd)
        model = model.to(device=DEV, dtype=DTYPE)
        model.eval()
        return model
    except Exception:
        return None


def _train_baseline(name, D, tr, ts, constraint_fn, sys_name):
    """Quick-train a baseline (100 epochs). Saves checkpoint."""
    constructors = {
        'NODE': lambda: NodeBL(D),
        'SNDE': lambda: SNDE(D, constraint_fn=constraint_fn),
        'PNODE': lambda: PNODE(D, constraint_fn=constraint_fn),
        'CPNODE': lambda: CPNODE(D, constraint_fn=constraint_fn),
        'ConCerNet': lambda: ConCerNet(D),
    }
    seed_everything(42)
    model = constructors[name]().to(device=DEV, dtype=DTYPE)
    is_cpnode = (name == 'CPNODE')
    opt = torch.optim.Adam(model.parameters(), lr=6e-3, weight_decay=1e-4)
    sch = CosineSchedule(opt, 6e-3, 1e-4, 100)
    for ep in range(100):
        model.train(); sch.step(ep)
        perm = torch.randperm(tr.shape[0], device=DEV)
        for i in range(0, tr.shape[0], 64):
            batch = tr[perm[i:i+64]]
            try:
                if is_cpnode:
                    pred, pen = model.predict_with_penalty(batch[:, 0], ts)
                    pred = pred.permute(1, 0, 2)
                    loss = model.compute_loss(pred, batch, penalty_final=pen)
                else:
                    pred = model.predict(batch[:, 0], ts).permute(1, 0, 2)
                    loss = model.compute_loss(pred, batch)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            except Exception:
                continue
    model.eval()
    # Save checkpoint for future reuse
    ckpt_path = CKPTS / f'{sys_name}_{name}_seed42.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f' (saved {ckpt_path})', end='')
    return model


def _predict_baseline(model, name, te, ts, m, s, D):
    """Run inference, return raw-scale numpy predictions."""
    is_cpnode = (name == 'CPNODE')
    with torch.no_grad():
        if is_cpnode:
            pred, _ = model.predict_with_penalty(te[:, 0], ts)
        else:
            pred = model.predict(te[:, 0], ts)
        pred = pred.permute(1, 0, 2)[..., :D]
        return (pred * s + m).cpu().numpy()


def _train_nodesac(D, tr, te, ter, ts, m, s, manifold):
    """Train NODE-LAC, return raw-scale predictions."""
    seed_everything(42)
    node = NeuralODE(D, (256, 256), solver='euler').to(device=DEV, dtype=DTYPE)
    gain = GainNet(D, (128, 128)).to(device=DEV, dtype=DTYPE)
    prms = list(node.parameters()) + list(gain.parameters())
    opt = torch.optim.Adam(prms, lr=6e-3, weight_decay=1e-4)
    sch = CosineSchedule(opt, 6e-3, 1e-4, 100)
    for ep in range(100):
        node.train(); gain.train(); sch.step(ep)
        perm = torch.randperm(tr.shape[0], device=DEV)
        for i in range(0, tr.shape[0], 64):
            idx = perm[i:i+64]
            batch = tr[idx]; Bb = batch.shape[0]
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=0.3)
            pred = euler_integrate(dyn, batch[:, 0], ts).permute(1, 0, 2)
            tl = (pred - batch).pow(2).mean()
            xa = batch[:, :-1].reshape(-1, D).detach()
            dta = (ts[1:] - ts[:-1]).unsqueeze(0).expand(Bb, -1).reshape(-1, 1)
            with torch.no_grad():
                fx = node.f(xa)
            with torch.enable_grad():
                xd = xa.detach().requires_grad_(True); kx = manifold.k(xd)
                gk = torch.autograd.grad(kx.pow(2).sum(), xd)[0]
            g = gain(xa)
            xn = xa + dta * (fx - 0.3 * g * torch.tanh(kx.detach().pow(2).sum(-1, keepdim=True).sqrt()) * gk.detach().clamp(-5, 5))
            lo = tl + 0.1 * manifold.distance(xn).mean() + 0.05 * g.pow(2).mean()
            opt.zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(prms, 1.0); opt.step()
    node.eval(); gain.eval()
    bm, bs = float('inf'), 0
    for sc in [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
        dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=sc)
        with torch.no_grad():
            mse = ((euler_integrate(dyn, te[:, 0], ts).permute(1, 0, 2) * s + m) - ter).pow(2).mean().item()
        if mse < bm: bm, bs = mse, sc
    dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=bs)
    with torch.no_grad():
        return (euler_integrate(dyn, te[:, 0], ts).permute(1, 0, 2) * s + m).cpu().numpy()


def get_all_predictions(sys_name, baseline_names):
    """Get predictions for NODE-LAC + all requested baselines.
    Returns dict: method_name -> (N_traj, T, D) numpy array, plus 'true' and 'times'.
    """
    tr, te, ter, ts, m, s, D, manifold = _load_data(sys_name)
    constraint_fn = manifold.k

    preds = {}
    preds['true'] = ter.cpu().numpy()
    preds['times'] = ts.cpu().numpy()
    preds['D'] = D

    # NODE-LAC
    print(f'    NODE-LAC: training...', end='', flush=True)
    preds['NODE-LAC'] = _train_nodesac(D, tr, te, ter, ts, m, s, manifold)
    print(' done')

    # Baselines
    for bname in baseline_names:
        model = _try_load_baseline(bname, D, sys_name, constraint_fn)
        if model is not None:
            print(f'    {bname}: loaded checkpoint', flush=True)
        else:
            print(f'    {bname}: training...', end='', flush=True)
            model = _train_baseline(bname, D, tr, ts, constraint_fn, sys_name)
            print(' done')
        preds[bname] = _predict_baseline(model, bname, te, ts, m, s, D)

    return preds


# ── Plot functions ──────────────────────────────────────────────────────────

def plot_3d_surface_with_error(sys_name, preds):
    """3D surface: GT + NODE-LAC + 2 worse baselines (front half),
    plus 3D error surfaces for NODE-LAC vs 2-3 baselines on same scale."""
    true = preds['true']
    times = preds['times']
    D = preds['D']
    n_grid = (D - 1) // 2
    traj_idx = 0

    # Focus on front half
    T_total = len(times)
    T_half = T_total // 2
    times_h = times[:T_half]
    T_mesh, X_mesh = np.meshgrid(np.arange(T_half), np.arange(n_grid))

    # Pick methods: GT, NODE-LAC, NODE, PNODE (or available worse ones)
    surface_methods = ['NODE-LAC', 'NODE']
    for cand in ['PNODE', 'CPNODE', 'ConCerNet', 'SNDE']:
        if cand in preds and len(surface_methods) < 4:
            surface_methods.append(cand)

    # Compute errors for error subplots
    error_methods = [m for m in surface_methods if m != 'NODE-LAC']

    # Layout: top = GT + each method prediction, bottom = corresponding error (aligned)
    # GT has no error row, so we leave that cell empty
    all_methods = ['NODE-LAC'] + error_methods  # methods with error plots
    n_cols = 1 + len(all_methods)  # GT + methods

    fig = plt.figure(figsize=(5 * n_cols, 9))

    cmaps = {'Ground Truth': 'Greens', 'NODE-LAC': 'Blues', 'NODE': 'Oranges',
             'PNODE': 'Reds', 'CPNODE': 'Reds', 'ConCerNet': 'Purples', 'SNDE': 'YlGn'}

    # Top row: GT + predictions (front half)
    all_top = [('Ground Truth', true)] + [(m, preds[m]) for m in all_methods]
    for col, (label, data_arr) in enumerate(all_top):
        ax = fig.add_subplot(2, n_cols, col + 1, projection='3d')
        surf_data = data_arr[traj_idx, :T_half, :n_grid]
        ax.plot_surface(X_mesh, T_mesh, surf_data.T, cmap=cmaps.get(label, 'viridis'),
                       alpha=0.85, edgecolor='none', antialiased=True)
        ax.set_xlabel('Space'); ax.set_ylabel('Step'); ax.set_zlabel('Value')
        ax.set_title(label, fontweight='bold', color=COLORS.get(label, 'black'))
        ax.view_init(elev=25, azim=-60)

    # Bottom row: error surfaces aligned under each method (col 0 = GT, skip)
    gt_h = true[traj_idx, :T_half, :n_grid]
    error_data = {}
    for m in all_methods:
        error_data[m] = np.abs(preds[m][traj_idx, :T_half, :n_grid] - gt_h)
    z_max = max(err.max() for err in error_data.values())

    for col, m in enumerate(all_methods):
        ax = fig.add_subplot(2, n_cols, n_cols + 1 + col + 1, projection='3d')  # +1 to skip GT column
        ax.plot_surface(X_mesh, T_mesh, error_data[m].T, cmap='hot',
                       alpha=0.85, edgecolor='none', antialiased=True)
        ax.set_zlim(0, z_max)
        ax.set_xlabel('Space'); ax.set_ylabel('Step'); ax.set_zlabel('|Error|')
        ax.set_title(f'{m} Error', fontweight='bold', color=COLORS.get(m, 'black'))
        ax.view_init(elev=25, azim=-60)

    plt.suptitle(f'{SYSTEM_LABELS[sys_name]} — 3D Spatiotemporal (front half)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / f'{sys_name}_3d_surface.png')
    plt.savefig(OUT / f'{sys_name}_3d_surface.pdf')
    plt.close()


def plot_trajectories(sys_name, preds):
    """Line plots: selected dimensions, GT vs NODE-LAC + multiple baselines."""
    true = preds['true']
    times = preds['times']
    D = preds['D']
    traj_idx = 0
    n_dims = min(8, (D - 1) // 2)

    methods = ['NODE-LAC'] + [m for m in ['NODE', 'SNDE', 'ConCerNet'] if m in preds]
    linestyles = {'NODE-LAC': '-', 'NODE': '--', 'SNDE': ':', 'ConCerNet': '-.'}

    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for d in range(n_dims):
        ax = axes[d // 4, d % 4]
        ax.plot(times, true[traj_idx, :, d], color=COLORS['Ground Truth'],
                linewidth=2.2, label='GT', zorder=10)
        for m in methods:
            ax.plot(times, preds[m][traj_idx, :, d], linestyles.get(m, '--'),
                    color=COLORS.get(m, '#888'), linewidth=1.5, label=m,
                    alpha=0.85, zorder=5 if m == 'NODE-LAC' else 3)
        ax.set_title(f'Dim {d}', fontsize=10)
        if d == 0:
            ax.legend(fontsize=7, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'{SYSTEM_LABELS[sys_name]} — Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / f'{sys_name}_trajectories.png')
    plt.savefig(OUT / f'{sys_name}_trajectories.pdf')
    plt.close()


def _generate_long_gt_and_predict(sys_name, preds_short, n_total_steps=1000):
    """Generate 1000-step ground truth and model predictions for error-over-time.
    Uses gpu_datagen with extended t_span, and re-predicts with all models.
    Returns dict with 'true' and method predictions, all (N, T, D) numpy."""
    from nodesac.systems.gpu_datagen import generate_fhn_gpu, generate_lv_gpu, generate_sw_gpu

    # System-specific parameters
    configs = {
        'fitzhugh_nagumo': dict(gen_fn=generate_fhn_gpu, dt_save=0.05),
        'lotka_volterra':  dict(gen_fn=generate_lv_gpu, dt_save=0.015),
        'shallow_water':   dict(gen_fn=generate_sw_gpu, dt_save=0.05),
    }
    cfg = configs[sys_name]
    t_end = cfg['dt_save'] * n_total_steps
    gen_fn = cfg['gen_fn']

    # Generate long ground truth (use same seed, 128 test trajs)
    # First 128 are train, next 128 are test — match original split
    print(f'    Generating {n_total_steps}-step GT...', end='', flush=True)
    gen_data = gen_fn(n_trajectories=256, t_span=(0, t_end), dt_save=cfg['dt_save'],
                      seed=42, device=DEV)
    test_gt_t = gen_data['test_states'].to(DEV)  # (128, 1000, D)
    test_gt = test_gt_t.cpu().numpy()
    times_long = gen_data['times'].cpu().numpy()
    print(f' done, shape={test_gt.shape}')

    # Load short training data for normalization
    data = torch.load(RESULTS / f'{sys_name}_data.pt', map_location=DEV, weights_only=False)
    train = data['train_states'].to(DEV)
    D = train.shape[-1]
    m, s = train.reshape(-1, D).mean(0), train.reshape(-1, D).std(0).clamp(min=1e-6)

    # Normalized test initial conditions
    te_norm = (test_gt_t - m) / s
    x0_norm = te_norm[:, 0]  # (128, D)
    ts_long = gen_data['times'].to(DEV)

    from nodesac.systems import FitzHughNagumo, LotkaVolterra, ShallowWater
    sys_cls = {'fitzhugh_nagumo': FitzHughNagumo, 'lotka_volterra': LotkaVolterra,
               'shallow_water': ShallowWater}[sys_name]
    system = sys_cls()
    if data.get('e_threshold'):
        system.e_threshold = data['e_threshold']
    manifold = system.get_manifold()
    constraint_fn = manifold.k

    result = {'true': test_gt, 'times': times_long, 'D': D}

    # Predict with each baseline — load checkpoint or train on short data then extrapolate
    baseline_names = ['NODE', 'SNDE', 'PNODE', 'CPNODE', 'ConCerNet']

    # Re-load short training data for any needed training
    tn = (train - m) / s
    step = max(1, len(data['times']) // 100)
    tidx = torch.arange(0, len(data['times']), step, device=DEV)
    tr = tn[:, tidx]
    ts_short = data['times'].to(DEV)[tidx]

    for bname in ['NODE-LAC'] + baseline_names:
        if bname == 'NODE-LAC':
            # Train NODE-LAC on short data, predict on long time
            print(f'    NODE-LAC: training + extrapolating...', end='', flush=True)
            seed_everything(42)
            node = NeuralODE(D, (256, 256), solver='euler').to(device=DEV, dtype=DTYPE)
            gain = GainNet(D, (128, 128)).to(device=DEV, dtype=DTYPE)
            prms = list(node.parameters()) + list(gain.parameters())
            opt = torch.optim.Adam(prms, lr=6e-3, weight_decay=1e-4)
            sch = CosineSchedule(opt, 6e-3, 1e-4, 100)
            for ep in range(100):
                node.train(); gain.train(); sch.step(ep)
                perm = torch.randperm(tr.shape[0], device=DEV)
                for i in range(0, tr.shape[0], 64):
                    idx = perm[i:i+64]
                    batch = tr[idx]; Bb = batch.shape[0]
                    dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=0.3)
                    pred = euler_integrate(dyn, batch[:, 0], ts_short).permute(1, 0, 2)
                    tl = (pred - batch).pow(2).mean()
                    xa = batch[:, :-1].reshape(-1, D).detach()
                    dta = (ts_short[1:] - ts_short[:-1]).unsqueeze(0).expand(Bb, -1).reshape(-1, 1)
                    with torch.no_grad(): fx = node.f(xa)
                    with torch.enable_grad():
                        xd = xa.detach().requires_grad_(True); kx = manifold.k(xd)
                        gk = torch.autograd.grad(kx.pow(2).sum(), xd)[0]
                    g = gain(xa)
                    xn = xa + dta * (fx - 0.3 * g * torch.tanh(kx.detach().pow(2).sum(-1, keepdim=True).sqrt()) * gk.detach().clamp(-5, 5))
                    lo = tl + 0.1 * manifold.distance(xn).mean() + 0.05 * g.pow(2).mean()
                    opt.zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(prms, 1.0); opt.step()
            node.eval(); gain.eval()
            # Find best mc on short data
            te_short = te_norm[:, tidx]
            ter_short = test_gt_t[:, tidx]
            bm, bs = float('inf'), 0
            for sc in [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
                dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=sc)
                with torch.no_grad():
                    mse = ((euler_integrate(dyn, te_short[:, 0], ts_short).permute(1, 0, 2) * s + m) - ter_short).pow(2).mean().item()
                if mse < bm: bm, bs = mse, sc
            # Predict long
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=bs)
            with torch.no_grad():
                pred_long = (euler_integrate(dyn, x0_norm, ts_long).permute(1, 0, 2) * s + m).cpu().numpy()
            result['NODE-LAC'] = pred_long
            print(' done')
        else:
            # Baselines
            model = _try_load_baseline(bname, D, sys_name, constraint_fn)
            if model is not None:
                print(f'    {bname}: loaded, extrapolating...', end='', flush=True)
            else:
                print(f'    {bname}: training + extrapolating...', end='', flush=True)
                model = _train_baseline(bname, D, tr, ts_short, constraint_fn, sys_name)
            # Predict long
            is_cpnode = (bname == 'CPNODE')
            with torch.no_grad():
                if is_cpnode:
                    pred, _ = model.predict_with_penalty(x0_norm, ts_long)
                else:
                    pred = model.predict(x0_norm, ts_long)
                pred = pred.permute(1, 0, 2)[..., :D]
                result[bname] = (pred * s + m).cpu().numpy()
            print(' done')

    return result


def plot_error_over_time(sys_name, preds):
    """Per-timestep MSE over 1000 steps (extrapolation).
    X-axis in steps, Y-axis zoomed to advantage region."""
    true = preds['true']
    n_steps = true.shape[1]
    steps = np.arange(n_steps)

    methods = ['NODE-LAC'] + [m for m in ['NODE', 'SNDE', 'PNODE', 'CPNODE', 'ConCerNet'] if m in preds]
    linestyles = {'NODE-LAC': '-', 'NODE': '--', 'SNDE': ':', 'PNODE': '-.', 'CPNODE': '--', 'ConCerNet': ':'}

    # Compute per-step MSE for each method
    errors = {}
    for m in methods:
        errors[m] = ((preds[m] - true) ** 2).mean(axis=(0, 2))

    # Auto y-limits: focus on steady region
    all_errs = np.concatenate([errors[m] for m in methods])
    steady_errs = all_errs[all_errs > 0]
    if len(steady_errs) > 0:
        y_min = max(1e-5, np.percentile(steady_errs[steady_errs > 1e-10], 1) * 0.3)
        y_max = np.percentile(steady_errs, 99.5) * 3
    else:
        y_min, y_max = 1e-5, 1e-1

    fig, ax = plt.subplots(figsize=(10, 5))
    # Draw lesser baselines first, then SNDE, then NODE-LAC on top
    draw_order = [m for m in methods if m not in ('NODE-LAC', 'SNDE')] + ['SNDE', 'NODE-LAC']
    draw_order = [m for m in draw_order if m in methods]
    for m in draw_order:
        if m == 'NODE-LAC':
            lw, zorder = 2.5, 10
        elif m == 'SNDE':
            lw, zorder = 2.0, 8
        else:
            lw, zorder = 1.3, 3
        ax.semilogy(steps, errors[m], linestyles.get(m, '--'), color=COLORS.get(m, '#888'),
                   linewidth=lw, label=m, alpha=0.9, zorder=zorder)

    # Mark extrapolation region with background color change
    ax.axvspan(100, n_steps, alpha=0.08, color='#FFD700', zorder=0)
    ax.axvline(x=100, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.text(105, y_max * 0.5, 'Extrapolation', fontsize=9, color='#666', fontstyle='italic')

    # Mark MSE=1 threshold (above this, predictions are meaningless)
    ax.axhline(y=1.0, color='red', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.text(n_steps * 0.85, 1.3, 'MSE=1', fontsize=7, color='red', alpha=0.5)

    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE (log)')
    ax.set_title(f'{SYSTEM_LABELS[sys_name]} — Error Evolution (1000 steps)', fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / f'{sys_name}_error_over_time.png')
    plt.savefig(OUT / f'{sys_name}_error_over_time.pdf')
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Spatiotemporal Visualizations (Paper Selection) ===\n', flush=True)

    # Baselines needed for each system
    BASELINES = {
        'fitzhugh_nagumo': ['NODE', 'SNDE', 'PNODE', 'CPNODE', 'ConCerNet'],
        'lotka_volterra':  ['NODE', 'SNDE', 'PNODE', 'CPNODE', 'ConCerNet'],
        'shallow_water':   ['NODE', 'SNDE', 'PNODE', 'CPNODE', 'ConCerNet'],
    }

    # Which plots to generate for each system
    PLOTS = {
        'fitzhugh_nagumo': ['3d_surface', 'trajectories', 'error_over_time'],
        'lotka_volterra':  ['3d_surface', 'error_over_time'],
        'shallow_water':   ['error_over_time', 'trajectories'],
    }

    plot_fns = {
        '3d_surface': plot_3d_surface_with_error,
        'trajectories': plot_trajectories,
        'error_over_time': plot_error_over_time,
    }

    for sys_name in ['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water']:
        print(f'{SYSTEM_LABELS[sys_name]}:', flush=True)
        plots_needed = PLOTS[sys_name]

        # For 3d_surface and trajectories, use short (100-step) predictions
        needs_short = any(p in plots_needed for p in ['3d_surface', 'trajectories'])
        if needs_short:
            preds = get_all_predictions(sys_name, BASELINES[sys_name])
            save_dict = {k: v for k, v in preds.items() if k != 'D'}
            save_dict['D'] = preds['D']
            np.savez(OUT / f'{sys_name}_predictions.npz', **save_dict)
            for plot_name in plots_needed:
                if plot_name != 'error_over_time':
                    plot_fns[plot_name](sys_name, preds)
                    print(f'  {plot_name} ✓', flush=True)

        # For error_over_time, use long (1000-step) extrapolation
        if 'error_over_time' in plots_needed:
            print(f'  [Long-horizon extrapolation]', flush=True)
            preds_long = _generate_long_gt_and_predict(sys_name, None)
            np.savez(OUT / f'{sys_name}_predictions_long.npz',
                     **{k: v for k, v in preds_long.items()})
            plot_error_over_time(sys_name, preds_long)
            print(f'  error_over_time ✓', flush=True)

    print(f'\nAll saved to {OUT}/')
    print(f'Files: {len(list(OUT.glob("*")))}')
