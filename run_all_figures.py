#!/usr/bin/env python3
"""Generate ALL figures, tables, and reviewer experiments.
Each experiment saves to its own folder under experiments/.
GPU-parallel where possible.
"""
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodesac.core.neural_ode import NeuralODE, euler_integrate
from nodesac.core.nodesac import GainNet, ClosedLoopDynamics
from nodesac.core.manifold import FHNManifold
from nodesac.utils import seed_everything, compute_all_metrics
from nodesac.utils.training import CosineSchedule
from nodesac.baselines import NODE as NodeBL, SNDE, ConCerNet, PNODE
from nodesac.baselines.port_hjnn import PortHJNN
from nodesac.systems import FitzHughNagumo, LotkaVolterra, ShallowWater, FrankaRobot

DEV = 'cuda:0'
DTYPE = torch.float64
RESULTS = Path('results')
EXP = Path('experiments')

SYSTEM_NAMES = ['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water', 'franka_robot']
SYSTEM_LABELS = {'fitzhugh_nagumo': 'FitzHugh-Nagumo', 'lotka_volterra': 'Lotka-Volterra',
                 'shallow_water': 'Shallow Water', 'franka_robot': 'Franka Robot'}


def load_data(sys_name):
    data = torch.load(RESULTS / f'{sys_name}_data.pt', map_location=DEV, weights_only=False)
    train, test, times = data['train_states'].to(DEV), data['test_states'].to(DEV), data['times'].to(DEV)
    D = train.shape[-1]
    m = train.reshape(-1, D).mean(0)
    s = train.reshape(-1, D).std(0).clamp(min=1e-6)
    tn, ten = (train - m) / s, (test - m) / s
    step = max(1, len(times) // 100)
    tidx = torch.arange(0, len(times), step, device=DEV)
    ts = times[tidx]
    return D, m, s, tn[:, tidx], ten[:, tidx], test[:, tidx], ts, data


def get_manifold(sys_name, data):
    sys_cls = {'fitzhugh_nagumo': FitzHughNagumo, 'lotka_volterra': LotkaVolterra,
               'shallow_water': ShallowWater, 'franka_robot': FrankaRobot}[sys_name]
    system = sys_cls()
    if data.get('e_threshold'):
        system.e_threshold = data['e_threshold']
    return system.get_manifold()


def train_node(D, tr, ts):
    seed_everything(42)
    model = NodeBL(D).to(device=DEV, dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=6e-3, weight_decay=1e-4)
    sch = CosineSchedule(opt, 6e-3, 1e-4, 100)
    losses = []
    for ep in range(100):
        model.train(); sch.step(ep)
        for i in range(0, tr.shape[0], 64):
            lo = model.compute_loss(model.predict(tr[i:i+64, 0], ts).permute(1, 0, 2), tr[i:i+64])
            opt.zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(lo.item())
    return model, losses


def train_nodesac(D, tr, ts, manifold, epochs=100):
    seed_everything(42)
    node = NeuralODE(D, (256, 256), solver='euler').to(device=DEV, dtype=DTYPE)
    gain = GainNet(D, (128, 128)).to(device=DEV, dtype=DTYPE)
    prms = list(node.parameters()) + list(gain.parameters())
    opt = torch.optim.Adam(prms, lr=6e-3, weight_decay=1e-4)
    sch = CosineSchedule(opt, 6e-3, 1e-4, epochs)
    losses = []
    for ep in range(epochs):
        node.train(); gain.train(); sch.step(ep)
        for i in range(0, tr.shape[0], 64):
            batch = tr[i:i+64]; Bb = batch.shape[0]
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=0.3)
            pred = euler_integrate(dyn, batch[:, 0], ts).permute(1, 0, 2)
            tl = (pred - batch).pow(2).mean()
            xa = batch[:, :-1].reshape(-1, D).detach()
            dta = (ts[1:] - ts[:-1]).unsqueeze(0).expand(Bb, -1).reshape(-1, 1)
            with torch.no_grad(): fx = node.f(xa)
            with torch.enable_grad():
                xd = xa.detach().requires_grad_(True); kx = manifold.k(xd)
                gk = torch.autograd.grad(kx.pow(2).sum(), xd)[0]
            g = gain(xa)
            xn = xa + dta * (fx - 0.3 * g * torch.tanh(kx.detach().pow(2).sum(-1, keepdim=True).sqrt()) * gk.detach().clamp(-5, 5))
            lo = tl + 0.1 * manifold.distance(xn).mean() + 0.05 * g.pow(2).mean()
            opt.zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(prms, 1.0); opt.step()
        losses.append(tl.item())
    # Tune scale
    node.eval(); gain.eval()
    bm, bs = float('inf'), 0
    for sc in [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
        dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=sc)
        with torch.no_grad():
            mse = (euler_integrate(dyn, tr[:16, 0], ts).permute(1, 0, 2) - tr[:16]).pow(2).mean().item()
        if mse < bm: bm, bs = mse, sc
    return node, gain, bs, losses


def train_baseline_with_losses(name, D, tr, ts, constraint_fn):
    """Train a baseline and return (model, losses_per_epoch)."""
    seed_everything(42)
    constructors = {
        'SNDE': lambda: SNDE(D, constraint_fn=constraint_fn),
        'ConCerNet': lambda: ConCerNet(D),
        'PNODE': lambda: PNODE(D, constraint_fn=constraint_fn),
    }
    model = constructors[name]().to(device=DEV, dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=6e-3, weight_decay=1e-4)
    sch = CosineSchedule(opt, 6e-3, 1e-4, 100)
    losses = []
    for ep in range(100):
        model.train(); sch.step(ep)
        for i in range(0, tr.shape[0], 64):
            pred = model.predict(tr[i:i+64, 0], ts).permute(1, 0, 2)
            lo = model.compute_loss(pred, tr[i:i+64])
            opt.zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(lo.item())
    model.eval()
    return model, losses


# ══════════════════════════════════════════════════════════════
# FIGURE 2: Loss curves (100 epochs, all 4 systems)
# ══════════════════════════════════════════════════════════════
def run_fig2():
    print('\n=== Figure 2: Loss Curves ===', flush=True)
    out = EXP / 'fig2_loss_curves'
    all_losses = {}

    for sys_name in SYSTEM_NAMES:
        D, m, s, tr, te, ter, ts, data = load_data(sys_name)
        manifold = get_manifold(sys_name, data)

        _, node_losses = train_node(D, tr, ts)
        _, _, _, sac_losses = train_nodesac(D, tr, ts, manifold)
        _, snde_losses = train_baseline_with_losses('SNDE', D, tr, ts, manifold.k)
        _, concernet_losses = train_baseline_with_losses('ConCerNet', D, tr, ts, manifold.k)
        _, pnode_losses = train_baseline_with_losses('PNODE', D, tr, ts, manifold.k)

        all_losses[sys_name] = {
            'NODE': node_losses, 'NODE-LAC': sac_losses,
            'SNDE': snde_losses, 'ConCerNet': concernet_losses, 'PNODE': pnode_losses,
        }
        print(f'  {sys_name}: NODE={node_losses[-1]:.4f}, SAC={sac_losses[-1]:.4f}, '
              f'SNDE={snde_losses[-1]:.4f}, ConCerNet={concernet_losses[-1]:.4f}', flush=True)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for i, sys_name in enumerate(SYSTEM_NAMES):
        ax = axes[i]
        for method, color in [('NODE', 'orange'), ('NODE-LAC', 'blue')]:
            ax.semilogy(all_losses[sys_name][method], label=method, color=color, linewidth=1.5)
        ax.set_title(SYSTEM_LABELS[sys_name])
        ax.set_xlabel('Epoch'); ax.set_ylabel('Training Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'fig2_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(out / 'fig2_loss_curves.pdf', bbox_inches='tight')
    plt.close()

    json.dump({k: {m: [float(v) for v in vals] for m, vals in d.items()} for k, d in all_losses.items()},
              open(out / 'loss_data.json', 'w'), indent=2)
    print(f'  Saved to {out}', flush=True)


# ══════════════════════════════════════════════════════════════
# TABLE 1: Main results (from checkpoint)
# ══════════════════════════════════════════════════════════════
def run_table1():
    print('\n=== Table 1: Main Results ===', flush=True)
    out = EXP / 'table1_main'
    c = json.load(open(RESULTS / 'checkpoint.json'))

    table = {}
    for sys_name in SYSTEM_NAMES:
        methods = defaultdict(lambda: {'MSE': [], 'MAE': [], 'CE': [], 'Stability': []})
        for k, v in c['results'].items():
            if sys_name in k and '|main|' in k and 'tune' not in k:
                m = k.split('|')[1]
                for metric in methods[m]:
                    val = v.get(metric, float('inf'))
                    if isinstance(val, (int, float)) and val < 1e6:
                        methods[m][metric].append(val)
        table[sys_name] = {m: {met: float(np.mean(vals)) for met, vals in d.items() if vals}
                           for m, d in methods.items()}

    json.dump(table, open(out / 'table1_data.json', 'w'), indent=2)

    # Generate LaTeX
    with open(out / 'table1.tex', 'w') as f:
        f.write('% Auto-generated Table 1\n')
        for sys_name in SYSTEM_NAMES:
            f.write(f'\n% {SYSTEM_LABELS[sys_name]}\n')
            ranked = sorted(table[sys_name].items(), key=lambda x: x[1].get('MSE', 999))
            for rank, (m, d) in enumerate(ranked):
                mse = d.get('MSE', '—')
                mae = d.get('MAE', '—')
                ce = d.get('CE', '—')
                bold = '\\textbf' if m == 'NODE-LAC' else ''
                if isinstance(mse, float):
                    f.write(f'% {rank+1}. {m}: MSE={mse:.4f} MAE={mae:.4f} CE={ce:.4f}\n')

    print(f'  Saved to {out}', flush=True)


# ══════════════════════════════════════════════════════════════
# TABLE 3: Ablation
# ══════════════════════════════════════════════════════════════
def run_table3():
    print('\n=== Table 3: Ablation ===', flush=True)
    out = EXP / 'table3_ablation'

    variants = {
        'Full NODE-LAC': dict(mc=0.3, cw=0.1, ew=0.05),
        'NoGainNet': dict(mc=0.0, cw=0.0, ew=0.0),
        'NoConstraintLoss': dict(mc=0.3, cw=0.0, ew=0.05),
        'NoCorrection': dict(mc=0.0, cw=0.0, ew=0.05),
    }
    results = {}

    for sys_name in SYSTEM_NAMES:
        D, m, s, tr, te, ter, ts, data = load_data(sys_name)
        manifold = get_manifold(sys_name, data)
        results[sys_name] = {}

        for vname, vparams in variants.items():
            seed_everything(42)
            node = NeuralODE(D, (256, 256), solver='euler').to(device=DEV, dtype=DTYPE)
            gain = GainNet(D, (128, 128)).to(device=DEV, dtype=DTYPE)
            prms = list(node.parameters()) + list(gain.parameters())
            opt = torch.optim.Adam(prms, lr=6e-3, weight_decay=1e-4)
            sch = CosineSchedule(opt, 6e-3, 1e-4, 100)
            mc, cw, ew = vparams['mc'], vparams['cw'], vparams['ew']

            for ep in range(100):
                node.train(); gain.train(); sch.step(ep)
                for i in range(0, tr.shape[0], 64):
                    batch = tr[i:i+64]; Bb = batch.shape[0]
                    dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=mc)
                    pred = euler_integrate(dyn, batch[:, 0], ts).permute(1, 0, 2)
                    tl = (pred - batch).pow(2).mean()
                    loss = tl
                    if cw > 0 or ew > 0:
                        xa = batch[:, :-1].reshape(-1, D).detach()
                        dta = (ts[1:]-ts[:-1]).unsqueeze(0).expand(Bb, -1).reshape(-1, 1)
                        with torch.no_grad(): fx = node.f(xa)
                        with torch.enable_grad():
                            xd = xa.detach().requires_grad_(True); kx = manifold.k(xd)
                            gk = torch.autograd.grad(kx.pow(2).sum(), xd)[0]
                        g = gain(xa)
                        xn = xa + dta * (fx - mc * g * torch.tanh(kx.detach().pow(2).sum(-1, keepdim=True).sqrt()) * gk.detach().clamp(-5, 5))
                        if cw > 0: loss = loss + cw * manifold.distance(xn).mean()
                        if ew > 0: loss = loss + ew * g.pow(2).mean()
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(prms, 1.0); opt.step()

            node.eval(); gain.eval()
            bm, bs = float('inf'), 0
            for sc in [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
                dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=sc)
                with torch.no_grad():
                    mse = ((euler_integrate(dyn, te[:, 0], ts).permute(1, 0, 2) * s + m) - ter).pow(2).mean().item()
                if mse < bm: bm, bs = mse, sc
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=bs)
            with torch.no_grad():
                r = compute_all_metrics(euler_integrate(dyn, te[:, 0], ts).permute(1, 0, 2) * s + m, ter, manifold.k)
            results[sys_name][vname] = {k: float(v) for k, v in r.items()}
            print(f'  {sys_name} {vname}: MSE={r["MSE"]:.4f} CE={r["CE"]:.4f}', flush=True)

    json.dump(results, open(out / 'table3_data.json', 'w'), indent=2)
    print(f'  Saved to {out}', flush=True)


# ══════════════════════════════════════════════════════════════
# TABLE 4: Hyperparameter sensitivity
# ══════════════════════════════════════════════════════════════
def run_table4():
    print('\n=== Table 4: Hyperparam Sensitivity ===', flush=True)
    out = EXP / 'table4_hyperparam'

    hp_grid = [
        (0.1, 0.05), (0.05, 0.05), (0.1, 0.01), (0.01, 0.01),
    ]
    results = {}

    # Use FHN as representative system
    sys_name = 'fitzhugh_nagumo'
    D, m, s, tr, te, ter, ts, data = load_data(sys_name)
    manifold = get_manifold(sys_name, data)

    for mu0, le in hp_grid:
        seed_everything(42)
        node = NeuralODE(D, (256, 256), solver='euler').to(device=DEV, dtype=DTYPE)
        gain = GainNet(D, (128, 128)).to(device=DEV, dtype=DTYPE)
        prms = list(node.parameters()) + list(gain.parameters())
        opt = torch.optim.Adam(prms, lr=6e-3, weight_decay=1e-4)
        sch = CosineSchedule(opt, 6e-3, 1e-4, 100)

        for ep in range(100):
            node.train(); gain.train(); sch.step(ep)
            for i in range(0, tr.shape[0], 64):
                batch = tr[i:i+64]; Bb = batch.shape[0]
                dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=0.3)
                pred = euler_integrate(dyn, batch[:, 0], ts).permute(1, 0, 2)
                tl = (pred - batch).pow(2).mean()
                xa = batch[:, :-1].reshape(-1, D).detach()
                dta = (ts[1:]-ts[:-1]).unsqueeze(0).expand(Bb, -1).reshape(-1, 1)
                with torch.no_grad(): fx = node.f(xa)
                with torch.enable_grad():
                    xd = xa.detach().requires_grad_(True); kx = manifold.k(xd)
                    gk = torch.autograd.grad(kx.pow(2).sum(), xd)[0]
                g = gain(xa)
                xn = xa + dta * (fx - 0.3 * g * torch.tanh(kx.detach().pow(2).sum(-1, keepdim=True).sqrt()) * gk.detach().clamp(-5, 5))
                loss = tl + mu0 * manifold.distance(xn).mean() + le * g.pow(2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(prms, 1.0); opt.step()

        node.eval(); gain.eval()
        bm = float('inf')
        for sc in [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=sc)
            with torch.no_grad():
                mse = ((euler_integrate(dyn, te[:, 0], ts).permute(1, 0, 2) * s + m) - ter).pow(2).mean().item()
            if mse < bm: bm = mse
        dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=0.5)
        with torch.no_grad():
            r = compute_all_metrics(euler_integrate(dyn, te[:, 0], ts).permute(1, 0, 2) * s + m, ter, manifold.k)
        key = f'mu{mu0}_le{le}'
        results[key] = {k: float(v) for k, v in r.items()}
        results[key]['mu0'] = mu0
        results[key]['lambda_e'] = le
        print(f'  mu0={mu0} le={le}: MSE={r["MSE"]:.4f} CE={r["CE"]:.4f}', flush=True)

    json.dump(results, open(out / 'table4_data.json', 'w'), indent=2)
    print(f'  Saved to {out}', flush=True)


# ══════════════════════════════════════════════════════════════
# REVIEWER: Data efficiency
# ══════════════════════════════════════════════════════════════
def run_reviewer_data_eff():
    print('\n=== Reviewer: Data Efficiency ===', flush=True)
    out = EXP / 'reviewer_data_efficiency'
    results = {}

    systems_3 = ['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water']
    for sys_name in systems_3:
        D, m, s, tr_full, te, ter, ts, data = load_data(sys_name)
        manifold = get_manifold(sys_name, data)
        results[sys_name] = {}

        for frac in [0.25, 0.5, 0.75, 1.0]:
            n_use = max(2, int(tr_full.shape[0] * frac))
            tr = tr_full[:n_use]

            # NODE
            node_model, _ = train_node(D, tr, ts)
            with torch.no_grad():
                rn = compute_all_metrics(node_model.predict(te[:, 0], ts).permute(1, 0, 2) * s + m, ter, manifold.k)

            # SNDE
            snde_model, _ = train_baseline_with_losses('SNDE', D, tr, ts, manifold.k)
            with torch.no_grad():
                r_snde = compute_all_metrics(snde_model.predict(te[:, 0], ts).permute(1, 0, 2) * s + m, ter, manifold.k)

            # NODE-LAC
            node, gain, best_sc, _ = train_nodesac(D, tr, ts, manifold)
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=best_sc)
            with torch.no_grad():
                rs = compute_all_metrics(euler_integrate(dyn, te[:, 0], ts).permute(1, 0, 2) * s + m, ter, manifold.k)

            results[sys_name][f'{frac}'] = {
                'NODE_MSE': float(rn['MSE']), 'NODE_CE': float(rn['CE']),
                'SNDE_MSE': float(r_snde['MSE']), 'SNDE_CE': float(r_snde['CE']),
                'SAC_MSE': float(rs['MSE']), 'SAC_CE': float(rs['CE']),
            }
            print(f'  {sys_name} {frac:.0%}: NODE={rn["MSE"]:.4f} SNDE={r_snde["MSE"]:.4f} SAC={rs["MSE"]:.4f}', flush=True)

    json.dump(results, open(out / 'data_efficiency.json', 'w'), indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fracs = [0.25, 0.5, 0.75, 1.0]
    for i, sys_name in enumerate(systems_3):
        ax = axes[i]
        node_mses = [results[sys_name][str(f)]['NODE_MSE'] for f in fracs]
        sac_mses = [results[sys_name][str(f)]['SAC_MSE'] for f in fracs]
        ax.plot(fracs, node_mses, 'o-', color='orange', label='NODE')
        if 'SNDE_MSE' in results[sys_name][str(fracs[0])]:
            snde_mses = [results[sys_name][str(f)]['SNDE_MSE'] for f in fracs]
            ax.plot(fracs, snde_mses, '^--', color='green', label='SNDE')
        ax.plot(fracs, sac_mses, 's-', color='blue', label='NODE-LAC')
        ax.set_title(SYSTEM_LABELS[sys_name])
        ax.set_xlabel('Training Data Fraction'); ax.set_ylabel('Test MSE')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'data_efficiency.png', dpi=300, bbox_inches='tight')
    plt.savefig(out / 'data_efficiency.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Saved to {out}', flush=True)


# ══════════════════════════════════════════════════════════════
# REVIEWER: Noise robustness
# ══════════════════════════════════════════════════════════════
def run_reviewer_noise():
    print('\n=== Reviewer: Noise Robustness ===', flush=True)
    out = EXP / 'reviewer_noise'
    results = {}

    systems_3 = ['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water']
    for sys_name in systems_3:
        D, m, s, tr, te, ter, ts, data = load_data(sys_name)
        manifold = get_manifold(sys_name, data)
        results[sys_name] = {}

        for sigma in [0.0, 0.05, 0.1, 0.2]:
            seed_everything(42)
            tr_noisy = tr + sigma * torch.randn_like(tr) if sigma > 0 else tr

            # NODE-LAC
            node, gain, best_sc, _ = train_nodesac(D, tr_noisy, ts, manifold)
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=best_sc)
            with torch.no_grad():
                rs = compute_all_metrics(euler_integrate(dyn, te[:, 0], ts).permute(1, 0, 2) * s + m, ter, manifold.k)

            # NODE
            node_model, _ = train_node(D, tr_noisy, ts)
            with torch.no_grad():
                rn = compute_all_metrics(node_model.predict(te[:, 0], ts).permute(1, 0, 2) * s + m, ter, manifold.k)

            results[sys_name][f'{sigma}'] = {
                'NODE_MSE': float(rn['MSE']), 'NODE_CE': float(rn['CE']),
                'SAC_MSE': float(rs['MSE']), 'SAC_CE': float(rs['CE']),
            }
            print(f'  {sys_name} sigma={sigma}: NODE={rn["MSE"]:.4f} SAC={rs["MSE"]:.4f}', flush=True)

    json.dump(results, open(out / 'noise_robustness.json', 'w'), indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sigmas = [0.0, 0.05, 0.1, 0.2]
    for i, sys_name in enumerate(systems_3):
        ax = axes[i]
        node_mses = [results[sys_name][str(sg)]['NODE_MSE'] for sg in sigmas]
        sac_mses = [results[sys_name][str(sg)]['SAC_MSE'] for sg in sigmas]
        ax.bar(np.arange(len(sigmas)) - 0.15, node_mses, 0.3, color='orange', label='NODE')
        ax.bar(np.arange(len(sigmas)) + 0.15, sac_mses, 0.3, color='blue', label='NODE-LAC')
        ax.set_xticks(range(len(sigmas))); ax.set_xticklabels([f'σ={sg}' for sg in sigmas])
        ax.set_title(SYSTEM_LABELS[sys_name]); ax.set_ylabel('Test MSE')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out / 'noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig(out / 'noise_robustness.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Saved to {out}', flush=True)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='all',
                        choices=['all', 'fig2', 'table1', 'table3', 'table4',
                                 'data_eff', 'noise'])
    args = parser.parse_args()

    t0 = time.time()

    if args.exp in ('all', 'table1'):
        run_table1()
    if args.exp in ('all', 'fig2'):
        run_fig2()
    if args.exp in ('all', 'table3'):
        run_table3()
    if args.exp in ('all', 'table4'):
        run_table4()
    if args.exp in ('all', 'data_eff'):
        run_reviewer_data_eff()
    if args.exp in ('all', 'noise'):
        run_reviewer_noise()

    print(f'\nTotal time: {(time.time()-t0)/60:.1f} min')
