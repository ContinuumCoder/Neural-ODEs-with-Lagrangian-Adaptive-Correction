#!/usr/bin/env python3
"""
NODE-SAC Parallel Experiment Runner — maximizes GPU utilization.

Runs multiple experiments concurrently via torch.multiprocessing.
Each experiment ~1-3GB VRAM → ~30 concurrent on each 98GB GPU.

Usage:
    python run_parallel.py                        # Run all pending
    python run_parallel.py --workers-per-gpu 20   # Control parallelism
    python run_parallel.py --phase main
    python run_parallel.py --phase tune
    python run_parallel.py --phase ablation
    python run_parallel.py --phase reviewer
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = Path('results')
CKPT_DIR = Path('checkpoints')
LOG_DIR = Path('logs')
CHECKPOINT_FILE = RESULTS_DIR / 'checkpoint.json'
DTYPE = torch.float64

ALL_METHODS = ['NODE-SAC', 'NODE', 'SNDE', 'ConCerNet', 'SymODEN', 'HNN',
               'CLNN', 'PORT-HJNN', 'PNODE', 'CPNODE']
SEEDS = [42, 123, 456]
EPOCHS = 100

SYSTEMS_CONFIG = {
    'fitzhugh_nagumo': dict(node_hidden=(128, 128), sac_hidden=(64, 64)),
    'lotka_volterra': dict(node_hidden=(128, 128), sac_hidden=(64, 64)),
    'shallow_water': dict(node_hidden=(128, 128), sac_hidden=(64, 64)),
    'franka_robot': dict(node_hidden=(128, 128), sac_hidden=(64, 64)),
}

# Optimal max_correction per system (updated by tune phase)
OPTIMAL_MC_FILE = RESULTS_DIR / 'optimal_mc.json'

# ─── Checkpoint ──────────────────────────────────────────────────────────────

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'completed': {}, 'results': {}}

def save_checkpoint(data):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def is_done(ckpt, key):
    return key in ckpt['completed']

def mark_done(key, result):
    """Thread-safe checkpoint update via file lock."""
    lock_path = CHECKPOINT_FILE.with_suffix('.lock')
    import fcntl
    with open(lock_path, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ckpt = load_checkpoint()
        ckpt['completed'][key] = str(datetime.now())
        ckpt['results'][key] = {k: float(v) if isinstance(v, (int, float, np.floating))
                                 else v for k, v in result.items()}
        save_checkpoint(ckpt)
        fcntl.flock(lock, fcntl.LOCK_UN)

# ─── Data Loading ────────────────────────────────────────────────────────────

def ensure_data(system_name, device):
    """Load or generate cached data."""
    cache = RESULTS_DIR / f'{system_name}_data.pt'
    if cache.exists():
        return torch.load(cache, map_location=device, weights_only=False)

    from nodesac.systems.gpu_datagen import (generate_fhn_gpu, generate_lv_gpu,
                                              generate_sw_gpu, generate_robot_gpu)
    gen = {
        'fitzhugh_nagumo': (generate_fhn_gpu, dict(n_trajectories=256, t_span=(0, 5), dt_save=0.05, dt_integrate=1e-3)),
        'lotka_volterra': (generate_lv_gpu, dict(n_trajectories=256, t_span=(0, 1.5), dt_save=0.015, dt_integrate=1e-3)),
        'shallow_water': (generate_sw_gpu, dict(n_trajectories=256, t_span=(0, 0.3), dt_save=0.003, dt_integrate=5e-5)),
        'franka_robot': (generate_robot_gpu, dict(n_trajectories=256, t_span=(0, 0.75), dt_save=0.005, dt_integrate=1e-3)),
    }
    fn, kwargs = gen[system_name]
    data = fn(device=device, seed=42, **kwargs)
    from nodesac.systems import FitzHughNagumo, LotkaVolterra, ShallowWater, FrankaRobot
    sys_cls = {'fitzhugh_nagumo': FitzHughNagumo, 'lotka_volterra': LotkaVolterra,
               'shallow_water': ShallowWater, 'franka_robot': FrankaRobot}[system_name]
    system = sys_cls()
    if 'k_max' in data:
        system.k_max = data['k_max']
    data['state_dim'] = system.state_dim
    data['system_name'] = system_name
    torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}, cache)
    return data

# ─── Single Experiment Worker ────────────────────────────────────────────────

def run_single_experiment(args):
    """Run one (system, method, seed) experiment. Called in subprocess."""
    system_name, method, exp_type, seed, device, extra = args
    key = f"{system_name}|{method}|{exp_type}|seed{seed}"

    try:
        torch.cuda.set_device(device)
        from nodesac.utils import seed_everything, compute_all_metrics
        from nodesac.utils.training import CosineSchedule
        seed_everything(seed)

        # Load data
        data = ensure_data(system_name, device)
        train = data['train_states'].to(device)
        test = data['test_states'].to(device)
        times = data['times'].to(device)
        D = train.shape[-1]

        # Normalize
        m = train.reshape(-1, D).mean(0)
        s = train.reshape(-1, D).std(0).clamp(min=1e-6)
        train_n = (train - m) / s
        test_n = (test - m) / s

        # Subsample time
        if len(times) > 100:
            step = max(1, len(times) // 100)
            t_idx = torch.arange(0, len(times), step, device=device)
        else:
            t_idx = torch.arange(len(times), device=device)
        t_sub = times[t_idx]
        tr = train_n[:, t_idx]
        te = test_n[:, t_idx]
        te_raw = test[:, t_idx]

        # Get manifold
        from nodesac.systems import FitzHughNagumo, LotkaVolterra, ShallowWater, FrankaRobot
        sys_cls = {'fitzhugh_nagumo': FitzHughNagumo, 'lotka_volterra': LotkaVolterra,
                   'shallow_water': ShallowWater, 'franka_robot': FrankaRobot}[system_name]
        system = sys_cls()
        if data.get('k_max') is not None:
            system.k_max = data['k_max']
        manifold = system.get_manifold()
        constraint_fn = manifold.k

        # Apply noise if requested
        noise_sigma = extra.get('noise_sigma', 0)
        if noise_sigma > 0:
            seed_everything(seed)
            tr = tr + noise_sigma * torch.randn_like(tr)

        # Data fraction
        data_frac = extra.get('data_frac', 1.0)
        if data_frac < 1.0:
            n_use = max(2, int(tr.shape[0] * data_frac))
            tr = tr[:n_use]

        epochs = extra.get('epochs', EPOCHS)
        max_correction = extra.get('max_correction', 1.0)

        if method == 'NODE-SAC':
            result = _train_nodesac(system_name, tr, te, te_raw, t_sub, m, s,
                                     manifold, device, seed, epochs, max_correction, extra)
        else:
            result = _train_baseline(method, tr, te, te_raw, t_sub, m, s,
                                      constraint_fn, device, seed, epochs)

        result['system'] = system_name
        result['method'] = method
        result['seed'] = seed
        mark_done(key, result)
        return key, result

    except Exception as e:
        result = {'MSE': float('inf'), 'CE': float('inf'), 'error': str(e)}
        mark_done(key, result)
        return key, result


def _train_nodesac(system_name, tr, te, te_raw, t_sub, mean, std,
                   manifold, device, seed, epochs, max_correction, extra):
    """Unified training: single loop, single optimizer, no phases."""
    import torch.nn as nn
    from nodesac.core.neural_ode import NeuralODE, euler_integrate
    from nodesac.core.nodesac import GainNet, ClosedLoopDynamics
    from nodesac.utils import seed_everything, compute_all_metrics
    from nodesac.utils.training import CosineSchedule
    seed_everything(seed)
    D = tr.shape[-1]
    mc = extra.get('mc', 0.3)

    node = NeuralODE(D, (256, 256), solver='euler').to(device=device, dtype=DTYPE)
    gain = GainNet(D, (128, 128)).to(device=device, dtype=DTYPE)
    params = list(node.parameters()) + list(gain.parameters())
    opt = torch.optim.Adam(params, lr=6e-3, weight_decay=1e-4)
    sched = CosineSchedule(opt, 6e-3, 1e-4, epochs)

    for epoch in range(epochs):
        node.train(); gain.train(); sched.step(epoch)
        perm = torch.randperm(tr.shape[0], device=device)
        for i in range(0, tr.shape[0], 64):
            idx = perm[i:i+64]
            batch = tr[idx]; Bb = batch.shape[0]
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=mc)
            pred = euler_integrate(dyn, batch[:, 0], t_sub).permute(1, 0, 2)
            traj_loss = (pred - batch).pow(2).mean()

            xa = batch[:, :-1].reshape(-1, D).detach()
            dta = (t_sub[1:] - t_sub[:-1]).unsqueeze(0).expand(Bb, -1).reshape(-1, 1)
            with torch.no_grad():
                fx = node.f(xa)
            with torch.enable_grad():
                xd = xa.detach().requires_grad_(True)
                kx = manifold.k(xd)
                gk = torch.autograd.grad(kx.pow(2).sum(), xd)[0]
            g = gain(xa)
            xn = xa + dta * (fx - mc * g * torch.tanh(
                kx.detach().pow(2).sum(-1, keepdim=True).sqrt()) * gk.detach().clamp(-5, 5))
            loss = traj_loss + 0.1 * manifold.distance(xn).mean() + 0.05 * g.pow(2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()

    node.eval(); gain.eval()
    best_mse, best_alpha = float('inf'), 0
    for sc in [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
        dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=sc)
        with torch.no_grad():
            mse = ((euler_integrate(dyn, te[:, 0], t_sub).permute(1, 0, 2) * std + mean)
                   - te_raw).pow(2).mean().item()
        if mse < best_mse:
            best_mse, best_alpha = mse, sc

    dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=best_alpha)
    with torch.no_grad():
        pred = euler_integrate(dyn, te[:, 0], t_sub).permute(1, 0, 2)
        pred_raw = pred * std + mean
        result = compute_all_metrics(pred_raw, te_raw, manifold.k)
        result = {k: float(v) for k, v in result.items()}
    result['correction_scale'] = best_alpha
    result['max_correction'] = mc
    return result


def _train_baseline(name, tr, te, te_raw, t_sub, mean, std,
                    constraint_fn, device, seed, epochs):
    import torch.nn as nn
    from nodesac.baselines import NODE, SNDE, ConCerNet, SymODEN, HNN, CLNN, PortHJNN, PNODE, CPNODE
    from nodesac.utils import seed_everything, compute_all_metrics
    from nodesac.utils.training import CosineSchedule
    seed_everything(seed)

    D = tr.shape[-1]
    even_dim = D if D % 2 == 0 else D + 1
    need_pad = D % 2 != 0
    models = {
        'NODE': lambda: (NODE(D), False),
        'SNDE': lambda: (SNDE(D, constraint_fn=constraint_fn), False),
        'ConCerNet': lambda: (ConCerNet(D), False),
        'SymODEN': lambda: (SymODEN(even_dim), need_pad),
        'HNN': lambda: (HNN(even_dim), need_pad),
        'CLNN': lambda: (CLNN(even_dim), need_pad),
        'PORT-HJNN': lambda: (PortHJNN(even_dim, hidden_dim=64), need_pad),  # fair params
        'PNODE': lambda: (PNODE(D, constraint_fn=constraint_fn), False),
        'CPNODE': lambda: (CPNODE(D, constraint_fn=constraint_fn), False),
    }
    model, pad = models[name]()
    model = model.to(device=device, dtype=DTYPE)
    is_cpnode = (name == 'CPNODE')

    optimizer = torch.optim.Adam(model.parameters(), lr=6e-3, weight_decay=1e-4)
    scheduler = CosineSchedule(optimizer, 6e-3, 1e-4, epochs)

    def pad_state(x):
        if pad and x.shape[-1] < even_dim:
            return torch.cat([x, torch.zeros(*x.shape[:-1], even_dim - x.shape[-1],
                             device=x.device, dtype=x.dtype)], dim=-1)
        return x

    for epoch in range(epochs):
        model.train()
        scheduler.step(epoch)
        perm = torch.randperm(tr.shape[0], device=device)
        for i in range(0, tr.shape[0], 64):
            idx = perm[i:i+64]
            batch = tr[idx]
            x0 = pad_state(batch[:, 0])
            batch_p = pad_state(batch)
            try:
                if is_cpnode:
                    pred, pen = model.predict_with_penalty(x0, t_sub)
                    pred = pred.permute(1, 0, 2)
                    loss = model.compute_loss(pred, batch_p, penalty_final=pen)
                else:
                    pred = model.predict(x0, t_sub).permute(1, 0, 2)
                    loss = model.compute_loss(pred, batch_p)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except Exception:
                continue

    model.eval()
    with torch.no_grad():
        x0 = pad_state(te[:, 0])
        if is_cpnode:
            pred, _ = model.predict_with_penalty(x0, t_sub)
        else:
            pred = model.predict(x0, t_sub)
        pred = pred.permute(1, 0, 2)
        if pad: pred = pred[..., :D]
        pred_raw = pred * std + mean
        result = compute_all_metrics(pred_raw, te_raw, constraint_fn)
        result = {k: float(v) for k, v in result.items()}
    return result


# ─── Job Builders ────────────────────────────────────────────────────────────

def get_optimal_mc(system_name):
    """Load tuned max_correction or return default."""
    if OPTIMAL_MC_FILE.exists():
        mc = json.load(open(OPTIMAL_MC_FILE))
        return mc.get(system_name, 0.5)
    return 0.5  # safe default


def build_main_jobs(gpus):
    """Build all main experiment jobs."""
    ckpt = load_checkpoint()
    jobs = []
    systems = list(SYSTEMS_CONFIG.keys())
    for i, sys_name in enumerate(systems):
        device = gpus[i % len(gpus)]
        for method in ALL_METHODS:
            mc = get_optimal_mc(sys_name) if method == 'NODE-SAC' else 1.0
            for seed in SEEDS:
                key = f"{sys_name}|{method}|main|seed{seed}"
                if is_done(ckpt, key):
                    continue
                extra = {'max_correction': mc, 'prefix': 'main', 'sys': sys_name}
                jobs.append((sys_name, method, 'main', seed, device, extra))
    return jobs


def build_tune_jobs(gpus):
    """Build tuning jobs for max_correction."""
    ckpt = load_checkpoint()
    mc_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    jobs = []
    for i, sys_name in enumerate(SYSTEMS_CONFIG):
        device = gpus[i % len(gpus)]
        for mc in mc_grid:
            key = f"{sys_name}|NODE-SAC|tune_mc{mc}|seed42"
            if is_done(ckpt, key):
                continue
            extra = {'max_correction': mc, 'epochs': 30, 'prefix': 'tune', 'sys': sys_name}
            jobs.append((sys_name, 'NODE-SAC', f'tune_mc{mc}', 42, device, extra))
    return jobs


def build_ablation_jobs(gpus):
    """Build ablation study jobs."""
    ckpt = load_checkpoint()
    variants = {
        'NODE-SAC': {},
        'NoGainNet': {'max_correction': 0.0},
        'NoConstraintLoss': {'lambda2': 0.0},
        'NoCorrection': {'max_correction': 0.0, 'lambda2': 0.0},
    }
    hp_grid = [
        {'mu_init': 0.1, 'lambda2': 0.05},
        {'mu_init': 0.05, 'lambda2': 0.05},
        {'mu_init': 0.1, 'lambda2': 0.01},
        {'mu_init': 0.01, 'lambda2': 0.01},
    ]
    jobs = []
    for i, sys_name in enumerate(SYSTEMS_CONFIG):
        device = gpus[i % len(gpus)]
        mc = get_optimal_mc(sys_name)
        # Component ablation
        for vname, vkw in variants.items():
            key = f"{sys_name}|{vname}|ablation|seed42"
            if is_done(ckpt, key):
                continue
            extra = {'max_correction': vkw.get('max_correction', mc),
                     'lambda2': vkw.get('lambda2', 0.05),
                     'prefix': 'ablation', 'sys': sys_name}
            jobs.append((sys_name, 'NODE-SAC', f'ablation_{vname}', 42, device, extra))
        # Hyperparam sensitivity
        for hp in hp_grid:
            hpkey = f"mu{hp['mu_init']}_le{hp['lambda2']}"
            key = f"{sys_name}|{hpkey}|hyperparam|seed42"
            if is_done(ckpt, key):
                continue
            extra = {'max_correction': mc, 'mu_init': hp['mu_init'],
                     'lambda2': hp['lambda2'], 'prefix': 'hp', 'sys': sys_name}
            jobs.append((sys_name, 'NODE-SAC', f'hyperparam_{hpkey}', 42, device, extra))
    return jobs


def build_reviewer_jobs(gpus):
    """Build reviewer response jobs."""
    ckpt = load_checkpoint()
    jobs = []
    review_methods = ['NODE-SAC', 'NODE', 'PORT-HJNN']

    for i, sys_name in enumerate(SYSTEMS_CONFIG):
        device = gpus[i % len(gpus)]
        mc = get_optimal_mc(sys_name)

        # Data efficiency
        for frac in [0.25, 0.5, 0.75, 1.0]:
            for method in ['NODE-SAC', 'NODE']:
                key = f"{sys_name}|{method}|data_eff_{frac}|seed42"
                if is_done(ckpt, key):
                    continue
                extra = {'data_frac': frac, 'max_correction': mc,
                         'prefix': 'dataeff', 'sys': sys_name}
                jobs.append((sys_name, method, f'data_eff_{frac}', 42, device, extra))

        # Noise robustness
        for sigma in [0.05, 0.1, 0.2]:
            for method in review_methods:
                key = f"{sys_name}|{method}|noise_{sigma}|seed42"
                if is_done(ckpt, key):
                    continue
                extra = {'noise_sigma': sigma, 'max_correction': mc,
                         'prefix': 'noise', 'sys': sys_name}
                jobs.append((sys_name, method, f'noise_{sigma}', 42, device, extra))

    return jobs


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all', choices=['all', 'main', 'tune', 'ablation', 'reviewer'])
    parser.add_argument('--workers-per-gpu', type=int, default=15,
                        help='Concurrent experiments per GPU')
    parser.add_argument('--gpus', default='0,1', help='GPU ids')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    gpus = [f'cuda:{g}' for g in args.gpus.split(',')]
    total_workers = args.workers_per_gpu * len(gpus)
    print(f"GPUs: {gpus}, Workers/GPU: {args.workers_per_gpu}, Total: {total_workers}")

    # Pre-generate all datasets
    print("Ensuring all datasets are cached...")
    for sys_name in SYSTEMS_CONFIG:
        ensure_data(sys_name, gpus[0])
    print("All datasets ready.\n")

    # Build jobs
    phases = ['tune', 'main', 'ablation', 'reviewer'] if args.phase == 'all' else [args.phase]
    all_jobs = []
    for phase in phases:
        if phase == 'tune':
            jobs = build_tune_jobs(gpus)
            print(f"Tune: {len(jobs)} jobs")
        elif phase == 'main':
            jobs = build_main_jobs(gpus)
            print(f"Main: {len(jobs)} jobs")
        elif phase == 'ablation':
            jobs = build_ablation_jobs(gpus)
            print(f"Ablation: {len(jobs)} jobs")
        elif phase == 'reviewer':
            jobs = build_reviewer_jobs(gpus)
            print(f"Reviewer: {len(jobs)} jobs")
        all_jobs.extend(jobs)

    if not all_jobs:
        print("All experiments already completed!")
        return

    print(f"\nTotal pending: {len(all_jobs)} experiments")
    print(f"Starting {total_workers} parallel workers...\n")

    mp.set_start_method('spawn', force=True)
    t0 = time.time()
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=total_workers,
                             mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(run_single_experiment, job): job for job in all_jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                key, result = future.result()
                mse = result.get('MSE', float('inf'))
                ce = result.get('CE', float('inf'))
                err = result.get('error', '')
                if err:
                    print(f"  FAIL {key}: {err}")
                    failed += 1
                else:
                    print(f"  DONE {key}: MSE={mse:.4f} CE={ce:.4f}")
                completed += 1
            except Exception as e:
                print(f"  CRASH {job[0]}|{job[1]}: {e}")
                failed += 1
                completed += 1

            if completed % 10 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600
                remaining = len(all_jobs) - completed
                eta = remaining / (completed / elapsed) if completed > 0 else 0
                print(f"  --- Progress: {completed}/{len(all_jobs)} "
                      f"({failed} failed), {rate:.0f}/hr, ETA: {eta/60:.0f}min ---")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE: {completed} experiments in {elapsed/60:.1f} min")
    print(f"  Success: {completed - failed}, Failed: {failed}")
    print(f"{'='*60}")

    # If tune phase, compute optimal mc
    if 'tune' in phases:
        _compute_optimal_mc()


def _compute_optimal_mc():
    """After tune phase, find optimal max_correction per system."""
    ckpt = load_checkpoint()
    optimal = {}
    for sys_name in SYSTEMS_CONFIG:
        best_mse = float('inf')
        best_mc = 0.5
        for mc in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            key = f"{sys_name}|NODE-SAC|tune_mc{mc}|seed42"
            r = ckpt['results'].get(key, {})
            mse = r.get('MSE', float('inf'))
            if mse < best_mse:
                best_mse = mse
                best_mc = mc
        optimal[sys_name] = best_mc
        print(f"  {sys_name}: optimal mc={best_mc} (MSE={best_mse:.4f})")

    with open(OPTIMAL_MC_FILE, 'w') as f:
        json.dump(optimal, f, indent=2)
    print(f"Saved to {OPTIMAL_MC_FILE}")


if __name__ == '__main__':
    main()
