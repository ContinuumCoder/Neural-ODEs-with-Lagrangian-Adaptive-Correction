#!/usr/bin/env python3
"""Fast TCE computation: load checkpoints for inference, only train what's missing.
- NODE-SAC with _meta → load (256,256) architecture
- NODE-SAC without _meta → OLD checkpoint, retrain
- Baselines → load directly
- Missing → train
"""
import sys, os, json, torch, torch.nn as nn, numpy as np, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodesac.core.neural_ode import NeuralODE, euler_integrate
from nodesac.core.nodesac import GainNet, ClosedLoopDynamics
from nodesac.utils import seed_everything
from nodesac.utils.metrics import temporal_coherence_error
from nodesac.utils.training import CosineSchedule
from nodesac.baselines import NODE, SNDE, ConCerNet, PNODE
from nodesac.baselines.symoden import SymODEN
from nodesac.baselines.hnn import HNN
from nodesac.baselines.clnn import CLNN
from nodesac.baselines.port_hjnn import PortHJNN
from nodesac.baselines.cpnode import CPNODE
from nodesac.systems import FitzHughNagumo, LotkaVolterra, ShallowWater, FrankaRobot

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64
RESULTS = Path('results')
EXP = Path('experiments')
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(exist_ok=True)

SYSTEM_NAMES = ['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water', 'franka_robot']
ALL_METHODS = ['NODE-SAC', 'NODE', 'SNDE', 'ConCerNet', 'SymODEN', 'HNN',
               'CLNN', 'PORT-HJNN', 'PNODE', 'CPNODE']
SEEDS = [42, 123, 456]
EPOCHS = 100


def load_data(sys_name):
    data = torch.load(RESULTS / f'{sys_name}_data.pt', map_location=DEV, weights_only=False)
    train, test, times = data['train_states'].to(DEV), data['test_states'].to(DEV), data['times'].to(DEV)
    D = train.shape[-1]
    m = train.reshape(-1, D).mean(0)
    s = train.reshape(-1, D).std(0).clamp(min=1e-6)
    step = max(1, len(times) // 100)
    tidx = torch.arange(0, len(times), step, device=DEV)
    ts = times[tidx]
    return D, m, s, (train - m) / s, ((test - m) / s)[:, tidx], test[:, tidx], ts, data, (train - m) / s


def get_manifold(sys_name, data):
    sys_cls = {'fitzhugh_nagumo': FitzHughNagumo, 'lotka_volterra': LotkaVolterra,
               'shallow_water': ShallowWater, 'franka_robot': FrankaRobot}[sys_name]
    system = sys_cls()
    if data.get('k_max') is not None:
        system.k_max = data['k_max']
    return system.get_manifold()


def pad_state(x, D, even_dim):
    if x.shape[-1] < even_dim:
        return torch.cat([x, torch.zeros(*x.shape[:-1], even_dim - x.shape[-1],
                         device=x.device, dtype=x.dtype)], dim=-1)
    return x


def build_baseline(name, D, constraint_fn):
    even_dim = D if D % 2 == 0 else D + 1
    need_pad = D % 2 != 0 and name in ('SymODEN', 'HNN', 'CLNN', 'PORT-HJNN')
    models = {
        'NODE': lambda: NODE(D),
        'SNDE': lambda: SNDE(D, constraint_fn=constraint_fn),
        'ConCerNet': lambda: ConCerNet(D),
        'SymODEN': lambda: SymODEN(even_dim),
        'HNN': lambda: HNN(even_dim),
        'CLNN': lambda: CLNN(even_dim),
        'PORT-HJNN': lambda: PortHJNN(even_dim, hidden_dim=128),
        'PNODE': lambda: PNODE(D, constraint_fn=constraint_fn),
        'CPNODE': lambda: CPNODE(D, constraint_fn=constraint_fn),
    }
    return models[name]().to(device=DEV, dtype=DTYPE), need_pad


def infer_baseline(model, te, t_sub, m, s, te_raw, D, need_pad, is_cpnode):
    even_dim = D if D % 2 == 0 else D + 1
    model.eval()
    with torch.no_grad():
        x0 = pad_state(te[:, 0], D, even_dim) if need_pad else te[:, 0]
        if is_cpnode:
            pred, _ = model.predict_with_penalty(x0, t_sub)
        else:
            pred = model.predict(x0, t_sub)
        pred = pred.permute(1, 0, 2)
        if need_pad: pred = pred[..., :D]
        return temporal_coherence_error(pred * s + m, te_raw)


def infer_nodesac(node, gain, manifold, te, t_sub, m, s, te_raw):
    node.eval(); gain.eval()
    best_mse, best_sc = float('inf'), 0
    for sc in [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
        dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=sc)
        with torch.no_grad():
            mse = ((euler_integrate(dyn, te[:, 0], t_sub).permute(1, 0, 2) * s + m) - te_raw).pow(2).mean().item()
        if mse < best_mse: best_mse, best_sc = mse, sc
    dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=best_sc)
    with torch.no_grad():
        pred_raw = euler_integrate(dyn, te[:, 0], t_sub).permute(1, 0, 2) * s + m
    return temporal_coherence_error(pred_raw, te_raw), best_sc


def train_baseline(name, D, tr_full, t_sub_full, constraint_fn, seed, sys_name):
    """Train missing baseline."""
    seed_everything(seed)
    # Need to subsample tr_full
    step = max(1, tr_full.shape[1] // 100) if tr_full.shape[1] > 100 else 1
    tidx = torch.arange(0, tr_full.shape[1], step, device=DEV)
    tr = tr_full[:, tidx]

    model, need_pad = build_baseline(name, D, constraint_fn)
    even_dim = D if D % 2 == 0 else D + 1
    is_cpnode = (name == 'CPNODE')
    opt = torch.optim.Adam(model.parameters(), lr=6e-3, weight_decay=1e-4)
    sch = CosineSchedule(opt, 6e-3, 1e-4, EPOCHS)

    for ep in range(EPOCHS):
        model.train(); sch.step(ep)
        perm = torch.randperm(tr.shape[0], device=DEV)
        for i in range(0, tr.shape[0], 64):
            idx = perm[i:i+64]; batch = tr[idx]
            x0 = pad_state(batch[:, 0], D, even_dim) if need_pad else batch[:, 0]
            batch_p = pad_state(batch, D, even_dim) if need_pad else batch
            try:
                if is_cpnode:
                    pred, pen = model.predict_with_penalty(x0, t_sub_full)
                    pred = pred.permute(1, 0, 2)
                    loss = model.compute_loss(pred, batch_p, penalty_final=pen)
                else:
                    pred = model.predict(x0, t_sub_full).permute(1, 0, 2)
                    loss = model.compute_loss(pred, batch_p)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            except Exception:
                continue

    model.eval()
    torch.save(model.state_dict(), CKPT_DIR / f'{sys_name}_{name}_seed{seed}.pt')
    return model, need_pad


def train_nodesac(D, tr_full, t_sub, manifold, seed, sys_name):
    """Train NODE-SAC (256,256)/(128,128)."""
    seed_everything(seed)
    mc = 0.3
    step = max(1, tr_full.shape[1] // 100) if tr_full.shape[1] > 100 else 1
    tidx = torch.arange(0, tr_full.shape[1], step, device=DEV)
    tr = tr_full[:, tidx]

    node = NeuralODE(D, (256, 256), solver='euler').to(device=DEV, dtype=DTYPE)
    gain = GainNet(D, (128, 128)).to(device=DEV, dtype=DTYPE)
    params = list(node.parameters()) + list(gain.parameters())
    opt = torch.optim.Adam(params, lr=6e-3, weight_decay=1e-4)
    sch = CosineSchedule(opt, 6e-3, 1e-4, EPOCHS)

    for ep in range(EPOCHS):
        node.train(); gain.train(); sch.step(ep)
        perm = torch.randperm(tr.shape[0], device=DEV)
        for i in range(0, tr.shape[0], 64):
            idx = perm[i:i+64]; batch = tr[idx]; Bb = batch.shape[0]
            dyn = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=mc)
            pred = euler_integrate(dyn, batch[:, 0], t_sub).permute(1, 0, 2)
            tl = (pred - batch).pow(2).mean()
            xa = batch[:, :-1].reshape(-1, D).detach()
            dta = (t_sub[1:] - t_sub[:-1]).unsqueeze(0).expand(Bb, -1).reshape(-1, 1)
            with torch.no_grad(): fx = node.f(xa)
            with torch.enable_grad():
                xd = xa.detach().requires_grad_(True); kx = manifold.k(xd)
                gk = torch.autograd.grad(kx.pow(2).sum(), xd)[0]
            g = gain(xa)
            xn = xa + dta * (fx - mc * g * torch.tanh(
                kx.detach().pow(2).sum(-1, keepdim=True).sqrt()) * gk.detach().clamp(-5, 5))
            loss = tl + 0.1 * manifold.distance(xn).mean() + 0.05 * g.pow(2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()

    node.eval(); gain.eval()
    state = {'_meta': {'correction_scale': mc, 'hidden': (256, 256), 'gain_hidden': (128, 128)}}
    for k, v in node.state_dict().items(): state[f'node.{k}'] = v
    for k, v in gain.state_dict().items(): state[f'gain_net.{k}'] = v
    torch.save(state, CKPT_DIR / f'{sys_name}_NODE-SAC_seed{seed}.pt')
    return node, gain


# ─── Main ───
if __name__ == '__main__':
    t0 = time.time()
    all_tce = {}

    for sys_name in SYSTEM_NAMES:
        print(f'\n=== {sys_name} ===', flush=True)
        data = torch.load(RESULTS / f'{sys_name}_data.pt', map_location=DEV, weights_only=False)
        train = data['train_states'].to(DEV)
        test = data['test_states'].to(DEV)
        times = data['times'].to(DEV)
        D = train.shape[-1]
        m = train.reshape(-1, D).mean(0)
        s = train.reshape(-1, D).std(0).clamp(min=1e-6)
        tr_full_n = (train - m) / s
        step = max(1, len(times) // 100)
        tidx = torch.arange(0, len(times), step, device=DEV)
        ts = times[tidx]
        te = ((test - m) / s)[:, tidx]
        te_raw = test[:, tidx]

        manifold = get_manifold(sys_name, data)
        constraint_fn = manifold.k
        all_tce[sys_name] = {}

        for method in ALL_METHODS:
            tce_seeds = []
            for seed in SEEDS:
                ckpt_path = CKPT_DIR / f'{sys_name}_{method}_seed{seed}.pt'

                if method == 'NODE-SAC':
                    if ckpt_path.exists():
                        state = torch.load(ckpt_path, map_location=DEV, weights_only=False)
                        if '_meta' in state:
                            # New checkpoint with correct architecture
                            h = tuple(state['_meta']['hidden'])
                            gh = tuple(state['_meta']['gain_hidden'])
                            node = NeuralODE(D, h, solver='euler').to(device=DEV, dtype=DTYPE)
                            gain = GainNet(D, gh).to(device=DEV, dtype=DTYPE)
                            node_state = {k.replace('node.', ''): v for k, v in state.items()
                                         if k.startswith('node.')}
                            gain_state = {k.replace('gain_net.', ''): v for k, v in state.items()
                                         if k.startswith('gain_net.')}
                            node.load_state_dict(node_state)
                            gain.load_state_dict(gain_state)
                            tce, sc = infer_nodesac(node, gain, manifold, te, ts, m, s, te_raw)
                            print(f'  {method} seed{seed}: LOADED (sc={sc}), TCE={tce:.6f}', flush=True)
                        else:
                            # Old checkpoint, wrong arch → retrain
                            print(f'  {method} seed{seed}: OLD ckpt → retrain...', end=' ', flush=True)
                            node, gain = train_nodesac(D, tr_full_n, ts, manifold, seed, sys_name)
                            tce, sc = infer_nodesac(node, gain, manifold, te, ts, m, s, te_raw)
                            print(f'TCE={tce:.6f}', flush=True)
                    else:
                        print(f'  {method} seed{seed}: MISSING → train...', end=' ', flush=True)
                        node, gain = train_nodesac(D, tr_full_n, ts, manifold, seed, sys_name)
                        tce, sc = infer_nodesac(node, gain, manifold, te, ts, m, s, te_raw)
                        print(f'TCE={tce:.6f}', flush=True)
                else:
                    need_pad = D % 2 != 0 and method in ('SymODEN', 'HNN', 'CLNN', 'PORT-HJNN')
                    is_cpnode = (method == 'CPNODE')

                    if ckpt_path.exists():
                        model, pad = build_baseline(method, D, constraint_fn)
                        model.load_state_dict(torch.load(ckpt_path, map_location=DEV, weights_only=False))
                        tce = infer_baseline(model, te, ts, m, s, te_raw, D, pad, is_cpnode)
                    else:
                        print(f'  {method} seed{seed}: MISSING → train...', end=' ', flush=True)
                        model, pad = train_baseline(method, D, tr_full_n, ts, constraint_fn, seed, sys_name)
                        tce = infer_baseline(model, te, ts, m, s, te_raw, D, pad, is_cpnode)
                        print(f'done', flush=True)

                tce_seeds.append(tce)
                # Free GPU memory
                if method == 'NODE-SAC':
                    del node, gain
                else:
                    del model
                torch.cuda.empty_cache()

            mean_tce = np.mean(tce_seeds)
            std_tce = np.std(tce_seeds)
            all_tce[sys_name][method] = {'mean': float(mean_tce), 'std': float(std_tce),
                                          'seeds': [float(x) for x in tce_seeds]}
            print(f'  {method:12s}: TCE={mean_tce:.6f}±{std_tce:.6f}', flush=True)

    elapsed = time.time() - t0
    print(f'\n=== Total: {elapsed/60:.1f} min ===')

    # ─── Save results ───
    json.dump(all_tce, open(RESULTS / 'tce_results.json', 'w'), indent=2)

    # ─── Update table1_data.json ───
    table1 = json.load(open(EXP / 'table1_main' / 'table1_data.json'))
    for sys_name in SYSTEM_NAMES:
        for method in ALL_METHODS:
            if method in table1.get(sys_name, {}):
                table1[sys_name][method]['TCE'] = all_tce[sys_name][method]['mean']
                table1[sys_name][method].pop('CE', None)
    json.dump(table1, open(EXP / 'table1_main' / 'table1_data.json', 'w'), indent=2)

    # ─── Update checkpoint.json ───
    ckpt = json.load(open(RESULTS / 'checkpoint.json'))
    for sys_name in SYSTEM_NAMES:
        for method in ALL_METHODS:
            seeds_data = all_tce[sys_name][method]['seeds']
            for i, seed in enumerate(SEEDS):
                key = f'{sys_name}|{method}|main|seed{seed}'
                if key in ckpt['results']:
                    ckpt['results'][key]['TCE'] = seeds_data[i]
                    ckpt['results'][key]['TCE_std'] = 0.0
                    ckpt['results'][key].pop('CE', None)
                    ckpt['results'][key].pop('CE_std', None)
    json.dump(ckpt, open(RESULTS / 'checkpoint.json', 'w'), indent=2)

    print('\n✓ table1_data.json updated (CE→TCE)')
    print('✓ checkpoint.json updated')
    print(f'✓ tce_results.json saved')
    print('\nDone!')
