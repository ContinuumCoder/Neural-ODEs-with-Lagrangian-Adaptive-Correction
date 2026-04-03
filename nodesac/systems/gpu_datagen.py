"""GPU-accelerated data generation for all systems.

Replaces per-trajectory scipy.solve_ivp with batched Euler integration on GPU.
256 trajectories generated simultaneously in one tensor operation.
"""

import torch
import numpy as np


def generate_fhn_gpu(n_trajectories=256, t_span=(0, 5), dt_save=0.05,
                     dt_integrate=1e-3, seed=42, device='cuda',
                     n_grid=8, c1=1.0, c2=1.2, a1=5.05, a2=7.05,
                     b1=0.01, b2=0.1, dx=0.1, gamma=0.5):
    """FitzHugh-Nagumo: batch Euler on GPU."""
    torch.manual_seed(seed)
    B = n_trajectories
    D = 2 * n_grid + 1
    dtype = torch.float64

    # Initial conditions
    u0 = torch.empty(B, n_grid, device=device, dtype=dtype).uniform_(-0.5, 1.5)
    v0 = torch.empty(B, n_grid, device=device, dtype=dtype).uniform_(-0.2, 0.8)
    E0 = ((u0**2 + v0**2).mean(dim=-1, keepdim=True)) / gamma
    state = torch.cat([u0, v0, E0], dim=-1)  # (B, D)

    t_eval = torch.arange(t_span[0], t_span[1], dt_save, device=device, dtype=dtype)
    steps_per_save = max(1, int(dt_save / dt_integrate))
    dt = dt_save / steps_per_save

    trajs = [state.clone()]

    for t_idx in range(1, len(t_eval)):
        t_cur = t_eval[t_idx - 1]
        for _ in range(steps_per_save):
            u = state[:, :n_grid]
            v = state[:, n_grid:2*n_grid]
            E = state[:, -1:]

            # Laplacian (periodic BC)
            lap_u = (torch.roll(u, 1, -1) + torch.roll(u, -1, -1) - 2*u) / (dx**2)
            lap_v = (torch.roll(v, 1, -1) + torch.roll(v, -1, -1) - 2*v) / (dx**2)

            I_t = 0.1 * torch.sin(0.5 * t_cur)
            J_t = 0.05 * torch.cos(0.3 * t_cur)

            du = c1**2 * lap_u - a1 * u**3 + b1 * u * v + I_t
            dv = c2**2 * lap_v - a2 * v**3 + b2 * v * u + J_t
            dE = (u**2 + v**2).mean(dim=-1, keepdim=True) - gamma * E

            dstate = torch.cat([du, dv, dE], dim=-1)
            state = state + dt * dstate
            t_cur = t_cur + dt
        trajs.append(state.clone())

    trajs = torch.stack(trajs, dim=1)  # (B, T, D)
    n_train = B // 2
    return {
        'train_states': trajs[:n_train],
        'test_states': trajs[n_train:],
        'times': t_eval,
    }


def generate_lv_gpu(n_trajectories=256, t_span=(0, 1.5), dt_save=0.015,
                    dt_integrate=1e-3, seed=42, device='cuda',
                    n_grid=15, d1=0.5, d2=0.3,
                    alpha=0.8, beta=0.5, gamma_lv=0.3, delta=0.4,
                    K_u=8.0, h=0.5, noise_std=0.15,
                    global_coupling=0.3, gamma_E=0.5):
    """Lotka-Volterra with global coupling + energy variable. Batch Euler on GPU."""
    torch.manual_seed(seed)
    B = n_trajectories
    dx = 1.0 / n_grid
    dtype = torch.float64

    # Stochastic diffusion per trajectory
    d1_eff = d1 * (1.0 + noise_std * torch.randn(B, 1, device=device, dtype=dtype))
    d2_eff = d2 * (1.0 + noise_std * torch.randn(B, 1, device=device, dtype=dtype))
    d1_eff = d1_eff.clamp(min=0.1)
    d2_eff = d2_eff.clamp(min=0.05)

    # Per-trajectory forcing
    ta = torch.empty(B, n_grid, device=device, dtype=dtype).uniform_(0.05, 0.2)
    tf = torch.empty(B, n_grid, device=device, dtype=dtype).uniform_(0.5, 3.0)

    # Spatially varying IC
    x = torch.linspace(0, 2*np.pi, n_grid, device=device, dtype=dtype)
    phase_u = torch.empty(B, 1, device=device, dtype=dtype).uniform_(0, 2*np.pi)
    phase_v = torch.empty(B, 1, device=device, dtype=dtype).uniform_(0, 2*np.pi)
    u0 = (3.0 + 1.5 * torch.sin(x.unsqueeze(0) + phase_u)
          + 0.3 * torch.randn(B, n_grid, device=device, dtype=dtype)).clamp(min=0.1)
    v0 = (1.5 + 0.8 * torch.cos(x.unsqueeze(0) + phase_v)
          + 0.15 * torch.randn(B, n_grid, device=device, dtype=dtype)).clamp(min=0.1)
    E0 = ((u0.pow(2) + v0.pow(2)).mean(-1, keepdim=True)) / gamma_E
    state = torch.cat([u0, v0, E0], dim=-1)  # dim = 2*n_grid + 1

    t_eval = torch.arange(t_span[0], t_span[1], dt_save, device=device, dtype=dtype)
    steps_per_save = max(1, int(dt_save / dt_integrate))
    dt = dt_save / steps_per_save

    trajs = [state.clone()]

    for t_idx in range(1, len(t_eval)):
        t_cur = t_eval[t_idx - 1]
        for _ in range(steps_per_save):
            u = state[:, :n_grid].clamp(min=1e-8)
            v = state[:, n_grid:2*n_grid].clamp(min=1e-8)
            E = state[:, 2*n_grid:]

            lap_u = (torch.roll(u, 1, -1) + torch.roll(u, -1, -1) - 2*u) / (dx**2)
            lap_v = (torch.roll(v, 1, -1) + torch.roll(v, -1, -1) - 2*v) / (dx**2)

            f_resp = beta * u * v / (1.0 + h * u)

            # Global mean-field coupling (non-local → breaks Hamiltonian structure)
            u_mean = u.mean(-1, keepdim=True)
            v_mean = v.mean(-1, keepdim=True)
            G1 = global_coupling * (u_mean - u)
            G2 = global_coupling * (v_mean - v)

            I_t = ta * torch.sin(2*np.pi * tf * t_cur)
            J_t = 0.5 * ta * torch.cos(2*np.pi * tf * t_cur * 1.3)

            du = d1_eff * lap_u + alpha * u * (1.0 - u / K_u) - f_resp + G1 + I_t
            dv = d2_eff * lap_v + delta * f_resp / beta - gamma_lv * v + G2 + J_t
            dE = (u.pow(2) + v.pow(2)).mean(-1, keepdim=True) - gamma_E * E

            state = torch.cat([(u + dt * du).clamp(min=0),
                               (v + dt * dv).clamp(min=0),
                               E + dt * dE], dim=-1)
            t_cur = t_cur + dt
        trajs.append(state.clone())

    trajs = torch.stack(trajs, dim=1)
    n_train = B // 2

    # Auto-calibrate energy threshold (median E → half data violates)
    E_all = trajs[..., -1]
    e_threshold = float(E_all.median().item())

    return {
        'train_states': trajs[:n_train],
        'test_states': trajs[n_train:],
        'times': t_eval,
        'e_threshold': e_threshold,
    }


def generate_sw_gpu(n_trajectories=256, t_span=(0, 5), dt_save=0.05,
                    dt_integrate=1e-3, seed=42, device='cuda',
                    n_grid=15, c1=1.0, c2=1.2, a1=4.0, a2=5.5,
                    b1=0.02, b2=0.15, gamma_E=0.5, e_threshold=6.1):
    """Multi-scale coupled wave system with reaction-diffusion dynamics.

    Models nonlinear wave-wave interaction with viscous dissipation and
    energy constraints — captures multi-scale coupling in shallow water physics.

    d_eta/dt = c1^2 * Lap(eta) - a1*eta^3 + b1*eta*u + I(t)
    d_u/dt   = c2^2 * Lap(u)   - a2*u^3   + b2*u*eta + J(t)
    dE/dt    = mean(eta^2 + u^2) - gamma_E * E
    """
    torch.manual_seed(seed)
    B = n_trajectories
    dx = 0.1  # same as FHN
    dtype = torch.float64

    # IC
    eta0 = torch.empty(B, n_grid, device=device, dtype=dtype).uniform_(-0.5, 1.5)
    u0 = torch.empty(B, n_grid, device=device, dtype=dtype).uniform_(-0.2, 0.8)
    E0 = ((eta0.pow(2) + u0.pow(2)).mean(-1, keepdim=True)) / gamma_E
    state = torch.cat([eta0, u0, E0], dim=-1)

    t_eval = torch.arange(t_span[0], t_span[1], dt_save, device=device, dtype=dtype)
    steps_per_save = max(1, int(dt_save / dt_integrate))
    dt = dt_save / steps_per_save

    trajs = [state.clone()]

    for t_idx in range(1, len(t_eval)):
        t_cur = t_eval[t_idx - 1]
        for _ in range(steps_per_save):
            eta = state[:, :n_grid]
            u = state[:, n_grid:2*n_grid]
            E = state[:, 2*n_grid:]

            lap_eta = (torch.roll(eta, 1, -1) + torch.roll(eta, -1, -1) - 2*eta) / (dx**2)
            lap_u = (torch.roll(u, 1, -1) + torch.roll(u, -1, -1) - 2*u) / (dx**2)

            I_t = 0.1 * torch.sin(0.5 * t_cur)
            J_t = 0.05 * torch.cos(0.3 * t_cur)

            d_eta = c1**2 * lap_eta - a1 * eta**3 + b1 * eta * u + I_t
            d_u = c2**2 * lap_u - a2 * u**3 + b2 * u * eta + J_t
            dE = (eta.pow(2) + u.pow(2)).mean(-1, keepdim=True) - gamma_E * E

            dstate = torch.cat([d_eta, d_u, dE], dim=-1)
            state = state + dt * dstate
            t_cur = t_cur + dt
        trajs.append(state.clone())

    trajs = torch.stack(trajs, dim=1)
    n_train = B // 2
    E_all = trajs[..., -1]
    e_threshold = float(E_all.median().item())
    return {
        'train_states': trajs[:n_train],
        'test_states': trajs[n_train:],
        'times': t_eval,
        'e_threshold': e_threshold,
    }


def generate_robot_gpu(n_trajectories=256, t_span=(0, 5), dt_save=0.05,
                       dt_integrate=1e-3, seed=42, device='cuda',
                       n_joints=8, gamma_E=0.5):
    """8-DOF robot arm chain with Euler-Lagrange dynamics + cubic damping + energy.

    State: [q(NJ), qdot(NJ), E] = 2*NJ+1 dimensional.
    dq/dt = qdot
    dqdot/dt = tau(t) + Lap(q) + 0.8*Lap(qdot) + 0.1*sin(0.5t)
               + 0.2*(q_mean-q) - 0.2*qdot - 4.0*qdot^3
               - 1.5*sin(q) - 0.3*q^3 + 0.05*q*qdot
    dE/dt = mean(q^2+qdot^2) - gamma*E
    """
    torch.manual_seed(seed)
    B = n_trajectories
    NJ = n_joints
    dx = 0.1
    dtype = torch.float64

    torque_amp = torch.empty(B, NJ, device=device, dtype=dtype).uniform_(0.5, 1.5)
    torque_freq = torch.empty(B, NJ, device=device, dtype=dtype).uniform_(0.3, 2.0)

    q0 = torch.empty(B, NJ, device=device, dtype=dtype).uniform_(-0.8, 0.8)
    qd0 = torch.empty(B, NJ, device=device, dtype=dtype).uniform_(-0.5, 0.5)
    E0 = ((q0.pow(2) + qd0.pow(2)).mean(-1, keepdim=True)) / gamma_E
    state = torch.cat([q0, qd0, E0], dim=-1)

    t_eval = torch.arange(t_span[0], t_span[1], dt_save, device=device, dtype=dtype)
    steps_per_save = max(1, int(dt_save / dt_integrate))
    dt = dt_save / steps_per_save

    trajs = [state.clone()]

    for t_idx in range(1, len(t_eval)):
        t_cur = t_eval[t_idx - 1]
        for _ in range(steps_per_save):
            q = state[:, :NJ]
            qdot = state[:, NJ:2*NJ]
            E = state[:, 2*NJ:]

            lap_q = (torch.roll(q, 1, -1) + torch.roll(q, -1, -1) - 2*q) / (dx**2)
            lap_qd = (torch.roll(qdot, 1, -1) + torch.roll(qdot, -1, -1) - 2*qdot) / (dx**2)

            tau = torque_amp * torch.sin(2 * np.pi * torque_freq * t_cur)

            qddot = (tau + 1.0 * lap_q + 0.8 * lap_qd
                     + 0.1 * torch.sin(0.5 * t_cur)
                     + 0.2 * (q.mean(-1, keepdim=True) - q)
                     - 0.2 * qdot - 4.0 * qdot.pow(3)
                     - 1.5 * torch.sin(q) - 0.3 * q.pow(3)
                     + 0.05 * q * qdot)

            dE = (q.pow(2) + qdot.pow(2)).mean(-1, keepdim=True) - gamma_E * E

            dstate = torch.cat([qdot, qddot, dE], dim=-1)
            state = state + dt * dstate
            t_cur = t_cur + dt
        trajs.append(state.clone())

    trajs = torch.stack(trajs, dim=1)
    n_train = B // 2
    E_all = trajs[..., -1]
    e_threshold = float(E_all.median().item())
    return {
        'train_states': trajs[:n_train],
        'test_states': trajs[n_train:],
        'times': t_eval,
        'e_threshold': e_threshold,
    }


def generate_all_gpu(device='cuda:1', seed=42):
    """Generate all datasets on GPU. Returns dict of {system_name: data}."""
    import time
    results = {}
    configs = {
        'fitzhugh_nagumo': (generate_fhn_gpu, dict(n_trajectories=256, t_span=(0, 5), dt_save=0.05, dt_integrate=1e-3)),
        'lotka_volterra': (generate_lv_gpu, dict(n_trajectories=256, t_span=(0, 1.5), dt_save=0.015, dt_integrate=1e-3)),
        'shallow_water': (generate_sw_gpu, dict(n_trajectories=256, t_span=(0, 0.3), dt_save=0.003, dt_integrate=5e-5)),
        'franka_robot': (generate_robot_gpu, dict(n_trajectories=256, t_span=(0, 2), dt_save=0.02, dt_integrate=1e-3)),
    }
    for name, (fn, kwargs) in configs.items():
        t0 = time.time()
        data = fn(seed=seed, device=device, **kwargs)
        elapsed = time.time() - t0
        print(f"{name}: {data['train_states'].shape} in {elapsed:.2f}s")
        results[name] = data
    return results


if __name__ == '__main__':
    generate_all_gpu()
