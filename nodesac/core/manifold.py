"""Constraint manifold definitions for each physical system."""

import torch
import torch.nn as nn


class ConstraintManifold:
    """Base class for constraint manifolds L = {x : k(x) = 0}."""

    def k(self, x):
        """Constraint function. Returns 0 on the manifold."""
        raise NotImplementedError

    def jacobian(self, x):
        """Jacobian of k w.r.t. x using autograd."""
        x_req = x.detach().requires_grad_(True)
        kx = self.k(x_req)
        J = torch.autograd.grad(kx.sum(), x_req, create_graph=True)[0]
        return J

    def distance(self, x):
        """Squared distance to manifold: ||k(x)||^2."""
        kx = self.k(x)
        return (kx ** 2).sum(dim=-1)

    def project(self, x, n_steps=5, lr=0.1):
        """Iterative projection onto manifold via gradient descent on ||k(x)||^2."""
        xp = x.clone().detach().requires_grad_(True)
        for _ in range(n_steps):
            loss = self.distance(xp).sum()
            grad = torch.autograd.grad(loss, xp)[0]
            xp = (xp - lr * grad).detach().requires_grad_(True)
        return xp.detach()


class FHNManifold(ConstraintManifold):
    """FitzHugh-Nagumo memory-dependent energy constraint.

    Energy E(t) evolves as dE/dt = (u^2 + v^2) - gamma*E.
    Constraint: E(t) <= E_threshold.
    k(x) = max(0, E - E_threshold).

    For the discretized system, state includes the energy variable E appended.
    """

    def __init__(self, n_grid=8, gamma=0.5, e_threshold=6.1):
        self.n_grid = n_grid
        self.gamma = gamma
        self.e_threshold = e_threshold

    def k(self, x):
        """x: (..., 2*n_grid + 1) where last dim is E.

        Constraint includes:
        1. Energy bound: relu(E - threshold)
        2. State norm bound: mean(u^2 + v^2) soft limit
        """
        E = x[..., -1]
        u = x[..., :self.n_grid]
        v = x[..., self.n_grid:2*self.n_grid]
        # Energy constraint
        k1 = torch.relu(E - self.e_threshold)
        # State amplitude constraint (soft, based on cubic nonlinearity stability)
        state_energy = (u.pow(2) + v.pow(2)).mean(dim=-1)
        k2 = torch.relu(state_energy - 2.0)  # tighter bound
        return torch.stack([k1, k2], dim=-1)


class LVManifold(ConstraintManifold):
    """Lotka-Volterra ecological constraints.

    Three biologically motivated constraints that align with trajectory accuracy:
    k1: Ratio stability — v/u should stay within [ratio_lo, ratio_hi]
        (NODE drift amplifies ratio deviation → correction reduces both CE and MSE)
    k2: Positivity — populations must be non-negative
        (NODE can predict negative populations at long horizons)
    k3: Biomass bound — total (u+v) should not exceed carrying capacity
        (NODE overpredicts growth → correction pulls back)
    """

    def __init__(self, n_grid=15, ratio_lo=0.45, ratio_hi=0.60,
                 biomass_max=None, dx=None, **kwargs):
        self.n_grid = n_grid
        self.ratio_lo = ratio_lo
        self.ratio_hi = ratio_hi
        self.biomass_max = biomass_max  # auto-calibrated from data
        self.dx = dx or 1.0 / n_grid

    def k(self, x):
        """x: (..., 2*n_grid)  [u(n_grid), v(n_grid)]."""
        u = x[..., :self.n_grid]
        v = x[..., self.n_grid:2 * self.n_grid]

        # k1: Predator-prey ratio constraint (tight bounds from data)
        u_mean = u.mean(dim=-1).clamp(min=1e-6)
        v_mean = v.mean(dim=-1).clamp(min=1e-6)
        ratio = v_mean / u_mean
        k1 = torch.relu(self.ratio_lo - ratio) + torch.relu(ratio - self.ratio_hi)

        # k2: Positivity
        k2 = torch.relu(-u).mean(dim=-1) + torch.relu(-v).mean(dim=-1)

        # k3: Biomass carrying capacity
        if self.biomass_max is not None:
            biomass = (u + v).sum(dim=-1) * self.dx
            k3 = torch.relu(biomass - self.biomass_max)
        else:
            k3 = torch.zeros_like(k1)

        return torch.stack([k1, k2, k3], dim=-1)


class LVManifoldRatio(ConstraintManifold):
    """LV constraint: predator-prey ratio only."""

    def __init__(self, n_grid=15, ratio_lo=0.1, ratio_hi=1.5, **kwargs):
        self.n_grid = n_grid
        self.ratio_lo = ratio_lo
        self.ratio_hi = ratio_hi

    def k(self, x):
        u = x[..., :self.n_grid]
        v = x[..., self.n_grid:2 * self.n_grid]
        u_mean = u.mean(dim=-1).clamp(min=1e-6)
        v_mean = v.mean(dim=-1).clamp(min=1e-6)
        ratio = v_mean / u_mean
        k1 = torch.relu(self.ratio_lo - ratio) + torch.relu(ratio - self.ratio_hi)
        return k1.unsqueeze(-1)


class LVManifoldPositivity(ConstraintManifold):
    """LV constraint: population positivity only."""

    def __init__(self, n_grid=15, **kwargs):
        self.n_grid = n_grid

    def k(self, x):
        u = x[..., :self.n_grid]
        v = x[..., self.n_grid:2 * self.n_grid]
        k1 = torch.relu(-u).mean(dim=-1)
        k2 = torch.relu(-v).mean(dim=-1)
        return torch.stack([k1, k2], dim=-1)


class LVManifoldSmooth(ConstraintManifold):
    """LV constraint: spatial smoothness only."""

    def __init__(self, n_grid=15, smooth_threshold=2.0, dx=None, **kwargs):
        self.n_grid = n_grid
        self.smooth_threshold = smooth_threshold
        self.dx = dx or 1.0 / n_grid

    def k(self, x):
        u = x[..., :self.n_grid]
        v = x[..., self.n_grid:2 * self.n_grid]
        du = (torch.roll(u, -1, -1) - torch.roll(u, 1, -1)) / (2 * self.dx)
        dv = (torch.roll(v, -1, -1) - torch.roll(v, 1, -1)) / (2 * self.dx)
        grad_norm = (du.pow(2) + dv.pow(2)).mean(dim=-1).sqrt()
        k1 = torch.relu(grad_norm - self.smooth_threshold)
        return k1.unsqueeze(-1)


class LVManifoldAmplitude(ConstraintManifold):
    """LV constraint: state amplitude bound (FHN-style).

    When NODE drifts, population amplitudes grow beyond physical range.
    Correction pulls amplitudes back → also reduces trajectory error.
    """

    def __init__(self, n_grid=15, amp_threshold=16.0, **kwargs):
        self.n_grid = n_grid
        self.amp_threshold = amp_threshold

    def k(self, x):
        u = x[..., :self.n_grid]
        v = x[..., self.n_grid:2 * self.n_grid]
        amp = (u.pow(2) + v.pow(2)).mean(dim=-1)
        k1 = torch.relu(amp - self.amp_threshold)
        return k1.unsqueeze(-1)


class SWManifoldAmplitude(ConstraintManifold):
    """SW constraint: state amplitude + energy conservation (FHN-style).

    k1: amplitude bound — prevents NODE from overshooting wave heights
    k2: energy drift — total energy should be approximately conserved
    """

    def __init__(self, n_grid=50, amp_threshold=1.0, energy_ref=None, **kwargs):
        self.n_grid = n_grid
        self.amp_threshold = amp_threshold
        self.energy_ref = energy_ref  # auto-calibrated from initial energy

    def k(self, x):
        eta = x[..., :self.n_grid]
        u = x[..., self.n_grid:2 * self.n_grid]
        dx = 1.0 / self.n_grid

        # k1: amplitude bound
        amp = (eta.pow(2) + u.pow(2)).mean(dim=-1)
        k1 = torch.relu(amp - self.amp_threshold)

        # k2: energy conservation (drift from reference)
        E = 0.5 * (9.81 * eta.pow(2) + 1.0 * u.pow(2)).sum(dim=-1) * dx
        if self.energy_ref is not None:
            k2 = (E - self.energy_ref).abs()
        else:
            k2 = torch.zeros_like(k1)

        return torch.stack([k1, k2], dim=-1)


class SWManifold(ConstraintManifold):
    """Shallow Water multi-scale energy conservation.

    Sum_i w_i * E_i(t) = E_total, with w=[1, 0.5, 0.25], E_total=0.1.
    Also cross-scale interaction Phi=0.
    """

    def __init__(self, n_grid=50, g=9.81, H=1.0, e_total=0.1):
        self.n_grid = n_grid
        self.g = g
        self.H = H
        self.e_total = e_total
        self.weights = [1.0, 0.5, 0.25]
        self.n_scales = 3

    def _decompose_scales(self, eta, u):
        """Spectral decomposition into 3 scales."""
        n = eta.shape[-1]
        eta_f = torch.fft.rfft(eta, dim=-1)
        u_f = torch.fft.rfft(u, dim=-1)
        freqs = eta_f.shape[-1]
        third = freqs // 3
        etas, us = [], []
        for i in range(3):
            mask = torch.zeros(freqs, device=eta.device, dtype=eta.dtype)
            lo = i * third
            hi = (i + 1) * third if i < 2 else freqs
            mask[lo:hi] = 1.0
            etas.append(torch.fft.irfft(eta_f * mask, n=n, dim=-1))
            us.append(torch.fft.irfft(u_f * mask, n=n, dim=-1))
        return etas, us

    def k(self, x):
        """x: (..., 2*n_grid) [eta(50), u(50)]."""
        eta = x[..., :self.n_grid]
        u = x[..., self.n_grid:2 * self.n_grid]
        etas, us = self._decompose_scales(eta, u)
        dx = 1.0 / self.n_grid
        total_e = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for i in range(3):
            Ei = 0.5 * (self.g * (etas[i] ** 2) + self.H * (us[i] ** 2)).sum(dim=-1) * dx
            total_e = total_e + self.weights[i] * Ei
        energy_err = (total_e - self.e_total).unsqueeze(-1)
        # Cross-scale: Phi = integral(d_x(eta1)*eta2 - eta3*d_x(u1))dx
        deta1 = torch.diff(etas[0], dim=-1, prepend=etas[0][..., -1:]) / dx
        du1 = torch.diff(us[0], dim=-1, prepend=us[0][..., -1:]) / dx
        n_min = min(deta1.shape[-1], etas[1].shape[-1], etas[2].shape[-1], du1.shape[-1])
        phi = (deta1[..., :n_min] * etas[1][..., :n_min] -
               etas[2][..., :n_min] * du1[..., :n_min]).sum(dim=-1) * dx
        return torch.stack([energy_err.squeeze(-1), phi], dim=-1)


class RobotManifold(ConstraintManifold):
    """Robot state amplitude + velocity constraints (FHN-style).

    State: [q(7), q_dot(7)] = 14D
    k1: state amplitude — mean(q^2 + qdot^2) bounded
        When NODE drifts, joint angles and velocities grow → k1 triggers
    k2: velocity smoothness — mean(qdot^2) bounded
        Prevents velocity blowup which is the main failure mode
    """

    def __init__(self, amp_threshold=3.0, vel_threshold=2.0):
        self.amp_threshold = amp_threshold
        self.vel_threshold = vel_threshold

    def k(self, x):
        """x: (..., 14) [q(7), qdot(7)]."""
        q = x[..., :7]
        qdot = x[..., 7:14]

        # k1: state amplitude bound (like FHN k2)
        amp = (q.pow(2) + qdot.pow(2)).mean(dim=-1)
        k1 = torch.relu(amp - self.amp_threshold)

        # k2: velocity energy bound
        vel_energy = qdot.pow(2).mean(dim=-1)
        k2 = torch.relu(vel_energy - self.vel_threshold)

        return torch.stack([k1, k2], dim=-1)
