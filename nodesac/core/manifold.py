"""Constraint residuals and gradient-based correction utilities."""

import torch
import torch.nn as nn


class ConstraintManifold:
    """Base interface for residuals defining the target set {x: k(x) = 0}."""

    def k(self, x):
        """Return the constraint residual, which vanishes on the target set."""
        raise NotImplementedError

    def jacobian(self, x):
        """Gradient of sum(k(x)) with respect to x, retaining the autograd graph."""
        x_req = x.detach().requires_grad_(True)
        kx = self.k(x_req)
        J = torch.autograd.grad(kx.sum(), x_req, create_graph=True)[0]
        return J

    def distance(self, x):
        """Return squared residual norm ||k(x)||^2 for each state."""
        kx = self.k(x)
        return (kx ** 2).sum(dim=-1)

    def project(self, x, n_steps=5, lr=0.1):
        """Apply n_steps of gradient descent to the summed squared residual norm."""
        xp = x.clone().detach().requires_grad_(True)
        for _ in range(n_steps):
            loss = self.distance(xp).sum()
            grad = torch.autograd.grad(loss, xp)[0]
            xp = (xp - lr * grad).detach().requires_grad_(True)
        return xp.detach()


class FHNManifold(ConstraintManifold):
    """Threshold violations for a two-field state and appended memory value.

    The residuals are relu(E - e_threshold) and
    relu(mean(u^2 + v^2) - 2). Thresholds use the coordinates supplied to k.
    """

    def __init__(self, n_grid=8, gamma=0.5, e_threshold=6.1):
        self.n_grid = n_grid
        self.gamma = gamma
        self.e_threshold = e_threshold

    def k(self, x):
        """Return two threshold violations for x shaped (..., 2*n_grid + 1)."""
        E = x[..., -1]
        u = x[..., :self.n_grid]
        v = x[..., self.n_grid:2*self.n_grid]
        # Memory-coordinate threshold.
        k1 = torch.relu(E - self.e_threshold)
        # Mean squared-amplitude threshold.
        state_energy = (u.pow(2) + v.pow(2)).mean(dim=-1)
        k2 = torch.relu(state_energy - 2.0)
        return torch.stack([k1, k2], dim=-1)


class LVManifold(ConstraintManifold):
    """Ratio, positivity, and optional biomass residuals for two fields.

    The ratio is mean(v) / mean(u), with each mean clamped below at 1e-6.
    Biomass is dx * sum(u + v); its residual is zero when biomass_max is None.
    """

    def __init__(self, n_grid=15, ratio_lo=0.45, ratio_hi=0.60,
                 biomass_max=None, dx=None, **kwargs):
        self.n_grid = n_grid
        self.ratio_lo = ratio_lo
        self.ratio_hi = ratio_hi
        self.biomass_max = biomass_max
        self.dx = dx or 1.0 / n_grid

    def k(self, x):
        """x: (..., 2*n_grid)  [u(n_grid), v(n_grid)]."""
        u = x[..., :self.n_grid]
        v = x[..., self.n_grid:2 * self.n_grid]

        # k1: Bounds on the ratio of field means.
        u_mean = u.mean(dim=-1).clamp(min=1e-6)
        v_mean = v.mean(dim=-1).clamp(min=1e-6)
        ratio = v_mean / u_mean
        k1 = torch.relu(self.ratio_lo - ratio) + torch.relu(ratio - self.ratio_hi)

        # k2: Positivity
        k2 = torch.relu(-u).mean(dim=-1) + torch.relu(-v).mean(dim=-1)

        # k3: Optional biomass upper bound.
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
    """Positive-part residual of a mean squared-amplitude upper bound."""

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
    """Amplitude violation and deviation from an optional quadratic reference.

    k1 is relu(mean(eta^2 + u^2) - amp_threshold).
    k2 is abs(E - energy_ref), or zero when energy_ref is None.
    """

    def __init__(self, n_grid=50, amp_threshold=1.0, energy_ref=None, **kwargs):
        self.n_grid = n_grid
        self.amp_threshold = amp_threshold
        self.energy_ref = energy_ref

    def k(self, x):
        eta = x[..., :self.n_grid]
        u = x[..., self.n_grid:2 * self.n_grid]
        dx = 1.0 / self.n_grid

        # k1: amplitude bound
        amp = (eta.pow(2) + u.pow(2)).mean(dim=-1)
        k1 = torch.relu(amp - self.amp_threshold)

        # k2: Absolute deviation from the supplied quadratic reference.
        E = 0.5 * (9.81 * eta.pow(2) + 1.0 * u.pow(2)).sum(dim=-1) * dx
        if self.energy_ref is not None:
            k2 = (E - self.energy_ref).abs()
        else:
            k2 = torch.zeros_like(k1)

        return torch.stack([k1, k2], dim=-1)


class SWManifold(ConstraintManifold):
    """Residuals for weighted spectral energy and cross-scale interaction.

    The spectral energy uses weights [1, 0.5, 0.25] and target e_total.
    The second residual is the cross-scale interaction Phi.
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
        """Return two residuals for x shaped (..., 2*n_grid), with fields eta and u."""
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
    """Amplitude and velocity thresholds for a 14-coordinate robot state.

    State ordering is [q(7), qdot(7)]. The two residuals bound
    mean(q^2 + qdot^2) and mean(qdot^2), respectively.
    """

    def __init__(self, amp_threshold=3.0, vel_threshold=2.0):
        self.amp_threshold = amp_threshold
        self.vel_threshold = vel_threshold

    def k(self, x):
        """x: (..., 14) [q(7), qdot(7)]."""
        q = x[..., :7]
        qdot = x[..., 7:14]

        # k1: Mean squared position-and-velocity bound.
        amp = (q.pow(2) + qdot.pow(2)).mean(dim=-1)
        k1 = torch.relu(amp - self.amp_threshold)

        # k2: Mean squared-velocity bound.
        vel_energy = qdot.pow(2).mean(dim=-1)
        k2 = torch.relu(vel_energy - self.vel_threshold)

        return torch.stack([k1, k2], dim=-1)
