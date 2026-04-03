"""FitzHugh-Nagumo model with memory-dependent energy constraints."""

import numpy as np
import torch
from scipy.integrate import solve_ivp
from .base import DynamicalSystem
from ..core.manifold import FHNManifold


class FitzHughNagumo(DynamicalSystem):
    """FitzHugh-Nagumo reaction-diffusion PDE discretized on spatial grid.

    du/dt = c1^2 * d2u/dx2 - a1*u^3 + b1*u*v + I(t)
    dv/dt = c2^2 * d2v/dx2 - a2*v^3 + b2*v*u + J(t)

    With memory-dependent energy constraint:
    dE/dt = mean(u^2 + v^2) - gamma*E, E <= E_threshold
    """

    def __init__(self, n_grid=8, c1=1.0, c2=1.2, a1=5.05, a2=7.05,
                 b1=0.01, b2=0.1, dx=0.1, gamma=0.5, e_threshold=6.1,
                 dtype=torch.float64, device='cpu'):
        super().__init__(dtype, device)
        self.n_grid = n_grid
        self.c1, self.c2 = c1, c2
        self.a1, self.a2 = a1, a2
        self.b1, self.b2 = b1, b2
        self.dx = dx
        self.gamma = gamma
        self.e_threshold = e_threshold
        self.state_dim = 2 * n_grid + 1  # u(n), v(n), E

    def _laplacian(self, y):
        """1D Laplacian with periodic boundary."""
        return (np.roll(y, 1) + np.roll(y, -1) - 2 * y) / (self.dx ** 2)

    def dynamics(self, t, state):
        u = state[:self.n_grid]
        v = state[self.n_grid:2 * self.n_grid]
        E = state[-1]

        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        I_t = 0.1 * np.sin(0.5 * t)
        J_t = 0.05 * np.cos(0.3 * t)

        du = self.c1 ** 2 * lap_u - self.a1 * u ** 3 + self.b1 * u * v + I_t
        dv = self.c2 ** 2 * lap_v - self.a2 * v ** 3 + self.b2 * v * u + J_t

        u2v2 = np.mean(u ** 2 + v ** 2)
        dE = u2v2 - self.gamma * E

        return np.concatenate([du, dv, [dE]])

    def sample_initial_condition(self, rng):
        u0 = rng.uniform(-0.5, 1.5, self.n_grid)
        v0 = rng.uniform(-0.2, 0.8, self.n_grid)
        E0 = np.array([np.mean(u0 ** 2 + v0 ** 2) / self.gamma])
        return np.concatenate([u0, v0, E0])

    def constraint_fn(self, state):
        return self.get_manifold().k(state)

    def get_manifold(self):
        return FHNManifold(self.n_grid, self.gamma, self.e_threshold)

    @classmethod
    def get_default_config(cls):
        return dict(n_grid=8, c1=1.0, c2=1.2, a1=5.05, a2=7.05,
                    b1=0.01, b2=0.1, dx=0.1, gamma=0.5, e_threshold=6.1)
