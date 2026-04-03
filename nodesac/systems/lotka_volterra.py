"""Lotka-Volterra ecosystem with global coupling and energy tracking.

Reaction-diffusion PDE with mean-field interaction:
  du/dt = D1*Lap(u) + alpha*u*(1-u/K) - beta*u*v/(1+h*u) + G1(u_mean) + I(t)
  dv/dt = D2*Lap(v) + delta*u*v/(1+h*u) - gamma*v + G2(v_mean) + J(t)
  dE/dt = mean(u^2+v^2) - gamma_E*E

Global mean-field coupling G1, G2 breaks locality → PORT-HJNN's Hamiltonian
prior is less effective. Energy variable E enables FHN-style constraint.
"""

import numpy as np
import torch
from .base import DynamicalSystem
from ..core.manifold import FHNManifold


class LotkaVolterra(DynamicalSystem):

    def __init__(self, n_grid=15, d1=0.5, d2=0.3,
                 alpha=0.8, beta=0.5, gamma_lv=0.3, delta=0.4,
                 K_u=8.0, h=0.5, gamma_E=0.5,
                 global_coupling=0.3, noise_std=0.15,
                 dtype=torch.float64, device='cpu'):
        super().__init__(dtype, device)
        self.n_grid = n_grid
        self.d1, self.d2 = d1, d2
        self.alpha, self.beta = alpha, beta
        self.gamma_lv, self.delta = gamma_lv, delta
        self.K_u, self.h = K_u, h
        self.gamma_E = gamma_E
        self.global_coupling = global_coupling
        self.noise_std = noise_std
        self.dx = 1.0 / n_grid
        self.state_dim = 2 * n_grid + 1  # u(n) + v(n) + E
        self.e_threshold = None  # auto-calibrated

    def get_manifold(self):
        return FHNManifold(n_grid=self.n_grid, gamma=self.gamma_E,
                           e_threshold=self.e_threshold or 1.0)

    def constraint_fn(self, state):
        return self.get_manifold().k(state)

    def dynamics(self, t, state):
        raise NotImplementedError("Use gpu_datagen.generate_lv_gpu()")

    def sample_initial_condition(self, rng):
        raise NotImplementedError("Use gpu_datagen.generate_lv_gpu()")

    def generate_data(self, **kwargs):
        raise NotImplementedError("Use gpu_datagen.generate_lv_gpu()")

    @classmethod
    def get_default_config(cls):
        return dict(n_grid=15, d1=0.5, d2=0.3,
                    alpha=0.8, beta=0.5, gamma_lv=0.3, delta=0.4)
