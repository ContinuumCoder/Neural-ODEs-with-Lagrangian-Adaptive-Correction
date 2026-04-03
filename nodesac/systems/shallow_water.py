"""Multi-scale coupled wave system with reaction-diffusion dynamics.

Models nonlinear wave-wave interaction with viscous dissipation and
energy constraints — captures multi-scale coupling effects.
"""

import torch
from .base import DynamicalSystem
from ..core.manifold import FHNManifold


class ShallowWater(DynamicalSystem):

    def __init__(self, n_grid=15, c1=1.0, c2=1.2, a1=4.0, a2=5.5,
                 b1=0.02, b2=0.15, gamma=0.5, e_threshold=6.1,
                 dtype=torch.float64, device='cpu'):
        super().__init__(dtype, device)
        self.n_grid = n_grid
        self.c1, self.c2 = c1, c2
        self.a1, self.a2 = a1, a2
        self.b1, self.b2 = b1, b2
        self.gamma = gamma
        self.e_threshold = e_threshold
        self.state_dim = 2 * n_grid + 1

    def get_manifold(self):
        return FHNManifold(self.n_grid, self.gamma, self.e_threshold)

    def constraint_fn(self, state):
        return self.get_manifold().k(state)

    def dynamics(self, t, state):
        raise NotImplementedError("Use gpu_datagen.generate_sw_gpu()")

    def sample_initial_condition(self, rng):
        raise NotImplementedError("Use gpu_datagen.generate_sw_gpu()")

    def generate_data(self, **kwargs):
        raise NotImplementedError("Use gpu_datagen.generate_sw_gpu()")

    @classmethod
    def get_default_config(cls):
        return dict(n_grid=15, c1=1.0, c2=1.2, a1=4.0, a2=5.5)
