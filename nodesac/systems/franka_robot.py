"""Coupled robot-arm benchmark with damping and a quadratic memory variable.

The state contains positions, velocities, and the memory value E.
Trajectory generation is provided by gpu_datagen.generate_robot_gpu.
"""

import torch
from .base import DynamicalSystem
from ..core.manifold import FHNManifold


class FrankaRobot(DynamicalSystem):

    def __init__(self, n_joints=8, D_q=1.0, D_v=0.8,
                 d_coeff=0.1, g_coeff=2.0, c_coeff=0.5,
                 gamma=0.5, e_threshold=None,
                 dtype=torch.float64, device='cpu'):
        super().__init__(dtype, device)
        self.n_joints = n_joints
        self.D_q, self.D_v = D_q, D_v
        self.d_coeff = d_coeff
        self.g_coeff = g_coeff
        self.c_coeff = c_coeff
        self.gamma = gamma
        self.e_threshold = e_threshold
        self.state_dim = 2 * n_joints + 1  # q + qdot + E

    def dynamics(self, t, state):
        raise NotImplementedError("Use gpu_datagen.generate_robot_gpu()")

    def sample_initial_condition(self, rng):
        raise NotImplementedError("Use gpu_datagen.generate_robot_gpu()")

    def get_manifold(self):
        return FHNManifold(self.n_joints, self.gamma,
                           self.e_threshold or 1.0)

    def constraint_fn(self, state):
        return self.get_manifold().k(state)

    def generate_data(self, **kwargs):
        raise NotImplementedError("Use gpu_datagen.generate_robot_gpu()")

    @classmethod
    def get_default_config(cls):
        return dict(n_joints=8, D_q=1.0, D_v=0.8)
