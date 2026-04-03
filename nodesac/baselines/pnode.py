"""Penalized Neural ODE (Massaroli et al. 2020)."""

import torch.nn as nn
from .base import BaselineModel


class PNODE(BaselineModel):
    """Standard NODE + penalty terms for constraint violation in loss."""

    def __init__(self, state_dim, hidden_dim=135, penalty_weight=10.0,
                 constraint_fn=None, **kwargs):
        super().__init__(state_dim, **kwargs)
        self.penalty_weight = penalty_weight
        self.constraint_fn = constraint_fn
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, x):
        return self.net(x)

    def compute_loss(self, x_pred, x_true, reg_lambda=1e-4):
        mse = (x_pred - x_true).pow(2).mean()
        l2_reg = sum(p.pow(2).sum() for p in self.parameters()) * reg_lambda
        penalty = 0.0
        if self.constraint_fn is not None:
            kx = self.constraint_fn(x_pred)
            penalty = self.penalty_weight * kx.pow(2).mean()
        return mse + l2_reg + penalty
