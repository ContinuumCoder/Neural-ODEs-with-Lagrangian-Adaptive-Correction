"""Continuously-Penalized NODE (Norcliffe et al. 2020)."""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from ..core.neural_ode import euler_integrate
from .base import BaselineModel


class CPNODE(BaselineModel):
    """Augmented state NODE with continuous penalty accumulator.

    dx/dt = f(x), dp/dt = ||k(x)||^2
    Loss = MSE + lambda * p(T)
    """

    def __init__(self, state_dim, hidden_dim=135, penalty_weight=10.0,
                 constraint_fn=None, **kwargs):
        super().__init__(state_dim, **kwargs)
        self.real_state_dim = state_dim
        self.penalty_weight = penalty_weight
        self.constraint_fn = constraint_fn
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, x_aug):
        """x_aug = [x, p] where p is penalty accumulator (scalar per batch)."""
        x = x_aug[..., :self.real_state_dim]
        dx = self.net(x)
        # Penalty accumulation
        if self.constraint_fn is not None:
            kx = self.constraint_fn(x)
            dp = kx.pow(2).sum(dim=-1, keepdim=True)
        else:
            dp = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        return torch.cat([dx, dp], dim=-1)

    def predict(self, x0, t_span):
        """Augment state with penalty accumulator."""
        p0 = torch.zeros(*x0.shape[:-1], 1, device=x0.device, dtype=x0.dtype)
        x0_aug = torch.cat([x0, p0], dim=-1)
        if self.solver == 'euler':
            x_aug = euler_integrate(self, x0_aug, t_span)
        else:
            x_aug = odeint(self, x0_aug, t_span, method=self.solver,
                           atol=self.atol, rtol=self.rtol)
        return x_aug[..., :self.real_state_dim]

    def predict_with_penalty(self, x0, t_span):
        p0 = torch.zeros(*x0.shape[:-1], 1, device=x0.device, dtype=x0.dtype)
        x0_aug = torch.cat([x0, p0], dim=-1)
        if self.solver == 'euler':
            x_aug = euler_integrate(self, x0_aug, t_span)
        else:
            x_aug = odeint(self, x0_aug, t_span, method=self.solver,
                           atol=self.atol, rtol=self.rtol)
        return x_aug[..., :self.real_state_dim], x_aug[-1, ..., -1]

    def compute_loss(self, x_pred, x_true, reg_lambda=1e-4, penalty_final=None):
        mse = (x_pred - x_true).pow(2).mean()
        l2_reg = sum(p.pow(2).sum() for p in self.parameters()) * reg_lambda
        pen = self.penalty_weight * penalty_final.mean() if penalty_final is not None else 0
        return mse + l2_reg + pen
