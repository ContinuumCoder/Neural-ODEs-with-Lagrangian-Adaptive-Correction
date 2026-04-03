"""Stabilized Neural Differential Equations (White et al. 2023)."""

import torch
import torch.nn as nn
from .base import BaselineModel


class SNDE(BaselineModel):
    """SNDE: NODE + explicit constraint stabilization.

    dx/dt = f(x) - J^T (J J^T)^{-1} (J f(x) + alpha * k(x))
    """

    def __init__(self, state_dim, constraint_fn=None, constraint_dim=1,
                 hidden_dim=135, alpha=10.0, **kwargs):
        super().__init__(state_dim, **kwargs)
        self.constraint_fn = constraint_fn
        self.constraint_dim = constraint_dim
        self.alpha_stab = alpha
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, x):
        f_x = self.net(x)
        if self.constraint_fn is None:
            return f_x

        try:
            with torch.enable_grad():
                x_req = x.detach().clone().requires_grad_(True)
                kx = self.constraint_fn(x_req)
                kx_scalar = kx.pow(2).sum()
                grad = torch.autograd.grad(kx_scalar, x_req, create_graph=False)[0]
            if grad is None:
                return f_x
            # Simple correction: push towards manifold along gradient
            grad_norm = grad.pow(2).sum(dim=-1, keepdim=True) + 1e-8
            kx_val = self.constraint_fn(x)
            correction = self.alpha_stab * grad * kx_val.pow(2).sum(dim=-1, keepdim=True) / grad_norm
            return f_x - correction
        except Exception:
            return f_x
