"""Hamiltonian Neural Networks (Greydanus et al. 2019)."""

import torch
import torch.nn as nn
from .base import BaselineModel


class HNN(BaselineModel):
    """Hamiltonian Neural Network with wider layers."""

    def __init__(self, state_dim, hidden_dim=140, **kwargs):
        super().__init__(state_dim, **kwargs)
        assert state_dim % 2 == 0
        self.half_dim = state_dim // 2
        self.H_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            H = self.H_net(x_req).sum()
            dH = torch.autograd.grad(H, x_req, create_graph=True)[0]

        dq = dH[..., self.half_dim:]
        dp = -dH[..., :self.half_dim]
        return torch.cat([dq, dp], dim=-1)
