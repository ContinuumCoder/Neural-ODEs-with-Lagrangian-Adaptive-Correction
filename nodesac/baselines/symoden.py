"""SymODEN: Symplectic ODE Networks (Zhong et al. 2019)."""

import torch
import torch.nn as nn
from .base import BaselineModel


class SymODEN(BaselineModel):
    """Symplectic ODE Network preserving Hamiltonian structure.

    dq/dt = dH/dp, dp/dt = -dH/dq
    """

    def __init__(self, state_dim, hidden_dim=128, **kwargs):
        super().__init__(state_dim, **kwargs)
        assert state_dim % 2 == 0, "State dim must be even (q, p pairs)"
        self.half_dim = state_dim // 2
        self.H_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def hamiltonian(self, x):
        return self.H_net(x).squeeze(-1)

    def forward(self, t, x):
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            H = self.hamiltonian(x_req)
            dH = torch.autograd.grad(H.sum(), x_req, create_graph=True)[0]

        dH_dq = dH[..., :self.half_dim]
        dH_dp = dH[..., self.half_dim:]

        # Hamilton's equations
        dq_dt = dH_dp
        dp_dt = -dH_dq
        return torch.cat([dq_dt, dp_dt], dim=-1)
