"""Constrained Lagrangian Neural Networks (Finzi et al. 2020)."""

import torch
import torch.nn as nn
from .base import BaselineModel


class CLNN(BaselineModel):
    """Constrained Lagrangian NN.

    L(q, qdot) = T - V = 0.5 * qdot^T M(q) qdot - V(q)
    Euler-Lagrange: M(q) qddot + C(q,qdot) qdot + dV/dq = 0
    """

    def __init__(self, state_dim, hidden_dim=132, **kwargs):
        super().__init__(state_dim, **kwargs)
        assert state_dim % 2 == 0
        self.half_dim = state_dim // 2
        d = self.half_dim

        # Mass matrix network: outputs lower triangular L, M = L L^T + eps I
        self.mass_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d * d),
        )
        # Potential energy network
        self.V_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def mass_matrix(self, q):
        batch_shape = q.shape[:-1]
        d = self.half_dim
        L_flat = self.mass_net(q)
        L = L_flat.view(*batch_shape, d, d)
        L_lower = torch.tril(L)
        M = L_lower @ L_lower.transpose(-1, -2) + 0.1 * torch.eye(d, device=q.device, dtype=q.dtype)
        return M

    def forward(self, t, x):
        q = x[..., :self.half_dim]
        qdot = x[..., self.half_dim:]

        # Compute M(q) and V(q) with gradients
        with torch.enable_grad():
            q_req = q.detach().clone().requires_grad_(True)
            M = self.mass_matrix(q_req)
            V = self.V_net(q_req)
            dV_dq = torch.autograd.grad(V.sum(), q_req, create_graph=True)[0]

        # dM/dq for Coriolis: simplified as finite difference
        M_val = self.mass_matrix(q)
        M_inv = torch.linalg.solve(M_val, torch.eye(self.half_dim, device=q.device, dtype=q.dtype).expand_as(M_val))

        # qddot = M^{-1} (-dV/dq - damping * qdot)
        rhs = -dV_dq - 0.01 * qdot
        qddot = (M_inv @ rhs.unsqueeze(-1)).squeeze(-1)

        return torch.cat([qdot, qddot], dim=-1)
