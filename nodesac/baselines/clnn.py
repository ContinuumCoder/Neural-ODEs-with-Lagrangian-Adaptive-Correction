"""Learned mass-and-potential baseline with linear damping."""

import torch
import torch.nn as nn
from .base import BaselineModel


class CLNN(BaselineModel):
    """Mass-and-potential vector field for position and velocity states.

    dq/dt = qdot
    dqdot/dt = M(q)^(-1) * (-grad V(q) - 0.01*qdot)
    M(q) = L(q) L(q)^T + 0.1 I, with lower-triangular L(q).
    """

    def __init__(self, state_dim, hidden_dim=132, **kwargs):
        super().__init__(state_dim, **kwargs)
        assert state_dim % 2 == 0
        self.half_dim = state_dim // 2
        d = self.half_dim

        # Mass network output is reshaped and lower-triangularized.
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

        # Evaluate the mass matrix and potential gradient.
        with torch.enable_grad():
            q_req = q.detach().clone().requires_grad_(True)
            M = self.mass_matrix(q_req)
            V = self.V_net(q_req)
            dV_dq = torch.autograd.grad(V.sum(), q_req, create_graph=True)[0]

        # Invert the mass matrix for the damped potential force.
        M_val = self.mass_matrix(q)
        M_inv = torch.linalg.solve(M_val, torch.eye(self.half_dim, device=q.device, dtype=q.dtype).expand_as(M_val))

        # qddot = M^{-1} (-dV/dq - damping * qdot)
        rhs = -dV_dq - 0.01 * qdot
        qddot = (M_inv @ rhs.unsqueeze(-1)).squeeze(-1)

        return torch.cat([qdot, qddot], dim=-1)
