"""Port-Hamiltonian Neural Networks (Desai et al. 2021).

Optimized: uses low-rank factorization for J and R when state_dim > 32,
avoiding O(d^2) network outputs and O(d^3) matrix operations.
"""

import torch
import torch.nn as nn
from .base import BaselineModel


class PortHJNN(BaselineModel):
    """Port-Hamiltonian NN: dx/dt = (J(x) - R(x)) * dH/dx.

    J: skew-symmetric (learned)
    R: positive semi-definite dissipation (R = S^T S)
    H: Hamiltonian (learned)

    For high-dim systems (d > 32), uses low-rank factorization:
      J = A B^T - B A^T  (rank-2r skew-symmetric)
      R = C^T C           (rank-r PSD)
    This reduces output dim from O(d^2) to O(d*r) and matmul from O(d^3) to O(d*r).
    """

    def __init__(self, state_dim, hidden_dim=128, rank=16, **kwargs):
        super().__init__(state_dim, **kwargs)
        self.d = state_dim
        self.use_lowrank = state_dim > 32

        # Hamiltonian network — same for all dims
        self.H_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        if self.use_lowrank:
            self.rank = min(rank, state_dim // 2)
            # J factors: A, B ∈ R^{d×r} → J = AB^T - BA^T
            self.A_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, state_dim * self.rank),
            )
            self.B_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, state_dim * self.rank),
            )
            # R factor: C ∈ R^{r×d} → R = C^T C
            self.C_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, self.rank * state_dim),
            )
        else:
            n_skew = state_dim * (state_dim - 1) // 2
            self.J_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, max(n_skew, 1)),
            )
            self.S_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, state_dim * state_dim),
            )

    def forward(self, t, x):
        # dH/dx via autograd
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            H = self.H_net(x_req).sum()
            dH = torch.autograd.grad(H, x_req, create_graph=True)[0]

        if self.use_lowrank:
            return self._forward_lowrank(x, dH)
        else:
            return self._forward_full(x, dH)

    def _forward_lowrank(self, x, dH):
        """Low-rank: J*dH = A(B^T dH) - B(A^T dH), R*dH = C^T(C dH)."""
        d, r = self.d, self.rank
        batch = x.shape[:-1]

        A = self.A_net(x).view(*batch, d, r)
        B = self.B_net(x).view(*batch, d, r)
        C = self.C_net(x).view(*batch, r, d)

        # J*dH = A(B^T dH) - B(A^T dH)  — O(d*r) instead of O(d^2)
        dH_col = dH.unsqueeze(-1)  # (..., d, 1)
        BtdH = (B.transpose(-1, -2) @ dH_col)  # (..., r, 1)
        AtdH = (A.transpose(-1, -2) @ dH_col)  # (..., r, 1)
        J_dH = (A @ BtdH - B @ AtdH).squeeze(-1)  # (..., d)

        # R*dH = C^T (C dH) * 0.01
        CdH = (C @ dH_col)  # (..., r, 1)
        R_dH = (C.transpose(-1, -2) @ CdH).squeeze(-1) * 0.01  # (..., d)

        return J_dH - R_dH

    def _forward_full(self, x, dH):
        """Full matrix for small dims."""
        d = self.d
        batch = x.shape[:-1]

        j_entries = self.J_net(x)
        J = torch.zeros(*batch, d, d, device=x.device, dtype=x.dtype)
        row_idx, col_idx = torch.triu_indices(d, d, offset=1, device=x.device)
        J[..., row_idx, col_idx] = j_entries
        J[..., col_idx, row_idx] = -j_entries

        S = self.S_net(x).view(*batch, d, d)
        R = S.transpose(-1, -2) @ S * 0.01

        JR = J - R
        return (JR @ dH.unsqueeze(-1)).squeeze(-1)
