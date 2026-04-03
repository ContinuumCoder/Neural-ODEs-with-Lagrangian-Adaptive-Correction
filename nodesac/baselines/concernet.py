"""ConCerNet: Conservation law discovery (Zhang et al. 2023)."""

import torch
import torch.nn as nn
from .base import BaselineModel


class ConCerNet(BaselineModel):
    """ConCerNet: learns dynamics + conserved quantity, then projects."""

    def __init__(self, state_dim, hidden_dim=128, proj_dim=64, **kwargs):
        super().__init__(state_dim, **kwargs)
        # Dynamics network
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Conservation quantity network h(x)
        self.conserve_net = nn.Sequential(
            nn.Linear(state_dim, proj_dim),
            nn.Tanh(),
            nn.Linear(proj_dim, proj_dim),
            nn.Tanh(),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, t, x):
        f_x = self.dynamics_net(x)

        # Compute gradient of h(x)
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            h = self.conserve_net(x_req)
            grad_h = torch.autograd.grad(h.sum(), x_req, create_graph=False)[0]

        if grad_h is None:
            return f_x

        # Project: f_proj = f - grad_h * (grad_h^T f) / (grad_h^T grad_h)
        gh_norm_sq = (grad_h * grad_h).sum(dim=-1, keepdim=True) + 1e-8
        gh_f = (grad_h * f_x).sum(dim=-1, keepdim=True)
        projection = grad_h * gh_f / gh_norm_sq

        return f_x - projection

    def conservation_loss(self, trajectory):
        """Loss to encourage h(x) to be conserved along trajectory."""
        h_values = self.conserve_net(trajectory)
        return (h_values - h_values[:, 0:1]).pow(2).mean()
