"""Vanilla Neural ODE (Chen et al. 2018)."""

import torch.nn as nn
from .base import BaselineModel


class NODE(BaselineModel):
    """Standard Neural ODE: dx/dt = f_theta(x)."""

    def __init__(self, state_dim, hidden_dim=135, **kwargs):
        super().__init__(state_dim, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, x):
        return self.net(x)
