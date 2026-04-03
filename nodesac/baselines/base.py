"""Base class for all baseline models."""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from ..core.neural_ode import euler_integrate


class BaselineModel(nn.Module):
    """Common base for all baselines."""

    def __init__(self, state_dim, solver='euler', atol=1e-5, rtol=1e-5):
        super().__init__()
        self.state_dim = state_dim
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def forward(self, t, x):
        raise NotImplementedError

    def predict(self, x0, t_span):
        if self.solver == 'euler':
            return euler_integrate(self, x0, t_span)
        return odeint(self, x0, t_span, method=self.solver,
                      atol=self.atol, rtol=self.rtol)

    def compute_loss(self, x_pred, x_true, reg_lambda=1e-4):
        mse = (x_pred - x_true).pow(2).mean()
        l2_reg = sum(p.pow(2).sum() for p in self.parameters()) * reg_lambda
        return mse + l2_reg

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
