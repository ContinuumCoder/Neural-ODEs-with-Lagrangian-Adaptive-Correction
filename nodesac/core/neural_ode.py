"""Neural ODE model with MLP-based vector field."""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class MLP(nn.Module):
    """Multi-layer perceptron for learning vector fields."""

    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128), activation=nn.Tanh):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), activation()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def euler_integrate(func, x0, t_span):
    """Simple Euler integration — fast and stable for training."""
    traj = [x0]
    x = x0
    for i in range(len(t_span) - 1):
        dt = t_span[i + 1] - t_span[i]
        x = x + dt * func(t_span[i], x)
        traj.append(x)
    return torch.stack(traj)  # (T, batch, D)


class NeuralODE(nn.Module):
    """Neural ODE: dx/dt = f_theta(x)."""

    def __init__(self, state_dim, hidden_dims=(128, 128), activation=nn.Tanh,
                 solver='euler', atol=1e-5, rtol=1e-5):
        super().__init__()
        self.state_dim = state_dim
        self.f = MLP(state_dim, state_dim, hidden_dims, activation)
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def forward(self, t, x):
        """Compute dx/dt = f_theta(x)."""
        return self.f(x)

    def predict(self, x0, t_span):
        """Integrate from x0 over t_span.

        Args:
            x0: (batch, state_dim) initial states
            t_span: (T,) time points
        Returns:
            (T, batch, state_dim) predicted trajectory
        """
        if self.solver == 'euler':
            return euler_integrate(self, x0, t_span)
        return odeint(self, x0, t_span, method=self.solver,
                      atol=self.atol, rtol=self.rtol)

    def set_tolerance(self, atol, rtol):
        self.atol = atol
        self.rtol = rtol
