"""Base class for dynamical system data generators."""

from abc import ABC, abstractmethod
import numpy as np
import torch
from scipy.integrate import solve_ivp


class DynamicalSystem(ABC):
    """Base class for physical dynamical systems."""

    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def dynamics(self, t, state):
        """True system dynamics dx/dt. Returns numpy array."""
        pass

    @abstractmethod
    def constraint_fn(self, state):
        """Constraint function k(x). Returns tensor."""
        pass

    @abstractmethod
    def get_manifold(self):
        """Return ConstraintManifold object."""
        pass

    def generate_data(self, n_trajectories=256, t_span=(0, 5), dt_save=0.05,
                      dt_integrate=1e-3, seed=42):
        """Generate trajectory data.

        Returns:
            dict with 'train_states', 'test_states', 'times'
            states shape: (n_traj, T, state_dim)
        """
        rng = np.random.RandomState(seed)
        t_eval = np.arange(t_span[0], t_span[1], dt_save)
        all_trajs = []

        for i in range(n_trajectories):
            x0 = self.sample_initial_condition(rng)
            try:
                sol = solve_ivp(self.dynamics, t_span, x0, t_eval=t_eval,
                                method='RK45', max_step=max(dt_integrate, 0.01),
                                rtol=1e-6, atol=1e-6)
                if sol.success:
                    all_trajs.append(sol.y.T)  # (T, state_dim)
            except Exception:
                continue

        # Pad if some trajectories failed
        if len(all_trajs) == 0:
            x0 = self.sample_initial_condition(rng)
            all_trajs.append(np.tile(x0, (len(t_eval), 1)))
        while len(all_trajs) < n_trajectories:
            all_trajs.append(all_trajs[0].copy())

        all_trajs = np.stack(all_trajs[:n_trajectories])  # (N, T, D)
        times = torch.tensor(t_eval, dtype=self.dtype)
        states = torch.tensor(all_trajs, dtype=self.dtype)

        n_train = n_trajectories // 2
        return {
            'train_states': states[:n_train],
            'test_states': states[n_train:],
            'times': times,
        }

    @abstractmethod
    def sample_initial_condition(self, rng):
        """Sample a random initial condition. Returns numpy array."""
        pass

    @classmethod
    def get_default_config(cls):
        return {}
