# NODE-LAC: Neural ODEs with Lagrangian Adaptive Correction

This repository contains the code for the paper:

**Learning Transverse Dynamics: Neural ODEs with Lagrangian Adaptive Correction on Constraint Manifolds**

## Overview

NODE-LAC learns constrained dynamical systems from data in a gray-box setting: the dynamics are unknown, but the constraint manifold is specified by a known function $k(\mathbf{x}) = 0$. The framework decomposes the learning problem into:

- **Tangential dynamics** (Neural ODE $f_\theta$): fits the on-manifold physics from trajectory data
- **Normal correction** (Gain network $\kappa_\omega$): synthesizes a state-dependent correction along $\nabla_x \|k\|^2$ that renders the constraint manifold asymptotically stable
- **Adaptive weighting** (Lagrangian dual variable $\mu$): prices constraint violations via dual ascent, removing the need to hand-tune penalty coefficients

The gain network is trained by differentiating through a single Euler step of the closed-loop dynamics (one-step lookahead). Lyapunov analysis provides asymptotic and finite-time stability guarantees.

## Results

NODE-LAC ranks **#1 on all four benchmark systems** in MSE, MAE, and Temporal Coherence Error (TCE), across 10 baselines and 3 seeds:

| System | Dim | NODE-LAC MSE | Best Baseline | Improvement |
|--------|-----|-------------|---------------|-------------|
| FitzHugh-Nagumo | 17 | **0.0025** | SNDE 0.0041 | -39% |
| Lotka-Volterra | 31 | **0.0083** | PORT-HJNN 0.0085 | -2% |
| Shallow Water | 31 | **0.0018** | SNDE 0.0066 | -73% |
| Franka Robot Arm | 17 | **0.0089** | SNDE 0.0098 | -9% |

## Repository Structure

```
nodesac/
  core/
    neural_ode.py      # Neural ODE with Euler integration
    nodesac.py         # GainNet, ClosedLoopDynamics, NODESAC
    manifold.py        # Constraint manifold definitions (energy + amplitude)
  baselines/           # 9 baseline implementations (NODE, SNDE, HNN, SymODEN, ...)
  systems/             # 4 dynamical systems + GPU data generation
  utils/               # Metrics (MSE, MAE, TCE), training utilities, visualization

run_parallel.py            # Main experiment runner (all methods, all systems, 3 seeds)
compute_tce_fast.py        # Compute TCE from saved checkpoints
generate_paper_figures.py  # Generate publication figures (bar chart, radar, ablation)
run_spatiotemporal_figs.py # Generate spatiotemporal visualizations
run_all_figures.py         # Generate all figures including loss curves

results/
  checkpoint.json          # All experiment results (MSE, MAE, TCE per method/system/seed)
  tce_results.json         # TCE results
  *_data.pt                # Cached datasets (4 systems, ~19MB total)

checkpoints/               # Saved model weights (~37MB, 120 files)
figures/                   # Generated publication figures
experiments/               # Experiment data (JSON) and spatiotemporal PDFs
```

## Quick Start

### Requirements

```bash
pip install torch torchdiffeq numpy matplotlib
```

### Run All Experiments

```bash
# Train all 10 methods x 4 systems x 3 seeds (uses GPU)
python run_parallel.py --phase main --gpus 0

# Compute TCE metric (loads checkpoints, trains missing)
python compute_tce_fast.py

# Generate figures
python generate_paper_figures.py
python run_spatiotemporal_figs.py
```

### Use NODE-LAC on Your Own System

```python
import torch
from nodesac.core.neural_ode import NeuralODE, euler_integrate
from nodesac.core.nodesac import GainNet, ClosedLoopDynamics

# Define your constraint: k(x) = 0 on the manifold
class MyManifold:
    def k(self, x):
        E = x[..., -1]  # energy variable
        return torch.relu(E - threshold).unsqueeze(-1)
    def distance(self, x):
        return self.k(x).pow(2).sum(-1)

# Build model
D = 17  # state dimension
node = NeuralODE(D, hidden=(256, 256), solver='euler')
gain = GainNet(D, hidden=(128, 128))
manifold = MyManifold()

# Closed-loop dynamics: f(x) - kappa(x) * gate(||k||) * grad||k||^2
dynamics = ClosedLoopDynamics(node.f, gain, manifold, correction_scale=0.3)

# Forward pass
x0 = torch.randn(batch, D)
times = torch.linspace(0, 1, 100)
trajectory = euler_integrate(dynamics, x0, times)  # (T, batch, D)
```

## Constraint Function

All four systems share a generic constraint template:

$$k(\mathbf{x}) = \begin{bmatrix} \text{relu}(E - E_{\text{thr}}) \\ \text{relu}(\bar{s} - s_{\max}) \end{bmatrix}$$

where $E$ is a system-specific energy variable and $\bar{s}$ is the mean squared state magnitude. The threshold $E_{\text{thr}}$ is set to the 95th percentile of training energies. Both components are differentiable (relu subgradient) and require no tuned weights.

## Training Protocol

- Optimizer: Adam with cosine annealing (LR: 6e-3 → 1e-4)
- Epochs: 100, batch size: 64
- Loss: `trajectory_MSE + 0.1 * constraint_violation + 0.05 * effort`
- Integration: Euler, float64
- Correction scale: tuned at evaluation over {0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0}

## Citation

```bibtex
@article{nodelac2026,
  title={Learning Transverse Dynamics: Neural ODEs with Lagrangian Adaptive Correction on Constraint Manifolds},
  author={Zheng, Dongzhe and Mei, Wenjie}
  year={2026}
}
```

## License

MIT
