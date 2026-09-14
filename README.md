# NODE-LAC: Neural ODEs with Lagrangian Adaptive Correction

Code for **Learning Transverse Dynamics: Neural ODEs with Lagrangian Adaptive Correction on Constraint Manifolds**, by Dongzhe Zheng and Wenjie Mei.

NODE-LAC combines a learned vector field with a state-dependent, gated constraint-gradient correction. A gain network learns from one-step predictions, and a Lagrangian multiplier adapts the weight of sampled constraint violation. An optional Lyapunov-residual loss supervises the corrected vector field on reference and predicted states.


## Installation

The supplied environment specifies PyTorch 2.7.1 with CUDA 11.8. Computation uses float64.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-revision.txt
```

The examples use a CUDA device; `--device cpu` selects CPU execution.

## Data

Use the four recorded datasets in the [repository data directory](https://github.com/ContinuumCoder/Neural-ODEs-with-Lagrangian-Adaptive-Correction/tree/6bd8aa176cffe3cad389267990d6ac8ba4e78790/results):

- `fitzhugh_nagumo_data.pt`
- `lotka_volterra_data.pt`
- `shallow_water_data.pt`
- `franka_robot_data.pt`

Place these files in a directory such as `data`. The runner checks their SHA-256 hashes against [revision/data_manifest.json](revision/data_manifest.json) before training. A dry run verifies the files and prints the experiment plan:

```bash
python reproduce.py --suite main --data-dir data --output-dir outputs --dry-run
```

## Reproduction

The entry point calls the training implementations used for the reported experiments. The main comparison uses four systems, NODE-LAC and nine comparators, and seeds 42, 123, and 456:

```bash
python reproduce.py --suite main --data-dir data --output-dir outputs \
  --seeds 42 123 456 --device cuda:0
```

The residual-loss comparison pairs each regularized model with its zero-weight control at the published settings:

```bash
python reproduce.py --suite residual --variant paired \
  --data-dir data --output-dir outputs --seeds 42 123 456 --device cuda:0
```

Use `--systems` and, for the main suite, `--methods` to select a subset. See [reproduction details](docs/reproduction.md) for configurations, outputs, and metric definitions.

## Interpretation

Prediction MSE and MAE, state-increment error (TCE), standardized threshold violation (CE), and Lyapunov residuals measure different properties. The experimental constraints define inequality feasible sets in standardized coordinates. The equality-manifold stability results assume the stated regularity, invariant-region, and pointwise decay conditions. Training behavior and sampled decay diagnostics are evaluated empirically across three seeds.
