# NODE-LAC: Neural ODEs with Lagrangian Adaptive Correction

Code for **Learning Transverse Dynamics: Neural ODEs with Lagrangian Adaptive Correction on Constraint Manifolds**, by Dongzhe Zheng and Wenjie Mei.

NODE-LAC augments a learned vector field with a gated constraint-gradient correction. A state-dependent gain learns through one-step predictions, while a scalar Lagrangian multiplier adapts the constraint-loss weight from trajectory violations. The optional Lyapunov-residual loss updates both the vector field and the gain network.

## Installation

Use Python 3.10-3.12 with the pinned requirements. The experiment environment uses PyTorch 2.7.1 with CUDA 11.8 and float64 computation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Commands below run from the repository root. Use `--device cpu` for CPU execution.

## Data and training

The four recorded datasets are included in `results/`. Their SHA-256 hashes are checked against [data_manifest.json](data_manifest.json) before execution. The data consist of 256 trajectories per system: 102 fitting, 26 validation, and 128 test trajectories. Standardization uses the fitting partition.

Verify the datasets and inspect the run plan:

```bash
python reproduce.py --suite main --output-dir outputs --dry-run
```

Train all ten methods with seeds 42, 123, and 456:

```bash
python reproduce.py --suite main --output-dir outputs --device cuda:0 --save-predictions
```

For a smaller comparison, add `--systems fitzhugh_nagumo --methods NODE-LAC NODE SNDE`. Use `--data-dir` to select another location for the same recorded datasets.

## Experiments

The same entry point provides the component ablations, data-efficiency, observation-noise, and paired residual-loss comparisons:

```bash
python reproduce.py --suite ablation --output-dir outputs --device cuda:0
python reproduce.py --suite data-efficiency --output-dir outputs --device cuda:0
python reproduce.py --suite noise --output-dir outputs --device cuda:0
python reproduce.py --suite residual --output-dir outputs --device cuda:0
```

Long-horizon evaluation reuses the trained main-suite models and their selected correction scales:

```bash
python reproduce.py --suite long-horizon --model-dir outputs --output-dir outputs --device cuda:0
```

See [reproduction details](docs/reproduction.md) for method settings, residual configurations, metric definitions, and output formats.

## Figures

Generate compact figures directly from completed experiment outputs:

```bash
python -m experiments.plotting --input-dir outputs --output-dir plots --kind all
```

Trajectory and surface figures require main-suite predictions saved with `--save-predictions`. Figures show three-seed means and sample standard deviations; the trajectory display applies a centered five-observation average to PNODE and CPNODE within each seed.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the training gradients, dual updates, residual computation, data handling, experiment dispatch, and plotting statistics.
