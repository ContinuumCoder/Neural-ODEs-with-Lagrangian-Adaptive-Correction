# Reproducing the reported comparisons

Run commands from the repository root after installing [requirements-revision.txt](../requirements-revision.txt). Supply the recorded datasets through `--data-dir`; [revision/data_manifest.json](../revision/data_manifest.json) lists their expected SHA-256 hashes. `--dry-run` verifies the files and displays the planned runs without training or writing outputs.

## Shared protocol

Each dataset contains 256 trajectories. The first 128 form the development partition, split deterministically into 102 fitting and 26 validation trajectories; the remaining 128 are used for evaluation. Means and scales are computed from the fitting partition. The model uses standardized states and the recorded observation times, with Euler integration and float64 arithmetic.

The published runs use 100 epochs, batch size 64, and seeds 42, 123, and 456. The main NODE-LAC configuration has initial multiplier 0.1, gain regularization 0.05, and training correction scale 0.3. Its evaluation scale is selected by validation MSE from the seven values specified in the paper. `--suite main` includes all ten methods by default.

For example, run only NODE-LAC on FitzHugh–Nagumo:

```bash
python reproduce.py --suite main --systems fitzhugh_nagumo --methods NODE-LAC \
  --data-dir data --output-dir outputs --seeds 42 123 456 --device cuda:0
```

## Lyapunov-residual comparison

`--suite residual --variant paired` uses the fixed settings below. Each pair shares its data, architecture, training schedule, and evaluation correction scales. The scale tuples follow seed order 42, 123, 456.

| System | Initial multiplier | Gain regularization | Residual weight | Evaluation scales |
|---|---:|---:|---:|---|
| FitzHugh–Nagumo | 0.01 | 0.01 | 0 | 0.3, 0.3, 0.4 |
| Lotka–Volterra | 0.05 | 0.05 | 0.001 | 0.4, 0.4, 0.4 |
| Shallow Water | 0.05 | 0.05 | 0.1 | 0.4, 0.5, 0.4 |
| Robot Arm | 0.05 | 0.05 | 0.1 | 0.3, 0.3, 0.3 |

The zero-weight control sets the residual weight to zero. FitzHugh–Nagumo has the same selected and control configuration, so paired execution trains it once. `--variant zero-control` or `--variant regularized` selects one member of each comparison.

The residual is `[dV/dt + 0.1 V]_+`, where `V = ||k||²/2`. Training averages its normalized square over equally weighted samples from the fitting trajectories and their current predictions. The normalization is `1 + V`, and the residual term updates both networks.

## Outputs and metrics

Runs are written under `OUTPUT/main/SYSTEM/METHOD/seedN` or `OUTPUT/residual/SYSTEM/VARIANT/seedN`. Each contains `model.pt`, `history.json`, `provenance.json`, and `result.json`. Suite summaries are written to `OUTPUT/SUITE/summary.csv`; dispersion is the sample standard deviation across seeds. Existing run directories are preserved, so use a new output directory for another execution.

MSE and MAE are prediction errors in recorded coordinates. TCE is the mean squared error of successive state increments. CE is the mean squared threshold violation in standardized coordinates. Derivative and residual diagnostics are evaluated on reference and predicted states, with active-state statistics reported separately.
