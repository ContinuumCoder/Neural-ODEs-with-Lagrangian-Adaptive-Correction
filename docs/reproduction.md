# Reproducing the benchmark experiments

Run commands from the repository root after installing [requirements.txt](../requirements.txt). Dataset hashes in [data_manifest.json](../data_manifest.json) are verified before training. `--dry-run` checks the data and displays the planned runs.

## Shared protocol

Each recorded dataset contains 256 trajectories and 100 observation times. The first 128 trajectories form the development partition; a permutation with seed 20260906 allocates 102 for fitting and 26 for validation. The remaining 128 form the test partition. Componentwise means and sample standard deviations use the 102 fitting trajectories, with a scale floor of 1e-6.

Training uses Adam for 100 epochs, batch size 64, initial learning rate 0.006 with cosine decay toward 0.0001, weight decay 0.0001, and gradient-norm clipping at 1. Integration uses forward Euler on the recorded observation grid and float64 arithmetic. The seeds are 42, 123, and 456. Neural-network widths and the baseline fields are implemented in `nodesac/` and `experiments/training.py`.

The NODE-LAC vector field has two 256-unit Tanh layers; its gain network has two 128-unit ReLU layers with softplus output. The main comparison initializes the multiplier at 0.1, uses gain regularization 0.05, and trains with correction scale 0.3. The effective multiplier is clipped to [0.01, 1]. A separate Adam update at learning rate 0.01 uses full-trajectory violation minus the target 0.01. The gain loss uses one-step predictions. The fitting loss differentiates through the vector field with the correction held fixed; the gain update holds the vector-field output fixed.

At evaluation, the correction scale minimizes recorded-coordinate validation MSE over {0, 0.1, 0.2, 0.5, 1, 1.5, 2}. The selected value is fixed for test prediction and extrapolation. The main suite includes NODE-LAC, NODE, SNDE, ConCerNet, SymODEN, HNN, CLNN, PORT-HJNN, PNODE, and CPNODE.

## Component, data-efficiency, and noise comparisons

These suites use FitzHugh-Nagumo, Lotka-Volterra, and Shallow Water, with the shared settings above.

| Suite | Methods or variants | Conditions |
|---|---|---|
| `ablation` | NODE-LAC | NoGainNet: constant gain 1; NoConstraintLoss: zero one-step constraint-loss weight; NoCorrection: zero training and evaluation correction scale |
| `data-efficiency` | NODE-LAC, NODE, SNDE | Fractions 0.25, 0.5, 0.75: 25, 51, 76 fitting trajectories |
| `noise` | NODE-LAC, NODE | Gaussian noise standard deviations 0.05, 0.1, 0.2 in standardized fitting observations |

Data-efficiency subsets are prefixes of the fitting permutation; normalization uses all 102 fitting trajectories. The full-data and zero-noise conditions use the corresponding main-suite outputs. Observation noise is shared between methods within each seed. Validation and test observations are unperturbed. The training random seed is reset after data preparation, so noise generation does not change model initialization.

## Lyapunov-residual comparison

`--suite residual --variant paired` uses the configurations below. Each zero-weight control and regularized model shares the data, architecture, schedule, and evaluation correction scales. Scale tuples follow seed order 42, 123, 456.

| System | Initial multiplier | Gain regularization | Residual weight | Evaluation scales |
|---|---:|---:|---:|---|
| FitzHugh-Nagumo | 0.01 | 0.01 | 0 | 0.3, 0.3, 0.4 |
| Lotka-Volterra | 0.05 | 0.05 | 0.001 | 0.4, 0.4, 0.4 |
| Shallow Water | 0.05 | 0.05 | 0.1 | 0.4, 0.5, 0.4 |
| Robot Arm | 0.05 | 0.05 | 0.1 | 0.3, 0.3, 0.3 |

FitzHugh-Nagumo's selected weight is zero, so paired execution trains it once. `--variant zero-control` or `--variant regularized` selects a single member.

For V = ||k||^2/2, the residual is [dV/dt + 0.1 V]_+. Training averages its normalized square, with denominator 1 + V, over equally weighted deterministic samples of up to 512 fitting and 512 rollout states. States, constraint gradients, and gates are fixed during parameter differentiation; both network outputs receive residual gradients.

The evaluation score is half the mean normalized squared residual on active reference states and half its mean over all predicted states. Active states have V > 1e-12. An empty active reference set yields no score. Full-grid diagnostics report raw residuals and decay fractions separately. A sampled decay condition is satisfied when dV/dt + 0.1 V <= 1e-10.

## Long-horizon evaluation

The long-horizon suite loads completed 100-epoch main runs for FitzHugh-Nagumo, Lotka-Volterra, and Shallow Water. It replays the 100-observation evaluation before predicting over 1000 observation times. The original generators use 256 trajectories, seed 42, integration step 0.001, and observation intervals 0.05, 0.015, and 0.05, respectively. Generated reference trajectories must reproduce both recorded partitions over their first 100 observations. Cached references are checked against the dataset and generator hashes.

The default methods are NODE-LAC, NODE, SNDE, PNODE, CPNODE, and ConCerNet. Use `--methods` to select a subset. `--model-dir` identifies the output root containing the main runs; `--reference-dir` optionally selects the reference cache directory. CUDA generation reproduces the recorded random-number stream. CPU inference can use an already verified cache.

## Outputs and metrics

Main runs are stored in `OUTPUT/main/SYSTEM/METHOD/seedN`. Other suites use `OUTPUT/SUITE/SYSTEM/METHOD/VARIANT/seedN`. Training runs contain `model.pt`, `history.json`, `provenance.json`, `curves.json`, and `result.json`. `--save-predictions` additionally writes `predictions.pt` with predictions, reference states, and observation times in recorded coordinates. Each suite writes `summary.csv` with means and sample standard deviations across seeds. Existing run directories are preserved; use a new output root for another execution.

MSE and MAE measure state prediction in recorded coordinates. TCE is mean squared error in successive state increments. CE is mean squared threshold violation in standardized coordinates. The filtered-memory coordinate has a separate prediction MSE. `near_feasible_fraction` counts states with squared standardized violation below 0.1. The threshold functions define inequality feasible regions; their violation is distinct from physical invariant drift or geometric distance. The equality-manifold attraction result uses its stated pointwise conditions. Empirical derivative and residual diagnostics characterize the evaluated states.

`experiments/training.py` and `experiments/residual.py` implement the benchmark updates. `nodesac/` provides model components, generators, and general-purpose interfaces; the reproduction commands above specify the reported training protocol. Runtime histories and generated figures stay in output directories and are ignored by Git.
