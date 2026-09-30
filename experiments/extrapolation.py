"""Long-horizon inference from validated main-suite checkpoints."""
import json
import math
import time
from pathlib import Path

import torch

from experiments import training
from nodesac.systems import gpu_datagen

OBSERVATIONS = 1000
OBSERVATION_STEPS = {
    "fitzhugh_nagumo": .05,
    "lotka_volterra": .015,
    "shallow_water": .05,
}
GENERATORS = {
    "fitzhugh_nagumo": gpu_datagen.generate_fhn_gpu,
    "lotka_volterra": gpu_datagen.generate_lv_gpu,
    "shallow_water": gpu_datagen.generate_sw_gpu,
}


def _read(path):
    return json.loads(Path(path).read_text())


def _public_metrics(values):
    values = dict(values)
    if "Stability" in values:
        values["near_feasible_fraction"] = values.pop("Stability")
    return values


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _prefix_checks(reference, data, system):
    """Validate both development and test prefixes against the recorded data."""
    dim = data["test_states"].shape[-1]
    if reference["true"].shape != (128, OBSERVATIONS, dim):
        raise ValueError("Long reference must contain 128 test trajectories and 1000 observations")
    if reference["train_prefix"].shape != data["train_states"].shape:
        raise ValueError("Reference development prefix has the wrong shape")
    times = reference["times"]
    if times.shape != (OBSERVATIONS,):
        raise ValueError("Long reference must contain 1000 observation times")
    dt = OBSERVATION_STEPS[system]
    expected = torch.arange(OBSERVATIONS, device=times.device, dtype=times.dtype) * dt
    if not torch.allclose(times, expected, atol=1e-12, rtol=0):
        raise ValueError("Long-reference observation times do not match the fixed grid")
    errors = {}
    for name, prefix in [("train_states", reference["train_prefix"]),
                         ("test_states", reference["true"][:, :100])]:
        target = data[name]
        errors[name] = float((prefix - target).abs().max())
        if not torch.allclose(prefix, target, atol=2e-11, rtol=2e-12):
            raise ValueError(f"Reference prefix mismatch for {system}/{name}: {errors[name]:.6g}")
    time_error = float((times[:100] - data["times"]).abs().max())
    if time_error > 1e-12:
        raise ValueError("Reference time prefix differs from the recorded observations")
    if not all(torch.isfinite(reference[key]).all() for key in ["true", "train_prefix", "times"]):
        raise FloatingPointError("Long-reference arrays contain nonfinite values")
    return errors, time_error


def _reference(system, args, dataset, data):
    directory = Path(args.reference_dir) if args.reference_dir is not None else Path(args.output_dir) / "reference_data"
    target = directory / f"{system}_long.pt"
    manifest_path = target.with_suffix(".json")
    dt = OBSERVATION_STEPS[system]
    generator = GENERATORS[system]
    expected = {
        "schema": "node-lac-long-reference-v1",
        "system": system,
        "dataset_sha256": dataset["sha256"],
        "generator_sha256": training.hash_file(gpu_datagen.__file__),
        "generator": generator.__name__,
        "seed": 42,
        "n_generated": 256,
        "observations": OBSERVATIONS,
        "dt_save": dt,
        "dt_integrate": .001,
        "interval": [0, OBSERVATIONS * dt],
    }
    if target.exists() or manifest_path.exists():
        if not (target.is_file() and manifest_path.is_file()):
            raise ValueError(f"Incomplete reference cache: {target}; use an empty cache directory")
        manifest = _read(manifest_path)
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Reference cache configuration or source hash differs: {target}")
        if training.hash_file(target) != manifest.get("reference_sha256"):
            raise ValueError(f"Reference cache SHA-256 mismatch: {target}")
        reference = torch.load(target, map_location=args.device, weights_only=True)
        if set(reference) != {"true", "train_prefix", "times"}:
            raise ValueError("Reference cache must contain true, train_prefix, and times")
        _prefix_checks(reference, data, system)
        return reference, manifest, target

    generated = generator(n_trajectories=256, t_span=(0, OBSERVATIONS * dt),
                          dt_save=dt, dt_integrate=.001, seed=42, device=args.device)
    _synchronize(args.device)
    if generated["train_states"].shape != (128, OBSERVATIONS, data["train_states"].shape[-1]):
        raise ValueError("Generator must return 128 development trajectories and 1000 observations")
    if not torch.isfinite(generated["train_states"]).all():
        raise FloatingPointError("Generated development trajectories contain nonfinite values")
    reference = {"true": generated["test_states"],
                 "train_prefix": generated["train_states"][:, :100],
                 "times": generated["times"]}
    try:
        errors, time_error = _prefix_checks(reference, data, system)
    except ValueError as error:
        if torch.device(args.device).type == "cpu":
            raise ValueError(
                f"{error}. CPU generation may use a different random sequence from the recorded data; "
                "generate a matching reference in a compatible CUDA environment or supply a verified reference cache."
            ) from error
        raise
    directory.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")
    torch.save({key: value.detach().cpu() for key, value in reference.items()}, temporary)
    temporary.replace(target)
    manifest = {
        **expected, "prefix_max_abs": errors, "prefix_time_max_abs": time_error,
        "reference_sha256": training.hash_file(target),
        "environment": {"torch": torch.__version__, "device": str(args.device)},
    }
    training.write_json(manifest_path, manifest)
    return reference, manifest, target


def _restore(job, args, datasets, engine):
    system, method, seed = job["system"], job["method"], job["seed"]
    path = Path(args.model_dir) / "main" / system / method / f"seed{seed}"
    for filename in ["model.pt", "result.json", "provenance.json"]:
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Missing completed main-suite artifact: {path / filename}")
    result = _read(path / "result.json")
    provenance = _read(path / "provenance.json")
    if (result.get("suite"), result.get("variant"), result.get("system"),
        result.get("method"), result.get("seed"), result.get("epochs")) != (
            "main", "main", system, method, seed, 100):
        raise ValueError("Long-horizon evaluation requires the matching completed 100-epoch main run")
    if result.get("published_epoch_schedule") is False:
        raise ValueError("Checkpoint was not trained with the 100-epoch schedule")
    dataset = datasets[system]
    if result.get("dataset", {}).get("sha256") != dataset["sha256"]:
        raise ValueError("Main-result dataset SHA-256 differs from the verified dataset")
    if provenance.get("source_sha256") != dataset["sha256"]:
        raise ValueError("Checkpoint provenance dataset SHA-256 differs")
    data_path = Path(args.data_dir) / f"{system}_data.pt"
    if engine.hash_file(data_path) != dataset["sha256"]:
        raise ValueError("Dataset changed after verification")
    data = torch.load(data_path, map_location=args.device, weights_only=False)
    dim = data["test_states"].shape[-1]
    if data["train_states"].shape != (128, 100, dim) or data["test_states"].shape != (128, 100, dim):
        raise ValueError("Recorded datasets must have 128 development and 128 test trajectories with 100 observations")
    if data["times"].shape != (100,):
        raise ValueError("Recorded dataset observation times have the wrong shape")
    if abs(float(data["times"][1] - data["times"][0]) - OBSERVATION_STEPS[system]) > 1e-12:
        raise ValueError("Recorded observation step differs from the long-horizon protocol")
    if provenance.get("constraint_coordinates") != "z=(raw_state-mean)/std":
        raise ValueError("Checkpoint constraints must use standardized model coordinates")
    if provenance.get("data_fraction") != 1. or provenance.get("normalized_noise_sigma") != 0.:
        raise ValueError("Long-horizon evaluation requires the full, unperturbed main training split")
    split = torch.randperm(128, generator=torch.Generator().manual_seed(20260906))
    fit_ids, val_ids = split[:102], split[102:]
    for key, expected in [("fit_indices", fit_ids.tolist()),
                          ("normalizer_fit_indices", fit_ids.tolist()),
                          ("validation_indices", val_ids.tolist())]:
        if provenance.get(key) != expected:
            raise ValueError(f"Checkpoint data split differs: {key}")
    mean = torch.tensor(provenance["mean"], device=args.device, dtype=engine.DTYPE)
    std = torch.tensor(provenance["std"], device=args.device, dtype=engine.DTYPE)
    fit = data["train_states"][fit_ids].reshape(-1, dim)
    expected_mean, expected_std = fit.mean(0), fit.std(0).clamp_min(1e-6)
    if mean.shape != (dim,) or std.shape != (dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Invalid checkpoint normalization")
    if not torch.allclose(mean, expected_mean, rtol=1e-12, atol=1e-14) or not torch.allclose(
            std, expected_std, rtol=1e-12, atol=1e-14):
        raise ValueError("Checkpoint normalization differs from the fitting split")
    classes = dict(zip(engine.SYSTEMS, [engine.FitzHughNagumo, engine.LotkaVolterra,
                                       engine.ShallowWater, engine.FrankaRobot]))
    physical = classes[system]()
    if data.get("k_max") is not None:
        physical.k_max = data["k_max"]
    physical = physical.get_manifold()
    if provenance["constraint_energy_threshold"] != physical.e_threshold or provenance["constraint_amplitude_threshold"] != 2.:
        raise ValueError("Checkpoint constraint thresholds differ")
    constraint = engine.StandardizedConstraint(physical, mean, std)
    checkpoint = torch.load(path / "model.pt", map_location=args.device, weights_only=True)
    if method == "NODE-LAC":
        options = checkpoint.get("options", {})
        if any(options.get(key) for key in ["fixed_gain", "no_correction", "no_constraint_loss"]):
            raise ValueError("An ablation checkpoint cannot serve as the main NODE-LAC model")
        node = engine.NeuralODE(dim, (256, 256), solver="euler").to(device=args.device, dtype=engine.DTYPE)
        gain = engine.GainNet(dim, (128, 128)).to(device=args.device, dtype=engine.DTYPE)
        node.load_state_dict(checkpoint["node"])
        gain.load_state_dict(checkpoint["gain"])
        scale = float(checkpoint["selected_scale"])
        if not math.isfinite(scale) or scale != float(result["selection"]["scale"]):
            raise ValueError("Checkpoint correction scale differs from the main result")
        model = engine.ClosedLoopDynamics(node.f, gain, constraint, correction_scale=scale)
        padded = False
    else:
        model, padded = engine.build_baseline(method, dim, constraint)
        model = model.to(device=args.device, dtype=engine.DTYPE)
        model.load_state_dict(checkpoint)
    model.eval()

    def predict(raw_initial, times):
        initial = (raw_initial - mean) / std
        with torch.no_grad():
            if method == "NODE-LAC":
                z = engine.euler_integrate(model, initial, times).permute(1, 0, 2)
            else:
                z = model.predict(engine.pad(initial, padded), times).permute(1, 0, 2)[..., :dim]
        return z, z * std + mean

    short_z, short_raw = predict(data["test_states"][:, 0], data["times"])
    measured = _public_metrics(engine.metrics(short_raw, data["test_states"], physical, short_z,
                                              (data["test_states"] - mean) / std))
    if not {"MSE", "MAE", "TCE", "CE"}.issubset(result["metrics"]):
        raise ValueError("Main result is missing prediction or constraint metrics")
    differences = {}
    for name, expected in result["metrics"].items():
        if name not in measured:
            if name == "Stability":
                continue
            raise ValueError(f"Cannot replay unknown main metric: {name}")
        actual = measured[name]
        if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=2e-10, rel_tol=2e-8):
            raise ValueError(f"Main checkpoint metric mismatch for {system}/{method}/{seed}/{name}: {actual} vs {expected}")
        differences[name] = abs(actual - expected)
    return path, result, provenance, data, mean, std, physical, constraint, predict, short_raw, differences


def evaluate(job, args, datasets, runtime):
    """Replay the main evaluation, then infer over 1000 matched observations."""
    if job["system"] not in OBSERVATION_STEPS:
        raise ValueError("Long-horizon evaluation supports the three field-system datasets")
    started = time.monotonic()
    engine = runtime[1]
    restored = _restore(job, args, datasets, engine)
    path, main_result, provenance, data, mean, std, physical, constraint, predict, short_raw, errors = restored
    reference, manifest, reference_path = _reference(job["system"], args, datasets[job["system"]], data)
    pred_z, pred_raw = predict(reference["true"][:, 0], reference["times"])
    _synchronize(args.device)
    if not torch.isfinite(pred_raw).all():
        raise FloatingPointError("Nonfinite long-horizon model prediction")
    if not torch.allclose(pred_raw[:, :100], short_raw, atol=2e-10, rtol=2e-8):
        raise ValueError("Long prediction does not reproduce the short prediction prefix")
    with torch.no_grad():
        metrics = _public_metrics(engine.metrics(pred_raw, reference["true"], physical, pred_z,
                                                 (reference["true"] - mean) / std))
        curves = {"times": reference["times"].tolist(),
                  "MSE": (pred_raw - reference["true"]).square().mean((0, 2)).tolist(),
                  "CE": constraint.k(pred_z).square().sum(-1).mean(0).tolist()}
    destination = Path(args.output_dir) / job["relative_output"]
    destination.mkdir(parents=True, exist_ok=False)
    engine.write_json(destination / "curves.json", curves)
    if args.save_predictions:
        torch.save({"pred": pred_raw.cpu(), "true": reference["true"].cpu(),
                    "times": reference["times"].cpu()}, destination / "predictions.pt")
    repo = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), Path(training.__file__), *sorted((repo / "nodesac").rglob("*.py"))]
    result = {"protocol": engine.PROTOCOL, **job, "epochs": 100, "metrics": metrics,
              "selection": main_result.get("selection", {}), "dataset": datasets[job["system"]],
              "observations": OBSERVATIONS, "training_performed": False,
              "main_replay": {"metrics_reproduced": True, "metric_absolute_differences": errors,
                              "model_sha256": engine.hash_file(path / "model.pt"),
                              "result_sha256": engine.hash_file(path / "result.json"),
                              "provenance_sha256": engine.hash_file(path / "provenance.json")},
              "reference": {"file": str(reference_path), **manifest},
              "source_sha256": {str(p.relative_to(repo)): engine.hash_file(p) for p in source_paths},
              "seconds": time.monotonic() - started,
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "device": str(args.device), "dtype": "float64"}}
    engine.write_json(destination / "result.json", result)
    print(json.dumps({"run": job["relative_output"], "metrics": metrics}, allow_nan=False), flush=True)
    return result
