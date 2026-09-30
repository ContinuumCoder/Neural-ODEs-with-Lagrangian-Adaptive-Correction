#!/usr/bin/env python3
"""Train and evaluate NODE-LAC and comparator models on the benchmark datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

REPO = Path(__file__).resolve().parent
SYSTEMS = ("fitzhugh_nagumo", "lotka_volterra", "shallow_water", "franka_robot")
METHODS = ("NODE-LAC", "NODE", "SNDE", "ConCerNet", "SymODEN", "HNN",
           "CLNN", "PORT-HJNN", "PNODE", "CPNODE")
SEEDS = (42, 123, 456)
RESIDUAL_WEIGHTS = dict(zip(SYSTEMS, (0.0, 0.001, 0.1, 0.1)))
RESIDUAL_SCALES = dict(zip(SYSTEMS, (
    (0.3, 0.3, 0.4), (0.4, 0.4, 0.4), (0.4, 0.5, 0.4), (0.3, 0.3, 0.3))))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("main", "residual", "ablation", "data-efficiency", "noise", "long-horizon"), default="main")
    parser.add_argument("--data-dir", type=Path, default=REPO / "results",
                        help="Directory containing the four benchmark *_data.pt files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, default=REPO / "data_manifest.json")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--variant", choices=("paired", "zero-control", "regularized"), default="paired",
                        help="Residual suite only.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs; reported runs use 100.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dir", type=Path, help="Main-suite output root for long-horizon evaluation.")
    parser.add_argument("--reference-dir", type=Path, help="Cache for long-horizon reference trajectories.")
    parser.add_argument("--save-predictions", action="store_true", help="Save recorded-coordinate predictions for plotting.")
    parser.add_argument("--dry-run", action="store_true", help="Verify dataset hashes and print the run plan.")
    return parser


def build_plan(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.suite != "residual" and args.variant != "paired":
        raise ValueError("--variant applies only to the residual suite")
    systems = args.systems or list(SYSTEMS if args.suite in ("main", "residual") else SYSTEMS[:3])
    defaults = {"main": METHODS, "residual": ("NODE-LAC",), "ablation": ("NODE-LAC",),
                "data-efficiency": ("NODE-LAC", "NODE", "SNDE"), "noise": ("NODE-LAC", "NODE"),
                "long-horizon": ("NODE-LAC", "NODE", "SNDE", "PNODE", "CPNODE", "ConCerNet")}
    methods = args.methods or list(defaults[args.suite])
    for name, values in (("systems", systems), ("methods", methods), ("seeds", args.seeds)):
        if len(values) != len(set(values)):
            raise ValueError("Duplicate --" + name + " entries are not allowed")
    if args.suite in ("residual", "ablation") and methods != ["NODE-LAC"]:
        raise ValueError("This suite supports only --methods NODE-LAC")
    if args.suite not in ("main", "residual") and "franka_robot" in systems:
        raise ValueError("This suite uses FitzHugh-Nagumo, Lotka-Volterra, and Shallow Water")
    if args.suite == "long-horizon" and args.model_dir is None:
        raise ValueError("--model-dir is required for long-horizon evaluation")
    args.systems = systems
    jobs = []
    for system in systems:
        for seed in args.seeds:
            for method in methods:
                variants = [("main", {}, 1.0, 0.0)]
                if args.suite == "ablation":
                    variants = [("NoGainNet", {"fixed_gain": True}, 1.0, 0.0),
                                ("NoConstraintLoss", {"no_constraint_loss": True}, 1.0, 0.0),
                                ("NoCorrection", {"no_correction": True}, 1.0, 0.0)]
                elif args.suite == "data-efficiency":
                    variants = [(f"fraction_{v}", {}, v, 0.0) for v in (0.25, 0.5, 0.75)]
                elif args.suite == "noise":
                    variants = [(f"sigma_{v}", {}, 1.0, v) for v in (0.05, 0.1, 0.2)]
                elif args.suite == "long-horizon":
                    variants = [("1000", {}, 1.0, 0.0)]
                elif args.suite == "residual":
                    weight = RESIDUAL_WEIGHTS[system]
                    names = ["zero-control", "regularized"] if args.variant == "paired" else [args.variant]
                    if args.variant == "paired" and weight == 0:
                        names = ["zero-control"]
                    variants = [(v, {}, 1.0, 0.0) for v in names]
                for variant, options, fraction, noise in variants:
                    options = {"mu_init": 0.1, "effort_weight": 0.05, **options}
                    residual_weight, scale = 0.0, None
                    if args.suite == "residual":
                        options = ({"mu_init": 0.01, "effort_weight": 0.01} if system == "fitzhugh_nagumo"
                                   else {"mu_init": 0.05, "effort_weight": 0.05})
                        residual_weight = 0.0 if variant == "zero-control" else RESIDUAL_WEIGHTS[system]
                        scale = RESIDUAL_SCALES[system][SEEDS.index(seed)]
                    folder = (f"main/{system}/{method}/seed{seed}" if args.suite == "main" else
                              f"{args.suite}/{system}/{method}/{variant}/seed{seed}")
                    jobs.append({"suite": args.suite, "system": system, "method": method, "variant": variant,
                                 "seed": seed, "epochs": args.epochs, "options": options, "fraction": fraction,
                                 "noise": noise, "lambda_L": residual_weight, "evaluation_scale": scale,
                                 "also_selected_model": args.suite == "residual" and RESIDUAL_WEIGHTS[system] == 0,
                                 "relative_output": folder})
    return jobs


def verify_data(args):
    manifest = json.loads(args.data_manifest.read_text())
    verified = {}
    for system in args.systems:
        filename = system + "_data.pt"
        expected = manifest["files"][filename]["sha256"]
        path = args.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing released dataset: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Dataset SHA-256 mismatch: {filename}")
        verified[system] = {"file": filename, "sha256": actual,
                            "source_commit": manifest.get("source_commit")}
    return verified


def load_runtime(data_dir):
    import torch
    from experiments import training, residual, diagnostics
    training.DATA = Path(data_dir).resolve()
    torch.set_num_threads(2)
    return torch, training, residual, diagnostics


def dispatch_training(job, tr, val, times, constraint, device, path, runtime):
    """Dispatch to the trajectory and residual training implementations."""
    _, training, residual, _ = runtime
    if job["method"] != "NODE-LAC":
        model, padded = training.train_baseline(
            job["method"], tr, times, constraint, device, job["epochs"], path)
        return model, padded, {}
    if job["suite"] == "residual" and job["lambda_L"] > 0:
        model, selection = residual.train_lac(
            tr, val, times, constraint, device, job["epochs"], path,
            job["options"], job["lambda_L"])
    else:
        model, selection = training.train_lac(
            tr, val, times, constraint, device, job["epochs"], path, job["options"])
    return model, False, selection


def use_published_scale(model, selection, job):
    selection = dict(selection)
    if job["suite"] == "residual":
        selection["training_validation_scale"] = selection["scale"]
        selection["scale"] = job["evaluation_scale"]
        selection["evaluation_rule"] = "Published paired scale, shared by control and regularized model"
        model.correction_scale = job["evaluation_scale"]
    elif job["options"].get("no_correction"):
        selection["evaluation_rule"] = "Fixed zero correction scale for NoCorrection"
    else:
        selection["evaluation_rule"] = "Minimum validation MSE on the published seven-point grid"
    return selection


def run_job(job, args, datasets, runtime):
    torch, training, residual, evaluation = runtime
    path = args.output_dir / job["relative_output"]
    path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    training.seed_everything(job["seed"])
    tr, val, test, raw, times, mean, std, physical, constraint, provenance = training.load_data(
        job["system"], args.device, fraction=job["fraction"], noise=job["noise"], seed=job["seed"])
    if provenance["source_sha256"] != datasets[job["system"]]["sha256"]:
        raise ValueError("Dataset changed after manifest verification")
    training.seed_everything(job["seed"])
    model, padded, selection = dispatch_training(
        job, tr, val, times, constraint, args.device, path, runtime)
    diagnostics = {}
    if job["method"] == "NODE-LAC":
        selection = use_published_scale(model, selection, job)
        if job["suite"] == "residual":
            checkpoint = torch.load(path / "model.pt", map_location=args.device, weights_only=False)
            checkpoint["training_validation_scale"] = checkpoint["selected_scale"]
            checkpoint["selected_scale"] = job["evaluation_scale"]
            checkpoint["lambda_L"] = job["lambda_L"]
            torch.save(checkpoint, path / "model.pt")
        with torch.no_grad():
            pred = training.euler_integrate(model, test[:, 0], times).permute(1, 0, 2)
        if job["suite"] == "residual":
            diagnostics = {
                "reference": evaluation.full_diagnostics(model, test, constraint),
                "rollout": evaluation.full_diagnostics(model, pred, constraint)}
        else:
            diagnostics = {"test_reference": training.diagnostics(model, test, constraint),
                           "test_rollout": training.diagnostics(model, pred, constraint)}
    else:
        with torch.no_grad():
            pred = model.predict(training.pad(test[:, 0], padded), times).permute(1, 0, 2)
            pred = pred[..., :tr.shape[-1]]
        if job["method"] in ("NODE", "SNDE"):
            diagnostics = {"test_reference": training.diagnostics(model, test, constraint),
                           "test_rollout": training.diagnostics(model, pred, constraint)}
    with torch.no_grad():
        pred_raw = pred * std + mean
        if not torch.isfinite(pred_raw).all():
            raise FloatingPointError("Nonfinite test prediction")
        metrics = training.metrics(pred_raw, raw, physical, pred, test)
        metrics["near_feasible_fraction"] = metrics.pop("Stability")
    if job["suite"] == "residual":
        metrics["residual_score"] = evaluation.score(diagnostics)
    manifest = json.loads((REPO / "source_manifest.json").read_text())
    sources = [Path(__file__), *(REPO / name for name in manifest["files"]),
               *sorted((REPO / "nodesac").rglob("*.py"))]
    result = {"protocol": "node-lac-benchmarks", **job, "metrics": metrics,
              "selection": selection, "diagnostics": diagnostics,
              "dataset": datasets[job["system"]], "seconds": time.monotonic() - started,
              "published_epoch_schedule": args.epochs == 100,
              "source_sha256": {str(f.relative_to(REPO)): file_sha256(f) for f in sources},
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "device": args.device, "dtype": "float64"}}
    curves = {"times": times.tolist(), "MSE": (pred_raw - raw).square().mean((0, 2)).tolist(),
              "CE": constraint.k(pred).square().sum(-1).mean(0).tolist()}
    training.write_json(path / "curves.json", curves)
    if args.save_predictions:
        torch.save({"pred": pred_raw.cpu(), "true": raw.cpu(), "times": times.cpu()}, path / "predictions.pt")
    training.write_json(path / "provenance.json", provenance)
    training.write_json(path / "result.json", result)
    print(json.dumps({"run": job["relative_output"], "metrics": metrics}, allow_nan=False), flush=True)
    return result


def write_summary(results, output_dir, suite):
    groups = {}
    for result in results:
        key = (result["system"], result["method"], result["variant"])
        groups.setdefault(key, []).append(result)
    rows = []
    metrics = ("MSE", "MAE", "TCE", "CE", "memory_E_MSE", "residual_score")
    for (system, method, variant), records in sorted(groups.items()):
        row = {"system": system, "method": method, "variant": variant, "n_seeds": len(records)}
        for name in metrics:
            values = [record["metrics"].get(name) for record in records]
            valid = values and all(v is not None for v in values)
            row[name + "_mean"] = statistics.mean(values) if valid else None
            row[name + "_sample_sd"] = statistics.stdev(values) if valid and len(values) > 1 else None
        rows.append(row)
    destination = output_dir / suite / "summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        args.data_dir = args.data_dir.resolve()
        args.output_dir = args.output_dir.resolve()
        jobs = build_plan(args)
        datasets = verify_data(args)
        existing = [str(args.output_dir / j["relative_output"]) for j in jobs
                    if (args.output_dir / j["relative_output"]).exists()]
        if existing:
            raise FileExistsError("Run directories already exist; choose a fresh output directory: "
                                  + ", ".join(existing[:3]))
        if args.dry_run:
            print(json.dumps({"suite": args.suite, "run_count": len(jobs), "datasets": datasets,
                              "runs": jobs}, indent=2, allow_nan=False))
            return 0
        runtime = load_runtime(args.data_dir)
        torch = runtime[0]
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose --device cpu or install the CUDA environment")
        if args.suite == "long-horizon":
            from experiments.extrapolation import evaluate
            results = [evaluate(job, args, datasets, runtime) for job in jobs]
        else:
            results = [run_job(job, args, datasets, runtime) for job in jobs]
        print(f"Summary: {write_summary(results, args.output_dir, args.suite)}")
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
