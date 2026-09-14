#!/usr/bin/env python3
"""Reproduce the published main comparison and paired residual ablation."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
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
    parser.add_argument("--suite", choices=("main", "residual"), default="main")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the four released *_data.pt files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path,
                        default=REPO / "revision" / "data_manifest.json")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--variant", choices=("paired", "zero-control", "regularized"),
                        default="paired", help="Residual suite only.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Published setting: 100. Other values are exploratory runs.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify data hashes and print the plan without loading tensors or training.")
    return parser


def build_plan(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.suite == "main" and args.variant != "paired":
        raise ValueError("--variant applies only to the residual suite")
    if args.suite == "residual" and args.methods not in (None, ["NODE-LAC"]):
        raise ValueError("The residual suite supports only --methods NODE-LAC")
    for name in ("systems", "seeds"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError("Duplicate --" + name + " entries are not allowed")
    methods = args.methods or (list(METHODS) if args.suite == "main" else ["NODE-LAC"])
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicate --methods entries are not allowed")
    jobs = []
    for system in args.systems:
        for seed in args.seeds:
            if args.suite == "main":
                for method in methods:
                    jobs.append({"suite": "main", "system": system, "method": method,
                                 "variant": "main", "seed": seed, "epochs": args.epochs,
                                 "options": {"mu_init": 0.1, "effort_weight": 0.05},
                                 "lambda_L": 0.0, "evaluation_scale": None,
                                 "relative_output": f"main/{system}/{method}/seed{seed}"})
            else:
                weight = RESIDUAL_WEIGHTS[system]
                variants = (["zero-control", "regularized"] if args.variant == "paired"
                            else [args.variant])
                if args.variant == "paired" and weight == 0:
                    variants = ["zero-control"]
                options = ({"mu_init": 0.01, "effort_weight": 0.01}
                           if system == "fitzhugh_nagumo"
                           else {"mu_init": 0.05, "effort_weight": 0.05})
                for variant in variants:
                    jobs.append({"suite": "residual", "system": system, "method": "NODE-LAC",
                                 "variant": variant, "seed": seed, "epochs": args.epochs,
                                 "options": dict(options),
                                 "lambda_L": 0.0 if variant == "zero-control" else weight,
                                 "evaluation_scale": RESIDUAL_SCALES[system][SEEDS.index(seed)],
                                 "also_selected_model": weight == 0,
                                 "relative_output": f"residual/{system}/{variant}/seed{seed}"})
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
    revision = str(REPO / "revision")
    if revision not in sys.path:
        sys.path.insert(0, revision)
    frozen = importlib.import_module("run_standardized_revision")
    residual = importlib.import_module("tune_lyapunov_loss")
    evaluation = importlib.import_module("evaluate_lyapunov_loss")
    # The caller chooses storage; the imported training functions stay unchanged.
    frozen.DATA = Path(data_dir).resolve()
    torch.set_num_threads(2)
    return torch, frozen, residual, evaluation


def dispatch_training(job, tr, val, times, constraint, device, path, runtime):
    """Call the published training routines without changing their update rules."""
    _, frozen, residual, _ = runtime
    if job["method"] != "NODE-LAC":
        model, padded = frozen.train_baseline(
            job["method"], tr, times, constraint, device, job["epochs"], path)
        return model, padded, {}
    if job["suite"] == "residual" and job["lambda_L"] > 0:
        model, selection = residual.train_lac(
            tr, val, times, constraint, device, job["epochs"], path,
            job["options"], job["lambda_L"])
    else:
        model, selection = frozen.train_lac(
            tr, val, times, constraint, device, job["epochs"], path, job["options"])
    return model, False, selection


def use_published_scale(model, selection, job):
    selection = dict(selection)
    if job["suite"] == "residual":
        selection["training_validation_scale"] = selection["scale"]
        selection["scale"] = job["evaluation_scale"]
        selection["evaluation_rule"] = "Published paired scale, shared by control and regularized model"
        model.correction_scale = job["evaluation_scale"]
    else:
        selection["evaluation_rule"] = "Minimum validation MSE on the published seven-point grid"
    return selection


def run_job(job, args, datasets, runtime):
    torch, frozen, residual, evaluation = runtime
    path = args.output_dir / job["relative_output"]
    path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen.seed_everything(job["seed"])
    tr, val, test, raw, times, mean, std, physical, constraint, provenance = frozen.load_data(
        job["system"], args.device, seed=job["seed"])
    if provenance["source_sha256"] != datasets[job["system"]]["sha256"]:
        raise ValueError("Dataset changed after manifest verification")
    frozen.seed_everything(job["seed"])
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
            pred = frozen.euler_integrate(model, test[:, 0], times).permute(1, 0, 2)
        if job["suite"] == "residual":
            diagnostics = {
                "reference": evaluation.full_diagnostics(model, test, constraint),
                "rollout": evaluation.full_diagnostics(model, pred, constraint)}
        else:
            diagnostics = {"test_reference": frozen.diagnostics(model, test, constraint),
                           "test_rollout": frozen.diagnostics(model, pred, constraint)}
    else:
        with torch.no_grad():
            pred = model.predict(frozen.pad(test[:, 0], padded), times).permute(1, 0, 2)
            pred = pred[..., :tr.shape[-1]]
        if job["method"] == "NODE":
            diagnostics = {"test_reference": frozen.diagnostics(model, test, constraint),
                           "test_rollout": frozen.diagnostics(model, pred, constraint)}
    with torch.no_grad():
        pred_raw = pred * std + mean
        if not torch.isfinite(pred_raw).all():
            raise FloatingPointError("Nonfinite test prediction")
        metrics = frozen.metrics(pred_raw, raw, physical, pred, test)
    if job["suite"] == "residual":
        metrics["residual_score"] = evaluation.score(diagnostics)
    manifest = json.loads((REPO / "revision" / "training_source_manifest.json").read_text())
    sources = [Path(__file__), *(REPO / name for name in manifest["files"]),
               *sorted((REPO / "nodesac").rglob("*.py"))]
    result = {"protocol": "published-main-and-residual", **job, "metrics": metrics,
              "selection": selection, "diagnostics": diagnostics,
              "dataset": datasets[job["system"]], "seconds": time.monotonic() - started,
              "published_epoch_schedule": args.epochs == 100,
              "source_sha256": {str(f.relative_to(REPO)): file_sha256(f) for f in sources},
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "device": args.device, "dtype": "float64"}}
    frozen.write_json(path / "provenance.json", provenance)
    frozen.write_json(path / "result.json", result)
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
        results = [run_job(job, args, datasets, runtime) for job in jobs]
        print(f"Summary: {write_summary(results, args.output_dir, args.suite)}")
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
