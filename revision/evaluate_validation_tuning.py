#!/usr/bin/env python3
"""Evaluate the immutable validation choice; never choose settings on test data."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_validation as tune
frozen = tune.frozen
OUTPUT = tune.ROOT / "test_evaluation"
VARIANTS = ["reference", "refined_scale", "selected_weights"]
METRICS = ["MSE", "MAE", "TCE", "CE", "CE_max", "reference_CE", "feasible_fraction", "memory_E_MSE"]

def verify_lock():
    lock_path = tune.ROOT / "selection_lock.json"
    lock = tune.read(lock_path)
    if not lock.get("locked") or lock["selection_split"] != "validation":
        raise ValueError("A completed validation selection lock is required")
    if lock["config_sha256"] != tune.CONFIG_HASH or lock["tuning_source_sha256"] != tune.SOURCE_HASH:
        raise ValueError("Selection protocol or search source differs")
    if lock["record_count"] != 48 or len(lock["candidate_inventory"]) != 48:
        raise ValueError("Incomplete selection inventory")
    for item in lock["candidate_inventory"]:
        if tune.sha(item["path"]) != item["sha256"]:
            raise ValueError("Candidate record changed after selection")
        record = tune.read(item["path"])
        if tune.sha(record["checkpoint"]["path"]) != record["checkpoint"]["sha256"]:
            raise ValueError("Candidate checkpoint changed after selection")
    tune.check_frozen()
    return lock, tune.sha(lock_path)

def near(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Reference replay differs for {name}: {actual} vs {expected}")

def run():
    torch.set_num_threads(2)
    lock, lock_hash = verify_lock()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evaluation_source_hash = tune.sha(__file__)
    existing = OUTPUT / "evaluation.json"
    if existing.exists():
        old = tune.read(existing)
        if old["selection_lock_sha256"] != lock_hash or old["evaluation_source_sha256"] != evaluation_source_hash:
            raise ValueError("Existing evaluation belongs to another selection/source")
        print(json.dumps({"status": "cached", "path": str(existing)}), flush=True)
        return
    rows = []
    replay_checks = []
    started = time.time()
    for system in frozen.SYSTEMS:
        tr, val, te, raw, times, mean, std, physical, constraint, provenance = frozen.load_data(system, "cuda:1")
        del tr, val
        selected_index = lock["selection_by_system"][system]["selected"]["candidate_index"]
        for seed in tune.SEEDS:
            reference = tune.read(tune.candidate_dir(system, 0, seed) / "candidate.json")
            selected = tune.read(tune.candidate_dir(system, selected_index, seed) / "candidate.json")
            tune.verify_preprocessing(provenance, reference["preprocessing"])
            tune.verify_preprocessing(provenance, selected["preprocessing"])
            for candidate in (reference, selected):
                if provenance["source_sha256"] != candidate["source_dataset_sha256"]:
                    raise ValueError("Evaluated dataset differs from the selected candidate source")
            requests = [
                ("reference", reference, reference["original_grid_selected"]),
                ("refined_scale", reference, reference["selected"]),
                ("selected_weights", selected, selected["selected"]),
            ]
            evaluated = {}
            for variant, candidate, calibration in requests:
                key = (candidate["checkpoint"]["sha256"], calibration["rho"])
                if key not in evaluated:
                    node, gain, ck = tune.model_from_checkpoint(
                        candidate["checkpoint"]["path"], te.shape[-1], constraint, "cuda:1", candidate["candidate_index"])
                    dyn = frozen.ClosedLoopDynamics(node.f, gain, constraint, correction_scale=calibration["rho"])
                    with torch.no_grad():
                        pred = frozen.euler_integrate(dyn, te[:, 0], times).permute(1, 0, 2)
                        pred_raw = pred * std + mean
                        if not torch.isfinite(pred_raw).all():
                            raise FloatingPointError(f"Non-finite locked test rollout: {system}/{seed}/{variant}")
                        values = frozen.metrics(pred_raw, raw, physical, pred, te)
                        values = {name: values[name] for name in METRICS}
                        if not all(math.isfinite(value) for value in values.values()):
                            raise FloatingPointError("Non-finite locked test metric")
                        errors = (pred_raw - raw).square().mean((0, 2)).tolist()
                    diag = {
                        "test_reference": frozen.diagnostics(dyn, te, constraint),
                        "test_rollout": frozen.diagnostics(dyn, pred, constraint),
                    }
                    evaluated[key] = {"metrics": values, "diagnostics": diag, "MSE_by_time": errors}
                result = evaluated[key]
                row = {
                    "system": system, "seed": seed, "variant": variant,
                    "candidate_index": candidate["candidate_index"], "options": candidate["options"],
                    "rho": calibration["rho"], "validation": calibration["validation"],
                    "checkpoint": candidate["checkpoint"], "mu_final": candidate["mu_final"],
                    "gain_validation_reference": candidate["gain"],
                    "dataset_sha256": candidate["source_dataset_sha256"],
                    **result,
                }
                rows.append(row)
                if variant == "reference":
                    prior = tune.read(tune.LEGACY / system / "main" / "NODE-LAC" / f"seed{seed}" / "result.json")
                    if prior["dataset_sha256"] != row["dataset_sha256"]:
                        raise ValueError("Historical reference dataset differs")
                    for name in METRICS:
                        near(row["metrics"][name], prior["metrics"][name], f"{system}/{seed}/{name}")
                    replay_checks.append({"system": system, "seed": seed, "all_reference_metrics_match": True})
                print(json.dumps({
                    "system": system, "seed": seed, "variant": variant,
                    "rho": row["rho"], "test_MSE": row["metrics"]["MSE"], "test_CE": row["metrics"]["CE"]
                }), flush=True)
    summaries = []
    comparisons = []
    for system in frozen.SYSTEMS:
        group = {}
        for variant in VARIANTS:
            subset = [r for r in rows if r["system"] == system and r["variant"] == variant]
            if [r["seed"] for r in subset] != tune.SEEDS:
                raise ValueError("Incomplete seed group")
            item = {
                "system": system, "variant": variant, "candidate_index": subset[0]["candidate_index"],
                "options": subset[0]["options"], "seeds": tune.SEEDS, "rhos": [r["rho"] for r in subset],
                "validation_MSE_mean": statistics.mean(r["validation"]["MSE"] for r in subset),
                "validation_MSE_sd": statistics.stdev(r["validation"]["MSE"] for r in subset),
            }
            for name in METRICS:
                item[name + "_mean"] = statistics.mean(r["metrics"][name] for r in subset)
                item[name + "_sd"] = statistics.stdev(r["metrics"][name] for r in subset)
            summaries.append(item)
            group[variant] = (item, subset)
        base, base_rows = group["reference"]
        for variant in VARIANTS[1:]:
            tuned, tuned_rows = group[variant]
            comparisons.append({
                "system": system, "variant": variant,
                "validation_mean_MSE_reduction_percent": 100 * (1 - tuned["validation_MSE_mean"] / base["validation_MSE_mean"]),
                "test_mean_MSE_reduction_percent": 100 * (1 - tuned["MSE_mean"] / base["MSE_mean"]),
                "test_MSE_paired_differences": [b["metrics"]["MSE"] - t["metrics"]["MSE"] for b, t in zip(base_rows, tuned_rows)],
                "test_CE_paired_differences": [b["metrics"]["CE"] - t["metrics"]["CE"] for b, t in zip(base_rows, tuned_rows)],
            })
    report = {
        "protocol": tune.PROTOCOL, "selection_lock_sha256": lock_hash,
        "evaluation_source_sha256": evaluation_source_hash,
        "tuning_source_sha256": tune.SOURCE_HASH,
        "frozen_runner_sha256": tune.RUNNER_HASH,
        "record_count": len(rows), "seed_group_count": len(summaries),
        "test_partition": "Previously evaluated original archived test_states, unchanged",
        "selection": "All configuration choices fixed by the prior validation-only lock",
        "reference_replay": replay_checks, "summaries": summaries, "comparisons": comparisons,
        "results": rows, "seconds": time.time() - started,
    }
    if len(rows) != 36 or len(replay_checks) != 12:
        raise ValueError("Incomplete evaluation")
    if verify_lock()[1] != lock_hash:
        raise ValueError("Selection lock changed during evaluation")
    tune.write(existing, report)
    for filename, data in [("summary.csv", summaries), ("paired_comparisons.csv", comparisons)]:
        with (OUTPUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            for item in data:
                writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in item.items()})
    tune.write(OUTPUT / "selected_configurations.json", {
        system: {
            "options": lock["selection_by_system"][system]["selected"]["options"],
            "seeds": tune.SEEDS,
            "rho_by_seed": lock["selection_by_system"][system]["selected"]["selected_rhos"],
            "checkpoint_by_seed": lock["selection_by_system"][system]["selected"]["checkpoint_by_seed"],
        } for system in frozen.SYSTEMS
    })
    print(json.dumps({"status": "complete", "rows": len(rows), "path": str(existing)}), flush=True)

if __name__ == "__main__":
    run()
