#!/usr/bin/env python3
"""Validate the complete canonical revision packet and its seed summaries."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_standardized_revision import DEFAULT_OUTPUT,PROTOCOL,SYSTEMS,METHODS,SEEDS,hash_file,write_json

def expected_jobs():
    jobs=[]
    for system in SYSTEMS:
        for method in METHODS:
            for seed in SEEDS:jobs.append((system,"main",method,seed,"main"))
    for system in SYSTEMS[:3]:
        for method in ["NODE-LAC","NODE","SNDE"]:
            for fraction in [.25,.5,.75]:
                for seed in SEEDS:jobs.append((system,f"data_eff_{fraction}",method,seed,"reviewer"))
        for method in ["NODE-LAC","NODE"]:
            for noise in [.05,.1,.2]:
                for seed in SEEDS:jobs.append((system,f"noise_{noise}",method,seed,"reviewer"))
        for name in ["NoGainNet","NoConstraintLoss","NoCorrection"]:
            for seed in SEEDS:jobs.append((system,"ablation_"+name,"NODE-LAC",seed,"ablation"))
    for mu,effort in [(.1,.05),(.05,.05),(.1,.01),(.01,.01)]:
        for seed in SEEDS:jobs.append(("fitzhugh_nagumo",f"hyper_mu{mu}_effort{effort}","NODE-LAC",seed,"hyperparam"))
    assert len(jobs)==294
    return jobs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--status",action="store_true")
    args=parser.parse_args()
    jobs=expected_jobs()
    expected={f"{s}/{v}/{m}/seed{seed}":phase for s,v,m,seed,phase in jobs}
    actual={str(p.parent.relative_to(args.root)):p for p in args.root.glob("*/*/*/seed*/result.json")}
    missing=sorted(set(expected)-set(actual))
    extra=sorted(set(actual)-set(expected))
    phases=Counter(expected[key] for key in actual if key in expected)
    failures=[str(p.relative_to(args.root)) for p in args.root.glob("*/*/*/seed*/failure.json")
              if not (p.parent/"result.json").exists()]
    report={"protocol":PROTOCOL,"expected_runs":294,"completed_runs":len(actual),
            "completed_by_phase":dict(phases),"expected_by_phase":dict(Counter(expected.values())),
            "missing_runs":missing,"unexpected_runs":extra,"unresolved_failures":failures}
    if args.status:
        print(json.dumps({k:v for k,v in report.items() if k not in ["missing_runs","unexpected_runs"]}))
        return
    if missing or extra or failures:
        write_json(args.root/"validation.json",report)
        raise RuntimeError("The canonical packet is incomplete or has unresolved failures")
    repository=Path(__file__).resolve().parents[1]
    source_manifest=json.loads((repository/"revision/source_manifest_v2.json").read_text())
    for path,expected_hash in source_manifest["source_sha256"].items():
        if hash_file(repository/path)!=expected_hash:raise ValueError(f"Source changed: {path}")
    runner_hash=source_manifest["source_sha256"]["revision/run_standardized_revision.py"]
    archive=json.loads((repository/"revision/legacy_artifacts.json").read_text())
    dataset_hashes={system:archive[f"results/{system}_data.pt"]["sha256"] for system in SYSTEMS}
    for system in SYSTEMS:
        if hash_file(archive[f"results/{system}_data.pt"]["server_path"])!=dataset_hashes[system]:
            raise ValueError("Archived dataset changed")
    records={}
    for system,variant,method,seed,phase in jobs:
        key=f"{system}/{variant}/{method}/seed{seed}"
        directory=args.root/key
        result=json.loads((directory/"result.json").read_text())
        if (result["protocol"],result["system"],result["variant"],result["method"],result["seed"],result["epochs"])!=(PROTOCOL,system,variant,method,seed,100):
            raise ValueError(f"Metadata mismatch: {key}")
        if result["source_sha256"]!=runner_hash or result["dataset_sha256"]!=dataset_hashes[system]:
            raise ValueError(f"Provenance mismatch: {key}")
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in result["metrics"].values()):
            raise ValueError(f"Invalid metric: {key}")
        provenance=json.loads((directory/"provenance.json").read_text())
        if set(provenance["fit_indices"])&set(provenance["validation_indices"]):
            raise ValueError("Fit and validation subsets overlap")
        if len(provenance["validation_indices"])!=26 or provenance["test_indices"]!=list(range(128)):
            raise ValueError("Evaluation partition changed")
        if provenance["constraint_coordinates"]!="z=(raw_state-mean)/std":
            raise ValueError("Constraint-coordinate mismatch")
        history=json.loads((directory/"history.json").read_text())
        curves=json.loads((directory/"curves.json").read_text())
        if [entry["epoch"] for entry in history]!=list(range(1,101)):
            raise ValueError("Training history does not contain all 100 epochs")
        if any(len(curves[k])!=100 for k in ["times","MSE","CE"]):
            raise ValueError("Prediction curve has an unexpected horizon")
        checkpoint=torch.load(directory/"model.pt",map_location="cpu",weights_only=True)
        if method=="NODE-LAC":
            selection=result["selection"]
            if checkpoint["selected_scale"]!=selection["scale"]:
                raise ValueError("Selected correction scale differs")
            if variant=="ablation_NoCorrection" and (checkpoint["training_scale"]!=0 or checkpoint["selected_scale"]!=0):
                raise ValueError("NoCorrection ablation re-enables correction")
            if variant=="ablation_NoGainNet" and (checkpoint["gain"] or not checkpoint["options"].get("fixed_gain")):
                raise ValueError("NoGainNet ablation is not constant-gain")
            if variant=="ablation_NoConstraintLoss" and not checkpoint["options"].get("no_constraint_loss"):
                raise ValueError("NoConstraintLoss option missing")
        records[key]=result
    default_replay=[]
    for seed in SEEDS:
        main_result=records[f"fitzhugh_nagumo/main/NODE-LAC/seed{seed}"]
        hyper_result=records[f"fitzhugh_nagumo/hyper_mu0.1_effort0.05/NODE-LAC/seed{seed}"]
        metric_differences={}
        for name,value in main_result["metrics"].items():
            reproduced=hyper_result["metrics"][name]
            if not math.isclose(value,reproduced,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError(f"Default hyperparameter run fails main replay: seed {seed}, {name}")
            metric_differences[name]=abs(value-reproduced)
        for name in ["scale","effective_mu","final_trajectory_loss"]:
            if not math.isclose(main_result["selection"][name],hyper_result["selection"][name],rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError(f"Default hyperparameter selection differs: seed {seed}, {name}")
        for scale,value in main_result["selection"]["validation_scale_scores"].items():
            if not math.isclose(value,hyper_result["selection"]["validation_scale_scores"][scale],rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError(f"Default validation-scale score differs: seed {seed}, scale {scale}")
        default_replay.append({"seed":seed,"maximum_metric_absolute_difference":max(metric_differences.values()),
                               "selected_scale":main_result["selection"]["scale"],
                               "selection_reproduced":True})
    summary=json.loads((args.root/"summary/summary.json").read_text())
    group_count=0
    for system,variants in summary["groups"].items():
        for variant,methods in variants.items():
            for method,group in methods.items():
                group_count+=1
                if not group["complete"] or group["seeds"]!=SEEDS:
                    raise ValueError("Incomplete three-seed summary")
                for metric,summary_metric in group["metrics"].items():
                    sample=[records[f"{system}/{variant}/{method}/seed{seed}"]["metrics"][metric] for seed in SEEDS]
                    if sample!=summary_metric["values"]:raise ValueError("Summary values differ from runs")
                    for name,value in [("mean",statistics.mean(sample)),("sample_sd",statistics.stdev(sample))]:
                        if not math.isclose(summary_metric[name],value,rel_tol=1e-12,abs_tol=1e-14):
                            raise ValueError("Seed summary arithmetic mismatch")
    if group_count!=98:raise ValueError("Expected 98 complete three-seed groups")
    for name,n_rows in [("main_results",40),("constraint_diagnostics",24),("memory_coordinate_error",4),
                        ("ablation_results",12),("hyperparameter_results",4),("selected_scales_and_dual_weights",12)]:
        with (args.root/"publication"/(name+".csv")).open() as stream:
            if len(list(csv.DictReader(stream)))!=n_rows:raise ValueError("Publication row count differs")
        if name!="selected_scales_and_dual_weights" and not (args.root/"publication"/(name+".tex")).exists():
            raise ValueError("Missing publication TeX")
    report.update(validated=True,validated_checkpoints=294,complete_seed_groups=98,
                  runner_source_sha256=runner_hash,dataset_sha256=dataset_hashes,
                  tables_validated=True,default_hyperparameter_replay=default_replay,validation_source_sha256=hash_file(__file__))
    write_json(args.root/"validation.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["missing_runs","unexpected_runs"]}))

if __name__=="__main__":main()
