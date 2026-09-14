#!/usr/bin/env python3
"""Aggregate independently completed seeds without mixing legacy/smoke results."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_standardized_revision import DEFAULT_OUTPUT,METHODS,SYSTEMS,SEEDS,PROTOCOL,write_json

def summarize(root):
    groups=defaultdict(list)
    for path in root.glob("*/*/*/seed*/result.json"):
        result=json.loads(path.read_text())
        if result["protocol"] != PROTOCOL or result["epochs"] !=100:continue
        result["_path"]=str(path)
        groups[(result["system"],result["variant"],result["method"])].append(result)
    output={}
    for (system,variant,method),values in sorted(groups.items()):
        values.sort(key=lambda v:v["seed"])
        seeds=[v["seed"] for v in values]
        if len(seeds)!=len(set(seeds)):raise ValueError("Duplicate optimization seed")
        entry={"seed_count":len(values),"seeds":seeds,
               "complete":seeds==sorted(SEEDS),
               "dataset_sha256s":sorted(set(v["dataset_sha256"] for v in values)),
               "source_sha256s":sorted(set(v["source_sha256"] for v in values)),
               "metrics":{},"runs":[v["_path"] for v in values]}
        if len(entry["dataset_sha256s"])!=1 or len(entry["source_sha256s"])!=1:
            raise ValueError("Provenance differs within a three-seed group")
        for metric in values[0]["metrics"]:
            sample=[v["metrics"][metric] for v in values]
            entry["metrics"][metric]={"mean":statistics.mean(sample),
                "sample_sd":statistics.stdev(sample) if len(sample)>1 else None,
                "values":sample}
        entry["diagnostics"]={}
        for value in values:
            supplement=Path(value["_path"]).parent/"diagnostics_supplement.json"
            if supplement.exists():
                extra=json.loads(supplement.read_text())
                if extra["protocol"]!=PROTOCOL or extra["runner_source_sha256"]!=value["source_sha256"] or extra["dataset_sha256"]!=value["dataset_sha256"]:
                    raise ValueError("Diagnostic supplement provenance differs")
                value["diagnostics"]=extra["diagnostics"]
        for kind in ["test_reference","test_rollout"]:
            if not all(kind in value.get("diagnostics",{}) for value in values):
                continue
            entry["diagnostics"][kind]={}
            for metric in values[0]["diagnostics"][kind]:
                sample=[value["diagnostics"][kind][metric] for value in values]
                if not all(x is None or isinstance(x,(int,float)) for x in sample):
                    continue
                available=[x for x in sample if x is not None]
                entry["diagnostics"][kind][metric]={
                    "mean":statistics.mean(available) if available else None,
                    "sample_sd":statistics.stdev(available) if len(available)>1 else None,
                    "maximum":max(available) if available else None,
                    "valid_seed_count":len(available),"values":sample}
        entry["selection"]=[{"seed":value["seed"],**value.get("selection",{})} for value in values]
        output.setdefault(system,{}).setdefault(variant,{})[method]=entry
    missing=[f"{system}/main/{method}/seed{seed}"
             for system in SYSTEMS for method in METHODS for seed in SEEDS
             if not (root/system/"main"/method/f"seed{seed}"/"result.json").exists()]
    failures=[json.loads(path.read_text()) for path in root.glob("*/*/*/seed*/failure.json")
              if not (path.parent/"result.json").exists()]
    return {"protocol":PROTOCOL,"definition":"arithmetic mean and sample standard deviation (ddof=1) across independent optimization seeds; datasets and splits fixed",
            "groups":output,"main_missing":missing,"unresolved_failures":failures}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--require-main-complete",action="store_true")
    args=parser.parse_args()
    data=summarize(args.root)
    destination=args.root/"summary"
    write_json(destination/"summary.json",data)
    output=io.StringIO()
    writer=csv.writer(output)
    writer.writerow(["system","variant","method","metric","n_seeds","mean","sample_sd","complete"])
    for system,variants in data["groups"].items():
        for variant,methods in variants.items():
            for method,group in methods.items():
                for metric,values in group["metrics"].items():
                    writer.writerow([system,variant,method,metric,group["seed_count"],
                                     values["mean"],values["sample_sd"],group["complete"]])
    (destination/"metrics.csv").write_text(output.getvalue())
    print(json.dumps({"main_completed":120-len(data["main_missing"]),
                      "main_total":120,"unresolved_failures":len(data["unresolved_failures"]),
                      "summary":str(destination/"summary.json")}))
    if args.require_main_complete and data["main_missing"]:sys.exit(2)

if __name__=="__main__":main()
