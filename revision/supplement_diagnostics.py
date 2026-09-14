#!/usr/bin/env python3
"""Evaluate missing SNDE diagnostics from final checkpoints without retraining."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_standardized_revision import (DEFAULT_OUTPUT,SYSTEMS,SEEDS,PROTOCOL,load_data,
    build_baseline,metrics,diagnostics,hash_file,write_json)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--systems",nargs="+",choices=SYSTEMS,default=SYSTEMS)
    args=parser.parse_args()
    torch.set_num_threads(2)
    statuses=[]
    for system in args.systems:
        for seed in SEEDS:
            directory=args.root/system/"main"/"SNDE"/f"seed{seed}"
            path=directory/"result.json"
            if not path.exists():
                statuses.append({"system":system,"seed":seed,"status":"pending"})
                continue
            result=json.loads(path.read_text())
            if result["protocol"]!=PROTOCOL or result["epochs"]!=100:
                raise ValueError("Only final 100-epoch v2 runs can be evaluated")
            checkpoint_hash=hash_file(directory/"model.pt")
            supplement=directory/"diagnostics_supplement.json"
            if supplement.exists():
                cached=json.loads(supplement.read_text())
                if cached["checkpoint_sha256"]==checkpoint_hash:
                    statuses.append({"system":system,"seed":seed,"status":"cached"})
                    continue
                raise ValueError("Supplement checkpoint hash differs")
            tr,val,te,raw,times,mean,std,base,constraint,provenance=load_data(system,"cpu")
            if provenance["source_sha256"]!=result["dataset_sha256"]:
                raise ValueError("Dataset hash differs from completed run")
            model,padded=build_baseline("SNDE",te.shape[-1],constraint)
            model=model.to(dtype=torch.float64)
            model.load_state_dict(torch.load(directory/"model.pt",map_location="cpu",weights_only=True))
            model.eval()
            with torch.no_grad():
                pred=model.predict(te[:,0],times).permute(1,0,2)
                replay=metrics(pred*std+mean,raw,base,pred,te)
            errors={key:abs(replay[key]-result["metrics"][key]) for key in replay}
            for key,error in errors.items():
                if error>1e-9+1e-7*abs(result["metrics"][key]):
                    raise ValueError(f"Metric replay differs for {system}/{seed}/{key}: {error}")
            data={"protocol":PROTOCOL,"checkpoint_sha256":checkpoint_hash,
                  "dataset_sha256":provenance["source_sha256"],
                  "runner_source_sha256":result["source_sha256"],
                  "evaluation_source_sha256":hash_file(__file__),
                  "replay_metric_absolute_errors":errors,
                  "diagnostics":{"test_reference":diagnostics(model,te,constraint),
                                 "test_rollout":diagnostics(model,pred,constraint)}}
            write_json(supplement,data)
            statuses.append({"system":system,"seed":seed,"status":"complete",
                             "replay_max_abs_error":max(errors.values())})
    print(json.dumps(statuses))

if __name__=="__main__":main()
