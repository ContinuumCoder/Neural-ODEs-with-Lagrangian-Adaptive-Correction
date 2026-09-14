#!/usr/bin/env python3
"""Export publication tables from complete, provenance-checked v2 summaries."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_standardized_revision import DEFAULT_OUTPUT,PROTOCOL,SYSTEMS,METHODS,SEEDS,write_json

NAMES={"fitzhugh_nagumo":"FitzHugh--Nagumo","lotka_volterra":"Lotka--Volterra",
       "shallow_water":"Shallow Water","franka_robot":"Robot Arm"}

def number(value):
    if value is None:return r"\text{--}"
    if value==0:return "0"
    if .001<=abs(value)<1000:
        return f"{value:.4g}"
    mantissa,exponent=f"{value:.3e}".split("e")
    return mantissa+r"\times10^{"+str(int(exponent))+"}"

def uncertainty(metric):
    mean,sd=metric["mean"],metric["sample_sd"]
    if mean is None:return r"$\text{--}$"
    if sd is None:raise ValueError("A displayed mean lacks a sample SD")
    if abs(sd)<=1e-12*max(abs(mean),1e-12):sd=0.0
    return "$"+number(mean)+r"\,\pm\,"+number(sd)+"$"

def compact_uncertainty(metric):
    mean,sd=metric["mean"],metric["sample_sd"]
    if mean is None:return r"$\text{--}$"
    if sd is None:raise ValueError("A displayed mean lacks a sample SD")
    if abs(sd)<=1e-12*max(abs(mean),1e-12):sd=0.
    exponent=math.floor(math.log10(abs(mean))) if mean else 0
    scale=10.**exponent
    mantissa=f"{mean/scale:.4g}";spread=f"{sd/scale:.3g}"
    if exponent==0:return "$"+mantissa+r"\,\pm\,"+spread+"$"
    return "$("+mantissa+r"\,\pm\,"+spread+r")\times10^{"+str(exponent)+"}$"

def table_start(caption,label,columns,head,long=False):
    environment="longtable" if long else "tabular"
    if long:
        return [r"\begingroup\footnotesize",r"\setlength{\tabcolsep}{4pt}",
                r"\begin{longtable}{"+columns+"}",
                r"\caption{"+caption+r"}\label{"+label+r"}\\",
                r"\toprule",head+r"\\",r"\midrule",r"\endfirsthead",
                r"\toprule",head+r"\\",r"\midrule",r"\endhead",
                r"\bottomrule",r"\endfoot"]
    return [r"\begin{table}[t]",r"\centering",r"\small",
            r"\caption{"+caption+"}",r"\label{"+label+"}",
            r"\setlength{\tabcolsep}{4pt}",r"\begin{tabular}{"+columns+"}",
            r"\toprule",head+r"\\",r"\midrule"]

def table_end(long=False):
    return [r"\end{longtable}",r"\endgroup"] if long else [
        r"\bottomrule",r"\end{tabular}",r"\end{table}"]

def write_csv(path,rows):
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    Path(path).write_text(stream.getvalue())

def get_group(summary,system,variant,method,expected_hash):
    group=summary["groups"].get(system,{}).get(variant,{}).get(method)
    if group is None or not group["complete"] or sorted(group["seeds"])!=sorted(SEEDS):
        raise LookupError(f"Incomplete three-seed group: {system}/{variant}/{method}")
    if group["source_sha256s"]!=[expected_hash]:
        raise ValueError("Runner source hash differs from canonical manifest")
    if len(group["dataset_sha256s"])!=1:raise ValueError("Mixed dataset provenance")
    return group

def main_table(summary,expected_hash):
    lines=table_start("Prediction performance on the unchanged test trajectories. Entries are mean and sample standard deviation over three seeded runs; prediction metrics use the original state units.",
                      "tab:main_results","lccc","Method & MSE & MAE & TCE",True)
    rows=[]
    for system in SYSTEMS:
        lines.append(r"\multicolumn{4}{l}{\textbf{"+NAMES[system]+r"}}\\")
        for method in METHODS:
            group=get_group(summary,system,"main",method,expected_hash)
            lines.append(method+" & "+" & ".join(uncertainty(group["metrics"][m]) for m in ["MSE","MAE","TCE"])+r"\\")
            row={"system":system,"method":method,"n_seeds":3}
            for metric in ["MSE","MAE","TCE"]:
                row[metric+"_mean"]=group["metrics"][metric]["mean"]
                row[metric+"_sample_sd"]=group["metrics"][metric]["sample_sd"]
            rows.append(row)
        lines.append(r"\addlinespace")
    return lines+table_end(True),rows

def diagnostic_table(summary,expected_hash):
    caption=("Test-rollout decay diagnostics in standardized model coordinates. CE averages all 12,800 saved test states per run; derivatives and residuals use 3,200 sampled rollout states per run. "
             "Entries with uncertainty are mean and sample standard deviation over three seeded runs. "
             r"The decay condition is $\dot V+0.1V\leq0$, with $V=\|k\|^2/2$ and numerical tolerance $10^{-10}$. "
             r"Active states satisfy $V>10^{-12}$. The last column is the largest sampled positive residual across all three runs.")
    lines=table_start(caption,"tab:constraint_diagnostics","lcccc",
                      r"Method & CE & \makecell{Decay condition\\all samples} & \makecell{Decay condition\\active samples} & \makecell{Max.\\residual}")
    rows=[]
    for system in SYSTEMS:
        reference=get_group(summary,system,"main","NODE-LAC",expected_hash)["metrics"]["reference_CE"]["mean"]
        lines.append(r"\multicolumn{5}{l}{\textbf{"+NAMES[system]+r"} (Ref. CE $="+number(reference)+r"$)}\\")
        for method in ["NODE-LAC","NODE","SNDE"]:
            group=get_group(summary,system,"main",method,expected_hash)
            diagnostic=group.get("diagnostics",{}).get("test_rollout")
            if not diagnostic:raise LookupError(f"Missing rollout diagnostics: {system}/{method}")
            active_fraction=diagnostic["active_residual_satisfied_fraction"]
            all_fraction=diagnostic["residual_satisfied_fraction"]
            if active_fraction["valid_seed_count"] not in [0,3]:
                raise ValueError("Cannot pool active-state fractions with missing runs")
            if diagnostic["sample_count"]["values"] != [3200,3200,3200]:
                raise ValueError("The diagnostic caption requires 3200 samples in each run")
            maximum=diagnostic["positive_residual_max"]["maximum"]
            fraction_tex=uncertainty(active_fraction) if active_fraction["valid_seed_count"] else r"$\text{--}$"
            lines.append(method+" & "+uncertainty(group["metrics"]["CE"])+" & "+
                         uncertainty(all_fraction)+" & "+fraction_tex+" & $"+number(maximum)+r"$\\")
            for scope in ["test_reference","test_rollout"]:
                data=group.get("diagnostics",{}).get(scope)
                if not data:raise LookupError(f"Missing diagnostics: {system}/{method}/{scope}")
                ce_key="reference_CE" if scope=="test_reference" else "CE"
                row={"system":system,"method":method,"scope":scope,"seeds":json.dumps(group["seeds"]),
                     "CE_scope_mean":group["metrics"][ce_key]["mean"],
                     "CE_scope_sample_sd":group["metrics"][ce_key]["sample_sd"],
                     "reference_CE":reference,"full_CE_states_per_run":12800,
                     "memory_E_MSE_mean":group["metrics"]["memory_E_MSE"]["mean"],
                     "memory_E_MSE_sample_sd":group["metrics"]["memory_E_MSE"]["sample_sd"],
                     "residual_rate":.1,"condition_tolerance":1e-10,"active_V_threshold":1e-12,
                     "source_sha256":group["source_sha256s"][0],"dataset_sha256":group["dataset_sha256s"][0]}
                for key in ["V_mean","dVdt_mean","dVdt_max","positive_residual_mean",
                            "positive_residual_max","residual_satisfied_fraction",
                            "active_residual_satisfied_fraction"]:
                    row[key+"_mean_across_runs"]=data[key]["mean"]
                    row[key+"_sample_sd"]=data[key]["sample_sd"]
                    row[key+"_by_seed"]=json.dumps(data[key]["values"])
                row["maximum_dVdt_across_three_seed_maxima"]=data["dVdt_max"]["maximum"]
                row["maximum_positive_residual_across_three_seed_maxima"]=data["positive_residual_max"]["maximum"]
                row["active_state_counts_by_seed"]=json.dumps(data["active_count"]["values"])
                row["sample_counts_by_seed"]=json.dumps(data["sample_count"]["values"])
                rows.append(row)
        lines.append(r"\addlinespace")
    return lines+table_end(),rows

def memory_table(summary,expected_hash):
    lines=table_start("Prediction MSE of the appended filtered-memory coordinate in the original state units. Entries are mean and sample standard deviation over three seeded runs.",
                      "tab:memory_error","lccc","System & NODE-LAC & NODE & SNDE")
    rows=[]
    for system in SYSTEMS:
        groups={method:get_group(summary,system,"main",method,expected_hash)
                for method in ["NODE-LAC","NODE","SNDE"]}
        lines.append(NAMES[system]+" & "+" & ".join(
            compact_uncertainty(groups[method]["metrics"]["memory_E_MSE"])
            for method in ["NODE-LAC","NODE","SNDE"])+r"\\")
        row={"system":system}
        for method,group in groups.items():
            row[method+"_MSE_mean"]=group["metrics"]["memory_E_MSE"]["mean"]
            row[method+"_MSE_sample_sd"]=group["metrics"]["memory_E_MSE"]["sample_sd"]
        rows.append(row)
    return lines+table_end(),rows

def scaled_uncertainty(metric,scale):
    return "$"+f"{metric['mean']/scale:.3f}"+r"\,\pm\,"+f"{metric['sample_sd']/scale:.3f}"+"$"

def ablation_table(summary,expected_hash):
    lines=table_start("Component ablations on the original datasets. Entries are mean and sample standard deviation over three seeded runs.",
                      "tab:ablation_results","lcc",r"Variant & MSE ($10^{-3}$) & MAE ($10^{-2}$)")
    rows=[]
    variants=[("Full","main"),("NoGainNet","ablation_NoGainNet"),
              ("NoConstraintLoss","ablation_NoConstraintLoss"),("NoCorrection","ablation_NoCorrection")]
    for system in SYSTEMS[:3]:
        lines.append(r"\multicolumn{3}{l}{\textbf{"+NAMES[system]+r"}}\\")
        for label,variant in variants:
            group=get_group(summary,system,variant,"NODE-LAC",expected_hash)
            lines.append(label+" & "+scaled_uncertainty(group["metrics"]["MSE"],1e-3)+" & "+
                         scaled_uncertainty(group["metrics"]["MAE"],1e-2)+r"\\")
            rows.append({"system":system,"variant":label,
                         "MSE_mean":group["metrics"]["MSE"]["mean"],"MSE_sample_sd":group["metrics"]["MSE"]["sample_sd"],
                         "MAE_mean":group["metrics"]["MAE"]["mean"],"MAE_sample_sd":group["metrics"]["MAE"]["sample_sd"]})
        lines.append(r"\addlinespace")
    return lines+table_end(),rows

def hyperparameter_table(summary,expected_hash):
    lines=table_start("Initial dual weight and gain-effort weight on FitzHugh--Nagumo. Entries are mean and sample standard deviation over three seeded runs. The feasible fraction uses componentwise standardized violation at most $10^{-6}$.",
                      "tab:hyperparameter_results","ccccc",r"$\mu_0$ & $\lambda_{\mathrm e}$ & MSE ($10^{-3}$) & CE & Feasible (\%)")
    rows=[]
    for mu,effort in [(.1,.05),(.05,.05),(.1,.01),(.01,.01)]:
        group=get_group(summary,"fitzhugh_nagumo",f"hyper_mu{mu}_effort{effort}","NODE-LAC",expected_hash)
        lines.append(f"{mu} & {effort} & "+scaled_uncertainty(group["metrics"]["MSE"],1e-3)+" & "+scaled_uncertainty(group["metrics"]["CE"],1.)+" & "+scaled_uncertainty(group["metrics"]["feasible_fraction"],.01)+r"\\")
        row={"initial_mu":mu,"effort_weight":effort}
        for metric in ["MSE","CE","feasible_fraction"]:
            row[metric+"_mean"]=group["metrics"][metric]["mean"]
            row[metric+"_sample_sd"]=group["metrics"][metric]["sample_sd"]
        rows.append(row)
    return lines+table_end(),rows

def selection_rows(summary,expected_hash):
    rows=[]
    for system in SYSTEMS:
        group=get_group(summary,system,"main","NODE-LAC",expected_hash)
        for selected in group["selection"]:
            rows.append({"system":system,"seed":selected["seed"],
                         "selected_correction_scale":selected["scale"],
                         "final_effective_mu":selected["effective_mu"],
                         "final_normalized_trajectory_loss":selected["final_trajectory_loss"]})
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-system",choices=SYSTEMS)
    parser.add_argument("--require",choices=["available","main","all"],default="available")
    args=parser.parse_args()
    summary=json.loads((args.root/"summary/summary.json").read_text())
    if summary["protocol"]!=PROTOCOL:raise ValueError("Only final standardized v2 summaries are accepted")
    manifest=json.loads((Path(__file__).parent/"source_manifest_v2.json").read_text())
    expected_hash=manifest["source_sha256"]["revision/run_standardized_revision.py"]
    if args.verify_system:
        for method in METHODS:get_group(summary,args.verify_system,"main",method,expected_hash)
        print(json.dumps({"verified_system":args.verify_system,"methods":len(METHODS),"seeds":SEEDS}))
        return
    destination=args.root/"publication";destination.mkdir(parents=True,exist_ok=True)
    status={}
    for name,builder in [("main_results",main_table),("constraint_diagnostics",diagnostic_table),
                         ("memory_coordinate_error",memory_table),("ablation_results",ablation_table),("hyperparameter_results",hyperparameter_table)]:
        try:
            lines,rows=builder(summary,expected_hash)
        except LookupError as error:
            status[name]={"status":"pending","reason":str(error)}
            continue
        (destination/(name+".tex")).write_text("\n".join(lines)+"\n")
        write_csv(destination/(name+".csv"),rows)
        status[name]={"status":"complete","rows":len(rows)}
    try:
        rows=selection_rows(summary,expected_hash)
        write_csv(destination/"selected_scales_and_dual_weights.csv",rows)
        status["selection"]={"status":"complete","rows":len(rows)}
    except LookupError as error:
        status["selection"]={"status":"pending","reason":str(error)}
    write_json(destination/"table_status.json",status)
    print(json.dumps(status))
    if args.require=="main" and status["main_results"]["status"]!="complete":sys.exit(2)
    if args.require=="all" and any(s["status"]!="complete" for s in status.values()):sys.exit(2)

if __name__=="__main__":main()
