#!/usr/bin/env python3
"""Evaluate a frozen validation selection and independently compute full-grid diagnostics."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_lyapunov_loss as study
import tune_validation as prior
frozen = study.frozen
OUTPUT = study.ROOT / 'test_evaluation'
METRICS = ['MSE','MAE','TCE','CE','CE_max','reference_CE','feasible_fraction','memory_E_MSE']
VARIANTS = ['zero_weight','selected_residual']

def near(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'{label}: {a} != {b}')

def verify_lock():
    path = study.ROOT / 'selection_lock.json'
    lock = prior.read(path)
    if prior.sha(study.__file__) != study.SOURCE_HASH:
        raise ValueError('Search source changed during evaluation')
    if not lock.get('locked') or lock.get('selection_split') != 'validation':
        raise ValueError('An immutable validation selection is required')
    if lock['config_sha256'] != study.CONFIG_HASH or lock['tuning_source_sha256'] != study.SOURCE_HASH:
        raise ValueError('Search source or configuration changed')
    inventory = lock['candidate_inventory']
    if lock['record_count'] != 48 or len(inventory) != 48:
        raise ValueError('Incomplete selection inventory')
    records = {}
    for item in inventory:
        if prior.sha(item['path']) != item['sha256']:
            raise ValueError('Candidate record changed after selection')
        rec = prior.read(item['path'])
        key = (rec['system'],rec['candidate_index'],rec['seed'])
        if key in records:
            raise ValueError('Duplicate candidate')
        if rec['tuning_source_sha256'] != study.SOURCE_HASH:
            raise ValueError('Candidate source differs')
        if rec['status'] == 'complete':
            if prior.sha(rec['checkpoint']['path']) != rec['checkpoint']['sha256']:
                raise ValueError('Candidate checkpoint changed')
            scores = rec['rho_scores']
            if [r['rho'] for r in scores] != prior.RHOS:
                raise ValueError('Calibration grid changed')
            finite = [r for r in scores if r['validation']['MSE'] is not None and math.isfinite(r['validation']['MSE'])]
            chosen_row = min(finite, key=lambda r:(r['validation']['MSE'],prior.RHOS.index(r['rho'])))
            if rec['selected'] != chosen_row:
                raise ValueError('Selected row differs from the validation MSE minimum')
            details = rec['selected']['diagnostics']
            a = details['reference']['active_normalized_residual_sq_mean']
            b = details['rollout']['normalized_residual_sq_mean']
            expected_score = None if a is None or b is None else .5*a+.5*b
            if rec['selected']['S'] != expected_score:
                raise ValueError('Cached selection score differs from its diagnostics')
        records[key] = rec
    expected = {(s,i,seed) for s in frozen.SYSTEMS for i in range(4) for seed in study.SEEDS}
    if set(records) != expected:
        raise ValueError('Wrong candidate matrix')
    for system in frozen.SYSTEMS:
        base = [records[system,0,seed] for seed in study.SEEDS]
        if any(r['status'] != 'complete' for r in base):
            raise ValueError('Incomplete zero-weight control')
        bm = statistics.mean(r['selected']['validation']['MSE'] for r in base)
        eligible = []
        empty_anchor = any(r['selected']['diagnostics']['reference']['active_count'] == 0 for r in base)
        for i in range(4):
            group = [records[system,i,seed] for seed in study.SEEDS]
            if any(r['status'] != 'complete' for r in group):
                continue
            mse = [r['selected']['validation']['MSE'] for r in group]
            score = [r['selected']['S'] for r in group]
            if empty_anchor:
                continue
            if any(v is None or not math.isfinite(v) for v in mse+score):
                continue
            if statistics.mean(mse) > 1.05*bm or any(m > 1.10*b['selected']['validation']['MSE'] for m,b in zip(mse,base)):
                continue
            eligible.append((statistics.mean(score),i))
        chosen = 0 if empty_anchor else min(eligible)[1]
        selected = lock['selection_by_system'][system]['selected']
        if selected['candidate_index'] != chosen or selected['lambda_L'] != study.LAMBDAS[chosen]:
            raise ValueError('Selection does not follow the frozen rule')
        for j, seed in enumerate(study.SEEDS):
            rec = records[system,chosen,seed]
            if selected['selected_rhos'][j] != rec['selected']['rho'] or selected['checkpoint_by_seed'][j] != rec['checkpoint']:
                raise ValueError('Selected model identity differs')
    prior.check_frozen()
    return lock, prior.sha(path), records

def full_diagnostics(dyn, states, constraint):
    """Independent reference calculation through the actual closed-loop field."""
    flat = states.detach().reshape(-1,states.shape[-1])
    values = []
    derivatives = []
    for part in flat.split(1024):
        with torch.enable_grad():
            z = part.detach().requires_grad_(True)
            v = .5*constraint.k(z).square().sum(-1)
            u = torch.autograd.grad(v.sum(),z)[0]
        with torch.no_grad():
            f = dyn(torch.zeros((),device=z.device,dtype=z.dtype),z)
            derivatives.append((u*f).sum(-1).detach())
            values.append(v.detach())
    v = torch.cat(values)
    dv = torch.cat(derivatives)
    signed = dv + .1*v
    raw = signed.clamp_min(0)
    norm = (raw/(1+v)).square()
    active = v > 1e-12
    if not torch.isfinite(torch.stack([v,dv,raw,norm])).all():
        raise FloatingPointError('Nonfinite full-grid diagnostic')
    out = {'sample_count':len(v),'active_count':int(active.sum()),'active_fraction':float(active.double().mean()),
           'V_mean':float(v.mean()),'dotV_mean':float(dv.mean()),'dotV_max':float(dv.max()),'dotV_min':float(dv.min()),
           'raw_R_mean':float(raw.mean()),'raw_R_max':float(raw.max()),
           'normalized_residual_sq_mean':float(norm.mean()),'normalized_residual_sq_max':float(norm.max()),
           'residual_satisfied_fraction':float((signed <= 1e-10).double().mean()),
           'active_normalized_residual_sq_mean':float(norm[active].mean()) if active.any() else None,
           'active_raw_R_mean':float(raw[active].mean()) if active.any() else None,
           'active_raw_R_max':float(raw[active].max()) if active.any() else None,
           'active_residual_satisfied_fraction':float((signed[active] <= 1e-10).double().mean()) if active.any() else None}
    return out

def score(diag):
    a = diag['reference']['active_normalized_residual_sq_mean']
    return None if a is None else .5*a+.5*diag['rollout']['normalized_residual_sq_mean']

def run(device):
    torch.set_num_threads(2)
    lock, lock_hash, records = verify_lock()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    output = OUTPUT/'evaluation.json'
    source_hash = prior.sha(__file__)
    if output.exists():
        old = prior.read(output)
        if old['selection_lock_sha256'] != lock_hash or old['evaluation_source_sha256'] != source_hash:
            raise ValueError('Existing output has different selection/source')
        print(json.dumps({'status':'cached','path':str(output)}),flush=True)
        return
    previous = prior.read(prior.ROOT/'test_evaluation/evaluation.json')
    old_rows = {(r['system'],r['seed']):r for r in previous['results'] if r['variant']=='selected_weights'}
    rows, replays = [], []
    start = time.time()
    for system in frozen.SYSTEMS:
        tr,val,te,raw,times,mean,std,physical,constraint,provenance = frozen.load_data(system,device)
        del tr,val
        index = lock['selection_by_system'][system]['selected']['candidate_index']
        for seed in study.SEEDS:
            cache = {}
            for variant,i in [('zero_weight',0),('selected_residual',index)]:
                rec = records[system,i,seed]
                prior.verify_preprocessing(provenance,rec['preprocessing'])
                if provenance['source_sha256'] != rec['source_dataset_sha256']:
                    raise ValueError('Test source differs from selected data')
                rho = rec['selected']['rho']
                key = (rec['checkpoint']['sha256'],rho)
                if key not in cache:
                    node,gain,ck = study.model_from_checkpoint(rec['checkpoint']['path'],te.shape[-1],constraint,device,system)
                    dyn = frozen.ClosedLoopDynamics(node.f,gain,constraint,correction_scale=rho)
                    with torch.no_grad():
                        pred = frozen.euler_integrate(dyn,te[:,0],times).permute(1,0,2)
                        pred_raw = pred*std+mean
                        if not torch.isfinite(pred_raw).all():
                            raise FloatingPointError('Nonfinite locked test rollout')
                        all_metrics = frozen.metrics(pred_raw,raw,physical,pred,te)
                        metrics = {k:all_metrics[k] for k in METRICS}
                        curve = (pred_raw-raw).square().mean((0,2)).tolist()
                    diag = {'reference':full_diagnostics(dyn,te,constraint),'rollout':full_diagnostics(dyn,pred,constraint)}
                    if any(d['sample_count'] != 12800 for d in diag.values()):
                        raise ValueError('Wrong full-grid diagnostic scope')
                    cache[key] = {'metrics':metrics,'diagnostics':diag,'S':score(diag),'MSE_by_time':curve}
                row = {'system':system,'seed':seed,'variant':variant,'candidate_index':i,'lambda_L':rec['lambda_L'],
                       'options':rec['options'],'rho':rho,'checkpoint':rec['checkpoint'],
                       'dataset_sha256':rec['source_dataset_sha256'],'validation':rec['selected'],**cache[key]}
                if variant == 'zero_weight':
                    old = old_rows[system,seed]
                    if old['checkpoint'] != rec['checkpoint'] or old['rho'] != rho:
                        raise ValueError('Zero weight differs from previous locked model')
                    for name in METRICS:
                        near(row['metrics'][name],old['metrics'][name],f'{system}/{seed}/{name}')
                    replays.append({'system':system,'seed':seed,'all_previous_metrics_match':True})
                rows.append(row)
                print(json.dumps({'system':system,'seed':seed,'variant':variant,'lambda_L':row['lambda_L'],
                                  'MSE':row['metrics']['MSE'],'S':row['S']}),flush=True)
    summaries, comparisons = [], []
    for system in frozen.SYSTEMS:
        groups = {}
        for variant in VARIANTS:
            group = [r for r in rows if r['system']==system and r['variant']==variant]
            if [r['seed'] for r in group] != study.SEEDS:
                raise ValueError('Incomplete seed group')
            out = {'system':system,'variant':variant,'lambda_L':group[0]['lambda_L'],'rhos':[r['rho'] for r in group]}
            fields = {k:[r['metrics'][k] for r in group] for k in METRICS}
            fields['S'] = [r['S'] for r in group]
            fields['validation_MSE'] = [r['validation']['validation']['MSE'] for r in group]
            fields['validation_S'] = [r['validation']['S'] for r in group]
            for scope in ['reference','rollout']:
                for k in group[0]['diagnostics'][scope]:
                    fields[scope+'_'+k] = [r['diagnostics'][scope][k] for r in group]
            for name,values in fields.items():
                out[name+'_mean'] = statistics.mean(values) if all(x is not None for x in values) else None
                out[name+'_sd'] = statistics.stdev(values) if all(x is not None for x in values) else None
            summaries.append(out)
            groups[variant] = (out,group)
        base,br = groups['zero_weight']; selected,sr = groups['selected_residual']
        item = {'system':system,'selected_lambda_L':selected['lambda_L']}
        for k in ['MSE','CE','TCE','S','reference_raw_R_mean','rollout_raw_R_mean']:
            b,s = base[k+'_mean'],selected[k+'_mean']
            item[k+'_change_percent'] = None if b is None or b==0 or s is None else 100*(s/b-1)
        item['MSE_paired_differences'] = [s['metrics']['MSE']-b['metrics']['MSE'] for b,s in zip(br,sr)]
        item['S_paired_differences'] = [None if s['S'] is None or b['S'] is None else s['S']-b['S'] for b,s in zip(br,sr)]
        comparisons.append(item)
    if len(rows)!=24 or len(replays)!=12 or verify_lock()[1]!=lock_hash:
        raise ValueError('Incomplete evaluation or modified lock')
    report = {'protocol':study.PROTOCOL,'selection_lock_sha256':lock_hash,'evaluation_source_sha256':source_hash,
              'tuning_source_sha256':study.SOURCE_HASH,'frozen_runner_sha256':prior.RUNNER_HASH,
              'record_count':len(rows),'seed_group_count':len(summaries),'reference_replay':replays,
              'test_partition':'Previously evaluated original archived test_states; unchanged',
              'diagnostic_scope':'All 12800 states in each reference/rollout scope; c=0.1,p=1',
              'selection':'Predetermined validation-only selection, locked before this evaluation',
              'summaries':summaries,'comparisons':comparisons,'results':rows,'seconds':time.time()-start}
    prior.write(output,report)
    for filename,data in [('summary.csv',summaries),('paired_comparisons.csv',comparisons)]:
        with (OUTPUT/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader()
            writer.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in data])
    prior.write(OUTPUT/'selected_configurations.json',lock['selection_by_system'])
    print(json.dumps({'status':'complete','path':str(output),'records':len(rows)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        _,sha,_=verify_lock();print(json.dumps({'verified':True,'selection_lock_sha256':sha}))
    else:
        run(args.device)
