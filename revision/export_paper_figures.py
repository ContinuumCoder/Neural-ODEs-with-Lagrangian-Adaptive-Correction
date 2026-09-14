#!/usr/bin/env python3
"""Export the original AIJ figure roster from standardized-v2 runs only."""
import argparse, csv, fcntl, hashlib, io, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_standardized_revision as run
from nodesac.systems.gpu_datagen import generate_fhn_gpu,generate_lv_gpu,generate_sw_gpu

LABELS={'fitzhugh_nagumo':'FitzHugh–Nagumo','lotka_volterra':'Lotka–Volterra','shallow_water':'Shallow Water','franka_robot':'Robot Arm'}
COLORS={'NODE-LAC':'#2166AC','NODE':'#E38D4A','SNDE':'#38966D','PORT-HJNN':'#8754A1','ConCerNet':'#D59A26','PNODE':'#9C665A','CPNODE':'#C84C64','SymODEN':'#949AA4','HNN':'#677787','CLNN':'#454E59'}
PDE=run.SYSTEMS[:3]
LONG_METHODS=['NODE-LAC','NODE','SNDE','PNODE','CPNODE','ConCerNet']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8,'savefig.dpi':200,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})

def read_json(p):return json.loads(Path(p).read_text())
def dump(p,obj):run.write_json(p,obj)
def fingerprint(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_write(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        w=csv.writer(handle);w.writerow(header);w.writerows(rows)
def finish(fig,out,stem,description,sources,rect=None,tight=True):
    if tight:fig.tight_layout(rect=rect)
    for ext in ['pdf','png']:fig.savefig(out/f'{stem}.{ext}')
    plt.close(fig)
    dump(out/f'{stem}.json',{'protocol':run.PROTOCOL,'description':description,'seed_ids':run.SEEDS,'uncertainty':'sample standard deviation across three independent training seeds (ddof=1)','sources':sources,'exporter_sha256':fingerprint(__file__)})

def publication_axes(axes):
    for ax in np.asarray(axes).ravel():
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(8.5);ax.yaxis.label.set_size(8.5);ax.title.set_size(9)

def shared_legend(fig,axis,ncol=3):
    handles,labels=axis.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,0),ncol=ncol,frameon=False,fontsize=8)

def mean_log_limits(ax,means,standard_deviations):
    means=np.asarray(means,float);standard_deviations=np.asarray(standard_deviations,float)
    if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():raise ValueError('Non-finite plotting statistics')
    positive=means[means>0]
    if not positive.size:raise ValueError('No positive means for log axis')
    ax.set_ylim(float(positive.min())/2,float((means+standard_deviations).max())*2)

def results(root,system,variant,method):
    rows=[]
    for seed in run.SEEDS:
        p=root/system/variant/method/f'seed{seed}'/'result.json'
        if not p.exists():return None
        r=read_json(p)
        if r['protocol']!=run.PROTOCOL or r['epochs']!=100:raise ValueError(f'Non-final record: {p}')
        if (r['system'],r['variant'],r['method'],r['seed'])!=(system,variant,method,seed):raise ValueError(f'Record identity mismatch: {p}')
        rows.append((p,r))
    if len({r['source_sha256'] for _,r in rows})!=1 or len({r['dataset_sha256'] for _,r in rows})!=1:raise ValueError(f'Mixed provenance: {system}/{variant}/{method}')
    return rows

def metric_group(root,s,v,m,key='MSE'):
    rows=results(root,s,v,m)
    if rows is None:return None
    values=np.array([r['metrics'][key] for _,r in rows],float)
    if not np.isfinite(values).all():raise ValueError(f'Non-finite metric {s}/{v}/{m}')
    return values.mean(),values.std(ddof=1),[str(p) for p,_ in rows]

def references(root,systems,device):
    out=root/'publication'/'raw'/'references';out.mkdir(parents=True,exist_ok=True)
    generators={'fitzhugh_nagumo':generate_fhn_gpu,'lotka_volterra':generate_lv_gpu,'shallow_water':generate_sw_gpu}
    for system in systems:
        if system not in generators:continue
        target=out/f'{system}_long.pt';meta=target.with_suffix('.json')
        archive=run.DATA/f'{system}_data.pt'
        source_hash=fingerprint(Path(run.__file__).parents[1]/'nodesac/systems/gpu_datagen.py')
        archive_hash=fingerprint(archive)
        if target.exists() and meta.exists():
            old=read_json(meta)
            if old['dataset_sha256']==archive_hash and old['generator_sha256']==source_hash:
                print(json.dumps({'reference':system,'status':'cached'}),flush=True);continue
            raise ValueError(f'Stale reference file: {target}')
        cached=torch.load(archive,map_location=device,weights_only=False)
        dt=float(cached['times'][1]-cached['times'][0])
        if abs(dt-(.015 if system=='lotka_volterra' else .05))>1e-12:raise ValueError('Unexpected archive observation step')
        started=time.time()
        generated=generators[system](n_trajectories=256,t_span=(0,1000*dt),dt_save=dt,dt_integrate=.001,seed=42,device=device)
        torch.cuda.synchronize(device)
        if generated['test_states'].shape[1]!=1000:raise ValueError('Expected 1000 observations')
        errors={}
        for part in ['train_states','test_states']:
            reference=generated[part][:,:100]
            err=float((reference-cached[part]).abs().max());errors[part]=err
            if not torch.allclose(reference,cached[part],atol=2e-11,rtol=2e-12):raise ValueError(f'Archive-prefix mismatch {system}/{part}: {err}')
            if not torch.isfinite(generated[part]).all():raise ValueError(f'Non-finite long truth {system}')
        time_err=float((generated['times'][:100]-cached['times']).abs().max())
        if time_err>1e-12:raise ValueError('Reference time-prefix mismatch')
        torch.save({'true':generated['test_states'].cpu(),'times':generated['times'].cpu()},target)
        manifest={'protocol':run.PROTOCOL,'system':system,'dataset_sha256':archive_hash,'generator_sha256':source_hash,'generator':generators[system].__name__,'seed':42,'n_generated':256,'test_indices':[128,255],'observations':1000,'dt_save':dt,'dt_integrate':.001,'interval':[0,1000*dt],'prefix_max_abs':errors,'prefix_time_max_abs':time_err,'finite':True,'seconds':time.time()-started,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device),'reference_sha256':fingerprint(target)}
        dump(meta,manifest)
        print(json.dumps({'reference':system,'status':'complete','prefix_max_abs':errors,'seconds':manifest['seconds']}),flush=True)
        del cached,generated

def checkpoint(path,device):
    r=read_json(path/'result.json');pv=read_json(path/'provenance.json')
    if r['protocol']!=run.PROTOCOL or r['epochs']!=100:raise ValueError('Only complete v2 records are accepted')
    if pv.get('constraint_coordinates')!='z=(raw_state-mean)/std':raise ValueError('Constraint-coordinate mismatch')
    data=torch.load(run.DATA/f"{r['system']}_data.pt",map_location=device,weights_only=False)
    if fingerprint(run.DATA/f"{r['system']}_data.pt")!=r['dataset_sha256']:raise ValueError('Dataset hash mismatch')
    mean=torch.tensor(pv['mean'],device=device,dtype=run.DTYPE);std=torch.tensor(pv['std'],device=device,dtype=run.DTYPE)
    classes=dict(zip(run.SYSTEMS,[run.FitzHughNagumo,run.LotkaVolterra,run.ShallowWater,run.FrankaRobot]))
    physical=classes[r['system']]().get_manifold();physical.e_threshold=pv['constraint_energy_threshold']
    constraint=run.StandardizedConstraint(physical,mean,std)
    dim=len(mean);state=torch.load(path/'model.pt',map_location=device,weights_only=False)
    if r['method']=='NODE-LAC':
        node=run.NeuralODE(dim,(256,256),solver='euler').to(device=device,dtype=run.DTYPE)
        gain=(run.FixedGain() if state.get('options',{}).get('fixed_gain') else run.GainNet(dim,(128,128))).to(device=device,dtype=run.DTYPE)
        node.load_state_dict(state['node']);gain.load_state_dict(state['gain'])
        scale=float(state['selected_scale'])
        if scale!=float(r['selection']['scale']):raise ValueError('Saved scale differs from result')
        model=run.ClosedLoopDynamics(node.f,gain,constraint,correction_scale=scale);padded=False
    else:
        model,padded=run.build_baseline(r['method'],dim,constraint)
        model=model.to(device=device,dtype=run.DTYPE);model.load_state_dict(state)
    model.eval()
    def predict(raw_x0,times):
        z0=(raw_x0-mean)/std
        with torch.no_grad():
            if r['method']=='NODE-LAC':pred=run.euler_integrate(model,z0,times).permute(1,0,2)
            else:pred=model.predict(run.pad(z0,padded),times).permute(1,0,2)[...,:dim]
        return pred,pred*std+mean
    return r,pv,data,mean,std,physical,constraint,predict

def predict_run(root,system,method,seed,device,long=True):
    path=root/system/'main'/method/f'seed{seed}'
    if not (path/'result.json').exists():return False
    out=root/'publication'/'raw'/'predictions'/system/method;out.mkdir(parents=True,exist_ok=True)
    verification=out/f'seed{seed}_verified.json'
    model_hash=fingerprint(path/'model.pt');result_hash=fingerprint(path/'result.json')
    short_path=out/f'seed{seed}_short.pt';long_path=out/f'seed{seed}_long.pt'
    if verification.exists() and short_path.exists() and (not long or long_path.exists()):
        saved=read_json(verification)
        reference_ok=not long or saved.get('long_reference_sha256')==fingerprint(root/'publication'/'raw'/'references'/f'{system}_long.pt')
        if saved.get('metrics_reproduced') and saved['model_sha256']==model_hash and saved['result_sha256']==result_hash and reference_ok:return True
        raise ValueError(f'Stale prediction cache: {verification}')
    r,pv,data,mean,std,physical,constraint,predict=checkpoint(path,device)
    pred_z,pred_raw=predict(data['test_states'][:,0],data['times'])
    measured=run.metrics(pred_raw,data['test_states'],physical,pred_z,(data['test_states']-mean)/std)
    errors={}
    for key,target in r['metrics'].items():
        errors[key]=abs(measured[key]-target)
        if not math.isclose(measured[key],target,abs_tol=2e-10,rel_tol=2e-8):raise ValueError(f'Checkpoint metric mismatch {system}/{method}/{seed}/{key}: {measured[key]} vs {target}')
    torch.save({'pred':pred_raw.cpu(),'true':data['test_states'].cpu(),'times':data['times'].cpu()},short_path)
    manifest={'protocol':run.PROTOCOL,'source_result':str(path/'result.json'),'model_sha256':model_hash,'result_sha256':result_hash,'metrics_reproduced':True,'metric_absolute_differences':errors,'constraint_coordinates':'standardized z','test_scale_tuning':False}
    if long:
        ref_path=root/'publication'/'raw'/'references'/f'{system}_long.pt'
        if not ref_path.exists():raise FileNotFoundError(ref_path)
        ref=torch.load(ref_path,map_location=device,weights_only=False)
        long_z,long_raw=predict(ref['true'][:,0],ref['times'])
        if not torch.isfinite(long_raw).all():raise FloatingPointError(f'Non-finite long prediction {system}/{method}/{seed}')
        curves={'MSE':(long_raw-ref['true']).square().mean((0,2)).cpu(),'CE':physical.k(long_z).square().sum(-1).mean(0).cpu()}
        torch.save({'representative_pred':long_raw[0].cpu(),'times':ref['times'].cpu(),'curves':curves},long_path)
        manifest['long_reference_sha256']=fingerprint(ref_path)
        manifest['long_finite']=True
        del ref,long_z,long_raw
    dump(verification,manifest)
    print(json.dumps({'prediction':f'{system}/{method}/seed{seed}','verified':True,'long':long}),flush=True)
    failure=out/f'seed{seed}_failure.json'
    if failure.exists():failure.unlink()
    return True

def aggregate(root,out):
    exported=[];missing=[]
    top=['NODE-LAC','SNDE','PORT-HJNN','NODE','ConCerNet','PNODE']
    groups={(s,m):metric_group(root,s,'main',m) for s in run.SYSTEMS for m in top}
    if all(groups.values()):
        fig,axes=plt.subplots(2,2,figsize=(5.5,5.8));axes=axes.ravel();rows=[];sources=[]
        for ax,s in zip(axes,run.SYSTEMS):
            vals=[groups[s,m][0] for m in top];sd=[groups[s,m][1] for m in top]
            ax.bar(np.arange(len(top)),vals,yerr=sd,capsize=2,color=[COLORS[m] for m in top]);ax.set_yscale('log');ax.set_xticks(range(len(top)),top,rotation=45,ha='right');ax.set_title(LABELS[s]);ax.set_ylabel('Test MSE');mean_log_limits(ax,vals,sd)
            for m,a,b in zip(top,vals,sd):rows.append([s,m,a,b]);sources+=groups[s,m][2]
        publication_axes(axes)
        finish(fig,out,'table1_bar_chart','Test MSE: mean and sample SD across three seeds.',sources);csv_write(out/'table1_bar_chart.csv',['system','method','mean_MSE','sample_SD'],rows);exported.append('table1_bar_chart')
    else:missing.append('table1_bar_chart')
    radar_methods=['NODE-LAC','SNDE','NODE','ConCerNet','PNODE','CPNODE']
    rg={(s,m,k):metric_group(root,s,'main',m,k) for s in run.SYSTEMS for m in radar_methods for k in ['MSE','MAE','TCE']}
    if all(rg.values()):
        fig,axes=plt.subplots(2,2,figsize=(5.5,5.8),subplot_kw={'projection':'polar'});axes=axes.ravel();angles=np.linspace(0,2*np.pi,3,endpoint=False);sources=[];rows=[]
        for ax,s in zip(axes,run.SYSTEMS):
            den=np.array([max(rg[s,m,k][0] for m in radar_methods) for k in ['MSE','MAE','TCE']])
            for m in radar_methods:
                av=np.array([rg[s,m,k][0] for k in ['MSE','MAE','TCE']])/den;sd=np.array([rg[s,m,k][1] for k in ['MSE','MAE','TCE']])/den
                aa=np.r_[angles,angles[0]];vv=np.r_[av,av[0]];ss=np.r_[sd,sd[0]]
                ax.plot(aa,vv,label=m,color=COLORS[m],lw=1.8 if m=='NODE-LAC' else 1);ax.fill_between(aa,np.maximum(vv-ss,0),vv+ss,color=COLORS[m],alpha=.08)
                for k,a,b in zip(['MSE','MAE','TCE'],av,sd):rows.append([s,m,k,a,b]);sources+=rg[s,m,k][2]
            ax.set_xticks(angles,['MSE','MAE','TCE']);ax.set_title(LABELS[s],pad=16)
        publication_axes(axes);shared_legend(fig,axes[0]);finish(fig,out,'summary_radar','Metrics normalized by the largest displayed mean in each task; bands show normalized sample SD.',sources,rect=[0,.08,1,1]);csv_write(out/'summary_radar.csv',['system','method','metric','normalized_mean','normalized_sample_SD'],rows);exported.append('summary_radar')
    else:missing.append('summary_radar')
    loss_methods=['NODE-LAC','NODE','SNDE','ConCerNet','PNODE'];hist={};sources=[]
    for s in run.SYSTEMS:
        for m in loss_methods:
            rr=results(root,s,'main',m)
            if rr is None:continue
            h=[read_json(p.parent/'history.json') for p,_ in rr]
            if all(len(v)==100 and all('trajectory_MSE' in e for e in v) for v in h):hist[s,m]=np.array([[e['trajectory_MSE'] for e in v] for v in h]);sources +=[str(p.parent/'history.json') for p,_ in rr]
    if len(hist)==len(run.SYSTEMS)*len(loss_methods):
        fig,axes=plt.subplots(2,2,figsize=(5.5,5.8));axes=axes.ravel();rows=[]
        for ax,s in zip(axes,run.SYSTEMS):
            for m in loss_methods:
                av=hist[s,m].mean(0);sd=hist[s,m].std(0,ddof=1);x=np.arange(1,101)
                ax.plot(x,av,label=m,color=COLORS[m]);ax.fill_between(x,np.maximum(av-sd,1e-14),av+sd,color=COLORS[m],alpha=.12)
                rows.extend([s,m,int(i),float(a),float(b)] for i,a,b in zip(x,av,sd))
            ax.set_yscale('log');ax.set_title(LABELS[s]);ax.set_xlabel('Epoch');ax.set_ylabel('Training trajectory MSE')
            mean_log_limits(ax,[hist[s,m].mean(0) for m in loss_methods],[hist[s,m].std(0,ddof=1) for m in loss_methods])
        publication_axes(axes);shared_legend(fig,axes[0]);finish(fig,out,'fig2_loss_curves','The same normalized trajectory-MSE statistic for every method; mean and sample SD across seeds. Minibatch MSEs are averaged without batch-size weighting. Log-axis limits follow positive means; SD outside the range is clipped.',sources,rect=[0,.08,1,1]);csv_write(out/'fig2_loss_curves.csv',['system','method','epoch','mean','sample_SD'],rows);exported.append('fig2_loss_curves')
    else:missing.append('fig2_loss_curves')
    for stem,kind,xs,methods in [('reviewer_data_efficiency','fraction',[.25,.5,.75,1.],['NODE-LAC','NODE','SNDE']),('reviewer_noise_robustness','noise',[0.,.05,.1,.2],['NODE-LAC','NODE'])]:
        vals={}
        for s in PDE:
            for m in methods:
                for x in xs:
                    variant='main' if (kind=='fraction' and x==1.) or(kind=='noise' and x==0.) else (f'data_eff_{x}' if kind=='fraction' else f'noise_{x}')
                    vals[s,m,x]=metric_group(root,s,variant,m)
        if not all(vals.values()):missing.append(stem);continue
        fig,axes=plt.subplots(3,1,figsize=(5.5,6.5));rows=[];sources=[]
        for ax,s in zip(axes,PDE):
            for i,m in enumerate(methods):
                av=[vals[s,m,x][0] for x in xs];sd=[vals[s,m,x][1] for x in xs]
                if kind=='fraction':ax.errorbar(np.array(xs)*100,av,yerr=sd,capsize=3,marker=['s','o','^'][i],label=m,color=COLORS[m])
                else:ax.bar(np.arange(4)+(i-.5)*.36,av,.36,yerr=sd,capsize=3,label=m,color=COLORS[m])
                for x,a,b in zip(xs,av,sd):rows.append([s,m,x,a,b]);sources+=vals[s,m,x][2]
            if kind=='noise':ax.set_xticks(range(4),[str(x) for x in xs])
            else:ax.set_xticks(np.array(xs)*100,[str(int(x*100)) for x in xs])
            ax.set_title(LABELS[s]);ax.set_xlabel('Training subset (%)' if kind=='fraction' else 'Training noise σ (standardized units)');ax.set_ylabel('Test MSE')
        publication_axes(axes);shared_legend(fig,axes[0]);finish(fig,out,stem,'Test MSE: mean and sample SD across three independent optimization seeds.',sources,rect=[0,.07,1,1]);csv_write(out/f'{stem}.csv',['system','method',kind,'mean_MSE','sample_SD'],rows);exported.append(stem)
    return exported,missing

def ablation(root,out):
    labels=['Full NODE-LAC','NoGainNet','NoConstraintLoss','NoCorrection']
    variants=['main','ablation_NoGainNet','ablation_NoConstraintLoss','ablation_NoCorrection']
    vals={(s,label):metric_group(root,s,variant,'NODE-LAC') for s in PDE for label,variant in zip(labels,variants)}
    if not all(vals.values()):return [],['table3_ablation']
    fig,axes=plt.subplots(3,1,figsize=(5.5,6.5));rows=[];sources=[]
    for ax,system in zip(axes,PDE):
        av=[vals[system,label][0] for label in labels];sd=[vals[system,label][1] for label in labels]
        ax.bar(range(4),av,yerr=sd,capsize=3,color=['#2166AC','#C84C64','#D59A26','#949AA4'])
        ax.set_xticks(range(4),['Full','Fixed gain','No constraint loss','No correction'],rotation=25,ha='right')
        ax.set_title(LABELS[system]);ax.set_ylabel('Test MSE')
        for label,a,b in zip(labels,av,sd):rows.append([system,label,a,b]);sources+=vals[system,label][2]
    publication_axes(axes)
    finish(fig,out,'table3_ablation','Ablation MSE: mean and sample SD across three seeds; the full model uses the main-run checkpoints.',sources)
    csv_write(out/'table3_ablation.csv',['system','variant','mean_MSE','sample_SD'],rows)
    return ['table3_ablation'],[]

def verified_prediction_paths(root,system,method,kind):
    paths=[]
    for seed in run.SEEDS:
        directory=root/'publication'/'raw'/'predictions'/system/method
        path=directory/f'seed{seed}_{kind}.pt'
        meta=directory/f'seed{seed}_verified.json'
        record=root/system/'main'/method/f'seed{seed}'
        if not path.exists() or not meta.exists():return None
        if (directory/f'seed{seed}_failure.json').exists():return None
        verified=read_json(meta)
        if not verified.get('metrics_reproduced'):raise ValueError(f'Unverified prediction: {path}')
        if verified['model_sha256']!=fingerprint(record/'model.pt') or verified['result_sha256']!=fingerprint(record/'result.json'):raise ValueError(f'Stale prediction: {path}')
        if kind=='long' and verified.get('long_reference_sha256')!=fingerprint(root/'publication'/'raw'/'references'/f'{system}_long.pt'):raise ValueError(f'Stale long reference: {path}')
        paths.append(path)
    return paths

def qualitative(root,out,system):
    methods=['NODE-LAC','NODE','PNODE','CPNODE'];arrays={};sources=[]
    for m in methods:
        paths=verified_prediction_paths(root,system,m,'short')
        if paths is None:return [],[f'{system}_qualitative']
        packed=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        arrays[m]=np.stack([v['pred'][0].numpy() for v in packed]);sources +=list(map(str,paths))
    true=packed[0]['true'][0].numpy();times=packed[0]['times'].numpy();n=(true.shape[-1]-1)//2
    exported=[]
    if system in ['fitzhugh_nagumo','lotka_volterra']:
        half=50;fig=plt.figure(figsize=(5.5,7.8));names=['Ground Truth']+methods
        xx,tt=np.meshgrid(np.arange(n),times[:half]);preds={m:arrays[m].mean(0) for m in methods}
        vals=[true[:half,:n]]+[preds[m][:half,:n] for m in methods];lo=min(v.min() for v in vals);hi=max(v.max() for v in vals)
        errs=[np.zeros_like(vals[0])]+[np.abs(preds[m][:half,:n]-true[:half,:n]) for m in methods];ehi=max(e.max() for e in errs)
        for col,(name,values,error) in enumerate(zip(names,vals,errs)):
            for row,data in enumerate([values,error]):
                ax=fig.add_subplot(5,2,col*2+row+1,projection='3d')
                lower,upper=(lo,hi) if row==0 else (0,max(ehi,1e-12))
                ax.plot_surface(xx,tt,data,cmap='viridis' if row==0 else 'magma',vmin=lower,vmax=upper,linewidth=0)
                ax.set_title(name if row==0 else name+' error',fontsize=8,pad=2)
                ax.set_xlabel('site',fontsize=7,labelpad=-5);ax.set_ylabel('time',fontsize=7,labelpad=-5)
                ax.set_zlabel('state' if row==0 else 'error',fontsize=7,labelpad=0)
                ax.set_xticks([0,n-1]);ax.set_yticks([0,times[half-1]]);ax.set_yticklabels(['0',f'{times[half-1]:.3g}'])
                ax.set_zlim(lower,upper);ax.set_zticks([lower,upper]);ax.set_zticklabels([f'{lower:.2g}',f'{upper:.2g}'])
                ax.tick_params(axis='both',labelsize=7,pad=-1,length=1)
                ax.view_init(elev=25,azim=-60);ax.set_box_aspect((1.4,1,.6),zoom=1.05)
        fig.subplots_adjust(left=.03,right=.94,bottom=.06,top=.96,hspace=.60,wspace=.32)
        stem=f'{system}_3d_surface'
        with plt.rc_context({'savefig.bbox':None}):finish(fig,out,stem,'First archived test trajectory, first50 observations; predictions averaged across three training seeds. Five rows: Ground Truth, NODE-LAC, NODE, PNODE, CPNODE. Left column: state. Right column: absolute error of the seed-mean prediction.',sources,tight=False)
        exported.append(stem)
    if system in ['fitzhugh_nagumo','shallow_water']:
        fig,axes=plt.subplots(4,2,figsize=(5.5,6.4),sharex=True)
        for site,ax in enumerate(axes.flat):
            ax.plot(times,true[:,site],color='black',lw=1.3,label='Ground Truth')
            for m in methods:
                av=arrays[m][:,:,site].mean(0);sd=arrays[m][:,:,site].std(0,ddof=1)
                ax.plot(times,av,color=COLORS[m],label=m,lw=1.0);ax.fill_between(times,av-sd,av+sd,color=COLORS[m],alpha=.10)
            ax.set_title(f'Site {site+1}',fontsize=8.5);ax.tick_params(labelsize=7.5)
            ax.locator_params(axis='y',nbins=3);ax.set_xlim(float(times[0]),float(times[-1]))
            ax.set_xticks([0,2.5,float(times[-1])],['0','2.5',f'{float(times[-1]):.3g}'])
            if site%2==0:ax.set_ylabel('state',fontsize=8)
            if site>=6:ax.set_xlabel('time',fontsize=8)
        shared_legend(fig,axes.flat[0]);stem=f'{system}_trajectories';finish(fig,out,stem,'First archived test trajectory, first eight coordinates; prediction mean and sample SD across three training seeds. Four rows and two columns, ordered by site.',sources,rect=[0,.08,1,1]);exported.append(stem)
    return exported,[]

def long_figures(root,out,system):
    curves={};sources=[]
    for m in LONG_METHODS:
        ps=verified_prediction_paths(root,system,m,'long')
        if ps is None:return [],[f'{system}_error_over_time']
        packs=[torch.load(p,map_location='cpu',weights_only=False) for p in ps]
        curves[m]=np.stack([p['curves']['MSE'].numpy() for p in packs]);sources+=list(map(str,ps))
    fig,ax=plt.subplots(figsize=(8,4.3));rows=[];steps=np.arange(1000)
    for m in LONG_METHODS:
        av=curves[m].mean(0);sd=curves[m].std(0,ddof=1)
        ax.plot(steps[1:],av[1:],label=m,color=COLORS[m],lw=1.8 if m=='NODE-LAC' else 1.2)
        ax.fill_between(steps[1:],np.maximum(av[1:]-sd[1:],1e-14),av[1:]+sd[1:],color=COLORS[m],alpha=.12)
        rows.extend([system,m,int(i),float(a),float(b)] for i,a,b in zip(steps,av,sd))
    ax.axvspan(99,999,color='#E9BB31',alpha=.10);ax.axvline(99,color='.5',ls='--',lw=1);ax.set_yscale('log');ax.set_xlabel('Observation index');ax.set_ylabel('Test MSE');ax.set_title(LABELS[system]);ax.legend(ncol=2)
    mean_log_limits(ax,[curves[m].mean(0)[1:] for m in LONG_METHODS],[curves[m].std(0,ddof=1)[1:] for m in LONG_METHODS])
    stem=f'{system}_error_over_time';finish(fig,out,stem,'1000 observations; last training observation index99. Curves: mean and sample SD across three training seeds; reference generator prefix verified against the unchanged archive.',sources);csv_write(out/f'{stem}.csv',['system','method','observation_index','mean_MSE','sample_SD'],rows)
    return [stem],[]

def combined_long_figure(root,out):
    all_curves={};sources=[]
    for system in PDE:
        for method in LONG_METHODS:
            paths=verified_prediction_paths(root,system,method,'long')
            if paths is None:return [],['error_over_time_combined']
            packed=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
            all_curves[system,method]=np.stack([p['curves']['MSE'].numpy() for p in packed])
            sources+=list(map(str,paths))
    with plt.rc_context({'font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8}):
        fig,axes=plt.subplots(3,1,figsize=(5.5,6.8),sharex=True)
        steps=np.arange(1000)
        for ax,system in zip(axes,PDE):
            for method in LONG_METHODS:
                data=all_curves[system,method];av=data.mean(0);sd=data.std(0,ddof=1)
                ax.plot(steps[1:],av[1:],label=method,color=COLORS[method],lw=1.4 if method=='NODE-LAC' else 1.0)
                ax.fill_between(steps[1:],np.maximum(av[1:]-sd[1:],1e-14),av[1:]+sd[1:],color=COLORS[method],alpha=.12)
            ax.axvspan(99,999,color='#E9BB31',alpha=.10);ax.axvline(99,color='.5',ls='--',lw=.8)
            ax.set_yscale('log');ax.set_ylabel('Test MSE');ax.set_title(LABELS[system],pad=5);ax.set_xlim(0,999);ax.set_xticks([0,200,400,600,800,999])
            mean_log_limits(ax,[all_curves[system,m].mean(0)[1:] for m in LONG_METHODS],[all_curves[system,m].std(0,ddof=1)[1:] for m in LONG_METHODS])
            ax.tick_params(labelbottom=True)
        axes[-1].set_xlabel('Observation index')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,0),ncol=3,frameon=False,fontsize=8)
        finish(fig,out,'error_over_time_combined','The same 1000-observation MSE data as the three task-specific figures. Means and sample SD across three seeds. Each log-axis range is set by positive means and mean-plus-SD; bands outside that range are clipped. Training ends at index99.',sources,rect=[0,.08,1,1])
    return ['error_over_time_combined'],[]

def compact_summaries(root,out):
    grouped={(system,method):results(root,system,'main',method) for system in run.SYSTEMS for method in run.METHODS}
    if all(grouped.values()):
        keys=['MSE','MAE','TCE','CE','Stability'];rows=[];sources=[]
        for system in run.SYSTEMS:
            ranking=sorted(run.METHODS,key=lambda method:np.mean([r['metrics']['MSE'] for _,r in grouped[system,method]]))
            for rank,method in enumerate(ranking,1):
                row=[system,rank,method]
                for key in keys:
                    values=np.array([r['metrics'][key] for _,r in grouped[system,method]],float)
                    if not np.isfinite(values).all():raise ValueError('Non-finite main summary metric')
                    row.extend([values.mean(),values.std(ddof=1)])
                rows.append(row);sources.extend(str(path) for path,_ in grouped[system,method])
        csv_write(out/'main_result_rankings.csv',['system','MSE_rank','method']+[name for key in keys for name in [key+'_mean',key+'_sample_SD']],rows)
        dump(out/'main_result_rankings_manifest.json',{'protocol':run.PROTOCOL,'seeds':run.SEEDS,'uncertainty':'sample standard deviation, ddof=1','sources':sources,'exporter_sha256':fingerprint(__file__)})
    rows=[];sources=[]
    for system in PDE:
        for method in LONG_METHODS:
            paths=verified_prediction_paths(root,system,method,'long')
            if paths is None:return
            curves=np.stack([torch.load(path,map_location='cpu',weights_only=False)['curves']['MSE'].numpy() for path in paths])
            row=[system,method]
            for values in [curves[:,99],curves[:,999],curves[:,100:].mean(1)]:
                if not np.isfinite(values).all():raise ValueError('Non-finite long summary metric')
                row.extend([values.mean(),values.std(ddof=1)])
            rows.append(row);sources.extend(map(str,paths))
    csv_write(out/'long_horizon_summary.csv',['system','method','index99_MSE_mean','index99_MSE_sample_SD','index999_MSE_mean','index999_MSE_sample_SD','index100_to999_average_MSE_mean','index100_to999_average_MSE_sample_SD'],rows)
    dump(out/'long_horizon_summary_manifest.json',{'protocol':run.PROTOCOL,'seeds':run.SEEDS,'uncertainty':'sample standard deviation, ddof=1; time averaging is performed within each seed before computing seed SD','sources':sources,'exporter_sha256':fingerprint(__file__)})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=run.DEFAULT_OUTPUT);parser.add_argument('--phase',choices=['references','predict','aggregate','all'],default='aggregate');parser.add_argument('--device',default='cuda:1');parser.add_argument('--systems',nargs='+',choices=run.SYSTEMS,default=run.SYSTEMS);parser.add_argument('--methods',nargs='+',default=LONG_METHODS);parser.add_argument('--seeds',nargs='+',type=int,default=run.SEEDS);parser.add_argument('--short-only',action='store_true');args=parser.parse_args()
    if args.root.name!=run.PROTOCOL:raise ValueError('Output root must identify final standardized-v2 protocol')
    torch.set_num_threads(2)
    out=args.root/'publication';out.mkdir(parents=True,exist_ok=True)
    export_lock=(out/'.figure_export.lock').open('w');fcntl.flock(export_lock,fcntl.LOCK_EX)
    if args.phase in ['references','all']:references(args.root,args.systems,args.device)
    failures=[]
    if args.phase in ['predict','all']:
        for s in args.systems:
            for m in args.methods:
                for seed in args.seeds:
                    try:predict_run(args.root,s,m,seed,args.device,long=s in PDE and not args.short_only)
                    except Exception as error:
                        failed={'system':s,'method':m,'seed':seed,'error':str(error),'status':'failed'}
                        failures.append(failed)
                        dump(out/'raw'/'predictions'/s/m/f'seed{seed}_failure.json',failed)
                        print(json.dumps(failed),flush=True)
        dump(out/'prediction_failures.json',failures)
    exported=[];missing=[]
    if args.phase in ['aggregate','all']:
        a,b=aggregate(args.root,out);exported+=a;missing+=b
        a,b=ablation(args.root,out);exported+=a;missing+=b
        for s in PDE:
            a,b=qualitative(args.root,out,s);exported+=a;missing+=b
            a,b=long_figures(args.root,out,s);exported+=a;missing+=b
        a,b=combined_long_figure(args.root,out);exported+=a;missing+=b
        compact_summaries(args.root,out)
        dump(out/'export_status.json',{'protocol':run.PROTOCOL,'exported':exported,'missing':missing,'exporter_sha256':fingerprint(__file__),'raw_arrays_remain_on_NAS':True,'prediction_failures':[read_json(p) for p in (out/'raw'/'predictions').glob('*/*/seed*_failure.json')]})
    print(json.dumps({'phase':args.phase,'exported':exported,'missing':missing,'output':str(out),'failures':failures}),flush=True)
    if failures:sys.exit(2)
if __name__=='__main__':main()
