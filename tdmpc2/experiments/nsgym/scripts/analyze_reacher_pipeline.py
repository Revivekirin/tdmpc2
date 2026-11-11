"""Seed-unit summaries and figures; missing measurements are never filled."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import reacher_pipeline as r

def plotting():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def grouped(rows,keys,metric):
    groups=defaultdict(list)
    for x in rows:
        if x.get(metric) not in ('',None):groups[tuple(x[k] for k in keys)].append(float(x[metric]))
    return {key:float(np.mean(v)) for key,v in groups.items()}

def summary(rows,metric,unit_keys=('family','severity','seed')):
    per=grouped(rows,unit_keys,metric);groups=defaultdict(list)
    for key,v in per.items():groups[key[:-1]].append(v)
    return [dict(zip(unit_keys[:-1],key),metric=metric,**r.stats(v)) for key,v in groups.items()]

def curves(rows,metrics,out,name,c,title):
    plt=plotting();fig,axes=plt.subplots(len(metrics),4,figsize=(19,4*len(metrics)),squeeze=False)
    for row,(metric,label) in enumerate(metrics):
        ss=summary(rows,metric)
        for col,(family,param) in enumerate(c['families'].items()):
            ax=axes[row,col];vv=sorted([x for x in ss if x['family']==family],key=lambda x:float(x['severity']))
            if vv:
                x=[float(v['severity']) for v in vv];y=[v['mean'] for v in vv]
                ax.plot(x,y,'o-',label='seed mean');ax.plot(x,[v['median'] for v in vv],'x--',label='seed median')
                if all(v['ci_low'] is not None for v in vv):ax.fill_between(x,[v['ci_low'] for v in vv],[v['ci_high'] for v in vv],alpha=.2,label='seed bootstrap 95% CI')
            matching=[x for x in rows if x['family']==family];nom=matching[0].get('nominal_parameter',matching[0].get('nominal','?')) if matching else '?'
            nominal_label=f'{float(nom):.6g}' if nom!='?' else '?'
            ax.set(title=f'{param}; nominal={nominal_label} '+('kg' if param.endswith('mass') else 'N m s/rad')+'\nactual = multiplier × nominal',xlabel='severity multiplier',ylabel=label)
            if vv:ax.legend(fontsize=7)
    fig.suptitle(title);fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig)

def physical_plots(source,out,c):
    rows=r.csvread(source/'reacher_physical_response.csv')
    metrics=[('delta_q_rad','Δ joint position norm [rad]'),('delta_qvel_rad_s','Δ joint velocity norm [rad/s]'),('delta_tip_m','Δ fingertip displacement [m]'),('delta_accel_rad_s2','Δ finite-difference acceleration [rad/s²]')]
    curves(rows,metrics,out,'reacher_physical_response_by_severity.png',c,'Same integration state/action; separate physical units')
    if any(x['shifted_latent_mse'] for x in rows):curves(rows,[('shifted_latent_mse','Fixed-bank latent MSE'),('delta_latent_mse','Shifted − nominal latent MSE')],out,'reacher_latent_error_by_severity.png',c,'Fixed calibration data; frozen encoder')
    ss=sum([summary(rows,m) for m,_ in metrics+[('shifted_latent_mse','')]],[])
    r.table(out/'physical_seed_summary.csv',ss)
    monotonic=[]
    for family in c['families']:
        for metric,_ in metrics:
            vv=sorted([x for x in ss if x['family']==family and x['metric']==metric],key=lambda x:float(x['severity']))
            for side in ('below_nominal','above_nominal'):
                points=[x for x in vv if (float(x['severity'])<=1 if side=='below_nominal' else float(x['severity'])>=1)]
                differences=np.diff([x['mean'] for x in points])
                monotonic.append(dict(family=family,metric=metric,side=side,condition_count=len(points),nonmonotonic=bool(np.any(differences>1e-12) if side=='below_nominal' else np.any(differences < -1e-12))))
    r.table(out/'physical_monotonicity.csv',monotonic)
    regime=defaultdict(int)
    for x in rows:regime[(x['family'],x['severity'],x['seed'],x['action_regime'])]+=1
    r.table(out/'physical_regime_counts.csv',[dict(family=k[0],severity=k[1],seed=k[2],regime=k[3],samples=v) for k,v in regime.items()])
    (out/'REACHER_SEVERITY_CALIBRATION.md').write_text('# Physical calibration\n\nAll predefined valid conditions retained. Integration state, simulation time, control and warmstart are restored; repeated nominal transitions agree to 1e-12. Moving-joint and nonzero-action bank used. CSV keeps raw vectors and next observations. Statistics first average within calibration seed. Bootstrap uncertainty is across seeds, not time steps. Regime counts and Tukey outliers are in companion CSVs. Non-monotonicity is permitted; multipliers are not matched difficulty. Latent MSE is not physical reconstruction error.\n')

def frozen_plots(source,out,c,probe=None):
    rows=r.csvread(source/'reacher_frozen_severity.csv')
    for name,metrics in [('reacher_frozen_return_vs_severity.png',[('episode_return','Episode return')]),('reacher_distance_vs_severity.png',[('distance_mean','Mean tip-target distance [m]'),('distance_final','Final distance [m]')]),('reacher_action_effort.png',[('action_abs','Mean |normalized action|'),('action_saturation','Action component saturation fraction'),('force_rms','Actuator force RMS'),('force_saturation','Force saturation (only limited actuators)')])]:
        curves(rows,metrics,out,name,c,'Frozen Reacher; paired evaluation seeds')
    metrics=['episode_return','distance_mean','distance_final','latent_mse','action_abs','action_saturation','force_rms']
    r.table(out/'frozen_seed_summary.csv',sum([summary(rows,m) for m in metrics],[]))
    plt=plotting();fig,axes=plt.subplots(1,4,figsize=(18,4));associations=[]
    base={int(x['seed']):float(x['episode_return']) for x in rows if x['family']=='D0' and float(x['severity'])==1}
    for ax,(family,param) in zip(axes,c['families'].items()):
        for k in c['severities']:
            vv=[x for x in rows if x['family']==family and float(x['severity'])==k]
            x=[float(v['latent_mse']) for v in vv];y=[float(v['episode_return'])-base[int(v['seed'])] for v in vv]
            ax.scatter(x,y,label=str(k),s=14)
            corr=float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None
            associations.append(dict(family=family,severity=k,n=len(x),within_severity_pearson=corr))
        nom=next(x['nominal_parameter'] for x in rows if x['family']==family)
        ax.set(xlabel='On-policy latent MSE',ylabel='Paired return − nominal',title=f'{param}; nominal {float(nom):.6g}');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'reacher_per_seed_error_vs_return.png',dpi=160);plt.close(fig);r.table(out/'prediction_control_associations.csv',associations)
    if probe:
        physical=r.csvread(probe/'reacher_physical_response.csv');fig,axes=plt.subplots(4,4,figsize=(18,15))
        across=[]
        for col,(family,param) in enumerate(c['families'].items()):
            for predictor in ('delta_qvel_rad_s','shifted_latent_mse'):
                xx={float(x['severity']):x['mean'] for x in summary(physical,predictor) if x['family']==family}
                yy={float(x['severity']):x['mean'] for x in summary(rows,'episode_return') if x['family']==family}
                kk=sorted(set(xx)&set(yy));vx=[xx[k] for k in kk];vy=[yy[k] for k in kk]
                corr=float(np.corrcoef(vx,vy)[0,1]) if len(kk)>2 and np.std(vx)>0 and np.std(vy)>0 else None
                across.append(dict(family=family,predictor=predictor,condition_means=len(kk),pearson=corr,interpretation='descriptive across severity; not independent seed replicates or causal evidence'))
            for ax,data,metric,label in zip(axes[:,col],[physical,physical,rows,rows],['delta_qvel_rad_s','shifted_latent_mse','episode_return','distance_final'],['Δ joint velocity [rad/s]','Fixed-data latent MSE','Frozen return','Final distance [m]']):
                ss=sorted([x for x in summary(data,metric) if x['family']==family],key=lambda x:float(x['severity']));ax.plot([float(x['severity']) for x in ss],[x['mean'] for x in ss],'o-');nom=next(x['nominal_parameter'] for x in data if x['family']==family);ax.set(title=f'{param}; nominal {float(nom):.6g}',xlabel='multiplier (actual = multiplier × nominal)',ylabel=label)
        fig.tight_layout();fig.savefig(out/'reacher_physical_latent_control.png',dpi=160);plt.close(fig)
        r.table(out/'across_severity_associations.csv',across)
    stepfile=source/'steps.csv'
    if stepfile.exists():
        steps=r.csvread(stepfile);fig,axes=plt.subplots(1,4,figsize=(18,4))
        for ax,(family,param) in zip(axes,c['families'].items()):
            for k in c['severities']:
                vv=[x for x in steps if x['family']==family and float(x['severity'])==k];g=grouped(vv,('t','seed'),'distance');ts=sorted(set(int(x[0]) for x in g));ax.plot(ts,[np.mean([v for key,v in g.items() if int(key[0])==t]) for t in ts],label=str(k))
            nom=next(x['nominal_parameter'] for x in rows if x['family']==family)
            ax.set(title=f'{param}; nominal {float(nom):.6g}',xlabel='environment step',ylabel='Tip-target distance [m]');ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/'reacher_distance_over_time.png');plt.close(fig)
    (out/'REACHER_FROZEN_GENERALIZATION.md').write_text('# Frozen generalization\n\nAll 24 labels retained. Four k=1 labels alias one canonical nominal evaluation per seed; aliases are not pooled as independent samples. Per-seed raw files are hashed. Fixed calibration error and on-policy error use different state distributions and are labeled separately. Correlations are descriptive, not causal. Force saturation is missing when no actuator has a force limit; normalized action saturation is a separate metric. Means, medians, bootstrap CI, sample counts and outliers are in frozen_seed_summary.csv.\n')

def aggregate(inputs,out,c):
    for p in inputs:
        if (p/'reacher_physical_response.csv').exists():physical_plots(p,out,c)
        if (p/'reacher_frozen_severity.csv').exists():
            probe=next((x for x in inputs if (x/'reacher_physical_response.csv').exists()),None);frozen_plots(p,out,c,probe)
    control=[];prediction=[]
    for p in inputs:
        for filename,dest in [('control.csv',control),('predict.csv',prediction)]:
            for f in p.rglob(filename):
                assert r.read(f.parent/'completion.json')['status']=='PASS';dest.extend(r.csvread(f))
    if not control and not prediction:
        (out/'REACHER_DYNAMICS_ONLY_TTT_RESULTS.md').write_text('# TTT results\n\nNOT_RUN: no held-out control/prediction files supplied. No recovery claim.\n');return
    # Exact duplicate evaluations are an error, never extra statistical samples.
    for data,keys in [(control,('family','severity','method','adaptation_seed','budget','seed','kind')),(prediction,('family','severity','method','adaptation_seed','budget','seed'))]:
        ids=[tuple(x[k] for k in keys) for x in data];assert len(ids)==len(set(ids)),'Duplicate evaluation identity'
    index={(x['family'],float(x['severity']),x['method'],int(x['adaptation_seed']),int(x['budget']),int(x['seed'])):x for x in prediction}
    prediction_baseline={(x['family'],float(x['severity']),int(x['seed'])):float(x['latent_mse']) for x in prediction if x['method']=='frozen' and int(x['budget'])==0}
    baseline={(x['family'],float(x['severity']),int(x['seed']),x['kind']):float(x['episode_return']) for x in control if x['method']=='frozen' and int(x['budget'])==0}
    joined=[]
    for x in control:
        key=(x['family'],float(x['severity']),x['method'],int(x['adaptation_seed']),int(x['budget']),int(x['seed']))
        base=baseline.get((key[0],key[1],key[-1],x['kind']));pr=index.get(key) if x['kind']=='shifted' else None
        predbase=prediction_baseline.get((key[0],key[1],key[-1]))
        joined.append(dict(x,delta_heldout_latent_mse=float(pr['latent_mse'])-predbase if pr and predbase is not None else None,paired_return_difference=float(x['episode_return'])-base if base is not None else None,heldout_fixed_latent_mse=float(pr['latent_mse']) if pr else None))
    failures=[]
    for root in inputs:
        for file in root.rglob('failure.json'):failures.append(dict(path=str(file),**r.read(file)))
    if failures:r.table(out/'run_failures.csv',failures)
    if joined:r.table(out/'reacher_adaptation_results.csv',joined)
    elif prediction:r.table(out/'reacher_adaptation_prediction_only.csv',prediction)
    plt=plotting()
    plot_specs=[('reacher_prediction_recovery.png',prediction,'latent_mse',None),('reacher_control_recovery.png',joined,'paired_return_difference','shifted'),('reacher_distance_recovery.png',joined,'distance_final','shifted'),('reacher_nominal_retention.png',joined,'paired_return_difference','nominal')]
    summaries=[]
    for filename,data,metric,kind in plot_specs:
        if not data:continue
        if kind:data=[x for x in data if x['kind']==kind]
        fig,axes=plt.subplots(4,6,figsize=(23,14),squeeze=False)
        for row,(family,param) in enumerate(c['families'].items()):
            for col,k in enumerate(c['severities']):
                ax=axes[row,col]
                for method in c['methods']:
                    vv=[x for x in data if x['family']==family and float(x['severity'])==k and x['method']==method]
                    # First average evaluation seeds within adaptation replicate.
                    per=grouped(vv,('budget','adaptation_seed'),metric);budgets=sorted(set(int(key[0]) for key in per));ss=[]
                    for b in budgets:
                        st=r.stats([v for key,v in per.items() if int(key[0])==b]);ss.append(st);summaries.append(dict(family=family,severity=k,method=method,budget=b,metric=metric,kind=kind,**st))
                    if ss:
                        ax.plot(budgets,[s['mean'] for s in ss],'o-',label=method)
                        if all(s['ci_low'] is not None for s in ss):ax.fill_between(budgets,[s['ci_low'] for s in ss],[s['ci_high'] for s in ss],alpha=.15)
                nominal=next((float(x['nominal_parameter']) for x in data if x['family']==family and x.get('nominal_parameter') not in ('',None)),None)
                title=f'{param} × {k}\nnominal {nominal:.6g}' if nominal is not None else f'{param} × {k} (missing)'
                ax.set(title=title,xlabel='adaptation transitions',ylabel=metric)
                if ax.lines:ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/filename,dpi=140);plt.close(fig)
    if summaries:r.table(out/'adaptation_replicate_summary.csv',summaries)
    outcomes=[]
    for cond in r.conditions(c):
        vv=[x for x in summaries if x['family']==cond['family'] and float(x['severity'])==cond['severity'] and x['method']=='decision' and x['metric']=='paired_return_difference' and x['kind']=='shifted']
        vv.sort(key=lambda x:x['budget'])
        source=[x for x in control if x['family']==cond['family'] and float(x['severity'])==cond['severity'] and x['method']=='decision' and x['kind']=='shifted']
        declared={int(x['declared_final_budget']) for x in source};assert len(declared)<=1
        final=next(iter(declared),None)
        complete=bool(vv) and vv[-1]['budget']==final and vv[-1]['n']==len(c['adaptation_seeds']) and len([x for x in source if int(x['budget'])==final])==len(c['adaptation_seeds'])*len(c['evaluation_seeds'])
        outcomes.append(dict(**cond,status='COMPLETE' if complete else 'INCOMPLETE' if vv else 'MISSING',declared_final_budget=final,last_measured_budget=vv[-1]['budget'] if vv else None,final_delta_return=vv[-1]['mean'] if complete else None,best_intermediate_delta=max(x['mean'] for x in vv[:-1]) if len(vv)>1 else None,auc=np.trapezoid([x['mean'] for x in vv],[x['budget'] for x in vv]) if len(vv)>1 else None))
    r.table(out/'adaptation_full_grid_outcomes.csv',outcomes)
    available=[x for x in joined if x['kind']=='shifted' and x['delta_heldout_latent_mse'] is not None and x['paired_return_difference'] is not None]
    if available:
        fig,ax=plt.subplots();ax.scatter([x['delta_heldout_latent_mse'] for x in available],[x['paired_return_difference'] for x in available],s=6);ax.set(xlabel='Held-out fixed-bank Δ latent MSE vs Frozen',ylabel='Paired return difference');fig.tight_layout();fig.savefig(out/'reacher_prediction_vs_control.png');plt.close(fig)
    (out/'REACHER_DYNAMICS_ONLY_TTT_RESULTS.md').write_text('# Dynamics-only results\n\nEvaluation seeds are averaged within each independently adapted model before uncertainty across adaptation replicas. One nominal training seed remains a limitation. Final available budget and best intermediate are separate in full-grid outcomes; verify final budget against preregistered budget_protocol.json before interpreting an incomplete matrix. Missing conditions remain explicitly MISSING. Online rollout return is not held-out performance. No causal claim follows from prediction/control association.\n')
