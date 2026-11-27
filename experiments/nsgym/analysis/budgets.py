#!/usr/bin/env python3
"""Aggregate only completed TTT evaluations. No simulator or model execution."""
from . import schema as _io
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


read = _io.read_required
records = _io.list_rows
def write(p, rs):
    return _io.save_table(p, rs, list(rs[0]))
def stats(v):
    from .statistics import summary
    s = summary(v, 'budgets')
    return dict(n_adaptation_seeds=s['n'], mean=s['mean'], median=s['median'],
                std=s['std'] if s['std'] is not None else np.nan,
                ci_low=s['ci_low'] if s['ci_low'] is not None else np.nan,
                ci_high=s['ci_high'] if s['ci_high'] is not None else np.nan,
                outliers_iqr=s['outliers'])


SYNTHETIC=False

def save(fig,path,**kwargs):
    if SYNTHETIC:fig.text(.01,.003,'SYNTHETIC CPU FIXTURE — not 750k performance',fontsize=7,color='darkred')
    fig.savefig(path,**kwargs)


def legend(axs):
    for ax in np.asarray(axs).flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7);break


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluations',required=True);p.add_argument('--output',required=True);p.add_argument('--runs',help='Optional root of online adaptation runs')
    a=p.parse_args();source=Path(a.evaluations);out=Path(a.output)
    evaluations=[];seen=set();perseed=[];steps=[]
    for path in sorted(source.rglob('manifest.json')):
        m=read(path)
        if 'snapshot_sha256' not in m:continue
        if read(path.parent/'completion.json')['status']!='PASS':raise ValueError('Incomplete evaluation')
        i=m['identity'];key=tuple(i[k] for k in ('parameter','severity','method','schedule','adaptation_seed'))+(m['budget'],)
        if key in seen:raise ValueError(f'Duplicate evaluation: {key}')
        seen.add(key);evaluations.append(m)
        base=dict(**i,budget=m['budget'],gradient_updates=m['gradient_updates'],update_seconds=m['update_seconds'])
        for r in records(path.parent/'heldout_episodes.csv'):
            perseed.append(dict(**base,kind=r['kind'],evaluation_seed=int(r['seed']),episode_return=float(r['episode_return']),latent_mse=np.nan,
                velocity=float(r['velocity']),control_cost=float(r['control_cost']),action_saturation=float(r['action_saturation'])))
        fixed=path.parent/'heldout_fixed_data.csv'
        if fixed.exists():
            for r in records(fixed):
                perseed.append(dict(**base,kind='fixed_data',evaluation_seed=int(r['seed']),episode_return=np.nan,latent_mse=float(r['latent_mse']),
                    velocity=np.nan,control_cost=np.nan,action_saturation=np.nan))
        # Seed means in 25-step bins, never treat individual steps as iid.
        bins=defaultdict(list)
        for r in records(path.parent/'heldout_steps.csv'):
            if r['kind']=='shifted':bins[(int(r['seed']),int(r['aligned_step'])//25*25)].append(r)
        for (seed,b),rs in bins.items():
            steps.append(dict(**base,evaluation_seed=seed,aligned_step=b,
                velocity=np.mean([float(r['velocity']) for r in rs]),latent_mse=np.mean([float(r['latent_mse_preupdate']) for r in rs])))
    if not evaluations:raise ValueError('No completed evaluations')
    if len(set(m['identity']['synthetic'] for m in evaluations))!=1:raise ValueError('Cannot pool synthetic and real')
    if len({json.dumps(m['configuration'],sort_keys=True) for m in evaluations})!=1:raise ValueError('Mixed evaluation configurations')
    global SYNTHETIC
    SYNTHETIC=evaluations[0]['identity']['synthetic']
    out.mkdir(parents=True,exist_ok=False)
    (out/'coverage.json').write_text(json.dumps(dict(evaluations=len(evaluations),synthetic=evaluations[0]['identity']['synthetic'],available_identities=[m['identity'] for m in evaluations]),indent=2))
    write(out/'per_seed_evaluation.csv',perseed)
    baseline={}
    for r in perseed:
        if r['method']=='frozen':baseline[(r['parameter'],r['severity'],r['schedule'],r['adaptation_seed'],r['budget'],r['kind'],r['evaluation_seed'])]=r['episode_return']
    metricrows=[]
    for r in perseed:
        for metric,value in [('heldout_latent_mse' if r['kind']=='fixed_data' else ('nominal_return' if r['kind']=='nominal' else 'heldout_return'),r['latent_mse'] if r['kind']=='fixed_data' else r['episode_return'])]:
            metricrows.append(dict(**r,metric=metric,value=value))
        if r['kind']=='shifted':
            k=tuple(r[t] for t in ('parameter','severity','schedule','adaptation_seed','budget','kind','evaluation_seed'))
            if k in baseline:metricrows.append(dict(**r,metric='paired_return_improvement',value=r['episode_return']-baseline[k]))
    initial_nominal={(r['parameter'],r['severity'],r['schedule'],r['method'],r['adaptation_seed'],r['evaluation_seed']):r['episode_return'] for r in perseed if r['kind']=='nominal' and r['budget']==0}
    for r in perseed:
        k=tuple(r[t] for t in ('parameter','severity','schedule','method','adaptation_seed','evaluation_seed'))
        if r['kind']=='nominal' and k in initial_nominal:metricrows.append(dict(**r,metric='nominal_retention_delta',value=r['episode_return']-initial_nominal[k]))
    grouped=defaultdict(list)
    for r in metricrows:
        key=tuple(r[k] for k in ('parameter','severity','schedule','method','budget','metric','adaptation_seed'))
        grouped[key].append(r['value'])
    seedmeans=[dict(zip(('parameter','severity','schedule','method','budget','metric','adaptation_seed'),k),value=np.mean(v),n_evaluation_seeds=len(v)) for k,v in grouped.items()]
    write(out/'adaptation_seed_means.csv',seedmeans)
    groups=defaultdict(list)
    for r in seedmeans:groups[tuple(r[k] for k in ('parameter','severity','schedule','method','budget','metric'))].append(r)
    summary=[dict(zip(('parameter','severity','schedule','method','budget','metric'),k),**stats([r['value'] for r in rs]),
        n_evaluation_seeds=min(r['n_evaluation_seeds'] for r in rs)) for k,rs in groups.items()]
    write(out/'seed_summary.csv',summary)
    plt.rcParams.update({'font.size':8})
    for metric in ('heldout_latent_mse','heldout_return','nominal_return','nominal_retention_delta','paired_return_improvement'):
        for schedule in sorted(set(r['schedule'] for r in summary)):
            rr=[r for r in summary if r['metric']==metric and r['schedule']==schedule]
            if not rr:continue
            fig,axs=plt.subplots(2,6,figsize=(18,6),squeeze=False)
            for row,param in enumerate(('bthigh_damping','torso_mass')):
                for col,k in enumerate((.5,.75,1.,1.25,1.5,2.)):
                    ax=axs[row,col]
                    for method in ('frozen','decision','episode'):
                        d=sorted([r for r in rr if r['parameter']==param and r['severity']==k and r['method']==method],key=lambda r:r['budget'])
                        if d:
                            ax.plot([r['budget'] for r in d],[r['mean'] for r in d],'.-',label=method)
                            ax.fill_between([r['budget'] for r in d],[r['ci_low'] for r in d],[r['ci_high'] for r in d],alpha=.15)
                    ax.set_title(f'{param}\nk={k:g}, nominal={6 if row==0 else 6.250209:g}');ax.set_xlabel('Adaptation transitions');ax.set_ylabel(metric)
            legend(axs);fig.suptitle(f"{'SYNTHETIC SMOKE — ' if evaluations[0]['identity']['synthetic'] else ''}{schedule}: seed means, 95% adaptation-stream bootstrap CI (conditional on fixed evaluation seeds)")
            fig.tight_layout();save(fig,out/f'{schedule}_{metric}.png',dpi=160);plt.close(fig)
    # Held-out shift alignment at the latest snapshot; stream/evaluation seed averaging precedes plotting.
    if steps:
        write(out/'shift_aligned_seed_bins.csv',steps)
        for metric in ('velocity','latent_mse'):
            fig,axs=plt.subplots(2,6,figsize=(18,6))
            for row,param in enumerate(('bthigh_damping','torso_mass')):
                for col,k in enumerate((.5,.75,1,1.25,1.5,2)):
                    ax=axs[row,col]
                    for method in ('frozen','decision','episode'):
                        rr=[r for r in steps if r['parameter']==param and r['severity']==k and r['schedule']=='abrupt' and r['method']==method]
                        if not rr:continue
                        budget=max(r['budget'] for r in rr);g=defaultdict(list)
                        for r in rr:
                            if r['budget']==budget:g[(r['adaptation_seed'],r['aligned_step'])].append(r[metric])
                        means=defaultdict(list)
                        for (_,t),v in g.items():means[t].append(np.mean(v))
                        ts=sorted(means);ss=[stats(means[t]) for t in ts]
                        ax.plot(ts,[s['mean'] for s in ss],label=method)
                        ax.fill_between(ts,[s['ci_low'] for s in ss],[s['ci_high'] for s in ss],alpha=.15)
                    ax.axvline(0,c='k',ls=':');ax.set_title(f'{param} k={k:g}');ax.set_xlabel('Steps from onset');ax.set_ylabel(metric)
            legend(axs);fig.suptitle('Frozen held-out abrupt evaluation at final snapshot; 25-step within-seed bins')
            fig.tight_layout();save(fig,out/f'shift_aligned_{metric}.png',dpi=160);plt.close(fig)
    if a.runs: online_plots(Path(a.runs),out)
    cost=[]
    for m in evaluations:cost.append(dict(**m['identity'],budget=m['budget'],gradient_updates=m['gradient_updates'],update_seconds=m['update_seconds']))
    write(out/'interaction_gradient_cost.csv',cost)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for method in ('frozen','decision','episode'):
        rr=[r for r in cost if r['method']==method];g=defaultdict(list)
        for r in rr:g[r['budget']].append(r)
        for ax,metric in zip(axs,('gradient_updates','update_seconds')):
            xx=sorted(g);ax.plot(xx,[np.mean([r[metric] for r in g[x]]) for x in xx],'.-',label=method);ax.set(xlabel='Adaptation transitions',ylabel=metric);ax.legend()
    fig.tight_layout();save(fig,out/'interaction_gradient_cost.png',dpi=160);plt.close(fig)
    # Final-budget severity response. All grid positions retained, missing data are gaps.
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,param in zip(axs,('bthigh_damping','torso_mass')):
        for schedule in ('fixed','abrupt'):
            for method in ('decision','episode'):
                rr=[r for r in summary if r['metric']=='paired_return_improvement' and r['parameter']==param and r['schedule']==schedule and r['method']==method]
                if not rr:continue
                budget=max(r['budget'] for r in rr);d=sorted([r for r in rr if r['budget']==budget],key=lambda r:r['severity'])
                ax.plot([r['severity'] for r in d],[r['mean'] for r in d],'.-',label=f'{schedule}/{method}')
        ax.set(title=param,xlabel='Severity multiplier',ylabel='Paired return gain vs Frozen');ax.legend()
    fig.tight_layout();save(fig,out/'severity_return_improvement.png',dpi=160);plt.close(fig)
    (out/'ANALYSIS.md').write_text('''# Dynamics-only TTT analysis

Only completed snapshots/evaluations are included; duplicate identity/budget is rejected.
Per-seed tables preserve evaluation seed and adaptation stream seed. First average held-out episodes within each adaptation stream, then use independent adaptation streams as bootstrap units (10,000 resamples). CI is conditional on the fixed evaluation seed bank; it does not estimate uncertainty over unseen evaluation seed banks. A single-stream smoke has no CI. All outliers remain; 1.5 IQR counts are descriptive. Repeated snapshots and time steps are not independent samples. Missing conditions remain empty panels; this is not evidence of no effect.

Online training return lives in each run's online_episodes.csv and is never pooled with held-out return. Shift-aligned plots here are frozen held-out evaluation, not online within-episode adaptation. Raw online_steps.csv separately records episode, pre-update error, update count and aligned step for within/across-episode analysis. No causal interpretation of prediction-error/return correlation. Frozen identical repetitions are pairing references, not additional independent policy evaluations. No S0/S1 legacy duplicates are ingested.

Fixed-data latent MSE is omitted if not supplied, never inferred from closed-loop errors. Synthetic smoke figures validate plotting only, not 750k behavior. Nominal return curves provide retention before/after; compare each stream to budget zero.
''')

def online_plots(root,out):
    data=[]; seen=set()
    for path in sorted(root.rglob('online_steps.csv')):
        m=read(path.parent/'manifest.json');i=m['identity']
        if read(path.parent/'completion.json')['status']!='PASS':raise ValueError('Incomplete online run')
        key=tuple(i[k] for k in ('parameter','severity','schedule','method','adaptation_seed'))
        if key in seen:raise ValueError('Duplicate online run')
        seen.add(key)
        g=defaultdict(list)
        for r in records(path):g[(int(r['episode']),int(r['aligned_step'])//25*25)].append(r)
        for (ep,t),rs in g.items():
            data.append(dict(**i,episode=ep,aligned_step=t,velocity=np.mean([float(r['velocity']) for r in rs]),
                latent_mse=np.mean([float(r['latent_mse_preupdate']) for r in rs]),samples=len(rs)))
    if not data:return
    write(out/'online_shift_aligned_seed_bins.csv',data)
    for episode in sorted(set(r['episode'] for r in data)):
        for metric in ('velocity','latent_mse'):
            fig,axs=plt.subplots(2,6,figsize=(18,6))
            for row,param in enumerate(('bthigh_damping','torso_mass')):
                for col,k in enumerate((.5,.75,1,1.25,1.5,2)):
                    ax=axs[row,col]
                    for method in ('frozen','decision','episode'):
                        g=defaultdict(list)
                        for r in data:
                            if r['parameter']==param and r['severity']==k and r['schedule']=='abrupt' and r['method']==method and r['episode']==episode:g[r['aligned_step']].append(r[metric])
                        ts=sorted(g)
                        if ts:
                            ss=[stats(g[t]) for t in ts];ax.plot(ts,[s['mean'] for s in ss],label=method)
                            ax.fill_between(ts,[s['ci_low'] for s in ss],[s['ci_high'] for s in ss],alpha=.15)
                    ax.axvline(0,c='k',ls=':');ax.set(title=f'{param} k={k:g}',xlabel='Steps from onset',ylabel=metric)
            legend(axs);fig.suptitle(f'ONLINE adaptation, episode {episode}: observed pre-update error; stream bootstrap CI')
            fig.tight_layout();save(fig,out/f'online_episode_{episode}_{metric}.png',dpi=160);plt.close(fig)

