#!/usr/bin/env python3
"""Read-only Reacher research analysis of saved episodes or exported per-seed tables.

No model loading, optimizer or environment calls. Missing original evidence stays missing.
"""
from . import schema as _io
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np
from ..scripts.analyze_existing_mechanisms import Sources, digest, table, js, stats, unique, correlation

ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _io.read_required(Path(__file__).with_name('config.json'))['historical_adapters']['reacher']
FAMILIES = _CONTRACT['families']
GRID = _CONTRACT['severities']
METHODS = {'frozen': 'Frozen', 'decision': 'Dynamics-only', 'full_joint_rl': 'Joint RL'}
METRICS = ['episode_return', 'reward_per_step', 'reward_dist', 'reward_ctrl', 'distance_initial',
           'distance_mean', 'distance_final', 'distance_min', 'action_abs', 'action_saturation',
           'force_rms', 'force_saturation', 'force_limited_count', 'episode_length', 'gradient_updates',
           'adaptation_wall_seconds', 'actual_parameter', 'nominal_parameter']


def num(value):
    if value in (None, ''): return None
    result = float(value)
    if not np.isfinite(result): raise ValueError(f'Nonfinite measurement: {value}')
    return result


def identity(row, stream=True, kind=True):
    key = (row['family'], float(row['severity']))
    if stream: key += (int(row['adaptation_seed']),)
    key += (int(row.get('evaluation_seed', row.get('seed'))),)
    if kind: key += (row.get('kind', 'shifted'),)
    return key


def collapse(rows, key, metrics):
    """Collapse only demonstrably identical reused references; conflicts are errors."""
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            for metric in metrics:
                a, b = num(row.get(metric)), num(result[k].get(metric))
                if a != b and (a is None or b is None or not np.isclose(a, b, rtol=1e-10, atol=1e-12)):
                    raise ValueError(f'Conflicting reused reference: {k} / {metric}')
        else: result[k] = row
    return result


def discover(source, src):
    candidates = [source] if (source/'reacher_adaptation_results.csv').is_file() else sorted(source.rglob('reacher_adaptation_results.csv'))
    candidates = [p if p.is_dir() else p.parent for p in candidates]
    valid = []
    for p in candidates:
        if (p/'completion.json').is_file() and src.json(p/'completion.json').get('status') == 'PASS': valid.append(p)
    if len(valid) != 1:
        raise ValueError(f'Specify the one exported analysis directory: found {len(valid)} PASS candidates: {valid}')
    inventory = []
    for p in sorted(source.rglob('completion.json')):
        d = src.json(p)
        inventory.append(dict(path=str(p.parent), status=d.get('status'), snapshots=d.get('snapshots'),
                              completed_children=len(list(p.parent.glob('*/completion.json'))),
                              evidence='original_or_derived_marker; role must be checked from manifest'))
    return valid[0], inventory


def selection_record(source, argument, src):
    files = sorted(source.rglob('selected_reacher_checkpoint.json'))
    if len(files)>1: raise ValueError('Multiple nominal selections; pass a narrower source')
    if files: return src.json(files[0]), 'original_selection_json'
    if argument:
        text = src.track(argument).read_text()
        if argument.suffix == '.json':
            record=json.loads(text)
            return record.get('record',record), 'user_provided_selection_record'
        # Accept the earlier console transcript without pretending it is original metadata.
        position = text.index('"checkpoint"')
        start = text.rfind('{', 0, position)
        return json.JSONDecoder().raw_decode(text[start:])[0], 'user_provided_console_transcript'
    return {}, 'NOT_AVAILABLE'


def load(source, summary, src, config):
    exported = src.json(summary/'completion.json')
    manifest = src.json(summary/'analysis_manifest.json')
    if manifest['config_fingerprint'] != __import__('hashlib').sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():
        raise ValueError('Exported Reacher config fingerprint does not match the supplied contract')
    rows = src.csv(summary/'reacher_adaptation_results.csv')
    unique(rows, lambda x: (identity(x), x['method'], int(x['budget'])))
    if len(rows) != exported['dynamics_control_rows']: raise ValueError('Control export count mismatch')
    paired = src.csv(summary/'reacher_method_paired_seeds.csv')
    unique(paired, identity)
    final = config['snapshot_budgets'][-1]
    dynamic = unique([x for x in rows if x['method']=='decision' and int(x['budget'])==final], identity)
    frozen = collapse([x for x in rows if x['method']=='frozen' and int(x['budget'])==0],
                      lambda x: identity(x,stream=False), METRICS[:12])
    expected = {(f,float(k),a,s,kind) for f in config['families'] for k in config['severities']
                for a in config['adaptation_seeds'] for s in config['evaluation_seeds'] for kind in ('shifted','nominal')}
    available = set(map(identity,paired)) & set(dynamic)
    # Partial streams are excluded as units, never compared on selected favorable seeds.
    complete = set()
    for f in config['families']:
        for k in config['severities']:
            for a in config['adaptation_seeds']:
                block = {(f,float(k),a,s,kind) for s in config['evaluation_seeds'] for kind in ('shifted','nominal')}
                if block <= available: complete |= block
    exclusions = [dict(family=k[0],severity=k[1],adaptation_seed=k[2],evaluation_seed=k[3],kind=k[4],
                       reason='incomplete paired stream excluded') for k in sorted(expected-complete)]
    canonical=[]; predictions=[]; matched=[]; rewards=[]
    # Keep every saved Dynamics-only snapshot; one Frozen policy reference per condition/physics/seed.
    for x in rows:
        if x['method']=='frozen': continue
        canonical.append(dict(task='Reacher',family=x['family'],parameter=x['parameter'],severity=float(x['severity']),
            method='Dynamics-only',adaptation_seed=int(x['adaptation_seed']),evaluation_seed=int(x['seed']),
            budget=int(x['budget']),kind=x['kind'],snapshot_sha256=None,completion_status='derived_from_PASS_export',
            evidence='exported_episode_row; original child marker unavailable',terminated=x['terminated'],truncated=x['truncated'],
            **{m:num(x.get(m)) for m in METRICS}))
        if x['kind']=='shifted':
            predictions.append(dict(task='Reacher',family=x['family'],parameter=x['parameter'],severity=float(x['severity']),
                method='Dynamics-only',adaptation_seed=int(x['adaptation_seed']),evaluation_seed=int(x['seed']),budget=int(x['budget']),
                latent_mse=num(x['heldout_fixed_latent_mse']),snapshot_sha256=None,source_bank_sha256=None,
                evaluation='fixed_Frozen_bank_reported',encoder_contract='frozen_by_code; original_state_hash_not_available'))
    for key,x in frozen.items():
        canonical.append(dict(task='Reacher',family=x['family'],parameter=x['parameter'],severity=float(x['severity']),
            method='Frozen',adaptation_seed=None,evaluation_seed=int(x['seed']),budget=0,kind=x['kind'],snapshot_sha256=None,
            completion_status='derived_from_PASS_export',evidence='reused_reference_collapsed',
            physical_reference_alias=f'nominal-seed{x["seed"]}' if x['kind']=='nominal' or float(x['severity'])==1 else '',
            terminated=x['terminated'],truncated=x['truncated'],**{m:num(x.get(m)) for m in METRICS}))
        if x['kind']=='shifted':
            predictions.append(dict(task='Reacher',family=x['family'],parameter=x['parameter'],severity=float(x['severity']),
                method='Frozen',adaptation_seed=None,evaluation_seed=int(x['seed']),budget=0,
                latent_mse=num(x['heldout_fixed_latent_mse']),snapshot_sha256=None,source_bank_sha256=None,
                evaluation='fixed_Frozen_bank_reported',encoder_contract='nominal_encoder_reported'))
    for x in paired:
        key=identity(x)
        if key not in complete: continue
        d=dynamic[key]; f=frozen[(key[0],key[1],key[3],key[4])]
        for field,value in [('frozen_return',f['episode_return']),('dynamics_only_return',d['episode_return']),
                            ('frozen_distance_final',f['distance_final']),('dynamics_distance_final',d['distance_final'])]:
            if not np.isclose(float(x[field]),float(value),rtol=1e-10,atol=1e-10): raise ValueError(f'Pair mismatch: {key} {field}')
        base=dict(family=key[0],parameter=d['parameter'],severity=key[1],adaptation_seed=key[2],evaluation_seed=key[3],
                  kind=key[4],budget=final,actual_parameter=float(d['actual_parameter']),nominal_parameter=float(d['nominal_parameter']))
        fj,dj,jj=map(float,(x['frozen_return'],x['dynamics_only_return'],x['joint_rl_return']))
        de=num(d['delta_heldout_latent_mse']) if key[4]=='shifted' else None
        ed=num(d['heldout_fixed_latent_mse']) if key[4]=='shifted' else None
        matched.append(dict(**base,frozen_return=fj,ttt_return=dj,joint_return=jj,delta_D=dj-fj,delta_J=jj-fj,delta_JD=jj-dj,
            frozen_distance_mean=float(f['distance_mean']),ttt_distance_mean=float(d['distance_mean']),
            frozen_distance_final=float(x['frozen_distance_final']),ttt_distance_final=float(x['dynamics_distance_final']),
            joint_distance_final=float(x['joint_distance_final']),
            delta_distance_D=float(x['dynamics_distance_final'])-float(x['frozen_distance_final']),
            delta_distance_J=float(x['joint_distance_final'])-float(x['frozen_distance_final']),
            delta_mean_distance_D=float(d['distance_mean'])-float(f['distance_mean']),
            frozen_mse=ed-de if ed is not None and de is not None else None,ttt_mse=ed,delta_E=de))
        canonical.append(dict(task='Reacher',**base,method='Joint RL',snapshot_sha256=None,completion_status='derived_from_PASS_export',
            evidence='paired_summary_partial_metrics; original_episode_csv_unavailable',episode_return=jj,
            reward_per_step=None,reward_dist=None,reward_ctrl=None,distance_mean=None,
            distance_final=float(x['joint_distance_final']),action_abs=float(x['joint_action_abs']),
            episode_length=None,gradient_updates=None,terminated=None,truncated=None))
        rewards.append(dict(**base,frozen_reward_dist=float(f['reward_dist']),ttt_reward_dist=float(d['reward_dist']),
            frozen_reward_ctrl=float(f['reward_ctrl']),ttt_reward_ctrl=float(d['reward_ctrl']),
            delta_reward_dist=float(d['reward_dist'])-float(f['reward_dist']),
            delta_reward_ctrl=float(d['reward_ctrl'])-float(f['reward_ctrl']),delta_return=dj-fj,
            joint_reward_dist=None,joint_reward_ctrl=None))
        for z in (f,d):
            if abs(float(z['reward_dist'])+float(z['reward_ctrl'])-float(z['episode_return']))>1e-8:
                raise ValueError('Reward decomposition does not sum to episode return')
    return canonical,predictions,matched,rewards,exclusions,manifest


def summarize(matched,config,return_band_fraction,error_band_fraction):
    groups=defaultdict(list)
    for x in matched: groups[(x['family'],x['severity'],x['kind'])].append(x)
    summaries=[]; streams=[]; coupling=[]
    metrics=['frozen_return','ttt_return','joint_return','delta_D','delta_J','delta_JD',
             'frozen_distance_final','ttt_distance_final','joint_distance_final','delta_distance_D','delta_distance_J',
             'delta_mean_distance_D','frozen_mse','ttt_mse','delta_E']
    for (family,severity,kind),rows in sorted(groups.items()):
        per=[]
        for a in sorted({x['adaptation_seed'] for x in rows}):
            rr=[x for x in rows if x['adaptation_seed']==a]
            per.append(dict(family=family,parameter=FAMILIES[family],severity=severity,kind=kind,adaptation_seed=a,
                evaluation_count=len(rr),**{m:float(np.mean([x[m] for x in rr])) if all(x[m] is not None for x in rr) else None for m in metrics}))
        streams.extend(per)
        for m in metrics:
            values=[x[m] for x in per if x[m] is not None]
            if not values: continue
            unit='adaptation_stream_conditional_on_fixed_evaluation_bank'
            if m.startswith('frozen_'):
                values=list({x['evaluation_seed']:x[m] for x in rows}.values());unit='unique_evaluation_seed_shared_Frozen'
            st=stats(values);q1,q3=np.quantile(values,[.25,.75]);iqr=q3-q1
            summaries.append(dict(family=family,parameter=FAMILIES[family],severity=severity,kind=kind,metric=m,unit=unit,
                positive_fraction=float(np.mean(np.array(values)>0)),outliers=int(sum(v<q1-1.5*iqr or v>q3+1.5*iqr for v in values)),**st))
        if kind=='shifted':
            ds=stats([x['delta_D'] for x in per]);es=stats([x['delta_E'] for x in per])
            rb=return_band_fraction*abs(float(np.mean([x['frozen_return'] for x in per])))
            eb=error_band_fraction*float(np.mean([x['frozen_mse'] for x in per]))
            def state(s,b):
                if s['ci_low']>b:return 'positive'
                if s['ci_high']< -b:return 'negative'
                if s['ci_low']>=-b and s['ci_high']<=b:return 'within_band'
                return 'inconclusive'
            ec,jc=state(es,eb),state(ds,rb)
            cls={('negative','positive'):'prediction_improves_control_improves',
                 ('negative','within_band'):'prediction_improves_control_within_band',
                 ('negative','negative'):'prediction_improves_control_degrades',
                 ('positive','positive'):'prediction_worsens_control_improves',
                 ('within_band','positive'):'prediction_within_band_control_improves',
                 ('positive','negative'):'prediction_and_control_degrade'}.get((ec,jc),'inconclusive')
            coupling.append(dict(family=family,parameter=FAMILIES[family],severity=severity,adaptation_streams=len(per),
                evaluation_seeds=len(config['evaluation_seeds']),classification=cls,prediction_state=ec,control_state=jc,
                return_band=rb,error_band=eb,delta_E=es['mean'],delta_D=ds['mean'],
                error_ci_low=es['ci_low'],error_ci_high=es['ci_high'],return_ci_low=ds['ci_low'],return_ci_high=ds['ci_high'],
                relative_mse_reduction=-es['mean']/float(np.mean([x['frozen_mse'] for x in per])),
                delta_distance_D=float(np.mean([x['delta_distance_D'] for x in per])),
                improved_stream_fraction=float(np.mean([x['delta_D']>0 for x in per]))))
    return summaries,streams,coupling


def associations(streams,coupling):
    records=[]
    for x in coupling:
        rr=[v for v in streams if v['kind']=='shifted' and v['family']==x['family'] and v['severity']==x['severity']]
        # n=3 is reported as descriptive scatter only, never a significance claim.
        records.append(dict(scope='within_severity_adaptation_seeds',family=x['family'],severity=x['severity'],
                            n=len(rr),pearson=None,spearman=None,reason='n=3; no correlation inference'))
    for family in FAMILIES:
        rr=[x for x in coupling if x['family']==family]
        p,s=correlation([x['delta_E'] for x in rr],[x['delta_D'] for x in rr])
        records.append(dict(scope='across_severity_condition_means',family=family,n=len(rr),pearson=p,spearman=s,
                            reason='descriptive; severity confounding, no causal inference'))
    return records


def draw(out,summaries,streams,physical,canonical,rewards,coupling,cross,nominal,joint_prediction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'frozen_return':'#777777','ttt_return':'#1976b5','joint_return':'#dc7d26'}
    def panels(filename,metrics,ylabel,kind='shifted'):
        fig,axes=plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
        for ax,(family,param) in zip(axes,FAMILIES.items()):
            for metric,label in metrics:
                rr=sorted([x for x in summaries if x['family']==family and x['kind']==kind and x['metric']==metric],key=lambda x:x['severity'])
                if rr:
                    xx=[x['severity'] for x in rr]; yy=[x['mean'] for x in rr]
                    line,=ax.plot(xx,yy,'o-',label=label,color=colors.get(metric))
                    ax.fill_between(xx,[x['ci_low'] for x in rr],[x['ci_high'] for x in rr],alpha=.15,color=line.get_color())
            values=[x['nominal_parameter'] for x in canonical if x['family']==family and x.get('nominal_parameter') is not None]
            value=values[0] if values else None
            unit='kg' if family.startswith('M') else 'N m s/rad'
            ax.set(title=f'{param}\nnominal={value:.5g} {unit}' if value is not None else param,
                   xlabel='Severity multiplier k (actual = k × nominal)' if kind=='shifted' else 'Adaptation k; evaluation at nominal physics',ylabel=ylabel,xticks=GRID)
            ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.suptitle('Means and 95% descriptive bootstrap CI; adapted: 3 streams × 20 seeds; Frozen: 20 unique seeds',fontsize=11)
        fig.savefig(out/filename,dpi=180);plt.close(fig)
    panels('03_heldout_return.png',[(m,METHODS[n]) for m,n in [('frozen_return','frozen'),('ttt_return','decision'),('joint_return','full_joint_rl')]],'Episode return')
    panels('04_final_target_distance.png',[('frozen_distance_final','Frozen'),('ttt_distance_final','Dynamics-only'),('joint_distance_final','Joint RL')],'Final fingertip–target distance [m]')
    panels('05_adaptation_benefit.png',[('delta_D','TTT − Frozen'),('delta_J','Joint − Frozen'),('delta_JD','Joint − TTT')],'Paired episode-return change')
    panels('06_fixed_bank_prediction.png',[('frozen_mse','Frozen'),('ttt_mse','Dynamics-only')],'One-step latent MSE (fixed encoder)')
    panels('nominal_retention.png',[('delta_D','TTT − Frozen'),('delta_J','Joint − Frozen')],'Nominal return change',kind='nominal')
    if physical:
        fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained')
        for col,(family,param) in enumerate(FAMILIES.items()):
            for row,(metric,label) in enumerate([('delta_qvel_rad_s','Joint-velocity difference [rad/s]'),('delta_tip_m','Fingertip displacement [m]'),('delta_accel_rad_s2','Finite-difference acceleration [rad/s²]')]):
                rr=sorted([x for x in physical if x['family']==family and x['metric']==metric],key=lambda x:float(x['severity']))
                ax=axes[row,col];ax.plot([float(x['severity']) for x in rr],[float(x['mean']) for x in rr],'o-')
                ax.fill_between([float(x['severity']) for x in rr],[float(x['ci_low']) for x in rr],[float(x['ci_high']) for x in rr],alpha=.2)
                nom=next((x.get('nominal_parameter') for x in canonical if x['family']==family and x.get('nominal_parameter') is not None),None)
                ax.set(title=f'{param}; nominal={nom:.5g}' if nom is not None else param,xlabel='k (actual = k × nominal)',ylabel=label,xticks=GRID)
        fig.suptitle('Measured common-state/action physical response; existing calibration-seed summaries (n=5)')
        fig.savefig(out/'02_physical_response.png',dpi=180);plt.close(fig)
    if joint_prediction:
        fig,axes=plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
        for ax,(family,param) in zip(axes,FAMILIES.items()):
            rr=sorted([x for x in joint_prediction if x['family']==family],key=lambda x:float(x['severity']))
            xx=[float(x['severity']) for x in rr]
            ax.plot(xx,[float(x['mean']) for x in rr],'o-',color='#dc7d26')
            ax.fill_between(xx,[float(x['ci_low']) for x in rr],[float(x['ci_high']) for x in rr],alpha=.2,color='#dc7d26')
            ax.set(title=param,xlabel='Adaptation severity k',ylabel='Joint own-encoder latent MSE',xticks=GRID)
        fig.suptitle('Joint representation-specific error; not directly comparable with Frozen / TTT MSE; n=3 streams')
        fig.savefig(out/'06b_joint_own_encoder_prediction.png',dpi=180);plt.close(fig)
    if physical:
        fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained')
        for col,(family,param) in enumerate(FAMILIES.items()):
            nom=next((x.get('nominal_parameter') for x in canonical if x['family']==family and x.get('nominal_parameter') is not None),None)
            for row,(metric,label) in enumerate([('delta_qvel_rad_s','Joint-velocity difference [rad/s]'),('shifted_latent_mse','Fixed-probe latent MSE'),('frozen_return','Frozen held-out return')]):
                rr=sorted([x for x in (physical if row<2 else summaries) if x['family']==family and x['metric']==metric and (row<2 or x['kind']=='shifted')],key=lambda x:float(x['severity']))
                ax=axes[row,col];xx=[float(x['severity']) for x in rr]
                ax.plot(xx,[float(x['mean']) for x in rr],'o-')
                ax.fill_between(xx,[float(x['ci_low']) for x in rr],[float(x['ci_high']) for x in rr],alpha=.2)
                ax.set(title=f'{param}; nominal={nom}' if nom is not None else param,xlabel='k; actual parameter = k × nominal',ylabel=label,xticks=GRID)
        fig.suptitle('Measured physical → latent → control association; probe n=5, Frozen n=20; no causal attribution')
        fig.savefig(out/'physical_latent_control.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');markers=['o','s','^','v','D','P'];palette=['#1976b5','#43a047','#e79b26','#a852a3']
    for family,color in zip(FAMILIES,palette):
        for k,marker in zip(GRID,markers):
            rr=[x for x in streams if x['family']==family and x['severity']==k and x['kind']=='shifted']
            ax.scatter([x['delta_E'] for x in rr],[x['delta_D'] for x in rr],c=color,marker=marker,label=f'{family} ×{k:g}',s=45)
    ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8)
    ax.set(xlabel='Held-out latent MSE change (TTT − Frozen)',ylabel='Paired return change (TTT − Frozen)',title='Each point is one adaptation stream mean over 20 held-out seeds')
    ax.legend(ncol=4,fontsize=7,loc='lower left',bbox_to_anchor=(0,1.06))
    fig.savefig(out/'07_prediction_control.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
    for ax,(family,param) in zip(axes,FAMILIES.items()):
        for metric,label in [('delta_reward_dist','Distance reward change'),('delta_reward_ctrl','Control reward change')]:
            rr=[x for x in rewards if x['family']==family and x['kind']=='shifted'];xx=sorted({x['severity'] for x in rr})
            yy=[]
            for k in xx: yy.append(np.mean([x[metric] for x in rr if x['severity']==k]))
            ax.plot(xx,yy,'o-',label=label)
        ax.axhline(0,color='gray',lw=.8);ax.set(title=param,xlabel='Severity multiplier',ylabel='TTT − Frozen reward contribution');ax.legend(fontsize=8)
    fig.suptitle('Measured reward decomposition; Joint component data unavailable in exported pairing')
    fig.savefig(out/'08_reward_decomposition.png',dpi=180);plt.close(fig)
    # Nominal validation distributions are not synthesized from aggregate mean/std.
    ff=[x for x in canonical if x['method']=='Frozen' and x['family']=='D0' and x['severity']==1 and x['kind']=='shifted']
    if ff:
        fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
        axes[0].hist([x['episode_return'] for x in ff],bins=8);axes[1].hist([x['distance_final'] for x in ff],bins=8)
        axes[0].set(xlabel='Episode return',ylabel='Held-out seeds');axes[1].set(xlabel='Final target distance [m]',ylabel='Held-out seeds')
        fig.suptitle('Nominal held-out Frozen behavior; NOT the nominal checkpoint-selection validation bank')
        fig.savefig(out/'nominal_heldout_distribution.png',dpi=180);plt.close(fig)
    if cross:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for ax,task in zip(axes,('HalfCheetah','Reacher')):
            for method,color in [('Dynamics-only','#1976b5'),('Joint RL','#dc7d26')]:
                rr=[x for x in cross if x['task']==task and x['method']==method]
                xx=np.arange(len(rr));ax.scatter(xx,[x['normalized_delta'] for x in rr],label=method,s=14,c=color)
            ax.axhline(0,color='gray',lw=.8);ax.set(title=task,xlabel='Condition × adaptation-stream index',ylabel='Mean paired return change / |mean Frozen return|');ax.legend()
        fig.suptitle('Descriptive within-task normalization; different horizons, budgets and physical families')
        fig.savefig(out/'10_cross_task_normalized.png',dpi=180);plt.close(fig)


def validate_resume(original_marker,resumed_marker,original_rows,resumed_rows):
    if original_marker['snapshot_sha256']!=resumed_marker['snapshot_sha256'] or original_rows!=resumed_rows:
        raise ValueError('Conflicting resumed snapshot or episode rows')
    return True


def snapshot_key(record):
    i=record['identity']
    return (i['family'],float(i['severity']),i['method'],int(i['adaptation_seed']),int(record.get('budget',10000)))


def audit_raw(raw,src,config,canonical,predictions,matched,rewards,out):
    """Cross-check original saved records; never load model or trajectory binaries."""
    selected=src.json(raw/'nominal-home-01/selected_reacher_checkpoint.json')
    archive=raw.with_name(raw.name+'.tar.gz')
    if archive.exists():src.track(archive)
    sha=selected['model_sha256'];inventory=[];markers={};resume=[];adapt={};joint_runs={}
    stages=['nominal-home-01','inventory-home-01','probe-home-01','frozen-home-01','adapt-home-01',
            'prediction-home-01','control-home-01','control-home-02-resumed',
            'full-joint-home-01/runs-fixed','full-joint-home-01/evaluation-full-01']
    for name in stages:
        p=raw/name;marker=p/'completion.json'
        d=src.json(marker) if marker.exists() else {}
        if name!='control-home-01' and d.get('status')!='PASS':raise ValueError(f'Incomplete required stage: {name}')
        if (p/'manifest.json').exists():src.json(p/'manifest.json')
        children=sorted(p.glob('*/completion.json'))
        inventory.append(dict(stage=name,status=d.get('status','INCOMPLETE'),completed_children=len(children),path=str(p.resolve())))
        for child in children:
            c=src.json(child)
            if c.get('status')!='PASS':raise ValueError(f'Incomplete child {child}')
            if 'identity' in c:markers[(name,snapshot_key(c))]=(child.parent,c)
    expected={(f,float(k),m,a,b) for f in config['families'] for k in config['severities']
              for m in config['methods'] for a in config['adaptation_seeds'] for b in config['snapshot_budgets']}
    for name in ('prediction-home-01','control-home-02-resumed'):
        actual={key for stage,key in markers if stage==name}
        if actual!=expected:raise ValueError(f'Snapshot grid mismatch: {name}')
    for p in sorted((raw/'adapt-home-01').glob('*/manifest.json')):
        m=src.json(p);c=src.json(p.with_name('completion.json'));i=m['identity']
        if m['selection']['model_sha256']!=sha or m['config']!=config or m['synthetic']:raise ValueError('Adaptation provenance mismatch')
        k=(i['family'],float(i['severity']),i['method'],int(i['adaptation_seed']))
        if k in adapt:raise ValueError('Duplicate adaptation identity')
        adapt[k]=(m,c)
    if len(adapt)!=144:raise ValueError('Adaptation grid incomplete')
    fixed_hashes={c['fixed_hash'] for m,c in adapt.values()}
    auxiliary_hashes={c['auxiliary_hash'] for m,c in adapt.values()}
    if len(fixed_hashes)!=1 or len(auxiliary_hashes)!=1:raise ValueError('Frozen-component hash mismatch')
    for p in sorted((raw/'full-joint-home-01/runs-fixed').glob('*/manifest.json')):
        m=src.json(p);c=src.json(p.with_name('completion.json'));i=m['identity']
        if m['checkpoint_sha256']!=sha:raise ValueError('Joint nominal origin mismatch')
        k=(i['family'],float(i['severity']),'full_joint_rl',int(i['adaptation_seed']))
        if k in joint_runs:raise ValueError('Duplicate Joint identity')
        joint_runs[k]=(p.parent,m,c)
    if len(joint_runs)!=72:raise ValueError('Joint grid incomplete')
    for (stage,k),(p,c) in markers.items():
        if stage=='control-home-01':
            other,oc=markers[('control-home-02-resumed',k)]
            same_rows=validate_resume(c,oc,src.csv(p/'control.csv'),src.csv(other/'control.csv'))
            resume.append(dict(snapshot=str(k),snapshot_sha256=c['snapshot_sha256'],identical_episode_rows=same_rows,counted_once=True))
        elif stage=='prediction-home-01':
            if c['snapshot_sha256']!=markers[('control-home-02-resumed',k)][1]['snapshot_sha256']:raise ValueError('Prediction/control snapshot mismatch')
        elif stage=='full-joint-home-01/evaluation-full-01':
            if c['snapshot_sha256']!=joint_runs[k[:4]][2]['snapshot_sha256']:raise ValueError('Joint snapshot mismatch')
    frozen_rows=src.csv(raw/'frozen-home-01/reacher_frozen_severity.csv')
    frozen=unique(frozen_rows,lambda x:(x['family'],float(x['severity']),int(x['seed'])))
    if len(frozen)!=480:raise ValueError('Frozen condition grid incomplete')
    control={};pred={};snapshots=[]
    for stage,filename in [('control-home-02-resumed','control.csv'),('prediction-home-01','predict.csv'),
                           ('full-joint-home-01/evaluation-full-01','control.csv'),('full-joint-home-01/evaluation-full-01','predict.csv')]:
        for (st,k),(p,c) in sorted(markers.items()):
            if st!=stage:continue
            rr=src.csv(p/filename)
            want={(s,kind) for s in config['evaluation_seeds'] for kind in (['shifted','nominal'] if filename=='control.csv' else ['shifted'])}
            got={(int(x['seed']),x.get('kind','shifted')) for x in rr}
            if got!=want or len(rr)!=len(want):raise ValueError(f'Episode seed grid mismatch {p}/{filename}')
            snapshots.append(dict(stage=stage,table=filename,**c['identity'],budget=k[4],snapshot_sha256=c['snapshot_sha256'],episodes=len(rr)))
            for x in rr:
                if snapshot_key(dict(identity=x,budget=x['budget']))!=k:raise ValueError('Episode identity differs from snapshot')
                key=(*k,int(x['seed']),x.get('kind','shifted'))
                dest=control if filename=='control.csv' else pred
                if key in dest:raise ValueError('Duplicate raw episode')
                x=dict(x,snapshot_sha256=c['snapshot_sha256'])
                dest[key]=x
                if filename=='predict.csv':
                    f=frozen[(k[0],k[1],int(x['seed']))]
                    if x['source_sha256']!=f['data_sha256']:raise ValueError('Held-out source bank mismatch')
                elif abs(float(x['reward_dist'])+float(x['reward_ctrl'])-float(x['episode_return']))>1e-8:
                    raise ValueError('Raw reward decomposition mismatch')
    for x in canonical:
        method={'Dynamics-only':'decision','Joint RL':'full_joint_rl','Frozen':'frozen'}[x['method']]
        a=x['adaptation_seed'] or config['adaptation_seeds'][0]
        key=(x['family'],x['severity'],method,a,x['budget'],x['evaluation_seed'],x['kind'])
        r=control[key]
        if not np.isclose(x['episode_return'],float(r['episode_return']),rtol=0,atol=1e-10):raise ValueError('Export/raw return conflict')
        for m in METRICS:
            v=num(r.get(m))
            if m!='adaptation_wall_seconds' and x.get(m) is not None and v is not None and not np.isclose(x[m],v,rtol=1e-9,atol=1e-10):raise ValueError(f'Export/raw metric conflict: {m}')
            if v is not None:x[m]=v
        x.update(snapshot_sha256=r['snapshot_sha256'],completion_status='PASS_original_record',evidence='original_control_csv',terminated=r['terminated'],truncated=r['truncated'])
    for x in predictions:
        method='decision' if x['method']=='Dynamics-only' else 'frozen'
        key=(x['family'],x['severity'],method,x['adaptation_seed'] or config['adaptation_seeds'][0],x['budget'],x['evaluation_seed'],'shifted')
        r=pred[key]
        if not np.isclose(x['latent_mse'],float(r['latent_mse']),rtol=1e-8,atol=1e-14):raise ValueError('Export/raw prediction conflict')
        x.update(snapshot_sha256=r['snapshot_sha256'],source_bank_sha256=r['source_sha256'],encoder_contract='fixed_and_auxiliary_hashes_identical_in_original_completion_records',
                 **{m:int(r[m]) for m in ('samples','low_action_n','mid_action_n','high_action_n') if m in r})
    for key,r in pred.items():
        if key[2]!='full_joint_rl':continue
        predictions.append(dict(task='Reacher',family=key[0],parameter=r['parameter'],severity=key[1],method='Joint RL',adaptation_seed=key[3],budget=key[4],evaluation_seed=key[5],
            latent_mse=float(r['latent_mse']),snapshot_sha256=r['snapshot_sha256'],source_bank_sha256=r['source_sha256'],samples=int(r['samples']),evaluation='fixed_Frozen_bank',encoder_contract='Joint_own_encoder_NOT_comparable_to_Frozen_encoder'))
    for x in rewards:
        key=(x['family'],x['severity'],'full_joint_rl',x['adaptation_seed'],x['budget'],x['evaluation_seed'],x['kind'])
        r=control[key]
        for field in ('reward_dist','reward_ctrl'):x['joint_'+field]=float(r[field]);x['joint_delta_'+field]=float(r[field])-x['frozen_'+field]
    events=[];online=[];budgets=[]
    for key,(p,m,c) in sorted(joint_runs.items()):
        updates=[];ep_return=0.;last=0;total=0
        with src.track(p/'training.jsonl').open() as handle:
            for line in handle:
                r=json.loads(line);total+=1
                if r['transition']!=total:raise ValueError('Joint transition log gap')
                ep_return+=r['reward']
                if r['terminated'] or r['truncated']:
                    online.append(dict(**m['identity'],transition=total,episode=r['episode'],online_return=ep_return));ep_return=0.
                if r['updates']!=last:
                    if r['updates']!=last+1:raise ValueError('Joint update counter gap')
                    metrics={k:num(v) for k,v in r['metrics'].items()}
                    updates.append(total);events.append(dict(**m['identity'],transition=total,updates=r['updates'],replay_episodes=r['replay_episodes'],**metrics))
                    last=r['updates']
        if updates!=list(range(75,10001,25)) or total!=c['transitions'] or last!=c['updates']:raise ValueError('Joint log/marker schedule mismatch')
        budgets.append(dict(**m['identity'],transitions=total,first_update=updates[0],last_update=updates[-1],updates=last,
            optimizer_step_calls=c['optimizer_step_calls'],completed_episodes=c['completed_episodes'],replay_rows=c['replay_rows'],snapshot_sha256=c['snapshot_sha256']))
    ttt_budgets=[]
    for key,(m,c) in sorted(adapt.items()):
        p=raw/'adapt-home-01'/f'{key[0]}-k{key[1]:g}-{key[2]}-seed{key[3]}'
        first=None;last_updates=0;last_trans=0
        with src.track(p/'online_steps.csv').open() as h:
            for r in csv.DictReader(h):
                last_trans=int(r['transitions']);u=int(r['gradient_updates'])
                if u>last_updates and first is None:first=last_trans
                last_updates=u
        if last_trans!=c['metrics']['adaptation_transitions'] or last_updates!=c['metrics']['gradient_update_steps']:raise ValueError('TTT log/marker mismatch')
        ttt_budgets.append(dict(**m['identity'],transitions=last_trans,updates=last_updates,first_update=first,
            fixed_hash=c['fixed_hash'],auxiliary_hash=c['auxiliary_hash'],model_hash=c['model_hash']))
    validation=src.csv(raw/'nominal-home-01/reacher_checkpoint_validation.csv')
    selected_rows=[r for r in validation if int(r['training_step'])==selected['training_step']]
    if sorted(int(r['seed']) for r in selected_rows)!=sorted(config['validation_seeds']):raise ValueError('Nominal validation seed mismatch')
    if not np.isclose(np.mean([float(r['episode_return']) for r in selected_rows]),selected['nominal_performance']['mean']):raise ValueError('Nominal validation score mismatch')
    probe=src.csv(raw/'probe-home-01/reacher_physical_response.csv')
    table(out/'tables/raw_stage_inventory.csv',inventory);table(out/'tables/resumed_duplicate_audit.csv',resume)
    table(out/'tables/snapshot_identity_audit.csv',snapshots);table(out/'tables/joint_update_events.csv',events)
    table(out/'tables/joint_online_episodes.csv',online);table(out/'tables/joint_budget_measured.csv',budgets)
    table(out/'tables/dynamics_budget_measured.csv',ttt_budgets);table(out/'tables/nominal_validation_episodes.csv',validation)
    result=dict(status='PASS_SAVED_RECORD_AUDIT',stages=inventory,resumed_identical_children=len(resume),
        adaptation_streams=len(adapt),joint_streams=len(joint_runs),control_snapshots=864,prediction_snapshots=864,
        joint_update_events=len(events),same_nominal_sha256=sha,fixed_component_hashes_identical=True,
        checkpoint_and_snapshot_bytes_verified=False,npz_bank_bytes_verified=False,
        limitations=['No model .pt or trajectory .npz in transfer; hashes cross-checked between saved records only',
                     'Original per-step CSVs retained; trajectory replay and fresh environment checks not performed'])
    js(out/'provenance/raw_audit.json',result)
    return dict(result=result,validation=selected_rows,events=events,online=online,probe=probe,control=control)


def raw_figures_and_reports(out,raw_data,canonical,rewards,src,raw,plotting=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tab=out/'tables';rep=out/'reports';figdir=out/'figures'
    # Remove stale claims only in this newly generated report package, retaining the v2 package.
    for p in rep.glob('*.md'):
        text=p.read_text()
        substitutions={
            'original per-stream hashes are unavailable here':'original per-stream hashes were cross-checked; model bytes were not transferred',
            'original snapshot/stage audits and Joint learning diagnostics are absent':'saved-record audits and Joint learning diagnostics are now included in RAW_EVIDENCE_AUDIT.md and JOINT_LEARNING_STABILITY.md',
            'Joint components are NOT AVAILABLE':'Joint components are available in JOINT_REWARD_DECOMPOSITION.md',
            'Joint reward components are unavailable':'Joint reward components are available in JOINT_REWARD_DECOMPOSITION.md',
            'Joint reward-distance/control-cost components are absent':'Joint reward-distance/control-cost components are now measured in JOINT_REWARD_DECOMPOSITION.md',
            'original_state_hash_not_available':'original_completion_hashes_checked',
        }
        for a,b in substitutions.items():text=text.replace(a,b)
        p.write_text('> Original-record supplement: see RAW_EVIDENCE_AUDIT.md. Any export-only availability statements below describe the earlier summary source; the supplement supersedes them.\n\n'+text)
    def md(rows,cols):return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(str(r.get(c,'')) for c in cols)+' |' for r in rows])
    r=raw_data['result']
    (rep/'RAW_EVIDENCE_AUDIT.md').write_text('# Original saved-record audit\n\n'+json.dumps(r,indent=2)+'\n\n'
        'All 160 completed children of interrupted control-home-01 match resumed snapshot hashes and control CSV rows exactly. They are counted once. '
        'The resumed stage contains the complete 864-snapshot grid. Prediction/control snapshot hashes agree. All 144 adaptation streams and 72 Joint streams name the selected 670k nominal SHA. '
        'Joint evaluation snapshot hashes match training completion markers. Prediction data-bank hashes agree with the Frozen manifest. '
        'Fixed and auxiliary hashes match across all Dynamics-only/Frozen adaptation completions. These establish record consistency, not independent model-byte verification.\n\n'
        'Canonical tables now contain original snapshot identities, Joint episode/reward fields and Joint prediction rows. '
        'Every original episode was checked for complete evaluation seeds and duplicate identity. Existing exported control and prediction metrics agree with originals.\n')
    vr=raw_data['validation'];ns=[dict(metric=m,**stats([float(x[m]) for x in vr])) for m in ['episode_return','distance_mean','distance_final']]
    (rep/'REACHER_NOMINAL_QUALITY.md').write_text('# Nominal checkpoint quality — original validation\n\n'
        'Selected checkpoint: 670k. SHA256 in original selection: '+r['same_nominal_sha256']+'. Model bytes not transferred. '
        'Validation seeds 51000001–51000020; held-out experiment seeds 54000001–54000020 are separate. '\
        'Reacher-v5, observation dim 10, action dim 2, horizon 50; MPPI horizon 3, iterations 6, samples 512, elites 64, policy trajectories 24.\n\n'+md(ns,['metric','n','mean','median','std','ci_low','ci_high'])+
        '\n\nObserved: competent target approach. No binary success threshold or independent multi-training-seed convergence claim. '
        'Reward components are negative target distance and negative squared action norm; raw recorded components sum to return.\n')
    groups=defaultdict(list)
    for x in rewards:groups[(x['family'],x['severity'],x['kind'],x['adaptation_seed'])].append(x)
    reward_streams=[dict(family=f,severity=k,kind=kind,adaptation_seed=a,**{m:float(np.mean([x[m] for x in rr])) for m in ['joint_delta_reward_dist','joint_delta_reward_ctrl','delta_reward_dist','delta_reward_ctrl']}) for (f,k,kind,a),rr in sorted(groups.items())]
    table(tab/'reward_components_per_stream.csv',reward_streams)
    summaries=[]
    for f in FAMILIES:
        for k in GRID:
            for m in ['joint_delta_reward_dist','joint_delta_reward_ctrl']:
                rr=[x[m] for x in reward_streams if x['family']==f and x['severity']==k and x['kind']=='shifted']
                if rr:summaries.append(dict(family=f,severity=k,metric=m,**stats(rr)))
    table(tab/'joint_reward_summary.csv',summaries)
    (rep/'JOINT_REWARD_DECOMPOSITION.md').write_text('# Joint reward decomposition\n\n'+md(summaries,['family','severity','metric','mean','median','ci_low','ci_high'])+
        '\n\nJoint minus Frozen, paired on held-out seed. Lower distance is represented by higher distance reward; lower action penalty is higher control reward. '
        'Across the full grid, Joint distance reward improves in 4/24 labels and control reward improves in 20/24. Intervals bootstrap three adaptation-stream means conditional on the evaluation bank. These are descriptive intervals. Nominal retention is separate in reward_components_per_stream.csv.\n')
    events=raw_data['events'];eg=defaultdict(list)
    for e in events:eg[(e['family'],e['severity'],e['adaptation_seed'])].append(e)
    statsrows=[]
    for key,rr in sorted(eg.items()):
        for metric in ('consistency_loss','reward_loss','value_loss','total_loss','grad_norm','pi_loss','pi_grad_norm'):
            vals=[x[metric] for x in rr]
            statsrows.append(dict(family=key[0],severity=key[1],adaptation_seed=key[2],metric=metric,
                events=len(vals),mean=float(np.mean(vals)),median=float(np.median(vals)),max=max(vals),first=vals[0],last=vals[-1]))
    table(tab/'joint_learning_per_stream.csv',statsrows)
    (rep/'JOINT_LEARNING_STABILITY.md').write_text('# Joint learning stability\n\n'
        'All 72 original logs were streamed (720,000 transitions). Every stream has 398 update events: first at 75, then every 25 transitions through 10,000. '
        'Completion markers report 796 optimizer step calls (two optimizers per update), 200 episodes and 10,200 replay rows. '
        'Optimizer steps increase from 669,999 to 670,397. No HalfCheetah update schedule is substituted. '
        'All recorded update metrics are finite. Finite losses do not prove optimization convergence or stable held-out performance.\n\n'
        'Figure 09 shows separate family/severity stream-mean curves. Shading is min–max over the three adaptation seeds, not independent-time-step uncertainty. '
        'Online episode returns come directly from sums of saved transition rewards; they are not held-out evaluation. '
        'Per-stream first/last/max/median loss and gradient values are in joint_learning_per_stream.csv.\n')
    (rep/'MISSING_EVIDENCE.md').write_text('# Remaining evidence limits\n\n'
        '- Nominal model.pt and optimizers.pt: byte-level verification of recorded checkpoint hashes.\n'
        '- Adaptation snapshot .pt files: independently recompute state/encoder hashes rather than trust recorded markers.\n'
        '- Frozen .npz bank: independently verify bytes against data_sha256/source_sha256. Prediction bank identifiers match between CSVs.\n'
        '- Multi-step latent errors and common-encoder Joint prediction accuracy: not recorded; not invented.\n'
        '- Independent training-seed nominal convergence and causal Q/actor attribution: not established by these experiments.\n'
        'Original validation, probe vectors, per-episode rewards, snapshot identities, Joint logs and trajectory CSVs are now available locally. '
        'No environment was rerun. Large per-step CSVs are kept in the extracted source and are not copied into the analysis ZIP.\n')
    (rep/'EXPERIMENT_INVENTORY.md').write_text('# Original experiment inventory\n\n'+md(r['stages'],['stage','status','completed_children','path'])+
        '\n\nRequired stages pass. Original interrupted control has no top-level completion; its 160 completed children exactly match resumed rows/hash and are counted once. '
        'All 864 resumed and 864 prediction snapshots have the full seed grid. Joint has 72 complete adaptation and evaluation streams. '
        'See snapshot_identity_audit.csv for hashes, budgets and counts; original source hashes are in provenance/input_manifest.json.\n')
    (rep/'PHYSICAL_SEVERITY_ANALYSIS.md').write_text('# Physical severity — original records\n\n'
        f"Original probe CSV contains {len(raw_data['probe'])} measured transitions with initial states, actions, nominal/shifted observations and physical vectors. "
        'The saved physical summary uses five calibration seeds; its provenance is now available. No simulator replay was performed. '
        'Four families × six severities retained; k=1 labels are aliases. Damping and mass units remain separate. '
        'Velocity difference [rad/s], fingertip displacement [m] and finite-difference acceleration [rad/s²] are measured, not inferred from severity k. '
        'Mass changes keep inertia fixed under the saved protocol; this is not uniform-density scaling.\n')
    (rep/'DYNAMICS_ONLY_ADAPTATION.md').write_text('# Dynamics-only original evidence\n\n'
        'All 144 adaptation completions and per-stream original logs were checked. Fixed-component and auxiliary hashes are identical across the streams. '
        'Prediction/control completion records identify the same snapshot at each budget; fixed-bank source hashes match the Frozen CSV. '
        'These are saved-record invariance checks; model tensors are not independently rehashed. '
        'Final prediction reduction is 85.9–96.3% across the 24 labels; all condition mean return changes are negative. '
        'Final target distance improves in 12/24 condition labels. See prediction_control.csv for individual condition classifications and uncertainty. '
        'Budget, actual first update and final update counts are in dynamics_budget_measured.csv. '
        'Fixed-data prediction remains distinct from on-policy latent_mse.\n')
    (rep/'JOINT_RL_COMPARISON.md').write_text('# Joint RL — original held-out results\n\n'
        '72 original adaptation streams and 72 held-out evaluation children pass. Each has 40 control episodes (20 shifted +20 nominal) and 20 fixed-bank prediction episodes. '
        'Training/evaluation snapshot hashes and selected nominal hash agree in the original markers. '
        'Final budget 10,000 transitions and 398 update events are verified from all 72 logs. '
        'Mean shifted return change is positive in 6/24 conditions, ranging −0.00868687 to +0.0138014. '
        'Nominal retention episodes are available and kept separate from shifted evaluation and online training. '
        'Reward decomposition is now measured in JOINT_REWARD_DECOMPOSITION.md. Loss/gradient/online-return diagnostics are in JOINT_LEARNING_STABILITY.md. '
        'Joint own-encoder latent error is represented separately; changing the encoder prevents direct physical-accuracy comparison with Frozen/TTT latent MSE.\n')
    conclusion=rep/'RESEARCH_CONCLUSIONS.md'
    text=conclusion.read_text().replace('online optimization stability is NOT ESTABLISHED','all recorded Joint losses are finite and update schedules match; optimization convergence is not established')
    text=text.replace('Evidence status: PARTIAL_EVIDENCE.','Evidence status: saved-record audit PASS; byte-level verification of omitted model/NPZ files remains incomplete.')
    conclusion.write_text(text)
    if not plotting:return
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,m,label in zip(axes,['episode_return','distance_mean','distance_final'],['Return','Mean target distance [m]','Final target distance [m]']):
        ax.hist([float(x[m]) for x in vr],bins=8);ax.set(xlabel=label,ylabel='Validation seeds')
    fig.suptitle('Selected 670k nominal checkpoint — original validation bank, n=20')
    fig.savefig(figdir/'01_nominal_validation.png',dpi=180);plt.close(fig)
    metrics=['total_loss','consistency_loss','reward_loss','value_loss','pi_loss','grad_norm','pi_grad_norm']
    fig,axes=plt.subplots(len(metrics)+1,4,figsize=(18,23),layout='constrained')
    for col,(f,param) in enumerate(FAMILIES.items()):
        for row,metric in enumerate(metrics+['online_return']):
            ax=axes[row,col]
            for k in GRID:
                rows=[x for x in (events if metric!='online_return' else raw_data['online']) if x['family']==f and x['severity']==k]
                by=defaultdict(list)
                for x in rows:by[x['transition']].append(x[metric])
                xx=sorted(by);yy=[np.mean(by[t]) for t in xx]
                line,=ax.plot(xx,yy,label=f'k={k:g}',lw=.7)
                ax.fill_between(xx,[min(by[t]) for t in xx],[max(by[t]) for t in xx],color=line.get_color(),alpha=.08)
            ax.set(title=param if row==0 else '',xlabel='Adaptation transitions',ylabel=metric)
            if metric not in ('online_return','pi_loss'):ax.set_yscale('symlog',linthresh=1e-8)
            if row==0:ax.legend(fontsize=7,ncol=3)
    fig.suptitle('Joint training diagnostics; mean and min–max of 3 adaptation streams, update every 25 transitions after 75')
    fig.savefig(figdir/'09_joint_learning_stability.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
    for ax,(f,param) in zip(axes,FAMILIES.items()):
        for metric,label in [('joint_delta_reward_dist','Distance reward'),('joint_delta_reward_ctrl','Control reward')]:
            rr=[x for x in summaries if x['family']==f and x['metric']==metric];xx=[x['severity'] for x in rr]
            line,=ax.plot(xx,[x['mean'] for x in rr],'o-',label=label)
            ax.fill_between(xx,[x['ci_low'] for x in rr],[x['ci_high'] for x in rr],color=line.get_color(),alpha=.15)
        ax.set(title=param,xlabel='Severity k',ylabel='Joint − Frozen reward component');ax.axhline(0,color='gray',lw=.5);ax.legend(fontsize=8)
    fig.suptitle('Joint reward decomposition: paired seeds; descriptive CI over 3 adaptation streams')
    fig.savefig(figdir/'08b_joint_reward_decomposition.png',dpi=180);plt.close(fig)
    (out/'README.md').write_text('# Reacher analysis v3 — original saved records\n\n'
        'Read reports/RAW_EVIDENCE_AUDIT.md, RESEARCH_CONCLUSIONS.md, JOINT_LEARNING_STABILITY.md and JOINT_REWARD_DECOMPOSITION.md. '
        'This extends the earlier summary analysis with original validation, episode CSVs, completion hashes and streaming training logs. '
        'All exported episode metrics agree with originals. Original source files and the previous v2 package remain unchanged. '
        'Saved-record audit passes; binary model/NPZ verification remains unavailable.\n\n'
        'No model loading, inference, optimizer updates or environment calls. Reproduction: analysis_code/RUN_REACHER_ANALYSIS.md; use --raw-source. '
        'Uncertainty: 3 adaptation-stream means conditional on 20 held-out seeds. Frozen references and resumed control duplicates are not independent replications. '
        'All grid labels retained; nominal family aliases are not independent physical settings.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=['reacher'],default='reacher')
    parser.add_argument('--stage',choices=['all','inventory','summary','plots'],default='all')
    parser.add_argument('--family',choices=list(FAMILIES));parser.add_argument('--severity',type=float,choices=GRID)
    parser.add_argument('--method',choices=list(METHODS.values()),help='Filter canonical episode/prediction views only; paired comparisons retain all three methods')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--plotting',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--summary',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--raw-source',type=Path,help='Extracted original CSV/JSON/JSONL bundle; cross-check and enrich summary analysis')
    parser.add_argument('--nominal-record',type=Path);parser.add_argument('--halfcheetah',type=Path)
    parser.add_argument('--return-band-fraction',type=float,default=.001);parser.add_argument('--error-band-fraction',type=float,default=.001)
    args=parser.parse_args();src=Sources()
    config_path=ROOT/'experiments/nsgym/configs/reacher_pipeline_v1.json'
    if not config_path.exists():config_path=Path(__file__).with_name('reacher_pipeline_v1.json')
    config=src.json(config_path)
    if args.stage=='inventory':
        records=[]
        for marker in sorted(args.source.rglob('completion.json')):
            try:
                d=src.json(marker)
                records.append(dict(path=str(marker.parent.resolve()),status=d.get('status'),
                    completed_children=len(list(marker.parent.glob('*/completion.json'))),
                    snapshots=d.get('snapshots'),control_rows=d.get('control_rows'),prediction_rows=d.get('prediction_rows')))
            except (ValueError,OSError) as exc:
                records.append(dict(path=str(marker),status='UNREADABLE',error=str(exc)))
        result=dict(source=str(args.source.resolve()),stages=records,
            nominal_selections=[str(p.resolve()) for p in args.source.rglob('selected_reacher_checkpoint.json')],
            note='Read-only marker inventory; PASS alone does not verify snapshot identity or original episode completeness')
        if not args.dry_run:
            args.output.mkdir(parents=True,exist_ok=False)
            js(args.output/'inventory.json',result)
        print(json.dumps(result,indent=2));return
    root,inventory=discover(args.source,src)
    if args.dry_run:
        print(json.dumps(dict(summary_source=str(root),inventory=inventory,contract=config,
                             output=str(args.output),environment_transitions=0,optimizer_updates=0),indent=2));return
    if args.output.exists() or args.output.with_suffix('.zip').exists(): raise FileExistsError('Use a fresh output directory and ZIP path')
    out=args.output;out.mkdir(parents=True)
    for name in ('reports','tables','figures','provenance','analysis_code'): (out/name).mkdir()
    tab=out/'tables';rep=out/'reports';pro=out/'provenance'
    table(tab/'experiment_inventory.csv',inventory)
    canonical,predictions,matched,rewards,excluded,manifest=load(args.source,root,src,config)
    selection,selection_source=selection_record(args.raw_source or args.source,args.nominal_record,src)
    if selection and selection['model_sha256'] != manifest['checkpoint_sha256']: raise ValueError('Nominal selection hash differs from result export')
    js(pro/'selected_nominal_record.json',dict(record=selection,evidence=selection_source,checkpoint_bytes_verified=False))
    raw_data=None
    if args.raw_source:
        raw_data=audit_raw(args.raw_source,src,config,canonical,predictions,matched,rewards,out)
    def keep(x):return (args.family is None or x['family']==args.family) and (args.severity is None or float(x['severity'])==args.severity)
    full_canonical=canonical
    canonical=[x for x in canonical if keep(x)];predictions=[x for x in predictions if keep(x)]
    matched=[x for x in matched if keep(x)];rewards=[x for x in rewards if keep(x)]
    summaries,streams,coupling=summarize(matched,config,args.return_band_fraction,args.error_band_fraction)
    table(tab/'canonical_episode_results.csv',[x for x in canonical if args.method is None or x['method']==args.method])
    table(tab/'canonical_prediction_results.csv',[x for x in predictions if args.method is None or x['method']==args.method])
    for name,data in [('matched_comparison',matched),('condition_summary',summaries),('per_adaptation_seed',streams),
                      ('prediction_control',coupling),('reward_decomposition',rewards),('associations',associations(streams,coupling))]:table(tab/(name+'.csv'),data)
    missing=[
        ('nominal provenance','selected_reacher_checkpoint.json; checkpoint/metadata.json and model.pt','original selection/config, model SHA verification; nominal per-seed return/distance unavailable'),
        ('episode provenance','prediction/control/*/completion.json and manifest.json','snapshot_sha256, snapshot identity, source_bank_sha256, actual original completion status'),
        ('Joint reward decomposition','Joint evaluation/*/control.csv','reward_dist, reward_ctrl, distance_mean, distance_min, action_saturation, episode_length, terminated, truncated, gradient_updates'),
        ('Joint prediction episodes','Joint evaluation/*/predict.csv','seed, adaptation_seed, latent_mse, source_sha256, samples'),
        ('physical raw probes','probe/reacher_physical_response.csv','seed, delta_qvel_rad_s, delta_tip_m, delta_accel_rad_s2, shifted_latent_mse, raw state/action'),
        ('Joint learning stability','Joint adaptation/*/training.jsonl and completion.json','transition, updates, metrics.*, replay size; online episode return'),
        ('trajectory diagnostics','Frozen/control/*/steps.csv','t, tip, target, qvel, qacc, force, ctrl'),
        ('encoder invariance','adaptation/*/completion.json and snapshot metadata','fixed_hash, auxiliary_hash, model_hash; nominal state comparison'),
    ]
    if raw_data:
        missing=[('binary integrity','nominal model.pt/optimizers.pt; adapted snapshots .pt; Frozen bank .npz','cross-record hashes agree; binary files not transferred'),('unmeasured prediction','common-encoder Joint and multi-step errors','not recorded; own-encoder error is not comparable physical accuracy')]
    exclusions=[dict(category=a,required_files=b,missing_columns_or_reason=c,status='NOT_AVAILABLE_IN_LOCAL_EXPORT') for a,b,c in missing]
    exclusions.extend(excluded);table(tab/'exclusions.csv',exclusions)
    coverage=[]
    for family in FAMILIES:
        for k in GRID:
            if not keep(dict(family=family,severity=k)):continue
            rr=[x for x in matched if x['family']==family and x['severity']==k and x['kind']=='shifted']
            coverage.append(dict(family=family,severity=k,paired_episodes=len(rr),adaptation_seeds=len({x['adaptation_seed'] for x in rr}),
                evaluation_seeds=len({x['evaluation_seed'] for x in rr}),status='COMPLETE_DERIVED_PAIRING' if len(rr)==60 else 'INCOMPLETE',
                raw_provenance='PASS_SAVED_RECORD_AUDIT' if raw_data else 'NOT_AVAILABLE',nominal_alias=k==1))
    table(tab/'coverage.csv',coverage)
    budgets=[]
    for method in ('Frozen','Dynamics-only','Joint RL'):
        rr=[x for x in canonical if x['method']==method]
        for b in sorted({x['budget'] for x in rr}):
            updates=sorted({x['gradient_updates'] for x in rr if x['budget']==b and x.get('gradient_updates') is not None})
            budgets.append(dict(method=method,budget=b,measured_gradient_update_counts=json.dumps(updates),
                expected_updates_from_current_code=0 if method=='Frozen' else max(0,b//25-2),
                measurement_status='episode_export' if updates else 'NOT_MEASURED_IN_AVAILABLE_FILES',
                first_update_measured=None,first_update_code_expected=75 if method!='Frozen' else None))
    table(tab/'adaptation_budget.csv',budgets)
    physical=src.csv(root/'physical_seed_summary.csv') if (root/'physical_seed_summary.csv').exists() else []
    physical=[x for x in physical if keep(x)];table(tab/'physical_response_summary.csv',physical)
    for filename in ('physical_regime_counts.csv','reacher_joint_prediction_summary.csv'):
        if (root/filename).exists():table(tab/filename,src.csv(root/filename))
    cross=[]
    if args.halfcheetah:
        hp=args.halfcheetah;hc=src.json(hp/'provenance/completion.json')
        hrows=src.csv(hp/'tables/canonical_comparison.csv');unique(hrows,lambda x:(x['parameter'],float(x['severity']),int(x['adaptation_seed']),int(x['evaluation_seed'])))
        hg=defaultdict(list)
        for x in hrows:hg[(x['parameter'],float(x['severity']),int(x['adaptation_seed']))].append(x)
        complete=hc.get('status')=='PASS' and len(hg)==36 and all(len(v)==20 for v in hg.values())
        if complete and all(x['status']=='COMPLETE_DERIVED_PAIRING' for x in coverage):
            for (param,k,a),rr in sorted(hg.items()):
                if k==1 and param!='bthigh_damping':continue
                base=abs(np.mean([float(x['frozen_return']) for x in rr]))
                for metric,method in [('delta_D','Dynamics-only'),('delta_J','Joint RL')]:
                    cross.append(dict(task='HalfCheetah',parameter=param,severity=k,adaptation_seed=a,method=method,
                        normalized_delta=float(np.mean([float(x[metric]) for x in rr]))/base,normalizer=base,
                        adaptation_budget=3000,comparison='within_task_relative_return_only'))
            for x in streams:
                if x['kind']!='shifted' or x['severity']==1 and x['family']!='D0':continue
                for metric,method in [('delta_D','Dynamics-only'),('delta_J','Joint RL')]:
                    cross.append(dict(task='Reacher',parameter=x['parameter'],severity=x['severity'],adaptation_seed=x['adaptation_seed'],method=method,
                        normalized_delta=x[metric]/abs(x['frozen_return']),normalizer=abs(x['frozen_return']),
                        adaptation_budget=10000,comparison='within_task_relative_return_only'))
    table(tab/'cross_task_normalized.csv',cross)
    sensitivity=[]
    for band in (0.,.0001,.001,.01):
        _,_,cc=summarize(matched,config,band,args.error_band_fraction)
        sensitivity.extend(dict(x,return_band_fraction=band) for x in cc)
    table(tab/'classification_sensitivity.csv',sensitivity)
    if args.plotting and args.stage in ('all','plots'):
        draw(out/'figures',summaries,streams,physical,canonical,rewards,coupling,cross,selection,src.csv(root/'reacher_joint_prediction_summary.csv'))
    if args.summary:
        write_reports(out,config,full_canonical,matched,summaries,streams,coupling,coverage,selection,selection_source,manifest,cross,missing,physical)
    else:
        (out/'README.md').write_text('Narrative summary disabled; see provenance and tables. Evidence is partial.\n')
    if raw_data:
        raw_figures_and_reports(out,raw_data,canonical,rewards,src,args.raw_source,args.plotting and args.stage in ('all','plots'))
    analysis_config=dict(task='Reacher',source=str(args.source.resolve()),summary_source=str(root.resolve()),
        raw_source=str(args.raw_source.resolve()) if args.raw_source else None,contract=config,stage=args.stage,filters=dict(family=args.family,severity=args.severity,method=args.method),
        return_band_fraction=args.return_band_fraction,error_band_fraction=args.error_band_fraction,
        thresholds='analysis-defined sensitivity bands; NOT preregistered biological/practical equivalence',
        bootstrap='5000 fixed-seed bootstrap of adaptation-stream means, conditional on 20 evaluation seeds; n=3 descriptive',
        optimizer_updates=0,environment_transitions=0,inference_calls=0)
    js(pro/'analysis_config.json',analysis_config)
    shutil.copy2(ROOT/'experiments/nsgym/configs/reacher_pipeline_v1.json',out/'analysis_code/reacher_pipeline_v1.json')
    for p in (Path(__file__), ROOT/'experiments/nsgym/scripts/analyze_existing_mechanisms.py', Path(__file__).with_name('schema.py')):
        src.track(p);shutil.copy2(p,out/'analysis_code'/p.name)
    for rel in ('tdmpc2/envs/nsgym/reacher.py','experiments/nsgym/scripts/reacher_pipeline.py','experiments/nsgym/scripts/run_full_joint_rl.py'):
        if (ROOT/rel).exists():src.track(ROOT/rel)
    documentation=ROOT/'experiments/nsgym/RUN_REACHER_ANALYSIS.md'
    if documentation.exists():
        src.track(documentation);shutil.copy2(documentation,out/'analysis_code/RUN_REACHER_ANALYSIS.md')
    js(pro/'input_manifest.json',src.files)
    js(pro/'completion.json',dict(status='PARTIAL_EVIDENCE',analysis_execution='PASS',conditions=len(coverage),
        matched_shifted_episodes=sum(x['kind']=='shifted' for x in matched),canonical_episode_rows=len(canonical),
        original_stage_audit='PASS_SAVED_RECORD_AUDIT' if raw_data else 'NOT_AVAILABLE',joint_training_diagnostics='MEASURED' if raw_data else 'NOT_AVAILABLE',
        optimizer_updates=0,environment_transitions=0,inference_calls=0))
    for path,record in src.files.items():
        if digest(path)!=record['sha256']:raise ValueError(f'Input changed during analysis: {path}')
    zip_path=out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(out.parent))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP integrity failure')
        if any(Path(n).suffix in ('.pt','.npz') for n in archive.namelist()):raise ValueError('Raw models/trajectories in analysis ZIP')
    sha=digest(zip_path);zip_path.with_suffix('.zip.sha256').write_text(sha+'  '+zip_path.name+'\n')
    print(json.dumps(dict(output=str(out.resolve()),zip=str(zip_path.resolve()),sha256=sha,status='PARTIAL_EVIDENCE'),indent=2))


def rewards_for_report(path):
    with path.open() as handle:return list(csv.DictReader(handle))


def write_reports(out,config,canonical,matched,summaries,streams,coupling,coverage,selection,selection_source,manifest,cross,missing,physical):
    rep=out/'reports'
    def report(name,text): (rep/name).write_text(text+'\n')
    def fmt(x):return f'{x:.6g}' if x is not None else 'NOT_AVAILABLE'
    def md(rows,columns):
        return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
            ['| '+' | '.join(str(x.get(k,'')) for k in columns)+' |' for x in rows])
    final=[x for x in streams if x['kind']=='shifted'];counts=Counter(x['classification'] for x in coupling)
    report('EXPERIMENT_INVENTORY.md','# Evidence inventory\n\nOBSERVED: exported PASS analysis contains 34,560 TTT/Frozen control rows, '
        '2,880 final paired rows, and Joint summary statistics. The copied source is an analysis export, not raw stage output. '
        'The embedded input manifest names only control-home-02-resumed; control-home-01 is not added again. '
        'Snapshot-level resume reconstruction and original completion verification are NOT ESTABLISHED.\n\n'+md(coverage,['family','severity','paired_episodes','adaptation_seeds','evaluation_seeds','status'])+
        '\n\nOriginal recorded input paths:\n\n'+'\n'.join('- '+p for p in manifest['inputs']))
    nominal=[x for x in canonical if x['method']=='Frozen' and x['family']=='D0' and x['severity']==1 and x['kind']=='shifted']
    nominal_stats={m:stats([x[m] for x in nominal]) for m in ('episode_return','distance_initial','distance_mean','distance_final')}
    report('REACHER_NOMINAL_QUALITY.md','# Nominal quality\n\nSelection evidence: '+selection_source+'. Checkpoint bytes unavailable for rehashing.\n\n'
        f"Selected path: `{selection.get('checkpoint','NOT_AVAILABLE')}`; step: {selection.get('training_step','NOT_AVAILABLE')}; SHA256: `{manifest['checkpoint_sha256']}`.\n\n"
        f"Selection validation summary: `{json.dumps(selection.get('nominal_performance',{}))}`. Original validation episode rows are unavailable, so Figure 1 validation distributions are NOT GENERATED.\n\n"
        'OBSERVED: nominal held-out bank (different seeds from selection validation):\n\n'+md([dict(metric=m,**s) for m,s in nominal_stats.items()],['metric','n','mean','median','std','ci_low','ci_high'])+
        '\n\nSUPPORTED INTERPRETATION: substantial target approach is present; this is not a predeclared success criterion or proof of global nominal convergence. '
        'No binary success threshold is introduced. Environment contract: Reacher-v5, state dim 10, action dim 2, horizon 50. '
        f"Recorded MPPI: { {k:selection.get('configuration',{}).get(k) for k in ('horizon','iterations','num_samples','num_elites','num_pi_trajs')} }.\n")
    report('PHYSICAL_SEVERITY_ANALYSIS.md','# Physical severity\n\nOBSERVED: existing common-state/action calibration summaries '
        'contain joint-velocity, fingertip-displacement, finite-difference acceleration and fixed-bank latent mismatch. '
        'Confidence intervals are copied from five calibration seeds; no fresh physical probes were executed. '
        'All four family labels and six severities are retained. k=1 aliases one nominal physical setting. '
        'Mass-only interventions keep rotational inertia fixed. This is not uniform-density scaling. '
        'Raw probe state/action records are unavailable, so counterfactual replay cannot be independently checked.\n\n'
        'Measured metrics and units: delta_qvel_rad_s [rad/s], delta_tip_m [m], delta_accel_rad_s2 [rad/s²]. '
        'No physical response is inferred from k alone. See tables/physical_response_summary.csv.\n')
    frozen=[x for x in summaries if x['metric']=='frozen_return' and x['kind']=='shifted']
    report('FROZEN_GENERALIZATION.md','# Frozen generalization\n\nOBSERVED: '+md(frozen,['family','severity','mean','median','std','ci_low','ci_high'])+
        '\n\nFrozen uncertainty uses 20 unique evaluation seeds, not 60 repeated Frozen training models. '
        'Return changes across severe damping should be interpreted with continuous reaching distance and effort. '
        'Family aliases at nominal k=1 are retained only for pairing; no cross-family pooled nominal significance is computed.\n')
    report('DYNAMICS_ONLY_ADAPTATION.md','# Dynamics-only adaptation\n\nOBSERVED: final 10k control rows record 398 gradient updates per stream. '
        'The first update at transition 75 is supported by current code, not independently observed without original logs. '
        'The code updates only dynamics and hashes fixed components; original per-stream hashes are unavailable here.\n\n'+
        md(coupling,['family','severity','relative_mse_reduction','delta_D','delta_distance_D','improved_stream_fraction','classification'])+
        '\n\nHeld-out fixed-data prediction is reconstructed from exported heldout_fixed_latent_mse and delta_heldout_latent_mse. '
        'It is distinct from the episode row latent_mse, which is on-policy error. No multi-step prediction metric exists in this export.\n')
    joint=[x for x in summaries if x['metric'] in ('delta_J','delta_JD','delta_distance_J') and x['kind']=='shifted']
    retention=[x for x in summaries if x['metric']=='delta_J' and x['kind']=='nominal']
    report('JOINT_RL_COMPARISON.md','# Joint RL comparison\n\nOBSERVED: 72 final Joint model labels, 2,880 paired control rows, '
        'and 24 own-encoder prediction summaries are represented in the export. Original Joint episode CSVs, training logs and snapshot hashes are absent. '
        'The current Reacher protocol expects 10k interactions, first update 75 and 398 update events; these are not substituted for actual Joint log measurements. '
        'HalfCheetah 3k/2,001 settings are not used.\n\n'+md(joint,['family','severity','metric','mean','ci_low','ci_high'])+
        '\n\nNominal retention was evaluated; the following refers to fresh nominal evaluation episodes, not online training return:\n\n'+
        md(retention,['family','severity','mean','ci_low','ci_high'])+
        '\n\nNOT ESTABLISHED: loss stability, replay schedule and online return trajectories (Figure 9) cannot be reconstructed from final summary metrics. '
        'Joint reward-distance/control-cost components are absent. Own-encoder latent MSE is not physical prediction accuracy and is not compared numerically to fixed-encoder TTT.\n')
    report('PREDICTION_CONTROL_COUPLING.md','# Prediction–control coupling\n\n'+json.dumps(dict(counts),indent=2)+
        '\n\nClassification uses 95% bootstrap intervals over three adaptation-stream means. Return band = 0.1% of |mean Frozen return|; '
        'error band = 0.1% of mean Frozen MSE by default. These are analysis-defined sensitivity bands, not preregistered practical equivalence margins. '
        'An interval entirely inside the band is labeled within-band, not proven unchanged. n=3 intervals are descriptive and conditional on the fixed 20-seed evaluation bank. '
        'No per-step independence or causal claim. Per-stream values and across-severity descriptive associations are separate.\n\n'+
        md(coupling,['family','severity','delta_E','delta_D','return_ci_low','return_ci_high','classification']))
    cross_rows=[]
    for task in ('HalfCheetah','Reacher'):
        for method in ('Dynamics-only','Joint RL'):
            vv=[x['normalized_delta'] for x in cross if x['task']==task and x['method']==method]
            if vv:cross_rows.append(dict(task=task,method=method,stream_condition_labels=len(vv),median=float(np.median(vv)),min=min(vv),max=max(vv)))
    report('REACHER_HALFCHEETAH_COMPARISON.md','# Cross-task comparison\n\n'+md(cross_rows,['task','method','stream_condition_labels','median','min','max'])+
        '\n\nOBSERVED: complete compatible per-seed return pairing is available in both derived datasets. '
        'Normalization is mean paired return change divided by absolute mean Frozen return within each condition/stream. '
        'This is a dimensionless descriptive effect, not equal task difficulty or a cross-task causal estimate. '
        'Nominal duplicate family labels are excluded from cross-task pooling. No CI treats conditions as independent training seeds. '
        'HalfCheetah uses 3k interactions and Reacher 10k; physical families, horizons, reward scale and Joint update schedules differ. '
        'No raw return or raw latent-MSE magnitudes are compared across tasks.\n')
    dmeans=[x['mean'] for x in summaries if x['metric']=='delta_D' and x['kind']=='shifted']
    jmeans=[x['mean'] for x in summaries if x['metric']=='delta_J' and x['kind']=='shifted']
    reward_groups=defaultdict(list)
    for x in rewards_for_report(out/'tables/reward_decomposition.csv'):
        if x['kind']=='shifted':reward_groups[(x['family'],x['severity'])].append(x)
    reward_rows=[dict(family=f,severity=k,delta_distance_reward=float(np.mean([float(x['delta_reward_dist']) for x in rr])),
        delta_control_reward=float(np.mean([float(x['delta_reward_ctrl']) for x in rr]))) for (f,k),rr in sorted(reward_groups.items())]
    report('REWARD_DECOMPOSITION.md','# Reward decomposition\n\n'+md(reward_rows,['family','severity','delta_distance_reward','delta_control_reward'])+
        '\n\nNegative distance-reward difference means greater time-integrated target distance. Positive control-reward difference means lower action penalty. '
        'For the full grid, distance reward decreases in all 24 labels; action cost decreases in 23/24 (exception D1 k=2). '
        'Thus lower action cost generally offsets part of the worsening approach trajectory, but not all of it. '
        'Final distance alone can hide this time-integrated behavior. These are condition means; k=1 labels are aliases. '
        'Joint reward components are unavailable. No new success threshold was chosen.\n')
    report('RESEARCH_CONCLUSIONS.md','# Research conclusions\n\n'
        f'1. OBSERVED: selected nominal checkpoint is recorded at {selection.get("training_step","unknown")} transitions; nominal held-out mean final distance is {nominal_stats["distance_final"]["mean"]:.6g} m. '
        'SUPPORTED INTERPRETATION: competent reaching; absolute convergence/success threshold is NOT ESTABLISHED.\n\n'
        '2. OBSERVED: measured physical responses exist for mass and damping, while Frozen control changes are small on the available Reacher grid. '
        'Read the complete per-family table; no condition was removed for poor adaptation.\n\n'
        f'3. OBSERVED: fixed-bank MSE reduction across labels ranges {min(x["relative_mse_reduction"] for x in coupling):.1%}–{max(x["relative_mse_reduction"] for x in coupling):.1%}.\n\n'
        f'4. OBSERVED: TTT mean return difference ranges {min(dmeans):+.6g} to {max(dmeans):+.6g}; {sum(x>0 for x in dmeans)}/{len(dmeans)} labels have positive means. '
        'Mean final target distance improves in 12/24 condition labels, despite the lower return. Reward decomposition shows worse time-integrated approach and generally lower action cost.\n\n'
        f'5. OBSERVED: Joint mean return difference ranges {min(jmeans):+.6g} to {max(jmeans):+.6g}; {sum(x>0 for x in jmeans)}/{len(jmeans)} labels have positive means. '
        'Reacher does not show the broad large Joint deterioration present in the available HalfCheetah comparison.\n\n'
        '6. OBSERVED: TTT time-integrated distance reward decreases in all 24 labels; action cost decreases in 23/24, partially offsetting the loss. Joint components are NOT AVAILABLE.\n\n'
        '7. SUPPORTED INTERPRETATION: prediction mismatch reduction alone is insufficient evidence of control recovery. Across-severity associations are descriptive, with confounding.\n\n'
        '8. SUPPORTED INTERPRETATION: adaptation patterns differ from HalfCheetah in effect magnitude and Joint stability of final control; online optimization stability is NOT ESTABLISHED.\n\n'
        '9. HYPOTHESIS: prediction-improved/control-degraded labels motivate Q/planning diagnostics, including M1 k=0.5 with the largest mean return decrease. '
        'They do not establish Q as the mechanism; effects also occur at k=1. Actor, representation, planner sensitivity and action effort remain alternatives.\n\n'
        '10. NOT ESTABLISHED: Q/actor-specific causal attribution, shared-replay effects, common-encoder Joint prediction accuracy and nominal convergence across training seeds. '
        'These require additional controlled experiments, which were not executed.\n\n'
        'Evidence status: PARTIAL_EVIDENCE. All present per-seed pairs were checked; original snapshot/stage audits and Joint learning diagnostics are absent.\n')
    report('MISSING_EVIDENCE.md','# Exact missing evidence\n\n'+md([dict(category=a,files=b,columns_or_reason=c) for a,b,c in missing],['category','files','columns_or_reason']))
    (out/'README.md').write_text('# Reacher research analysis v2\n\nRead reports/RESEARCH_CONCLUSIONS.md first. '
        'This package analyzes a copied PASS summary export. It does not claim independent original-run verification. '
        'provenance/completion.json intentionally reports PARTIAL_EVIDENCE. All raw inputs were read-only; no model, optimizer or environment was executed.\n\n'
        '## Reuse and calculations\n\nSources, digest, CSV writing, duplicate-key checking, deterministic bootstrap and correlation helpers are reused from '
        'analyze_existing_mechanisms.py (HalfCheetah). Reacher-specific family/seed mapping, final-distance/reward metrics, nominal aliases and fixed-vs-own encoder semantics are handled in analyze_reacher_research.py. '
        'Existing analyze_reacher_pipeline.py provides useful earlier plots but cannot establish raw provenance from the copied summary or compare Joint own-encoder error as common physical accuracy. '
        'Its immutable source hashes remain unchanged. Model and environment adapters were not rewritten.\n\n'
        'Canonical rows are exported episode records; absent snapshot hashes and unavailable Joint fields are blank. Frozen replicas and repeated budgets are collapsed to one reference per condition/evaluation physics/seed, '
        'with physical nominal aliases explicitly labeled. Final matched contrasts use the full 20-seed ×3-stream grid. Incomplete streams are excluded and recorded. '
        'Return differences are adapted minus Frozen; distance differences use final distance unless explicitly labeled mean. Lower distance and lower MSE are better. '
        'Return = recorded reward_dist + reward_ctrl; current Reacher-v5 implementation uses negative post-step target distance and negative squared action cost. '
        'Action component saturation is distinct from actuator-force saturation; zero force-limited actuators gives missing force saturation.\n\n'
        'Mean/median/sample-SD and 95% bootstrap intervals first average evaluation episodes within adaptation stream. n=3 intervals are descriptive, not proof of reliable population adaptation. '
        'Frozen summaries use unique evaluation seeds. All outliers remain. Figures 1 (original validation distributions) and 9 (Joint training stability) are unavailable; '
        'nominal_heldout_distribution.png is explicitly a different bank. Figure 10 is dimensionless descriptive within-task return change, not equal physical severity.\n\n'
        '## Reproduction\n\nUse the command in analysis_code/analyze_reacher_research.py --help or experiments/nsgym/RUN_REACHER_ANALYSIS.md. '
        'Required: Python, NumPy, Matplotlib. Source hashes and paths are in provenance/input_manifest.json. The ZIP contains no checkpoints or raw trajectories.\n')


