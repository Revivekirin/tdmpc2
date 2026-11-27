"""Regenerate measured summaries/figures from saved raw data; never invent data."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from severity_runtime import csv_write, js


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def group(rows, extra=()):
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ('family','parameter','severity','schedule',*extra))].append(r)
    return groups


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),std=float(a.std(ddof=1)) if len(a)>1 else 0.)


def bootstrap(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(90210)
    return np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()


def aggregate(rows,metrics,extra=()):
    result=[]
    for key,items in group(rows,extra).items():
        r=dict(zip(('family','parameter','severity','schedule',*extra),key))
        r.update(observations=len(items),seeds=len({x['seed'] for x in items}))
        for metric in metrics:
            for name,val in stats([float(x[metric]) for x in items]).items():r[f'{metric}_{name}']=val
        result.append(r)
    return result


def curve(out,rows,metrics,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(metrics),1,figsize=(9,3.2*len(metrics)),squeeze=False)
    lines=defaultdict(list)
    for r in rows:lines[(r['parameter'],r['schedule'])].append(r)
    for ax,(metric,label) in zip(axes.flat,metrics):
        for key,items in lines.items():
            items=sorted(items,key=lambda r:float(r['severity']));x=[float(r['severity']) for r in items]
            ax.errorbar(x,[r[metric+'_mean'] for r in items],yerr=[r[metric+'_std'] for r in items],
                        marker='o',capsize=3,label='/'.join(key))
        ax.set(xlabel='Parameter multiplier (not equal difficulty across families)',ylabel=label)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Mean ± descriptive SD; episode seeds are the sampling unit for return')
    fig.tight_layout();fig.savefig(out/name,dpi=150);plt.close(fig)


def summarize(c,out):
    source=Path(c['input']);manifest=json.loads((source/'manifest.json').read_text())
    outputs=[]
    if (source/'physical_response_pairs.csv').exists():
        rows=read(source/'physical_response_pairs.csv')
        metrics=['joint_position_l2','joint_velocity_l2','root_translation_l2','root_velocity_l2',
                 'root_acceleration_difference','mixed_state_l2','contact_count_difference','contact_force_difference']
        table=aggregate(rows,metrics)
        csv_write(out/'physical_response_by_severity.csv',table)
        curve(out,table,[('joint_velocity_l2','Joint velocity difference (rad/s)'),
                         ('root_velocity_l2','Root linear velocity difference (m/s)'),
                         ('root_acceleration_difference','Root acceleration difference (m/s²)'),
                         ('contact_force_difference','Contact force norm difference (N)')],'physical_response_vs_severity.png')
        outputs+=['physical_response_by_severity.csv','physical_response_vs_severity.png']
        validity=[]
        for key,items in group(rows).items():
            r=dict(zip(('family','parameter','severity','schedule'),key))
            values=[float(x['numerically_valid']=='True') for x in items]
            r.update(numerical_valid_fraction_mean=float(np.mean(values)),numerical_valid_fraction_std=float(np.std(values)))
            validity.append(r)
        csv_write(out/'numerical_validity_by_severity.csv',validity)
        curve(out,table,[('joint_position_l2','Joint position difference (rad)'),
                         ('mixed_state_l2','Mixed-coordinate state L2 (descriptive only)')],
              'physical_state_difference_vs_severity.png')
        curve(out,validity,[('numerical_valid_fraction','Observed numerical validity fraction')],
              'numerical_validity_vs_severity.png')
        outputs+=['numerical_validity_by_severity.csv','physical_state_difference_vs_severity.png','numerical_validity_vs_severity.png']
    if (source/'latent_errors.csv').exists():
        rows=read(source/'latent_errors.csv')
        table=aggregate(rows,['latent_mse','reward_squared_error'],extra=('horizon',))
        nominal={(r['parameter'],r['schedule'],r['horizon']):r['latent_mse_mean'] for r in table if float(r['severity'])==1}
        for r in table:
            baseline=nominal.get((r['parameter'],r['schedule'],r['horizon']))
            r['latent_error_ratio_to_nominal']=r['latent_mse_mean']/baseline if baseline and baseline>1e-12 else None
        csv_write(out/'latent_error_by_severity.csv',table)
        csv_write(out/'latent_error_by_regime.csv',aggregate(rows,['latent_mse','reward_squared_error'],
                   extra=('horizon','action_regime','velocity_regime')))
        curve(out,[r for r in table if int(r['horizon'])==1],
              [('latent_mse','One-step latent MSE'),('reward_squared_error','Reward squared error')],'latent_error_vs_severity.png')
        outputs+=['latent_error_by_severity.csv','latent_error_by_regime.csv','latent_error_vs_severity.png']
        # For nonstationary data, preserve onset-aligned errors separately.
        aligned=aggregate(rows,['latent_mse'],extra=('horizon','aligned_step'))
        csv_write(out/'latent_error_aligned.csv',aligned)
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5))
        for key,items in group([r for r in aligned if int(r['horizon'])==1]).items():
            items=sorted(items,key=lambda r:int(r['aligned_step']))
            ax.plot([int(r['aligned_step']) for r in items],[r['latent_mse_mean'] for r in items],label=f'{key[1]} k={key[2]}')
        ax.axvline(0,color='black',ls='--',alpha=.3);ax.legend(fontsize=7,ncol=2)
        ax.set(xlabel='Step relative to shift onset (nominal: reference onset)',ylabel='One-step latent MSE')
        fig.tight_layout();fig.savefig(out/'latent_error_aligned.png',dpi=150);plt.close(fig)
        outputs+=['latent_error_aligned.csv','latent_error_aligned.png']
    if (source/'episodes.csv').exists():
        rows=read(source/'episodes.csv')
        metrics=['episode_return','return_per_step','forward_velocity','control_cost','episode_length',
                 'action_abs_mean','action_rms','action_saturation','action_smoothness_rms','exposure_fraction','reward_time_integral']
        table=aggregate(rows,metrics)
        baselines={(r['parameter'],r['seed']):float(r['episode_return']) for r in rows if float(r['severity'])==1}
        for r in table:
            items=[x for x in rows if all(str(x[k])==str(r[k]) for k in ('parameter','severity','schedule'))]
            values=[float(x['episode_return']) for x in items]
            r['return_ci95_low'],r['return_ci95_high']=bootstrap(values)
            diffs=[float(x['episode_return'])-baselines[(x['parameter'],x['seed'])] for x in items if (x['parameter'],x['seed']) in baselines]
            r['paired_return_difference']=float(np.mean(diffs)) if len(diffs)==len(items) else None
            ci=bootstrap(diffs) if len(diffs)==len(items) else [None,None]
            r['paired_difference_ci95_low'],r['paired_difference_ci95_high']=ci
            r['terminated_count']=sum(x['terminated']=='True' for x in items)
            r['truncated_count']=sum(x['truncated']=='True' for x in items)
        name='nonstationary_evaluation.csv' if manifest['stage']=='nonstationary' else 'frozen_generalization.csv'
        csv_write(out/name,table)
        csv_write(out/'action_diagnostics_by_severity.csv',aggregate(rows,
                  ['action_abs_mean','action_rms','action_saturation','action_smoothness_rms']+
                  [f'action_{i}_{s}' for i in range(6) for s in ('mean','std')]))
        curve(out,table,[('episode_return','Episode return'),('forward_velocity','Forward velocity (m/s)'),
                        ('control_cost','Episode control cost')],'frozen_return_vs_severity.png')
        curve(out,table,[('action_abs_mean','Mean |action|'),('action_saturation','Fraction |action| ≥ .99')],
              'action_statistics_vs_severity.png')
        step_rows=[]
        for entry in json.loads((source/'dataset_index.json').read_text()):step_rows+=read(source/entry['steps'])
        aligned=aggregate(step_rows,['reward','forward_velocity','control_cost','action_abs_mean','action_change_rms'],extra=('aligned_step',))
        csv_write(out/'shift_aligned_timeseries.csv',aligned)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5))
        for r in table:
            ax.scatter(r['control_cost_mean']/r['episode_length_mean'],r['forward_velocity_mean'])
            ax.annotate(f'{r["parameter"]} k={r["severity"]}',
                        (r['control_cost_mean']/r['episode_length_mean'],r['forward_velocity_mean']),fontsize=7)
        ax.set(xlabel='Mean control cost per step',ylabel='Forward velocity (m/s)')
        fig.tight_layout();fig.savefig(out/'velocity_vs_control_cost.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for key,items in group(aligned).items():
            items=sorted(items,key=lambda r:int(r['aligned_step']))
            for ax,metric in zip(axes,['forward_velocity','reward','action_change_rms']):
                ax.plot([int(r['aligned_step']) for r in items],[r[metric+'_mean'] for r in items],label=f'{key[1]} k={key[2]}')
                ax.set_ylabel(metric);ax.axvline(0,color='black',ls='--',alpha=.3)
        axes[0].legend(fontsize=7,ncol=2);axes[-1].set_xlabel('Episode step minus scheduled onset')
        fig.tight_layout();fig.savefig(out/'shift_aligned_timeseries.png',dpi=150);plt.close(fig)
        outputs += [name,'action_diagnostics_by_severity.csv','frozen_return_vs_severity.png',
                    'action_statistics_vs_severity.png','velocity_vs_control_cost.png','shift_aligned_timeseries.csv','shift_aligned_timeseries.png']
        physical_ref=c.get('physical_reference')
        if physical_ref:
            response=aggregate(read(Path(physical_ref)/'physical_response_pairs.csv'),['joint_velocity_l2'])
            fig,ax=plt.subplots(figsize=(7,5))
            for r in table:
                matches=[p for p in response if p['parameter']==r['parameter'] and float(p['severity'])==float(r['severity'])]
                if matches:
                    x=matches[0]['joint_velocity_l2_mean'];y=r['episode_return_mean']
                    ax.scatter(x,y);ax.annotate(f'{r["parameter"]} {r["severity"]}',(x,y),fontsize=7)
            ax.set(xlabel='Controlled joint velocity response (rad/s)',ylabel='Closed-loop mean return')
            fig.tight_layout();fig.savefig(out/'physical_response_vs_control.png',dpi=150);plt.close(fig)
            outputs.append('physical_response_vs_control.png')
    if not outputs:raise ValueError('Input has no supported measured data')
    js(out/'analysis_scope.json',dict(input=str(source),outputs=outputs,
        uncertainty='Return CIs resample episode seeds. Probe/latent SD is descriptive; windows within seed are correlated.',
        limitations='One training seed; no adaptation inference. Different checkpoint ages are not independent training seeds.'))
    return dict(transitions=0,outputs=outputs)
