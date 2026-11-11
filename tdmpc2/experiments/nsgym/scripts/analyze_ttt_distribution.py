#!/usr/bin/env python3
"""Analyze saved final-snapshot TTT CSVs; never load a model or run physics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

EPS = 1e-10
METHODS = ('Frozen', 'Dynamics-only', 'Q-only', 'Dynamics+Q',
           'Encoder+Dynamics', 'Full Joint RL')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as handle:
        yield from csv.DictReader(handle)


def write(path, records, fields=None):
    records = list(records)
    fields = fields or list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def number(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Nonfinite {name}: {value}')
    return result


def key(row):
    return row['parameter'], float(row['severity']), int(row['adaptation_seed'])


def quantiles(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {name: None for name in ('mean', 'median', 'std', 'q05', 'q25', 'q50', 'q75', 'q95')}
    q = np.quantile(a, [.05, .25, .5, .75, .95])
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std(ddof=1)) if len(a) > 1 else None,
                **dict(zip(('q05', 'q25', 'q50', 'q75', 'q95'), map(float, q))))


def stream_ci(values):
    import numpy as np
    if len(values) < 2:
        return None, None
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261111)
    samples = a[rng.integers(len(a), size=(10000, len(a)))].mean(axis=1)
    return tuple(map(float, np.quantile(samples, [.025, .975])))


def stage(root, mode, expected, audit, source_index, frozen_episodes):
    root = Path(root)
    top = read(root / 'completion.json') if (root / 'completion.json').is_file() else {}
    manifest = read(root / 'manifest.json') if (root / 'manifest.json').is_file() else {}
    if manifest and (manifest.get('mode') != mode or manifest.get('checkpoint_sha256') != audit['checkpoint_sha256'] or
                     manifest.get('audit_sha256') != digest(Path(audit['_path']))):
        raise ValueError(f'{mode} stage manifest is incompatible with PASS audit')
    output = {}
    status = {}
    for identity in expected:
        parameter, severity, seed = identity
        name = f'{parameter}-k{severity:g}-decision-seed{seed}'
        child = root / name
        done = child / 'completion.json'
        if not child.exists():
            status[identity] = 'MISSING'
            continue
        if not done.is_file() or read(done).get('status') != 'PASS':
            status[identity] = 'PARTIAL'
            continue
        try:
            meta = read(child / 'manifest.json')
            ident = meta['identity']
            if (ident['parameter'], float(ident['severity']), int(ident['adaptation_seed'])) != identity:
                raise ValueError('identity mismatch')
            if meta['mode'] != mode or meta['checkpoint_sha256'] != audit['checkpoint_sha256']:
                raise ValueError('checkpoint/mode mismatch')
            if meta['snapshot_sha256'] != expected[identity]:
                raise ValueError('snapshot hash mismatch with audit')
            data = list(rows(child / f'{mode}.csv'))
            if len(data) != len(audit['configuration']['evaluation_seeds']):
                raise ValueError('evaluation seed count mismatch')
            seen = set()
            for row in data:
                evaluation_seed = int(row['evaluation_seed'])
                if key(row) != identity or evaluation_seed in seen or evaluation_seed not in audit['configuration']['evaluation_seeds']:
                    raise ValueError('row identity/seed mismatch')
                seen.add(evaluation_seed)
                base_key = parameter, severity, evaluation_seed
                if mode == 'prediction':
                    if row['bank_file'] != frozen_episodes[base_key]['data_file'] or row['bank_sha256'] != source_index[row['bank_file']]:
                        raise ValueError('prediction bank not paired with Frozen source')
                    if int(row['samples']) != audit['configuration']['horizon']:
                        raise ValueError('prediction sample count differs from horizon')
                    for field in ('frozen_latent_mse', 'decision_latent_mse', 'delta_latent_mse'):
                        number(row[field], field)
                    if not math.isclose(number(row['delta_latent_mse'], 'delta_latent_mse'),
                                        number(row['decision_latent_mse'], 'decision_latent_mse') - number(row['frozen_latent_mse'], 'frozen_latent_mse'),
                                        rel_tol=1e-6, abs_tol=1e-10):
                        raise ValueError('prediction delta mismatch')
                else:
                    if int(row['episode_length']) != audit['configuration']['horizon'] or row['terminated'] != 'False' or row['truncated'] != 'True':
                        raise ValueError('incomplete episode')
                    for field in ('actual_parameter_min','actual_parameter_max'):
                        if not math.isclose(number(row[field],field),
                                            number(frozen_episodes[base_key][field],field),
                                            rel_tol=1e-5,abs_tol=1e-5):
                            raise ValueError('applied physical parameter differs from Frozen reference')
                    if not math.isclose(number(row['frozen_return'], 'frozen_return'),
                                        number(frozen_episodes[base_key]['episode_return'], 'episode_return'),
                                        rel_tol=1e-6, abs_tol=1e-5):
                        raise ValueError('Frozen return does not match paired seed')
                    for field in ('decision_return', 'paired_return_difference', 'decision_velocity', 'decision_control_cost'):
                        number(row[field], field)
                    if not math.isclose(number(row['paired_return_difference'], 'paired_return_difference'),
                                        number(row['decision_return'], 'decision_return') - number(row['frozen_return'], 'frozen_return'),
                                        rel_tol=1e-6, abs_tol=1e-5):
                        raise ValueError('return delta mismatch')
            output[identity] = dict(meta=meta, rows=data)
            status[identity] = 'COMPLETE'
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            status[identity] = 'INCOMPATIBLE'
            output[identity] = dict(error=str(exc))
    return output, status, top.get('status', 'IN_PROGRESS')


def summarize_streams(records, metric):
    by_condition = {}
    for row in records:
        by_condition.setdefault((row['parameter'], float(row['severity'])), []).append(row)
    result = []
    for (parameter, severity), group in sorted(by_condition.items()):
        values = [float(row[metric]) for row in group]
        low, high = stream_ci(values)
        result.append(dict(parameter=parameter, severity=severity, n_adaptation_streams=len(group),
                           **quantiles(values), ci_low=low, ci_high=high,
                           **{f'seed_{row["adaptation_seed"]}': row[metric] for row in group}))
    return result


def prediction_tables(stage_data, c):
    stream, per_seed = [], []
    for (parameter, severity, seed), item in sorted(stage_data.items()):
        if 'rows' not in item:
            continue
        rr = item['rows']
        frozen = [number(r['frozen_latent_mse'], 'frozen_latent_mse') for r in rr]
        adapted = [number(r['decision_latent_mse'], 'decision_latent_mse') for r in rr]
        gain = [a - b for a, b in zip(frozen, adapted)]
        fs, ts, gs = quantiles(frozen), quantiles(adapted), quantiles(gain)
        relative = (fs['mean'] - ts['mean']) / max(fs['mean'], EPS)
        stream.append(dict(parameter=parameter, severity=severity, adaptation_seed=seed,
            n_evaluation_seeds=len(rr), frozen_mean_mse=fs['mean'], ttt_mean_mse=ts['mean'],
            delta_mse=ts['mean']-fs['mean'], mse_improvement=gs['mean'], relative_mse_reduction=relative,
            relative_denominator_near_zero=fs['mean'] <= EPS,
            fraction_eval_seed_means_improved=sum(x > 0 for x in gain)/len(gain),
            frozen_median=fs['median'], ttt_median=ts['median'],
            frozen_std=fs['std'], ttt_std=ts['std'],
            **{f'frozen_{k}':fs[k] for k in ('q05','q25','q50','q75','q95')},
            **{f'ttt_{k}':ts[k] for k in ('q05','q25','q50','q75','q95')}))
        for r, f, ttt, g in zip(rr, frozen, adapted, gain):
            per_seed.append(dict(parameter=parameter, severity=severity, adaptation_seed=seed,
                evaluation_seed=int(r['evaluation_seed']), bank_file=r['bank_file'], bank_sha256=r['bank_sha256'],
                samples=int(r['samples']), frozen_mean_mse=f, ttt_mean_mse=ttt,
                mse_improvement=g, relative_mse_reduction=g/max(f,EPS),
                relative_denominator_near_zero=f <= EPS))
    distribution = summarize_streams(stream, 'mse_improvement')
    return stream, distribution, per_seed


def control_tables(stage_data):
    stream, per_seed = [], []
    for (parameter, severity, seed), item in sorted(stage_data.items()):
        if 'rows' not in item:
            continue
        rr = item['rows']
        frozen = [number(r['frozen_return'], 'frozen_return') for r in rr]
        adapted = [number(r['decision_return'], 'decision_return') for r in rr]
        gain = [b-a for a,b in zip(frozen,adapted)]
        fs,ts,gs = quantiles(frozen),quantiles(adapted),quantiles(gain)
        stream.append(dict(parameter=parameter,severity=severity,adaptation_seed=seed,
            n_evaluation_seeds=len(rr),frozen_mean_return=fs['mean'],ttt_mean_return=ts['mean'],
            delta_return=gs['mean'],return_percent_difference=100*gs['mean']/max(abs(fs['mean']),EPS),
            percent_denominator_near_zero=abs(fs['mean'])<=EPS,
            fraction_evaluation_seeds_improved=sum(x>0 for x in gain)/len(gain),
            frozen_median=fs['median'],ttt_median=ts['median'],frozen_std=fs['std'],ttt_std=ts['std'],
            **{f'frozen_{k}':fs[k] for k in ('q05','q25','q50','q75','q95')},
            **{f'ttt_{k}':ts[k] for k in ('q05','q25','q50','q75','q95')},
            velocity_difference=st.mean(number(r['decision_velocity'],'decision_velocity')-number(r['frozen_velocity'],'frozen_velocity') for r in rr),
            control_cost_difference=st.mean(number(r['decision_control_cost'],'decision_control_cost')-number(r['frozen_control_cost'],'frozen_control_cost') for r in rr),
            action_abs_difference=st.mean(number(r['decision_action_abs'],'decision_action_abs')-number(r['frozen_action_abs'],'frozen_action_abs') for r in rr),
            action_saturation_difference=st.mean(number(r['decision_action_saturation'],'decision_action_saturation')-number(r['frozen_action_saturation'],'frozen_action_saturation') for r in rr),
            return_per_step_difference=gs['mean']/1000,
            decision_force_saturation=st.mean(number(r['decision_force_saturation'],'force_saturation') for r in rr if r['decision_force_saturation'] not in ('',None)) if any(r['decision_force_saturation'] not in ('',None) for r in rr) else None))
        for r, g in zip(rr,gain):
            per_seed.append(dict(parameter=parameter,severity=severity,adaptation_seed=seed,
                evaluation_seed=int(r['evaluation_seed']),frozen_return=r['frozen_return'],
                ttt_return=r['decision_return'],return_improvement=g,
                frozen_velocity=r['frozen_velocity'],ttt_velocity=r['decision_velocity'],
                frozen_control_cost=r['frozen_control_cost'],ttt_control_cost=r['decision_control_cost'],
                episode_length=r['episode_length'],terminated=r['terminated'],truncated=r['truncated'],
                actual_parameter_min=r['actual_parameter_min'],actual_parameter_max=r['actual_parameter_max']))
    return stream,summarize_streams(stream,'delta_return'),per_seed


def optimization(root, expected, config):
    result=[]; trends=[]
    for parameter,severity,seed in sorted(expected):
        child=Path(root)/'fixed-decision'/f'{parameter}-k{severity:g}-decision-seed{seed}'
        done=read(child/'completion.json');meta=read(child/'manifest.json')
        if done['status']!='PASS' or meta['checkpoint_sha256']!=config['checkpoint_sha256']:
            raise ValueError(f'Adaptation source invalid: {child}')
        updates=0;loss=[];grad=[];wall=0.;events=[];episode_loss={0:[],1:[],2:[]}
        for step,row in enumerate(rows(child/'online_steps.csv'),start=1):
            if int(row['adaptation_steps'])!=step:raise ValueError(f'Online step gap: {child}')
            hit=int(row['updates_this_transition']);updates+=hit
            if int(row['gradient_updates'])!=updates:raise ValueError('Update count mismatch')
            wall+=number(row['transition_seconds'],'transition_seconds')
            if hit:
                events.extend([step]*hit)
                l=number(row['training_loss'],'training_loss');g=number(row['gradient_norm'],'gradient_norm')
                loss.append(l);grad.append(g);episode_loss[(step-1)//1000].append(l)
                trends.append(dict(parameter=parameter,severity=severity,adaptation_seed=seed,
                                   transition=step,training_consistency_loss=l,gradient_norm=g))
        if step!=3000 or events!=list(range(75,3001,25)):
            raise ValueError(f'Wrong adaptation schedule: {child}')
        metrics=done['metrics'];ls=quantiles(loss);gs=quantiles(grad)
        result.append(dict(parameter=parameter,severity=severity,adaptation_seed=seed,
            adaptation_transitions=step,gradient_update_steps=updates,updates_per_transition=updates/step,
            completed_episodes=metrics['adaptation_episodes'],replay_size_final=metrics.get('replay_size'),
            batch_size=meta['configuration']['batch_size'],training_loss_mean=ls['mean'],training_loss_median=ls['median'],
            training_loss_std=ls['std'],training_loss_first_episode_mean=st.mean(episode_loss[0]),
            training_loss_last_episode_mean=st.mean(episode_loss[2]),
            gradient_norm_mean=gs['mean'],gradient_norm_median=gs['median'],gradient_norm_q95=gs['q95'],
            optimizer_seconds=metrics['update_compute_time'],observed_transition_seconds=wall,
            parameter_update_norm=None,valid_transition_rows=step,invalid_transition_rows=0))
    return result,trends


def make_pairs(prediction,control,optimization_rows):
    c={(r['parameter'],float(r['severity']),int(r['adaptation_seed'])):r for r in control}
    o={(r['parameter'],float(r['severity']),int(r['adaptation_seed'])):r for r in optimization_rows}
    pairs=[]
    for p in prediction:
        identity=(p['parameter'],float(p['severity']),int(p['adaptation_seed']))
        if identity not in c:continue
        r=c[identity];op=o[identity]
        pg=float(p['mse_improvement']);cg=float(r['delta_return'])
        pb=max(1e-8,.001*abs(float(p['frozen_mean_mse'])))
        cb=max(1e-6,.001*max(1.,abs(float(r['frozen_mean_return']))))
        ps='+' if pg>pb else '-' if pg<-pb else '0'
        cs='+' if cg>cb else '-' if cg<-cb else '0'
        pairs.append(dict(parameter=identity[0],severity=identity[1],adaptation_seed=identity[2],
            prediction_improvement=pg,return_improvement=cg,quadrant=f'P{ps}/C{cs}',
            prediction_neutral_band=pb,return_neutral_band=cb,
            frozen_mse=p['frozen_mean_mse'],ttt_mse=p['ttt_mean_mse'],
            frozen_return=r['frozen_mean_return'],ttt_return=r['ttt_mean_return'],
            velocity_difference=r['velocity_difference'],control_cost_difference=r['control_cost_difference'],
            optimizer_updates=op['gradient_update_steps'],adaptation_transitions=op['adaptation_transitions'],
            prediction_status='COMPLETE',control_status='COMPLETE'))
    return pairs


def draw(out,conditions,prediction,pred_dist,pred_seed,control,control_dist,pairs,opt,trends):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    families=('bthigh_damping','torso_mass')
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        group=[r for r in prediction if r['parameter']==param]
        for seed in sorted({r['adaptation_seed'] for r in group}):
            g=sorted([r for r in group if r['adaptation_seed']==seed],key=lambda r:r['severity'])
            ax.plot([r['severity'] for r in g],[r['ttt_mean_mse'] for r in g],'.-',alpha=.45,label=f'TTT seed {seed}')
        g=sorted({r['severity']:r['frozen_mean_mse'] for r in group}.items())
        ax.plot([x for x,_ in g],[y for _,y in g],'k--',label='Frozen')
        d=sorted([r for r in pred_dist if r['parameter']==param],key=lambda r:r['severity'])
        if d:
            ax.errorbar([r['severity'] for r in d],[r['mean'] for r in d],
                yerr=[[r['mean']-r['ci_low'] for r in d],[r['ci_high']-r['mean'] for r in d]],
                fmt='none',ecolor='tab:blue',alpha=.6,capsize=2)
        ax.set(title=param,xlabel='Severity multiplier',ylabel='One-step latent MSE (per-seed means)');ax.legend(fontsize=7)
    save(fig,'prediction_vs_severity.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        group=[r for r in prediction if r['parameter']==param]
        for seed in sorted({r['adaptation_seed'] for r in group}):
            g=sorted([r for r in group if r['adaptation_seed']==seed],key=lambda r:r['severity'])
            ax.plot([r['severity'] for r in g],[r['relative_mse_reduction'] for r in g],'.-',alpha=.55,label=str(seed))
        ax.axhline(0,color='gray',lw=.8)
        ax.set(title=param,xlabel='Severity multiplier',ylabel='Relative MSE reduction; 3 streams')
        ax.legend(fontsize=7)
    save(fig,'relative_prediction_improvement.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        data=[r for r in pred_seed if r['parameter']==param]
        frozen=[r['frozen_mean_mse'] for r in data]; adapted=[r['ttt_mean_mse'] for r in data]
        if frozen and adapted:
            ax.boxplot([frozen,adapted],showfliers=True)
            ax.set_xticks([1,2],['Frozen','Dynamics-only'])
            if min(frozen+adapted)>0:ax.set_yscale('log')
        ax.set(title=param,ylabel='One-step latent MSE; per-evaluation-seed means')
    save(fig,'prediction_error_by_family.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        data=[r for r in pred_seed if r['parameter']==param]
        severities=sorted(set(r['severity'] for r in data))
        for i,k in enumerate(severities):
            x=[r['mse_improvement'] for r in data if r['severity']==k]
            if x:ax.boxplot(x,positions=[i],showfliers=True)
        ax.axhline(0,color='gray',lw=.8);ax.set(title=param,xticks=range(len(severities)),xticklabels=[str(x) for x in severities],
            xlabel='Severity',ylabel='Frozen − TTT MSE; 20 seed means ×3 streams')
    save(fig,'prediction_improvement_distribution.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        group=[r for r in prediction if r['parameter']==param]
        if group:
            seeds=sorted({r['adaptation_seed'] for r in group});sev=sorted({r['severity'] for r in group})
            matrix=np.full((len(seeds),len(sev)),np.nan)
            for r in group:matrix[seeds.index(r['adaptation_seed']),sev.index(r['severity'])]=r['relative_mse_reduction']
            im=ax.imshow(matrix,aspect='auto',cmap='coolwarm');fig.colorbar(im,ax=ax)
            ax.set(yticks=range(len(seeds)),yticklabels=seeds,xticks=range(len(sev)),xticklabels=sev,title=param,
                xlabel='Severity',ylabel='Adaptation seed')
    save(fig,'prediction_seed_heatmap.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        group=[r for r in control if r['parameter']==param]
        if group:
            sev=sorted({r['severity'] for r in group})
            ax.plot(sev,[st.mean(r['frozen_mean_return'] for r in group if r['severity']==k) for k in sev],'k--',label='Frozen')
            for seed in sorted({r['adaptation_seed'] for r in group}):
                g=sorted([r for r in group if r['adaptation_seed']==seed],key=lambda r:r['severity'])
                ax.plot([r['severity'] for r in g],[r['ttt_mean_return'] for r in g],'.-',alpha=.55,label=f'TTT {seed}')
        else:ax.text(.5,.5,'PENDING',ha='center',va='center',transform=ax.transAxes)
        ax.set(title=param,xlabel='Severity',ylabel='Held-out episode return')
        if group:ax.legend(fontsize=7)
    save(fig,'control_vs_severity.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        group=[r for r in control if r['parameter']==param]
        for r in group:ax.scatter(r['severity'],r['delta_return'],s=25)
        ax.axhline(0,color='gray',lw=.8);ax.set(title=param,xlabel='Severity',ylabel='TTT − Frozen return; one point/stream')
        if not group:ax.text(.5,.5,'PENDING',ha='center',va='center',transform=ax.transAxes)
    save(fig,'control_improvement_distribution.png')
    if control:
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for ax,param in zip(axs,families):
            group=[r for r in control if r['parameter']==param]
            seeds=sorted({r['adaptation_seed'] for r in group})
            ax.boxplot([[r['delta_return'] for r in group if r['adaptation_seed']==seed] for seed in seeds],
                       showfliers=True)
            ax.set_xticks(range(1,len(seeds)+1),[str(seed) for seed in seeds])
            ax.axhline(0,color='gray',lw=.8)
            ax.set(title=param,xlabel='Adaptation seed',ylabel='TTT − Frozen return; 6 severity means')
        save(fig,'control_by_adaptation_seed.png')
    fig,ax=plt.subplots(figsize=(7,5))
    markers={.5:'o',.75:'s',1.:'^',1.25:'D',1.5:'P',2.:'X'}
    for param,color in zip(families,('tab:blue','tab:orange')):
        group=[r for r in pairs if r['parameter']==param]
        for i,r in enumerate(group):
            ax.scatter(r['prediction_improvement'],r['return_improvement'],color=color,
                       marker=markers[r['severity']],label=param if i==0 else None)
            ax.annotate(str(r['adaptation_seed'])[-2:],
                        (r['prediction_improvement'],r['return_improvement']),fontsize=5,alpha=.7)
    ax.axhline(0,color='gray');ax.axvline(0,color='gray')
    if not pairs:ax.text(.5,.5,'PENDING CONTROL',ha='center',va='center',transform=ax.transAxes)
    ax.set(xlabel='Frozen − TTT latent MSE',ylabel='TTT − Frozen return',
           title='One point per adaptation stream; marker denotes severity')
    if pairs:ax.legend()
    save(fig,'prediction_control_quadrants.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        group=[r for r in pairs if r['parameter']==param]
        if group:
            seeds=sorted({r['adaptation_seed'] for r in group});sev=sorted({r['severity'] for r in group})
            matrix=np.full((len(seeds),len(sev)),np.nan)
            for r in group:matrix[seeds.index(r['adaptation_seed']),sev.index(r['severity'])]=r['return_improvement']
            im=ax.imshow(matrix,aspect='auto',cmap='coolwarm');fig.colorbar(im,ax=ax)
            ax.set(yticks=range(len(seeds)),yticklabels=seeds,xticks=range(len(sev)),xticklabels=sev,title=param,
                xlabel='Severity',ylabel='Adaptation seed')
        else:ax.text(.5,.5,'PENDING',ha='center',va='center',transform=ax.transAxes)
    save(fig,'control_seed_heatmap.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        group=[r for r in trends if r['parameter']==param]
        for seed in sorted({r['adaptation_seed'] for r in group}):
            g=[r for r in group if r['adaptation_seed']==seed and r['severity']==1.]
            if g:ax.plot([r['transition'] for r in g],[r['training_consistency_loss'] for r in g],alpha=.5,label=str(seed))
        ax.set(title=param,xlabel='Adaptation transition (k=1)',ylabel='Training consistency loss');ax.legend(fontsize=7)
    save(fig,'optimization_diagnostics.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axs,families):
        group=[r for r in trends if r['parameter']==param]
        if group:
            ax.hist([r['gradient_norm'] for r in group],bins=25,alpha=.7)
        ax.set(title=param,xlabel='Predictor gradient norm',ylabel='Logged update events')
    save(fig,'gradient_norm_distribution.png')
    fig,axs=plt.subplots(3,2,figsize=(11,10))
    for ax,param in zip(axs[0],families):
        g=[r for r in prediction if r['parameter']==param]
        for r in g:ax.scatter(r['severity'],r['mse_improvement'],s=20)
        ax.axhline(0,color='gray',lw=.8);ax.set(title=param,xlabel='Severity',ylabel='Frozen − TTT MSE')
    for ax,param in zip(axs[1],families):
        g=[r for r in pairs if r['parameter']==param]
        for r in g:ax.scatter(r['severity'],r['return_improvement'],s=20)
        ax.axhline(0,color='gray',lw=.8);ax.set(title=param,xlabel='Severity',ylabel='TTT − Frozen return')
        if not g:ax.text(.5,.5,'PENDING CONTROL',ha='center',va='center',transform=ax.transAxes)
    for ax,param in zip(axs[2],families):
        g=[r for r in opt if r['parameter']==param]
        for r in g:ax.scatter(r['severity'],r['gradient_update_steps'],s=20)
        ax.set(title=f'{param}: update cost',xlabel='Severity',ylabel='Optimizer updates / stream')
        ax.text(.02,.08,'Nominal retention: NOT_EVALUATED',transform=ax.transAxes,fontsize=8)
    save(fig,'mechanism_summary.png')


def report(out,a,complete,prediction,control,pairs,opt,control_top):
    from collections import Counter
    import numpy as np
    npred=sum(r['prediction_status']=='COMPLETE' for r in complete)
    ncontrol=sum(r['control_status']=='COMPLETE' for r in complete)
    status='COMPLETE' if len(control)==len(prediction)==36 and ncontrol==npred==36 and control_top=='PASS' else 'PREDICTION_ONLY'
    def association(group):
        if len(group)<6:
            return f'n={len(group)}; insufficient independent streams'
        x=np.asarray([r['prediction_improvement'] for r in group],dtype=float)
        y=np.asarray([r['return_improvement'] for r in group],dtype=float)
        if np.std(x)==0 or np.std(y)==0:
            return f'n={len(group)}; undefined (constant variable)'
        rank_x=np.argsort(np.argsort(x)).astype(float)
        rank_y=np.argsort(np.argsort(y)).astype(float)
        return f'n={len(group)}; Pearson {np.corrcoef(x,y)[0,1]:.3f}; Spearman {np.corrcoef(rank_x,rank_y)[0,1]:.3f}'
    prediction_summary=[]
    for family in ('bthigh_damping','torso_mass'):
        group=[r for r in prediction if r['parameter']==family]
        prediction_summary.append(f'- {family}: {len(group)} streams; {sum(r["mse_improvement"]>0 for r in group)} improved, '
            f'{sum(r["mse_improvement"]<0 for r in group)} worsened; mean stream MSE improvement '
            f'{st.mean(r["mse_improvement"] for r in group):.6g}.' if group else f'- {family}: no complete streams.')
    quadrant_counts=Counter(r['quadrant'] for r in pairs)
    pair_summary=['Overall association: '+association(pairs)]
    for family in ('bthigh_damping','torso_mass'):
        group=[r for r in pairs if r['parameter']==family]
        pair_summary.append(f'- {family}: {association(group)}; quadrants {dict(Counter(r["quadrant"] for r in group))}.')
    pair_summary.append('Within a single family × severity, n=3 streams; no per-severity correlation is computed.')
    heading=['# Frozen vs Dynamics-only TTT distribution analysis','',
        f'Status: **{status}**. Prediction {npred}/36 streams; control {ncontrol}/36 streams; control stage {control_top}.',
        f'Inputs: `{a.final}`, `{a.runs}`, `{a.reference}`. No simulation or training was run.','',
        'The existing PASS compatibility audit is reused. Per-child snapshot SHA, checkpoint SHA, bank SHA, episode seeds, horizon, and applied fixed parameter are checked before aggregation.','',
        'Prediction MSE is a mean of 1,000 transitions per evaluation seed. The saved TTT CSV has no per-transition errors; all prediction box plots and quantiles describe **per-seed mean MSE**, not transition-level errors.',
        'The Frozen and TTT per-seed MSEs use the same hashed Frozen observation/action bank. Sample-level paired error differences and fraction of improved individual transitions are **NOT_AVAILABLE**.',
        'Adaptation variability uses three independently adapted streams per physical condition. The 20 evaluation seeds estimate each stream conditionally; 720 episodes are not 720 adaptation replicates.',
        'Relative MSE reduction uses max(Frozen MSE, 1e-10); near-zero denominators are flagged. Return percent difference uses max(abs(Frozen return), 1e-10).',
        'Quadrant neutral bands are max(1e-8, 0.1% of |Frozen MSE|) and max(1e-6, 0.1% of max(1, |Frozen return|)); categories are descriptive, not significance tests.','',
        '## Prediction',f'Complete stream estimates: {len(prediction)}. Negative MSE improvements are retained.',
        *prediction_summary,
        'See `prediction_per_stream.csv`, `prediction_distribution.csv`, `prediction_per_seed.csv` and the prediction figures.','',
        '## Control',f'Complete stream estimates used: {len(control)}. {"Control analysis is pending a complete PASS stage." if not control else "See paired per-seed and per-stream return/control metrics."}',
        *(f'- {family}: {st.mean(r["delta_return"] for r in group):.4g} mean paired return difference across {len(group)} streams; {sum(r["delta_return"]>0 for r in group)} improved, {sum(r["delta_return"]<0 for r in group)} worsened.'
          for family in ('bthigh_damping','torso_mass') if (group:=[r for r in control if r['parameter']==family])),
        'The TTT evaluator saved no TTT root/joint trajectories or per-step prediction errors. Actuator-force saturation is available only for the adapted rollout; it is not inferred from action saturation.','',
        '## Prediction–control',f'Complete paired streams: {len(pairs)}. Quadrant counts: {dict(quadrant_counts)}. Quadrants and associations are descriptive, not causal.',
        *pair_summary,'',
        '## Optimization',f'Existing online adaptation logs analyzed for {len(opt)} streams. '
        f'Observed update counts: {sorted(set(r["gradient_update_steps"] for r in opt))}; '
        f'first eligible actual update at step 75, then every 25 transitions to 3,000. '
        'Training loss is not held-out generalization.','',
        '## Missing evidence',
        '- Nominal retention: NOT_EVALUATED unless a separate nominal batch exists.',
        '- Q/value mismatch, encoder drift, replay-distribution shift and planning candidate ranking: HYPOTHESES requiring separate evidence.',
        '- The single nominal training seed limits cross-training generalization.','']
    (out/'DYNAMICS_TTT_DISTRIBUTION_REPORT.md').write_text('\n'.join(heading))
    (out/'PREDICTION_DISTRIBUTION.md').write_text('\n'.join(['# Held-out prediction distribution','',
        f'{len(prediction)}/36 adaptation streams have complete prediction CSVs.',
        'Each error observation here is a 1,000-transition mean for one evaluation seed. Individual transition-level TTT errors were not saved.',
        'Confidence intervals, when shown, resample only the three adaptation-stream means within a physical condition.','']))
    if control:
        (out/'CONTROL_DISTRIBUTION.md').write_text('\n'.join(['# Held-out shifted control distribution','',
            f'{len(control)}/36 adaptation streams have complete 20-seed control CSVs.',
            'Return differences are paired by evaluation seed against the existing Frozen episode.',
            'Return quantiles describe evaluation-seed variation within a model; uncertainty across adapted models uses three stream means.','']))
        cases=[r for r in pairs if r['quadrant'] in ('P+/C-','P-/C+','P-/C-','P+/C0')]
        lines=['# Prediction–control cases','',
               'Quadrants use the neutral bands documented in the main report. These are descriptive signs, not causal diagnoses.','']
        for r in cases:
            lines.append(f"- {r['parameter']} ×{r['severity']:g}, adaptation seed {r['adaptation_seed']}: {r['quadrant']}; Frozen/TTT MSE {r['frozen_mse']:.6g}/{r['ttt_mse']:.6g}; Frozen/TTT return {r['frozen_return']:.4g}/{r['ttt_return']:.4g}; Δvelocity {r['velocity_difference']:.4g}; Δcontrol cost {r['control_cost_difference']:.4g}; {r['adaptation_transitions']} transitions, {r['optimizer_updates']} updates.")
        if not cases:lines.append('No listed discordant/worsening cases among complete paired streams.')
        (out/'PREDICTION_CONTROL_FAILURE_CASES.md').write_text('\n'.join(lines)+'\n')
    (out/'JOINT_RL_ABLATION_READINESS.md').write_text('\n'.join(['# Joint RL and component ablation readiness','',
        'No Joint RL or ablation run was launched by this analysis. The companion CSV is a schema only.','',
        'Dynamics-only prediction uses a frozen encoder. A method that updates the encoder cannot be compared by its own self-consistency latent MSE as if the representation were shared.',
        'Use `frozen_encoder_latent_mse`, `model_self_consistency_mse`, or `common_reference_prediction_metric` only when that metric is actually measured.',
        'Q-only, Dynamics+Q, Encoder+Dynamics and Full Joint RL are different update contracts; keep their identities separate.',
        'Follow-up diagnostics, if warranted by observed discordant cases: planning candidate ranking; frozen Q/value compatibility; encoder drift; gradient variance; replay-distribution shift; update schedule sensitivity. All remain hypotheses.','']))
    return status


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True,help='Existing dynamics-only-750k-v1/runs root')
    p.add_argument('--final',required=True,help='Existing final-snapshot-v2 root')
    p.add_argument('--reference',required=True,help='Existing physical-severity-750k-v1 root')
    p.add_argument('--output',required=True,help='New versioned analysis directory; never overwrites')
    p.add_argument('--mode',choices=('prediction','full','auto'),default='auto',
                   help='prediction ignores pending control; full requires shifted PASS; auto uses control only if PASS')
    a=p.parse_args(argv)
    out=Path(a.output)
    final=Path(a.final);reference=Path(a.reference)
    audit=read(final/'audit/audit.json');audit['_path']=str(final/'audit/audit.json')
    if audit['status']!='PASS' or read(final/'audit/completion.json')['status']!='PASS':
        raise ValueError('Existing compatibility audit is not PASS')
    c=audit['configuration']
    expected_keys={(x['parameter'],float(x['severity']),seed) for x in c['conditions'] for seed in c['adaptation_seeds']}
    checks=list(rows(final/'audit/run_integrity.csv'))
    expected={key(row):row['snapshot_sha256'] for row in checks}
    if len(expected_keys)!=36 or len(expected)!=36 or set(expected)!=expected_keys or len(c['evaluation_seeds'])!=20:
        raise ValueError('Unexpected condition/adaptation/evaluation seed coverage')
    integrity=read(reference/'frozen/model_integrity.json')
    if integrity['file_sha256']['model.pt']!=audit['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint mismatch')
    index={r['file']:r['sha256'] for r in read(reference/'frozen/dataset_index.json')}
    frozen={(r['parameter'],float(r['severity']),int(r['seed'])):r for r in rows(reference/'frozen/episodes.csv')}
    if len(frozen)!=12*20:
        raise ValueError('Frozen seed/condition coverage differs')
    pred,pstatus,pred_top=stage(final/'prediction','prediction',expected,audit,index,frozen)
    if a.mode=='full' and (pred_top!='PASS' or sum(x=='COMPLETE' for x in pstatus.values())!=36):
        raise ValueError('Full mode requires 36/36 compatible prediction streams')
    shifted_root=final/'shifted'
    control_top=read(shifted_root/'completion.json').get('status','IN_PROGRESS') if (shifted_root/'completion.json').is_file() else 'IN_PROGRESS'
    if a.mode=='full' and control_top!='PASS':
        raise ValueError('Full mode requires shifted completion PASS; use prediction or auto while running')
    control,cstatus,control_top=stage(shifted_root,'shifted',expected,audit,index,frozen)
    if a.mode=='full' and sum(x=='COMPLETE' for x in cstatus.values())!=36:
        raise ValueError('Full mode requires all 36 compatible shifted streams')
    if a.mode=='prediction' or control_top!='PASS' or sum(x=='COMPLETE' for x in cstatus.values())!=36:
        control={}
    completeness=[dict(parameter=k[0],severity=k[1],adaptation_seed=k[2],
                       prediction_status=pstatus[k],control_status=cstatus[k],
                       prediction_error=pred.get(k,{}).get('error'),control_error=control.get(k,{}).get('error'))
                  for k in sorted(expected)]
    prediction,pdist,pseed=prediction_tables(pred,c)
    optimization_rows,trends=optimization(a.runs,expected,audit)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'condition_completeness.csv',completeness)
    write(out/'prediction_per_stream.csv',prediction)
    write(out/'prediction_distribution.csv',pdist)
    write(out/'prediction_per_seed.csv',pseed)
    write(out/'adaptation_optimization_statistics.csv',optimization_rows)
    write(out/'adaptation_update_events.csv',trends)
    control_rows=[];control_dist=[];control_seed=[];pairs=[]
    if control and a.mode!='prediction':
        control_rows,control_dist,control_seed=control_tables(control)
        write(out/'control_per_stream.csv',control_rows)
        write(out/'control_distribution.csv',control_dist)
        write(out/'control_per_seed.csv',control_seed)
        pairs=make_pairs(prediction,control_rows,optimization_rows)
        write(out/'prediction_control_pairs.csv',pairs)
    schema_fields=('task','checkpoint_sha256','physical_shift_family','parameter','severity','adaptation_seed',
                   'adaptation_transitions','evaluation_seed','adaptation_method','optimizer_update_count',
                   'heldout_return','forward_velocity','control_cost','action_abs','action_component_saturation',
                   'actuator_force_saturation','prediction_metric_type','prediction_value','replay_protocol','data_status')
    write(out/'future_comparison_schema.csv',
          [dict(task='nsgym-halfcheetah',adaptation_method=m,data_status='SCHEMA_ONLY') for m in METHODS],schema_fields)
    draw(out,c['conditions'],prediction,pdist,pseed,control_rows,control_dist,pairs,optimization_rows,trends)
    status=report(out,a,completeness,prediction,control_rows,pairs,optimization_rows,control_top)
    (out/'analysis_manifest.json').write_text(json.dumps(dict(status=status,mode=a.mode,
        source_paths=dict(runs=str(Path(a.runs).resolve()),final=str(final.resolve()),reference=str(reference.resolve())),
        audit_sha256=digest(final/'audit/audit.json'),checkpoint_sha256=audit['checkpoint_sha256'],
        source_revision=audit.get('source_revision'),analysis_script_sha256=digest(__file__),
        prediction_streams=len(prediction),control_streams=len(control_rows),paired_streams=len(pairs),
        prediction_csv_status=pred_top,control_csv_status=control_top,
        limitations=['no per-transition TTT error','no TTT root/joint trajectories','nominal retention not evaluated here']),indent=2)+'\n')
    print(f'{status}: prediction {len(prediction)}/36, control {len(control_rows)}/36, output {out}')


if __name__=='__main__':main()
