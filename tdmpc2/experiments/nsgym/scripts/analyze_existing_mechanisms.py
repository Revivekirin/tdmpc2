#!/usr/bin/env python3
"""Task B: saved HalfCheetah artifacts, optional bounded inference, no environment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import importlib.metadata
import time
import zipfile

import numpy as np

SHA = 'f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b'
PARAMETERS = ('bthigh_damping', 'torso_mass')
SEVERITIES = (.5, .75, 1., 1.25, 1.5, 2.)


def rankdata(values):
    _,inverse,counts=np.unique(np.asarray(values),return_inverse=True,return_counts=True)
    end=np.cumsum(counts)
    return ((2*end-counts+1)/2)[inverse]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


class Sources:
    def __init__(self): self.files = {}
    def track(self, path):
        path = Path(path).resolve()
        if str(path) not in self.files:
            self.files[str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
        return path
    def json(self, path): return json.loads(self.track(path).read_text())
    def csv(self, path):
        with self.track(path).open(newline='') as f: return list(csv.DictReader(f))


def table(path, data):
    if not data: return
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader(); w.writerows(data)


def js(path, data): path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
def ident(r): return r['parameter'], float(r['severity']), int(r['adaptation_seed'])
def episode(r): return (*ident(r), int(r['evaluation_seed']))
def finite(x):
    value = float(x)
    if not np.isfinite(value): raise ValueError(f'Nonfinite input: {x}')
    return value


def unique(data, key):
    result = {key(r): r for r in data}
    if len(result) != len(data): raise ValueError('Duplicate comparison key')
    return result


def stats(values):
    a = np.array(values, dtype=float)
    if not len(a) or not np.isfinite(a).all(): raise ValueError('Invalid statistics input')
    rng = np.random.default_rng(20261112)
    boot = a[rng.integers(len(a), size=(5000, len(a)))].mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std(ddof=1)) if len(a)>1 else None,
                ci_low=float(lo), ci_high=float(hi), n=len(a))


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x)<6 or np.std(x)==0 or np.std(y)==0: return None, None
    return float(np.corrcoef(x,y)[0,1]), float(np.corrcoef(rankdata(x),rankdata(y))[0,1])


def classify(summary, band):
    if summary['ci_low'] > band: return 'positive'
    if summary['ci_high'] < -band: return 'negative'
    return 'inconclusive'


def load_stage(root, mode, src):
    top = src.json(root/'completion.json')
    if top['status'] != 'PASS': raise ValueError(f'Incomplete stage: {root}')
    manifest = src.json(root/'manifest.json')
    if manifest['checkpoint_sha256'] != SHA: raise ValueError('Checkpoint mismatch')
    result=[]; metas={}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith('interrupted-') or not (child/'completion.json').exists(): continue
        done=src.json(child/'completion.json')
        if done['status']!='PASS': continue
        m=src.json(child/'manifest.json') if (child/'manifest.json').exists() else dict(manifest,snapshot_sha256=done['snapshot_sha256'])
        if m['checkpoint_sha256']!=SHA or m['mode']!=mode: raise ValueError('Child provenance mismatch')
        rr=src.csv(child/f'{mode}.csv')
        if len(rr)!=20: raise ValueError(f'Incomplete 20-seed stream: {child}')
        k=ident(rr[0])
        if k in metas or any(ident(r)!=k for r in rr): raise ValueError('Duplicate/invalid stream')
        metas[k]=m;result.extend(rr)
    if len(metas)!=36 or len(result)!=720: raise ValueError(f'Expected 36/720 in {root}')
    unique(result,episode)
    return result,metas,manifest


def audit(a, src):
    root=a.results_root
    frozen=root/'physical-severity-750k-v1/frozen'
    final=root/'dynamics-only-750k-v1/final-snapshot-v2'
    joint=root/'joint-final-evaluation-v1'
    fm=src.json(frozen/'manifest.json');fi=src.json(frozen/'model_integrity.json')
    if src.json(frozen/'completion.json')['status']!='PASS' or fi['file_sha256']['model.pt']!=SHA:
        raise ValueError('Frozen baseline not PASS/750k')
    if fi['optimizer_updates']!=0: raise ValueError('Frozen updates are nonzero')
    fa=src.json(final/'audit/audit.json');ja=src.json(joint/'audit/audit.json')
    if fa['status']!='PASS' or ja['status']!='PASS':
        raise ValueError('Compatibility audit mismatch')
    if fa['checkpoint_sha256']!=SHA or ja['checkpoint_sha256']!=SHA: raise ValueError('750k required')
    c=fa['configuration']
    if c!=ja['configuration'] or set(c['adaptation_seeds']) & set(c['evaluation_seeds']):
        raise ValueError('Configuration or seed split mismatch')
    if len(c['adaptation_seeds'])!=3 or len(c['evaluation_seeds'])!=20:raise ValueError('Expected3 adaptation and20 evaluation seeds')
    F=unique(src.csv(frozen/'episodes.csv'),lambda r:(r['parameter'],float(r['severity']),int(r['seed'])))
    P,pm,pmeta=load_stage(final/'prediction','prediction',src)
    D,dm,dmeta=load_stage(final/'shifted','shifted',src)
    J,jm,jmeta=load_stage(joint/'shifted','shifted',src)
    JP,jpm,jpmeta=load_stage(joint/'prediction','prediction',src)
    for manifest,audit_path in ((pmeta,final/'audit/audit.json'),(dmeta,final/'audit/audit.json'),(jmeta,joint/'audit/audit.json'),(jpmeta,joint/'audit/audit.json')):
        if manifest['audit_sha256']!=digest(audit_path):raise ValueError('Stage audit fingerprint mismatch')
    keys={(p,k,s,e) for p in PARAMETERS for k in SEVERITIES for s in c['adaptation_seeds'] for e in c['evaluation_seeds']}
    if any(set(episode(r) for r in data)!=keys for data in (P,D,J,JP)):
        raise ValueError('Condition/seed coverage mismatch')
    if len(F)!=240 or set(fm['seeds'])!=set(c['evaluation_seeds']): raise ValueError('Frozen seed coverage mismatch')
    checks=unique(src.csv(final/'audit/run_integrity.csv'),ident)
    jchecks={ident(r):r for r in ja['checks']}
    for k in dm:
        if pm[k]['snapshot_sha256']!=dm[k]['snapshot_sha256'] or dm[k]['snapshot_sha256']!=checks[k]['snapshot_sha256']:
            raise ValueError('TTT snapshot differs between audit/prediction/control')
        if jm[k]['snapshot_sha256']!=jchecks[k]['snapshot_sha256'] or jpm[k]['snapshot_sha256']!=jm[k]['snapshot_sha256']: raise ValueError('Joint snapshot differs from audit/prediction')
    index={r['file']:r for r in src.json(frozen/'dataset_index.json')}
    for row in P:
        f=F[(row['parameter'],float(row['severity']),int(row['evaluation_seed']))]
        if row['bank_file']!=f['data_file'] or row['bank_sha256']!=index[f['data_file']]['sha256']:
            raise ValueError('Unpaired prediction bank')
        if int(row['samples'])!=1000: raise ValueError('Wrong prediction horizon')
    JP=unique(JP,episode)
    for row in J:
        pred=JP[episode(row)]
        ref=F[(row['parameter'],float(row['severity']),int(row['evaluation_seed']))]
        if pred['bank_file']!=ref['data_file'] or pred['bank_sha256']!=index[ref['data_file']]['sha256']:raise ValueError('Joint prediction bank mismatch')
        row['joint_self_consistency_mse']=finite(pred['joint_self_consistency_mse'])
    return F,unique(P,episode),unique(D,episode),unique(J,episode),c,checks,jchecks,fi


def canonical(F,P,D,J):
    records=[]
    for k in sorted(D):
        f=F[(k[0],k[1],k[3])];d=D[k];j=J[k];p=P[k]
        if not np.isclose(float(f['episode_return']),float(d['frozen_return']),atol=1e-6,rtol=0) or not np.isclose(float(f['episode_return']),float(j['frozen_return']),atol=1e-6,rtol=0):
            raise ValueError('Frozen return mismatch')
        for row in (f,d,j):
            if int(row['episode_length'])!=1000 or row['terminated']!='False' or row['truncated']!='True':
                raise ValueError('Incomplete episode or termination mismatch')
            for field in ('actual_parameter_min','actual_parameter_max'):
                if not np.isclose(finite(row[field]),finite(f[field]),rtol=1e-7,atol=1e-7):raise ValueError('Physical parameter mismatch')
        r=dict(parameter=k[0],shift_family='damping' if k[0]=='bthigh_damping' else 'mass',severity=k[1],
               adaptation_seed=k[2],evaluation_seed=k[3],condition_type='nominal' if k[1]==1 else 'shifted',
               frozen_reference_id=f['data_file'],episode_length=1000,
               actual_parameter=finite(f['actual_parameter_min']),nominal_parameter=finite(f['actual_parameter_min'])/k[1],
               frozen_return=finite(f['episode_return']),ttt_return=finite(d['decision_return']),joint_return=finite(j['joint_return']),
               frozen_mse=finite(p['frozen_latent_mse']),ttt_mse=finite(p['decision_latent_mse']),
               joint_self_consistency_mse=finite(j['joint_self_consistency_mse']),joint_prediction_metric='own_encoder_not_common_physical_error')
        for method,row,velocity,cost,action,sat in (
            ('frozen',f,'forward_velocity','control_cost','action_abs_mean','action_saturation'),
            ('ttt',d,'decision_velocity','decision_control_cost','decision_action_abs','decision_action_saturation'),
            ('joint',j,'joint_forward_velocity','joint_control_cost','joint_action_abs','joint_action_component_saturation')):
            for output,field in [('velocity',velocity),('control_cost',cost),('action_abs',action),('action_saturation',sat)]:
                r[f'{method}_{output}']=finite(row[field])
        r.update(delta_D=r['ttt_return']-r['frozen_return'],delta_J=r['joint_return']-r['frozen_return'],
                 delta_JD=r['joint_return']-r['ttt_return'],delta_E=r['ttt_mse']-r['frozen_mse'])
        records.append(r)
    return records


def summarize(records):
    streams=[];conditions=[]
    for key in sorted({ident(r) for r in records}):
        rr=[r for r in records if ident(r)==key]
        s=dict(parameter=key[0],severity=key[1],adaptation_seed=key[2],n_evaluation_seeds=len(rr))
        for field in ('frozen_return','ttt_return','joint_return','frozen_mse','ttt_mse','delta_D','delta_J','delta_JD','delta_E'):
            for stat,value in stats([r[field] for r in rr]).items(): s[f'{field}_{stat}']=value
        pband=max(1e-8,.001*s['frozen_mse_mean']);cband=max(1e-6,.001*max(1,abs(s['frozen_return_mean'])))
        s['prediction_class']=classify(stats([r['delta_E'] for r in rr]),pband)
        s['control_class']=classify(stats([r['delta_D'] for r in rr]),cband)
        s['classification_scope']='conditional_evaluation_seed_bootstrap_not_adaptation_reliability'
        s['prediction_band']=pband;s['return_band']=cband
        s['relative_mse_reduction']=-s['delta_E_mean']/max(s['frozen_mse_mean'],1e-10)
        streams.append(s)
    for p in PARAMETERS:
        for k in SEVERITIES:
            rr=[s for s in streams if s['parameter']==p and s['severity']==k]
            c=dict(parameter=p,severity=k,n_adaptation_streams=len(rr))
            for field in ('frozen_return','ttt_return','joint_return','delta_D','delta_J','delta_JD','delta_E'):
                for stat,value in stats([r[field+'_mean'] for r in rr]).items():c[field+'_'+stat]=value
            for field in ('delta_D','delta_J','delta_JD'):c[field+'_positive_streams']=sum(r[field+'_mean']>0 for r in rr)
            c['prediction_class']=classify(stats([r['delta_E_mean'] for r in rr]),max(1e-8,.001*np.mean([r['frozen_mse_mean'] for r in rr])))
            c['control_class']=classify(stats([r['delta_D_mean'] for r in rr]),max(1e-6,.001*max(1,abs(c['frozen_return_mean']))))
            conditions.append(c)
    return streams,conditions


def online(root, src):
    summaries=[];blocks=[]
    if src.json(root/'completion.json')['status']!='PASS':raise ValueError('Joint training stage incomplete')
    for child in sorted(root.iterdir()):
        if not child.is_dir(): continue
        m=src.json(child/'manifest.json');d=src.json(child/'completion.json')
        if m['checkpoint_sha256']!=SHA:raise ValueError('Joint training checkpoint mismatch')
        if d['status']!='PASS' or d['transitions']!=3000 or d['updates']!=2001 or d['optimizer_step_calls']!=4002:
            raise ValueError('Joint training budget/completion mismatch')
        id={k:m['identity'][k] for k in ('parameter','severity','adaptation_seed')}
        path=src.track(child/'training.jsonl');episode_returns=[0.,0.,0.];bins={};count=0
        with path.open() as f:
            for line in f:
                r=json.loads(line);count+=1
                if r['transition']!=count or r['updates']!=max(0,count-999):raise ValueError('Joint update cadence mismatch')
                episode_returns[(count-1)//1000]+=finite(r['reward'])
                metrics=dict(r['metrics'],model_update_calls=r['updates'],replay_episodes=r['replay_episodes'])
                for metric,value in metrics.items():bins.setdefault(((count-1)//100,metric),[]).append(finite(value))
        if count!=3000:raise ValueError('Incomplete joint log')
        for (b,metric),values in sorted(bins.items()):
            blocks.append(dict(**id,transition_start=b*100+1,transition_end=(b+1)*100,metric=metric,
                               mean=float(np.mean(values)),median=float(np.median(values)),std=float(np.std(values)),samples=len(values)))
        summaries.append(dict(**id,episode1_return=episode_returns[0],episode2_return=episode_returns[1],
                              episode3_return=episode_returns[2],online_last_minus_first=episode_returns[2]-episode_returns[0],
                              updates=2001,optimizer_steps=4002,transitions=3000,
                              update_seconds=d['update_seconds'],wall_seconds=d['wall_seconds'],replay_rows_final=d.get('replay_rows')))
    if len(summaries)!=36:raise ValueError('Expected 36 Joint logs')
    return summaries,blocks


def planning(root, src):
    if src.json(root/'completion.json')['status']!='PASS':raise ValueError('Planning source incomplete')
    comparisons=src.csv(root/'planning_comparisons.csv');summary=[];candidates=[]
    for n,row in enumerate(comparisons):
        for observation in ('same_observation','nominal_next_observation','shifted_next_observation'):
            path=src.track(root/f'example-{n:02d}-{observation}.npz')
            with np.load(path) as data:
                for i in sorted(int(k.split('_')[-1]) for k in data.files if k.startswith('score_')):
                    reward=data[f'reward_sum_{i}'].reshape(-1);q=data[f'discounted_terminal_q_{i}'].reshape(-1);score=data[f'score_{i}'].reshape(-1)
                    np.testing.assert_allclose(reward+q,score,rtol=1e-5,atol=1e-4)
                    n_elites=len(data[f'elite_indices_{i}'])
                    fingerprint=hashlib.sha256()
                    for array in (data['encoded'],data[f'actions_{i}'],reward,q,score):fingerprint.update(np.ascontiguousarray(array).tobytes())
                    top=lambda x:set(np.argsort(x)[-n_elites:])
                    sr=float(np.corrcoef(rankdata(reward),rankdata(score))[0,1]) if np.std(reward)>0 and np.std(score)>0 else None
                    base=dict(example=n,parameter=row['parameter'],severity=float(row['severity']),seed=int(row['seed']),observation=observation,iteration=i,
                              trace_fingerprint=fingerprint.hexdigest())
                    summary.append(dict(**base,reward_mean=float(reward.mean()),terminal_q_mean=float(q.mean()),
                        reward_std=float(reward.std()),terminal_q_std=float(q.std()),score_std=float(score.std()),
                        reward_only_vs_full_spearman=sr,top_k_overlap=len(top(reward)&top(score))/n_elites,
                        actual_ranking='ACTUAL_RANKING_NOT_MEASURED'))
                    for idx in range(len(score)):
                        candidates.append(dict(**base,candidate=idx,reward_sum=float(reward[idx]),discounted_terminal_q=float(q[idx]),
                                               total_score=float(score[idx]),elite=idx in set(data[f'elite_indices_{i}'].tolist())))
    return summary,candidates


def plots(out, conditions, streams, blocks, plan, reward, qdiag, online_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.tight_layout();fig.savefig(out/name,dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,p in zip(axs,PARAMETERS):
        group=[r for r in conditions if r['parameter']==p]
        for method in ('frozen','ttt','joint'):
            line,=ax.plot(SEVERITIES,[r[method+'_return_mean'] for r in group],'.-',label=method)
            ss=[r for r in streams if r['parameter']==p]
            ax.scatter([r['severity'] for r in ss],[r[method+'_return_mean'] for r in ss],s=10,color=line.get_color(),alpha=.35)
        ax.set(title=p,xticks=SEVERITIES,xlabel='Severity (k=1: nominal continual update)',ylabel='Held-out episode return');ax.legend()
    save(fig,'control_vs_severity.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,p in zip(axs,PARAMETERS):
        group=[r for r in conditions if r['parameter']==p]
        for metric,label in [('delta_D','TTT - Frozen'),('delta_J','Joint - Frozen'),('delta_JD','Joint - TTT')]:
            y=np.array([r[metric+'_mean'] for r in group]);lo=np.array([r[metric+'_ci_low'] for r in group]);hi=np.array([r[metric+'_ci_high'] for r in group])
            ax.plot(SEVERITIES,y,'.-',label=label);ax.fill_between(SEVERITIES,lo,hi,alpha=.12)
        ax.axhline(0,color='gray',lw=.7);ax.set(title=p,xticks=SEVERITIES,xlabel='Severity',ylabel='Paired return difference (n=3 streams)');ax.legend()
    save(fig,'joint_vs_ttt.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    markers=dict(zip(SEVERITIES,['o','s','^','D','P','X']))
    for ax,p in zip(axs,PARAMETERS):
        for k in SEVERITIES:
            g=[r for r in streams if r['parameter']==p and r['severity']==k]
            ax.scatter([r['delta_E_mean'] for r in g],[r['delta_D_mean'] for r in g],marker=markers[k],label=f'k={k:g}')
        ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set(title=p,xlabel='TTT - Frozen MSE (negative = improvement)',ylabel='TTT - Frozen return');ax.legend(fontsize=7)
    save(fig,'prediction_control_quadrants.png')
    metrics=sorted({r['metric'] for r in blocks})
    fig,axs=plt.subplots((len(metrics)+2)//3,3,figsize=(13,3*((len(metrics)+2)//3)),squeeze=False)
    for ax,metric in zip(axs.flat,metrics):
        for p in PARAMETERS:
            rr=[r for r in blocks if r['parameter']==p and r['metric']==metric]
            xs=sorted({r['transition_end'] for r in rr})
            ax.plot(xs,[np.median([r['mean'] for r in rr if r['transition_end']==x]) for x in xs],label=p)
        ax.axvline(1000,color='gray',ls='--');ax.set(title=metric,xlabel='Transition; 100-step bins',ylabel='Median stream-bin mean');ax.legend(fontsize=6)
    for ax in list(axs.flat)[len(metrics):]:ax.axis('off')
    save(fig,'joint_optimization.png')
    fig,axs=plt.subplots(2,6,figsize=(17,6))
    for axes,p in zip(axs,PARAMETERS):
        for ax,k in zip(axes,SEVERITIES):
            for row in online_rows:
                if row['parameter']==p and float(row['severity'])==k:
                    ax.plot([1,2,3],[row[f'episode{i}_return'] for i in (1,2,3)],'.-',alpha=.6,label=str(row['adaptation_seed']))
            ax.set(title=f'{p} x{k:g}',xticks=[1,2,3],xlabel='Online episode',ylabel='Exploration return')
    fig.suptitle('3 online episodes per stream; learning starts after collecting episode 1',y=1.01)
    save(fig,'joint_online_returns.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,p in zip(axs,PARAMETERS):
        g=[r for r in plan if r['parameter']==p and r['observation']=='same_observation']
        for metric in ('reward_std','terminal_q_std'):ax.plot([r[metric] for r in g],label=metric)
        ax.set(title=p+' Frozen candidates',xlabel='Example / MPPI iteration',ylabel='Candidate score component std');ax.legend()
    save(fig,'planning_decomposition.png')
    if reward:
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for ax,p in zip(axs,PARAMETERS):
            for method in ('frozen','ttt','joint'):
                g=[r for r in reward if r['parameter']==p and r['method']==method]
                ax.scatter([r['severity'] for r in g],[r['reward_mae'] for r in g],label=method,alpha=.5)
            ax.set(title=p,xlabel='Severity',ylabel='Reward MAE on common real transitions');ax.legend()
        save(fig,'reward_prediction.png')
    if qdiag:
        fig,axs=plt.subplots(2,3,figsize=(13,8))
        for axes,p in zip(axs,PARAMETERS):
            for ax,metric in zip(axes,('q_mean','q_ensemble_std','bellman_abs_mean')):
                for method in ('frozen','ttt','joint'):
                    g=[r for r in qdiag if r['method']==method and r['parameter']==p]
                    ax.scatter([r['severity'] for r in g],[r[metric] for r in g],label=method,alpha=.4)
                ax.set(title=p,xlabel='Severity',ylabel=metric);ax.legend()
        save(fig,'q_diagnostics.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,default=Path('results/nsgym'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inference',action='store_true',help='Bounded forward-only models; no optimizer/environment construction')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=p.parse_args();start=time.perf_counter();src=Sources()
    if a.output.exists() or a.output.with_suffix('.zip').exists():raise FileExistsError(a.output)
    F,P,D,J,c,checks,jchecks,fi=audit(a,src)
    records=canonical(F,P,D,J);streams,conditions=summarize(records)
    online_rows,blocks=online(a.results_root/'full-joint-750k-v1/runs-fixed',src)
    if set(unique(online_rows,ident))!=set(checks):raise ValueError('Joint log condition coverage mismatch')
    planning_rows,candidates=planning(a.results_root/'physical-severity-750k-v1/planning',src)
    reward=[];qdiag=[];drift=[];predicted=[]
    if a.inference:
        from mechanism_forward import run
        reward,qdiag,drift,predicted=run(a,src,F,c,checks,jchecks,fi)
    # Failed input/inference checks do not consume the requested output name.
    out=a.output
    for directory in ('reports','tables','figures','provenance','scripts'):(out/directory).mkdir(parents=True)
    for name,rr in [('canonical_comparison',records),('paired_control_differences',records),('per_stream',streams),
        ('per_condition',conditions),('prediction_control',streams),('joint_optimization_diagnostics',blocks),
        ('joint_online_episodes',online_rows),('planning_score_decomposition',candidates),('planning_summary',planning_rows),
        ('reward_prediction',reward),('q_diagnostics',qdiag),('representation_drift',drift),('common_candidate_predictions',predicted)]:
        table(out/'tables'/f'{name}.csv',rr)
    table(out/'tables/condition_completeness.csv',[
        dict(parameter=s['parameter'],severity=s['severity'],adaptation_seed=s['adaptation_seed'],
             prediction='COMPLETE',ttt_control='COMPLETE',joint_control='COMPLETE',episodes=20) for s in streams])
    ranking=[]
    for key in sorted(checks):
        groups={m:sorted([r for r in predicted if ident(r)==key and r['method']==m],key=lambda r:r['candidate']) for m in ('frozen','ttt','joint')}
        if not groups['frozen']:continue
        for left,right in (('frozen','ttt'),('frozen','joint'),('ttt','joint')):
            x=[r['total_score'] for r in groups[left]];y=[r['total_score'] for r in groups[right]]
            ranking.append(dict(parameter=key[0],severity=key[1],adaptation_seed=key[2],left=left,right=right,
                predicted_rank_spearman=correlation(x,y)[1],top8_overlap=len(set(np.argsort(x)[-8:])&set(np.argsort(y)[-8:]))/8,
                actual_ranking='ACTUAL_RANKING_NOT_MEASURED'))
    table(out/'tables/predicted_ranking_comparison.csv',ranking)
    associations=[]
    for label,data,metric in [('reward',reward,'reward_mae'),('Q',qdiag,'bellman_abs_mean'),
                              ('Q',qdiag,'q_ensemble_std'),('encoder',drift,'feature_mse_vs_frozen')]:
        values={}
        for row in data:
            if row['method']=='joint' and float(row['severity'])!=1:values.setdefault(ident(row),[]).append(row[metric])
        ss=[s for s in streams if ident(s) in values]
        if not ss:continue
        x=np.array([np.mean(values[ident(s)]) for s in ss]);y=np.array([s['delta_J_mean'] for s in ss])
        for scope in ('across_shifted_conditions','within_condition_centered'):
            xx=x.copy();yy=y.copy()
            if scope=='within_condition_centered':
                for p0 in PARAMETERS:
                    for k0 in SEVERITIES:
                        indices=[i for i,s in enumerate(ss) if s['parameter']==p0 and s['severity']==k0]
                        if indices:xx[indices]-=xx[indices].mean();yy[indices]-=yy[indices].mean()
            pear,spear=correlation(xx,yy)
            associations.append(dict(diagnostic=metric,scope=scope,n_streams=len(ss),pearson=pear,spearman=spear,
                                      interpretation='descriptive association with Joint-Frozen return; not causal'))
    table(out/'tables/diagnostic_control_associations.csv',associations)
    coverage=[dict(mechanism=k,status=s,reason=r) for k,s,r in [
        ('heldout_control','MEASURED','36 streams x20 seeds; 240 reused Frozen references'),
        ('fixed_encoder_prediction','MEASURED','Paired bank hashes; per-seed mean errors'),
        ('joint_online_learning','MEASURED','All 3000 logged transitions; 100-step loss summaries'),
        ('reward_accuracy','MEASURED' if reward else 'NOT_MEASURED','Bounded common bank; enable --inference'),
        ('q_bellman_entropy','MEASURED' if qdiag else 'NOT_MEASURED','Forward-only target-state check; enable --inference'),
        ('q_empirical_calibration','NOT_MEASURED','Frozen continuation is off-policy for actor Q; no valid matching target'),
        ('representation_drift','MEASURED' if drift else 'NOT_MEASURED','Final vs750k on common observations; no intermediate encoder'),
        ('planning_decomposition','PARTIALLY_MEASURED','Frozen traces; identical cross-model tapes only with --inference'),
        ('actual_candidate_ranking','NOT_MEASURED','Candidate-specific simulator returns not saved'),
        ('shift_trained_nominal_retention','NOT_MEASURED','k=1 is nominal training+evaluation, not retention after each shifted training')]]
    table(out/'tables/analysis_coverage.csv',coverage)
    plots(out/'figures',conditions,streams,blocks,planning_rows,reward,qdiag,online_rows)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5));colors={'MEASURED':'#217a54','PARTIALLY_MEASURED':'#c39125','NOT_MEASURED':'#999999'}
    ax.barh([r['mechanism'] for r in coverage],[1]*len(coverage),color=[colors[r['status']] for r in coverage])
    for i,r in enumerate(coverage):ax.text(.02,i,r['status'],va='center',color='white',fontsize=9)
    ax.set_xticks([]);fig.tight_layout();fig.savefig(out/'figures/coverage.png',dpi=170);plt.close(fig)
    write_reports(out,a,conditions,streams,online_rows,planning_rows,reward,qdiag,drift)
    selected=a.results_root/'nominal-convergence-audit-01/selected_nominal_checkpoint.json'
    if selected.exists():
        old=src.json(selected)
        with (out/'reports/DATA_PROVENANCE.md').open('a') as f:
            f.write(f'\nLegacy nominal selection metadata: {old.get("checkpoint")}, SHA {old.get("sha256")}, status {old.get("status")}. '
                    f'Current comparison remains fixed to audited750k SHA {SHA}; legacy selection is not used to reselect a model.\n')
    config=dict(checkpoint_sha256=SHA,seed=c['evaluation_seeds'],adaptation_seeds=c['adaptation_seeds'],
        bootstrap=dict(repetitions=5000,seed=20261112,condition_unit='adaptation_stream',stream_unit='paired_evaluation_seed'),
        neutral_bands=dict(mse='max(1e-8,0.001*Frozen MSE)',return_='max(1e-6,0.001*max(1,abs(Frozen return)))'),
        inference=a.inference,device=a.device,maximum_banks_per_condition=2,maximum_transitions_per_bank=128,
        identical_action_candidates=32,planning_horizon=3)
    project=Path(__file__).resolve().parents[3]
    config['analysis_source_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    config['versions']={}
    for package in ('numpy','matplotlib','torch','tensordict'):
        try:config['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:config['versions'][package]=None
    for relative in ('tdmpc2.py','common/world_model.py','common/math.py','common/layers.py','common/init.py'):
        src.track(project/'tdmpc2'/relative)
    js(out/'provenance/analysis_config.json',config)
    here=Path(__file__).parent
    for filename in ('analyze_existing_mechanisms.py','mechanism_forward.py'):
        source=src.track(here/filename);shutil.copy2(source,out/'scripts'/filename)
    for path,record in src.files.items():
        if digest(path)!=record['sha256']:raise ValueError(f'Input changed during analysis: {path}')
    js(out/'provenance/input_manifest.json',src.files)
    js(out/'provenance/completion.json',dict(status='PASS',conditions=12,streams=36,paired_episodes=720,
        frozen_reference_rows=240,inference=a.inference,actual_candidate_ranking='NOT_MEASURED',
        elapsed_seconds=time.perf_counter()-start,optimizer_updates=0,environment_transitions=0))
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        count=len(z.namelist())
    print(json.dumps(dict(status='PASS',zip=str(archive.resolve()),files=count,bytes=archive.stat().st_size,sha256=digest(archive)),indent=2))


def write_reports(out,a,conditions,streams,online_rows,plan,reward,qdiag,drift):
    from collections import Counter
    def report(name,text):(out/'reports'/name).write_text(text+'\n')
    rows=['| Parameter | k | Frozen | TTT | Joint | D−F | J−F | J−D |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in conditions:rows.append('| '+r['parameter']+' | '+str(r['severity'])+' | '+' | '.join(f'{r[k+"_mean"]:.2f}' for k in ('frozen_return','ttt_return','joint_return','delta_D','delta_J','delta_JD'))+' |')
    report('PRIMARY_CONTROL_FINDINGS.md','# Primary held-out control findings\n\n'+'\n'.join(rows)+
        '\n\nMeans first average 20 matched evaluation episodes within each stream, then 3 adaptation streams. '
        'All 12 IDs are retained; the two k=1 labels share nominal physics and are not independent physical conditions. '
        'Do not interpret Frozen reuse as 720 independent Frozen episodes. See per_condition.csv for exploratory n=3 stream-bootstrap intervals.\n\n'+
        '\n'.join(f'{metric}: negative mean in {sum(r[metric+"_mean"]<0 for r in conditions)}/12 labelled conditions; positive stream means {sum(r[metric+"_mean"]>0 for r in streams)}/36.' for metric in ('delta_D','delta_J','delta_JD')))
    shifted=[r for r in streams if r['severity']!=1]
    lines=['# Prediction–control coupling','', 'X = TTT−Frozen MSE; negative means prediction improves. Y = TTT−Frozen return.',
        'Stream classes require paired-evaluation-seed bootstrap CI outside the documented neutral band. These describe each fixed adapted model, not adaptation reliability. Condition uncertainty resamples only 3 stream means.']
    for family in ('all',*PARAMETERS):
        g=[r for r in shifted if family=='all' or r['parameter']==family]
        pear,spear=correlation([r['delta_E_mean'] for r in g],[r['delta_D_mean'] for r in g])
        lines.append(f'{family}, shifted only: n={len(g)}, Pearson={pear}, Spearman={spear}; descriptive, no p-value.')
    x=[];y=[]
    for p in PARAMETERS:
        for k in SEVERITIES:
            if k==1:continue
            g=[r for r in streams if r['parameter']==p and r['severity']==k]
            x.extend(np.array([r['delta_E_mean'] for r in g])-np.mean([r['delta_E_mean'] for r in g]))
            y.extend(np.array([r['delta_D_mean'] for r in g])-np.mean([r['delta_D_mean'] for r in g]))
    lines.append(f'Within-family/severity centered descriptive association: {correlation(x,y)}. No n=3 per-condition correlation or causal inference.')
    lines.append(f'Improved stream mean prediction: {sum(r["delta_E_mean"]<0 for r in streams)}/36. Improved mean control: {sum(r["delta_D_mean"]>0 for r in streams)}/36.')
    lines.append('Exploratory condition-level n=3 stream-CI classes (prediction sign, control sign): '+str(dict(Counter((r['prediction_class'],r['control_class']) for r in conditions))))
    report('PREDICTION_CONTROL_COUPLING.md','\n\n'.join(lines))
    online_conditions=sum(np.mean([r['online_last_minus_first'] for r in online_rows if r['parameter']==p and float(r['severity'])==k])<0 for p in PARAMETERS for k in SEVERITIES)
    report('JOINT_LEARNING_STABILITY.md',f'# Joint online learning\n\n36 complete 3000-transition logs; exactly 2001 update calls, starting at1000. '
        f'Episode3−episode1 mean is negative in {online_conditions}/12 labelled conditions. These exploration returns are not held-out returns. '
        'Only 3 episodes exist per stream; no precise learning curve or collapse claim. Loss plots use 100-transition bins of all logged updates; onset bin contains only transition1000. '
        'Full loss/reward/value/policy/gradient/entropy/scale columns are retained when recorded. Final displacement/feature statistics, if inference enabled, describe movement, not loss of useful information.')
    report('REWARD_Q_DIAGNOSTICS.md',f'# Reward and Q diagnostics\n\nReward rows: {len(reward)}; Q rows: {len(qdiag)}; drift rows: {len(drift)}. '+
        ('Bounded inference used common Frozen real observation/action/reward banks across methods. ' if reward else 'NOT_MEASURED: rerun with --inference and saved snapshots to measure. ')+
        'Frozen and Dynamics-only reward/Q heads and encoder are fixed, so direct real-state reward/Q diagnostics should match; learned rollout inputs can differ. '
        'Joint own-coordinate consistency is not comparable to fixed-encoder MSE.\n\n'
        'Actual code uses 101 symlog bins from −10 to10 for both reward and Q, spacing0.2, decoded by symexp(softmax(logits)·bins). '
        'Five Q ensemble members. Planner terminal Q averages two random members; TD target takes minimum of two target members with actor-generated bootstrap action. '
        'Discount0.995; episodic=false, no artificial TimeLimit terminal mask. Target state must be saved; online Q is never substituted. '
        'Bellman residual uses observed reward, encoded next observation and stochastic actor action. Small residual does not establish calibration. '
        'Q MC calibration: NOT_MEASURED. Frozen continuation returns are finite-horizon and off-policy relative to Joint/actor bootstrap. '
        'No Q accuracy or uncertainty calibration is inferred from magnitude, entropy or spread.')
    if reward:
        with (out/'reports/REWARD_Q_DIAGNOSTICS.md').open('a') as f:
            f.write('\n\nBound: first2 declared evaluation seeds,128 evenly spaced transitions/bank, all36 snapshots. '
                    'Only3072 labelled condition/seed/timestep observations are sampled; their reuse across models/adaptation seeds is not independent evidence.\n')
            for method in ('frozen','ttt','joint'):
                rr=[r for r in reward if r['method']==method];qq=[r for r in qdiag if r['method']==method]
                f.write(f'\n{method}: mean bank reward MAE={np.mean([r["reward_mae"] for r in rr]):.6g}; '
                        f'mean absolute Bellman residual={np.mean([r["bellman_abs_mean"] for r in qq]):.6g}; '
                        f'mean Q ensemble std={np.mean([r["q_ensemble_std"] for r in qq]):.6g}. Descriptive equal bank weights.\n')
            f.write('\nSee diagnostic_control_associations.csv for across-severity and within-condition-centered associations; no causal attribution or p-values.\n')
    unique_traces={r['trace_fingerprint']:r for r in plan if r['observation']=='same_observation'}
    overlap=[r['top_k_overlap'] for r in unique_traces.values()]
    report('PLANNING_MISMATCH.md',f'# Planning decomposition\n\n{len(plan)} example/observation/iteration records. '
        f'Frozen reward-only versus full-score elite overlap on same-observation traces: mean {np.mean(overlap):.3f}, range {min(overlap):.3f}–{max(overlap):.3f}. '
        f'{len(unique_traces)} unique same-observation trace/iteration fingerprints; duplicated family traces are not independent evidence. '
        'This measures Q contribution to predicted ranking, not usefulness of chosen actions. '
        'Next-observation traces have different candidate sets: no cross-trace ranking correctness is computed. '
        'Optional inference scores identical 32 candidate tapes per stream from a shared held-out observation (not independent random tapes by model). '
        'ACTUAL_RANKING_NOT_MEASURED: no candidate-specific actual returns. No candidate regret. '
        'Existing run_severity.py planning does not implement a true-candidate-return audit; there is no valid existing command to provide for that audit.')
    report('CONFOUNDERS_AND_LIMITATIONS.md','# Confounders\n\n| Property | Frozen | Dynamics-only | Joint |\n|---|---|---|---|\n'
        '| Checkpoint |750k|750k|750k|\n| Adaptation interactions |0 (separate 3000-step reference streams exist)|3000|3000|\n'
        '| Model update calls |0|118|2001|\n| Adam step calls |0|118|4002|\n| Collection |Frozen MPPI|eval_mode=True|eval_mode=False|\n'
        '| Optimizer |None|New Adam|Restored nominal Adam|\n| Learned parts |None|Predictor|Encoder/dynamics/reward/Q/prior, target Q, scale|\n'
        '| Replay |None|Transition replay|Completed episode sequence replay|\n| Reward supervision |None|None|Yes|\n\n'
        'Interaction matched, not update/compute/visitation matched. 2001/118≈16.96 is an update-call ratio, not wall-time ratio. '
        'One nominal training seed; three adaptation seeds per condition. k=1 is nominal continual updating, not retention of all shift-trained models. '
        'Entropy, representation movement, loss and return associations do not identify causal failure mechanisms. '
        'The older nominal-convergence selection file may describe950k; current comparison is fixed by the audited750k SHA and never changes selection.')
    report('NEXT_EXPERIMENT_DECISIONS.md','# At most two controlled follow-ups\n\n'
        '1. Match Joint update cadence/count (118) and action collection protocol to Dynamics-only on the predeclared grid. '
        'This isolates protocol differences more cleanly than interpreting the current2001-update result as a parameter-subset effect. Do not duplicate the separate Dynamics+Q work.\n\n'
        '2. On the observed P-improves/C-worsens conditions and mass×2 positive exception, run a separately approved bounded candidate-return audit '
        'with identical saved simulator states/action tapes. Compare predicted reward-only/full-Q scores to actual finite-horizon candidate returns. '
        'Define finite-horizon target and terminal handling first; that target differs from bootstrapped planner score. Neither follow-up is executed here.')
    report('DATA_PROVENANCE.md','# Provenance\n\nThe input manifest hashes every consumed CSV/JSON/log/bank/snapshot. Existing PASS audits are reused; condition/seed, checkpoint, final snapshot and bank identities are checked. '
        'Canonical keys are parameter,severity,adaptation_seed,evaluation_seed. Frozen references are reused and have their own data_file ID. '
        'All source files are rehashed before completion. No optimizer construction, environment step, checkpoint write or source modification. '
        'Missing prerequisite or incompatible source aborts before publishing a successful completion record.')
    (out/'README.md').write_text('# Mechanism analysis750k\n\nStart with reports/PRIMARY_CONTROL_FINDINGS.md and tables/analysis_coverage.csv. '
        'No new training or simulator evaluation. A future rerun must choose a new versioned output directory.\n\n'
        'Reproduce from the project root:\n```bash\npython experiments/nsgym/scripts/analyze_existing_mechanisms.py '
        f'--results-root {a.results_root} --output results/nsgym/mechanism-analysis-750k-v2'
        +(' --inference --device '+a.device if a.inference else '')+'\n```\n')


if __name__=='__main__': main()
