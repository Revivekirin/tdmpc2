"""Saved Q/planning aggregation extracted from the original measurement runner.

No model loading. Heads/time steps -> evaluation seeds -> adaptation streams.
"""
from collections import defaultdict
import numpy as np
from .statistics import bootstrap


def stats(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite metric')
    return dict(mean=float(x.mean()),median=float(np.median(x)),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)),n=len(x))


def summarize(data,group,fields):
    episodes=defaultdict(list)
    for r in data:episodes[tuple(r[k] for k in group+['adaptation_seed','evaluation_seed'])].append(r)
    per_seed=[]
    for keys,rr in episodes.items():
        per_seed.append(dict(zip(group+['adaptation_seed','evaluation_seed'],keys),
            **{f:float(np.mean([float(r[f]) for r in rr])) for f in fields}))
    streams=defaultdict(list)
    for r in per_seed:streams[tuple(r[k] for k in group+['adaptation_seed'])].append(r)
    streamrows=[dict(zip(group+['adaptation_seed'],k),**{f:float(np.mean([r[f] for r in rr])) for f in fields},
                    evaluation_seeds=len(rr)) for k,rr in streams.items()]
    groups=defaultdict(list)
    for r in streamrows:groups[tuple(r[k] for k in group)].append(r)
    summaries=[]
    for k,rr in groups.items():
        for metric in fields:
            vals=[r[metric] for r in rr];unit='adaptation_seed'
            if 'method' in group and dict(zip(group,k))['method']=='F':
                # Frozen is identical across adaptation replicas; use shared eval seeds once.
                ep=[r for r in per_seed if tuple(r[g] for g in group)==k and r['adaptation_seed']==rr[0]['adaptation_seed']]
                vals=[r[metric] for r in ep];unit='shared_frozen_evaluation_seed'
            low, high = bootstrap(vals, "q", single=True)
            summaries.append(dict(zip(group,k),metric=metric,**stats(vals),sampling_unit=unit,
                ci_low=low,ci_high=high))
    return streamrows,summaries



def ingest_archives(rec):
    """Recompute categorical diagnostics with the original TD-MPC2 math on saved arrays.

    Observation tables supply identities, not precomputed aggregate inputs. Missing
    arrays/targets are explicit DERIVED_ONLY metrics. No checkpoint is loaded.
    """
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    from .discovery import load, passed, METHODS
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    from q_planning_core import distributions, js, ranks
    specifications={
        'immediate_q_metrics.csv':('immediate',['q','entropy','ensemble_std','boundary_mass','interhead_js','bellman_residual','bellman_soft_ce']),
        'q_distribution_divergence.csv':('divergence',['js','q_difference']),
        'terminal_q_metrics.csv':('terminal',['q','entropy','ensemble_std']),
        'planning_score_components.csv':('planning',['reward_sum','terminal_contribution','total_score','ess']),
        'candidate_ranking_comparison.csv':('ranking',['top_k_overlap','argmax_disagreement','spearman','kendall']),
        'terminal_counterfactuals.csv':('counterfactual',['dynamics_effect','q_head_effect','interaction'])}
    fallback_names={'immediate_q_metrics.csv':'immediate_per_stream.csv','q_distribution_divergence.csv':'divergence_per_stream.csv',
        'terminal_q_metrics.csv':'terminal_q_metrics_per_stream.csv','planning_score_components.csv':'planning_score_components_per_stream.csv',
        'candidate_ranking_comparison.csv':'candidate_ranking_comparison_per_stream.csv','terminal_counterfactuals.csv':'terminal_counterfactuals_per_stream.csv'}
    candidates={p.parent.parent for name in (*specifications,*fallback_names.values()) for p in rec.c.by_name.get(name,[]) if p.parent.name=='tables'}
    seen=set()
    for root in sorted(candidates,key=lambda p:(not (p/'immediate').is_dir(),str(p))):
        done=load(root/'provenance/completion.json')
        if not passed(done) or done.get('stages',{}).get('immediate')!='PASS':continue
        cfg=load(root/'provenance/analysis_config.json')
        checkpoint=rec.c.resolve(cfg.get('checkpoint'))
        checkpoint_hash=rec.c.digest(checkpoint/'model.pt') if checkpoint and (checkpoint/'model.pt').exists() else ''
        if checkpoint and (checkpoint/'metadata.json').is_file():rec.c.digest(checkpoint/'metadata.json')
        checkpoint_cfg=load(checkpoint/'metadata.json').get('config',{}) if checkpoint else {}
        cache={};table_fingerprints=[]
        for name,(kind,metrics) in specifications.items():
            p=root/'tables'/name
            if not p.is_file():
                fallback=root/'tables'/fallback_names[name]
                if fallback.is_file():
                    fingerprint=rec.c.digest(fallback)
                    if (name,fingerprint) in seen:continue
                    seen.add((name,fingerprint))
                    for i,row in enumerate(rec.c.table(fallback),2):
                        method='dynamics_q' if kind=='counterfactual' else METHODS.get(row.get('method'),row.get('method',row.get('left','')+'→'+row.get('right','')))
                        base=rec.base(fallback,row,dict(task='halfcheetah',checkpoint_sha256=checkpoint_hash),done,method)
                        for metric in metrics:rec.emit(base,metric,row.get(metric),fallback,i,'DERIVED_ONLY',kind='q_'+kind)
                continue
            fingerprint=rec.c.digest(p)
            if (name,fingerprint) in seen:
                rec.duplicates.append(dict(source=rec.c.rel(p),canonical='identical diagnostic table',reason='byte-identical Q/planning export'));continue
            seen.add((name,fingerprint));table_fingerprints.append(fingerprint)
            groups=defaultdict(list);verification={}
            for i,row in enumerate(rec.c.table(p),2):
                archive_label=f'{row["parameter"]}-k{float(row["severity"]):g}-seed{row["adaptation_seed"]}-eval{row["evaluation_seed"]}'
                immediate=kind in ('immediate','divergence')
                ap=root/('immediate' if immediate else 'terminal')/(archive_label+('' if immediate else '-terminal')+'.npz')
                if ap not in cache:
                    cache.clear() # tables are sorted by archive; keep memory bounded
                    if ap.is_file():
                        rec.c.digest(ap)
                        with np.load(ap,allow_pickle=False) as z:arrays=dict(z)
                        pack={}
                        if immediate:
                            support=arrays['symlog_support'];conf=SimpleNamespace(num_bins=len(support),vmin=float(support[0]),vmax=float(support[-1]))
                            for m in ('F','D','DQ','J'):
                                pack[m]=distributions(torch.from_numpy(arrays[m+'_logits']),conf)
                        elif checkpoint_cfg.get('num_bins'):
                            support=torch.linspace(checkpoint_cfg['vmin'],checkpoint_cfg['vmax'],checkpoint_cfg['num_bins'],dtype=torch.float32)
                            for field in arrays:
                                if field.endswith('_fixed_action_probability'):
                                    m=field[:-len('_fixed_action_probability')];prob=torch.from_numpy(arrays[field]);expected=(prob*support).sum(-1)
                                    scalar=expected.sign()*(expected.abs().exp()-1)
                                    pack[m]=dict(scalar=scalar,entropy=-(prob*prob.clamp_min(1e-30).log()).sum(-1),ensemble_std=scalar.std(0,unbiased=False))
                        cache[ap]=(arrays,pack)
                    else:cache[ap]=({}, {})
                arrays,packs=cache[ap];values={k:row.get(k) for k in metrics};evidence={k:'RAW_REPRODUCED' for k in metrics}
                if kind=='immediate' and packs:
                    m=row['method'];h=int(row['head']);j=list(arrays['timestep']).index(int(row['timestep']));pack=packs[m]
                    for metric,key in [('q','scalar'),('entropy','entropy'),('boundary_mass','boundary')]:values[metric]=float(pack[key][h,j])
                    for metric in ('ensemble_std','interhead_js'):values[metric]=float(pack[metric][j])
                elif kind=='divergence' and packs:
                    j=list(arrays['timestep']).index(int(row['timestep']));a=packs[row['left']];b=packs[row['right']]
                    values['js']=float(js(a['mixture'][j],b['mixture'][j]));values['q_difference']=float(b['scalar'][:,j].mean()-a['scalar'][:,j].mean())
                elif kind=='terminal' and packs:
                    j=int(row['candidate']);pack=packs[row['method']]
                    values.update(q=float(pack['scalar'][:,j].mean()),entropy=float(pack['entropy'][:,j].mean()),ensemble_std=float(pack['ensemble_std'][j]))
                elif kind=='counterfactual' and packs:
                    j=int(row['candidate']);qm=lambda m:packs[m]['scalar'].mean(0)[j]
                    values.update(dynamics_effect=float(qm('DQ-dyn-only')-qm('F')),q_head_effect=float(qm('DQ-full')-qm('DQ-dyn-only')),interaction=float(qm('DQ-full')-qm('DQ-dyn-only')-qm('F+DQ-Q')+qm('F')))
                elif kind=='ranking' and arrays:
                    values.update({k:v for k,v in ranks(arrays[row['left']+'_score'],arrays[row['right']+'_score'],int(row['top_k'])).items() if k in metrics})
                elif kind=='planning' and arrays:
                    values['total_score']=float(arrays[row['method']+'_score'][int(row['candidate'])])
                    rec.check(p,f'reward_plus_terminal[{i}]',row['total_score'],float(row['reward_sum'])+float(row['terminal_contribution']),rtol=1e-6,atol=1e-4)
                if kind in ('immediate','divergence') and not packs:
                    evidence={k:'DERIVED_ONLY' for k in metrics}
                if kind in ('terminal','counterfactual') and not packs:
                    # These distributions lack logits/support; logged scalar diagnostics remain usable.
                    evidence={k:'DERIVED_ONLY' for k in metrics}
                for k in ('bellman_residual','bellman_soft_ce'):
                    if k in evidence:evidence[k]='DERIVED_ONLY'
                method='dynamics_q' if kind=='counterfactual' else METHODS.get(row.get('method'),row.get('method',f'{row.get("left","")}→{row.get("right","")}'))
                meta=dict(task='halfcheetah',checkpoint_sha256=checkpoint_hash)
                base=rec.base(p,row,meta,done,method)
                if base and arrays:base['raw_dependencies']=rec.c.rel(ap)
                for metric,v in values.items():
                    if v in ('',None) or not np.isfinite(float(v)):continue
                    if evidence[metric]=='RAW_REPRODUCED' and metric in row:
                        # GPU-produced float32 vs CPU float32: preserve and report a declared arithmetic tolerance.
                        expected=float(row[metric]);difference=abs(expected-float(v))
                        item=verification.setdefault(metric,dict(n=0,failures=0,max_difference=0.,exact=True,roundoff_bound=0.))
                        item['n']+=1;item['failures']+=not np.isclose(expected,float(v),rtol=3e-5,atol=2e-4)
                        item['max_difference']=max(item['max_difference'],difference);item['exact'] &= difference==0
                        if metric in ('ensemble_std','q_difference','dynamics_effect','q_head_effect','interaction') and packs:
                            scale=max(float(pack['scalar'].abs().max()) for pack in packs.values())
                            item['roundoff_bound']=max(item['roundoff_bound'],32*np.finfo(np.float32).eps*max(scale,1.))
                    key=(row['parameter'],row['severity'],row['adaptation_seed'],row['evaluation_seed'],method,metric,evidence[metric])
                    # Retain original recorded float32 measurements after independent CPU checks,
                    # so statistical reproduction does not silently redefine a GPU diagnostic.
                    measured=row.get(metric)
                    groups[key].append((base,float(measured) if measured not in (None,'') else float(v),i))
            for metric,v in verification.items():
                rec.checks.append(dict(source=rec.c.rel(p),metric=metric,status=('NUMERICAL_DIFFERENCE' if v['max_difference']<=v['roundoff_bound'] and v['roundoff_bound']>0 else 'FAIL') if v['failures'] else 'EXACT' if v['exact'] else 'TOLERANCE',roundoff_bound=v['roundoff_bound'],compared_values=v['n'],failed_values=int(v['failures']),max_absolute_difference=v['max_difference'],rtol=3e-5,atol=2e-4))
            for key,items in groups.items():
                rec.emit(items[0][0],key[-2],np.mean([x[1] for x in items]),p,f'{items[0][2]}..{items[-1][2]}',key[-1],kind='q_'+kind)
        rec.c.experiments.append(dict(path=rec.c.rel(root),task='halfcheetah',stage='q_planning',status='PASS',reason='completed diagnostic provenance',records=len(table_fingerprints)))
