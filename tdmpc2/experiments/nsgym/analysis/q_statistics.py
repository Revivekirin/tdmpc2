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

