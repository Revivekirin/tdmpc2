"""Paired stream aggregation; episodes are nested within adaptation streams."""
import numpy as np
import pandas as pd
from ..metrics import require

KEY=['task','family','severity','adaptation_seed','evaluation_seed']


def hierarchical_interval(frame,column,seed=20261116,repetitions=4000):
    groups=[g[column].to_numpy(float) for _,g in frame.groupby('adaptation_seed')]
    require(bool(groups) and all(np.isfinite(g).all() for g in groups),'Missing/invalid paired observations')
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repetitions):
        selected=rng.integers(len(groups),size=len(groups))
        samples.append(np.mean([rng.choice(groups[i],size=len(groups[i]),replace=True).mean() for i in selected]))
    means=np.array([g.mean() for g in groups]);lo,hi=np.quantile(samples,[.025,.975])
    return dict(mean=float(means.mean()),median=float(np.median(means)),ci_low=float(lo),ci_high=float(hi),
                adaptation_streams=len(groups),positive_streams=int((means>0).sum()),episodes=len(frame))


def return_bridge(episodes,historical):
    require(not episodes.duplicated(KEY+['method']).any(),'Duplicate controller episode')
    require(set(episodes.method)=={'DQ','DQ_F'},'Both intervention controllers required')
    wide=episodes.pivot(index=KEY,columns='method',values='episode_return').reset_index()
    require(not wide[['DQ','DQ_F']].isna().any().any(),'Unpaired controllers')
    d=historical[historical.method=='D'][KEY+['return']].rename(columns={'return':'D'})
    require(not d.duplicated(KEY).any(),'Duplicate D reference')
    result=wide.merge(d,on=KEY,how='left',validate='one_to_one')
    require(result.D.notna().all(),'Missing historical D episode')
    result['critic']=result.DQ-result.DQ_F
    result['predictor']=result.DQ_F-result.D
    result['total']=result.DQ-result.D
    require(np.allclose(result.critic+result.predictor,result.total,atol=1e-10,rtol=1e-12),'Bridge identity failed')
    records=[]
    for keys,frame in result.groupby(['task','family','severity']):
        for metric in ('critic','predictor','total'):
            records.append(dict(zip(['task','family','severity'],keys),metric=metric,**hierarchical_interval(frame,metric)))
    return result,pd.DataFrame(records)
