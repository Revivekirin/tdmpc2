"""Migrated historical computations; numerical formulas preserved."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from experiments.nsgym.analysis.control_q.config import SCRIPTS
from experiments.nsgym.analysis.reacher_dq_figures import ci, probability_js
from q_planning_core import ranks


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def write(path, obj): Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def stable_seed(*key): return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:4], 'big')


def numpy(pack):return {k:v.detach().cpu().numpy() for k,v in pack.items() if k in ('probability','scalar','entropy')}


def pack_metrics(pack, frozen):
    q=pack['scalar'].mean(0);fq=frozen['scalar'].mean(0)
    return dict(scalar_q=q,head_std=pack['scalar'].std(0),entropy=pack['entropy'].mean(0),
                js_frozen=probability_js(pack['probability'].mean(0),frozen['probability'].mean(0)),
                signed_delta=q-fq,absolute_delta=np.abs(q-fq))


def decomposition(values, discount):
    v={m:np.asarray(a,dtype=np.float64) for m,a in values.items()}
    d=discount*(v['D']-v['F']);dr=discount*(v['DQ_F']-v['D']);q=discount*(v['DQ']-v['DQ_F'])
    total=discount*(v['DQ']-v['F'])
    require(np.allclose(d+dr+q,total,rtol=1e-10,atol=1e-8),'Terminal additive identity failed')
    return dict(predictor_D=d,predictor_D_to_DQ=dr,critic_DQ=q,total_DQ_minus_F=total,
                reconstruction_error=d+dr+q-total)


def ranking_metrics(a,b,actions,top):
    x=ranks(a,b,top);sa=set(np.argsort(-a,kind='stable')[:top]);sb=set(np.argsort(-b,kind='stable')[:top])
    x['top_k_jaccard']=len(sa&sb)/len(sa|sb)
    x['top_action_l2']=float(np.linalg.norm(actions[int(np.argmax(a))]-actions[int(np.argmax(b))]))
    return x


def aggregate(frame, group, metrics):
    out=[]
    if frame.empty:return pd.DataFrame()
    for key,g in frame.groupby(group,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        identity=dict(zip(group,key))
        for metric in metrics:
            matrix=g.pivot(index='adaptation_seed',columns='evaluation_seed',values=metric).to_numpy(float)
            if not np.isfinite(matrix).all():continue
            if identity.get('method')=='F':
                require(np.allclose(matrix,matrix[:1]),'Frozen references unexpectedly vary between streams')
                matrix=matrix[:1]
            m,lo,hi=ci(matrix,(*key,metric))
            out.append(dict(**identity,metric=metric,mean=m,median=float(np.median(matrix.mean(1))),ci_low=lo,ci_high=hi,
                n_adaptation_streams=matrix.shape[0],n_evaluation_seeds=matrix.shape[1]))
    return pd.DataFrame(out)


def centered_components(reward, terminal, total):
    """Center each component within a state, retaining original score identity."""
    require(np.allclose(reward + terminal, total, rtol=2e-6, atol=2e-4), 'Score components do not reconstruct')
    return tuple(x - np.median(x, axis=-1, keepdims=True) for x in (reward, terminal, total))


def probability_window(support, probabilities, tail=.001):
    """Union of each head/method central 99.8% probability support, plus one bin."""
    cumulative=np.cumsum(probabilities.reshape(-1,len(support)),axis=-1)
    left=max(0,int((cumulative < tail).sum(-1).min())-1)
    right=min(len(support)-1,int((cumulative < 1-tail).sum(-1).max())+1)
    return float(support[left]),float(support[right])

# Public hash utility retained for provenance callers.
from experiments.nsgym.analysis.reacher_dq_figures import sha256
