"""Forward-only Q instrumentation; no agent/optimizer/environment construction."""
from copy import copy
from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np
from mechanism_forward import Config, state_hash
from analyze_existing_mechanisms import rankdata

CORE = Path(__file__).resolve().parents[3]/'tdmpc2'
sys.path.insert(0, str(CORE))


def require(ok, message):
    if not ok: raise ValueError(message)


def support(cfg):
    require(cfg.num_bins > 1, 'Categorical support required')
    return (cfg.num_bins, cfg.vmin, cfg.vmax)


def distributions(logits, cfg):
    import torch
    from common import math
    require(torch.isfinite(logits).all().item(), 'Nonfinite Q logits')
    require(logits.shape[-1] == support(cfg)[0], 'Support size mismatch')
    logp=logits.log_softmax(-1);p=logp.exp()
    scalar=math.two_hot_inv(logits,cfg).squeeze(-1)
    mix=p.mean(0); logmix=mix.clamp_min(1e-30).log()
    return dict(probability=p,scalar=scalar,entropy=-(p*logp).sum(-1),
        maximum=p.max(-1).values,boundary=p[...,:2].sum(-1)+p[...,-2:].sum(-1),
        ensemble_std=scalar.std(0,unbiased=False),
        interhead_js=(p*(logp-logmix)).sum(-1).mean(0),mixture=mix)


def js(p,q):
    require(p.shape==q.shape, 'Cannot compare different supports')
    mid=(p+q)/2
    return .5*((p*(p.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(-1)
              +(q*(q.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(-1))


def candidate_bank(horizon,count,action_dim,seed,std):
    # Analysis-only exogenous candidates: Gaussian first-iteration proposal,
    # no policy prior, warm start, future observations or executed action tapes.
    rng=np.random.default_rng(seed)
    return np.clip(rng.normal(0,std,(horizon,count,action_dim)),-1,1).astype('float32')


def component_hashes(model):
    return {name:state_hash(getattr(model,attr)) for name,attr in
        [('encoder','_encoder'),('dynamics','_dynamics'),('reward','_reward'),
         ('q','_Qs'),('actor','_pi'),('target_q','_target_Qs_params')]}


def compatible(models):
    hashes={m:component_hashes(v) for m,v in models.items()}
    for m,parts in [('D',('encoder','reward','q','actor','target_q')),
                    ('DQ',('encoder','reward','actor'))]:
        for part in parts:require(hashes['F'][part]==hashes[m][part],f'{m} changed fixed {part}')
    for m,parts in [('D',('dynamics',)),('DQ',('dynamics','q','target_q')),
                    ('J',('encoder','dynamics','reward','q','actor'))]:
        for part in parts:require(hashes['F'][part]!=hashes[m][part],f'{m} expected changed {part}')
    require(len({support(x.cfg) for x in models.values()})==1,'Cross-support JS unavailable')
    return hashes


class ReadOnlyHybrid:
    """Calls native modules; never mutates or registers swapped trainable state."""
    def __init__(self, dynamics_model, q_model):
        require(component_hashes(dynamics_model)['encoder']==component_hashes(q_model)['encoder'],
                'Hybrid encoder mismatch')
        self.native=dynamics_model;self.critic=q_model
    def __getattr__(self,name):return getattr(self.native,name)
    def Q(self,*args,**kwargs):return self.critic.Q(*args,**kwargs)


def score(model, obs, candidates, cfg, discount, seed):
    """Invoke unchanged TDMPC2._estimate_value, capturing native Q inputs/output."""
    import torch
    from tdmpc2 import TDMPC2
    device=obs.device
    require(not cfg.multitask and not cfg.episodic,'This analysis targets non-episodic single-task HalfCheetah')
    require(candidates.ndim==3 and candidates.shape[2]==cfg.action_dim,'Candidate dimensions')
    require(torch.isfinite(candidates).all().item() and candidates.abs().max().item()<=1,'Action bounds')
    local=copy(cfg);local.horizon=candidates.shape[0];local.num_samples=candidates.shape[1]
    capture={}
    class Instrument:
        def __getattr__(self,name):return getattr(model,name)
        def Q(self,z,a,task,**kwargs):
            capture['terminal_latent']=z.detach().clone();capture['terminal_action']=a.detach().clone()
            value=model.Q(z,a,task,**kwargs)
            capture['native_terminal_q']=value.detach().clone()
            capture['terminal_logits']=model.Q(z,a,task,return_type='all').detach().clone()
            return value
    trace=SimpleNamespace(value=lambda r,q:capture.update(reward=r.detach().clone(),terminal=q.detach().clone()))
    obj=SimpleNamespace(cfg=local,discount=discount,model=Instrument(),_mechanism_trace=trace)
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices),torch.inference_mode():
        torch.manual_seed(seed)
        z=model.encode(obs.reshape(1,-1),None).repeat(local.num_samples,1)
        total=TDMPC2._estimate_value(obj,z,candidates,None)
        torch.testing.assert_close(total,capture['reward']+capture['terminal'])
        require(torch.isfinite(total).all().item(),'Nonfinite planning score')
    capture['score']=total.flatten()
    # Exactly native elite exponent weighting on this fixed bank; not endogenous MPPI.
    elite=torch.topk(capture['score'],min(cfg.num_elites,local.num_samples)).indices
    weight=torch.softmax(cfg.temperature*(capture['score'][elite]-capture['score'][elite].max()),0)
    capture.update(elite=elite,weight=weight,ess=float(1/weight.square().sum()))
    return capture


def ranks(a,b,top):
    from scipy.stats import kendalltau
    a=np.asarray(a);b=np.asarray(b);ra=rankdata(a);rb=rankdata(b)
    spearman=None if ra.std()==0 or rb.std()==0 else float(np.corrcoef(ra,rb)[0,1])
    tau=float(kendalltau(a,b).statistic)
    return dict(spearman=spearman,kendall=tau if np.isfinite(tau) else None,
        top_k=top,top_k_overlap=len(set(np.argsort(-a)[:top])&set(np.argsort(-b)[:top]))/top,
        argmax_disagreement=int(a.argmax()!=b.argmax()))
