"""Bounded model-only diagnostics. Never imports an environment or TDMPC2 agent."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from analyze_existing_mechanisms import rankdata


def state_hash(model):
    import torch
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


class Config(SimpleNamespace):
    def get(self,key,default=None):return getattr(self,key,default)


def run(a,src,F,c,checks,jchecks,fi):
    import torch
    # Import only network definitions; no agent, adapter, optimizer, environment.
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tdmpc2'))
    from common.world_model import WorldModel
    from common import math as tdmath
    root=a.results_root
    checkpoint=root/'nominal-server4-20261108-nohup-01/checkpoint-000750000'
    meta=src.json(checkpoint/'metadata.json');cfg_dict=meta['config'];cfg=Config(**cfg_dict)
    if cfg_dict!=fi['checkpoint_metadata']['config']:raise ValueError('Nominal configuration mismatch')
    nominal_path=src.track(checkpoint/'model.pt')
    if src.files[str(nominal_path)]['sha256']!=fi['file_sha256']['model.pt']:raise ValueError('Nominal hash mismatch')
    nominal=torch.load(nominal_path,map_location='cpu',weights_only=False)['model']
    if not any(k.startswith('_target_Qs_params.') for k in nominal):raise ValueError('Saved target Q state missing')
    if cfg.episodic or cfg.horizon!=3 or cfg.num_bins!=101 or cfg.vmin!=-10 or cfg.vmax!=10:
        raise ValueError('Unexpected HalfCheetah model contract')
    discount=min(max((cfg.episode_length/cfg.discount_denom-1)/(cfg.episode_length/cfg.discount_denom),cfg.discount_min),cfg.discount_max)
    reward_rows=[];qrows=[];drift=[];plans=[]
    frozen_bank=root/'physical-severity-750k-v1/frozen'
    bank_index={r['file']:r['sha256'] for r in src.json(frozen_bank/'dataset_index.json')}
    def model_from(state):
        model=WorldModel(cfg).to(a.device)
        model.load_state_dict(state,strict=True);model.eval();model.requires_grad_(False)
        return model
    def entropy(logits):
        logp=logits.log_softmax(-1);p=logp.exp()
        return -(p*logp).sum(-1),p[...,0]+p[...,-1]
    for key in sorted(checks):
        p,k,s=key
        name=f'{p}-k{k:g}'
        tp=src.track(root/f'dynamics-only-750k-v1/runs/fixed-decision/{name}-decision-seed{s}/snapshot-0003000.pt')
        jp=src.track(root/f'full-joint-750k-v1/runs-fixed/{name}-full-joint-seed{s}/snapshot-0003000.pt')
        if src.files[str(tp)]['sha256']!=checks[key]['snapshot_sha256'] or src.files[str(jp)]['sha256']!=jchecks[key]['snapshot_sha256']:
            raise ValueError('Final snapshot hash mismatch')
        ttt=torch.load(tp,map_location='cpu',weights_only=False)
        joint=torch.load(jp,map_location='cpu',weights_only=False)
        for payload in (ttt,joint):
            identity=payload['identity']
            if (identity['parameter'],float(identity['severity']),int(identity['adaptation_seed']))!=key or identity['schedule']!='fixed':
                raise ValueError('Snapshot identity mismatch')
            if payload['checkpoint_sha256']!=fi['file_sha256']['model.pt']:raise ValueError('Snapshot initial checkpoint mismatch')
        if ttt['configuration']!=c:raise ValueError('TTT configuration mismatch')
        if joint['configuration']!=cfg_dict or joint['transitions']!=3000 or joint['updates']!=2001:
            raise ValueError('Joint snapshot contract mismatch')
        if ttt['adapter']['counters'][1]!=3000 or ttt['adapter']['counters'][3]!=118:
            raise ValueError('TTT snapshot budget mismatch')
        if not any(x.startswith('_target_Qs_params.') for x in joint['model']):raise ValueError('Joint target Q absent')
        data=[]
        for e in c['evaluation_seeds'][:2]:
            filename=F[(p,k,e)]['data_file'];path=src.track(frozen_bank/filename)
            if src.files[str(path)]['sha256']!=bank_index[filename]:raise ValueError('Bank content changed')
            with np.load(path) as raw:
                ids=np.linspace(0,len(raw['actions'])-1,128,dtype=int)
                data.append((e,filename,ids,raw['observations'][ids],raw['actions'][ids],raw['observations'][ids+1],raw['rewards'][ids]))
                if len(data)==1:
                    starts=np.linspace(0,len(raw['actions'])-cfg.horizon,32,dtype=int)
                    tapes=np.stack([raw['actions'][i:i+cfg.horizon] for i in starts],axis=1)
                    initial=raw['observations'][0].copy()
        tensors=lambda x:torch.as_tensor(x,device=a.device,dtype=torch.float32)
        base_features={};base_diagnostics={}
        for method in ('frozen','ttt','joint'):
            model=model_from(joint['model'] if method=='joint' else nominal)
            if method=='ttt':model._dynamics.load_state_dict(ttt['adapter']['predictor'],strict=True)
            before=state_hash(model)
            with torch.inference_mode():
                for e,filename,ids,o,act,n,reward in data:
                    z=model.encode(tensors(o),None);zn=model.encode(tensors(n),None)
                    logits=model.reward(z,tensors(act),None);pred=tdmath.two_hot_inv(logits,cfg).flatten()
                    error=pred-tensors(reward)
                    qlogits=model.Q(z,tensors(act),None,return_type='all')
                    qs=tdmath.two_hot_inv(qlogits,cfg).squeeze(-1)
                    torch.manual_seed(20261112+e)
                    ap,info=model.pi(zn,None)
                    target=tensors(reward)+discount*model.Q(zn,ap,None,return_type='min',target=True).flatten()
                    residual=qs-target[None,:]
                    rh,re=entropy(logits);qh,qe=entropy(qlogits)
                    if not all(torch.isfinite(v).all() for v in (z,zn,error,qs,residual,rh,re,qh,qe)):
                        raise FloatingPointError('Nonfinite forward diagnostic')
                    common=dict(parameter=p,severity=k,adaptation_seed=s,evaluation_seed=e,method=method,
                                bank_file=filename,bank_sha256=bank_index[filename],samples=len(ids),
                                timestep_selection='128 evenly spaced executed transitions; common across models')
                    reward_rows.append(dict(**common,reward_mae=float(error.abs().mean()),reward_bias=float(error.mean()),
                        error_q05=float(torch.quantile(error,.05)),error_q50=float(torch.quantile(error,.5)),error_q95=float(torch.quantile(error,.95)),
                        reward_categorical_entropy=float(rh.mean()),reward_edge_probability=float(re.mean()),
                        within_model_latent_mse=float((model.next(z,tensors(act),None)-zn).square().mean())))
                    qrows.append(dict(**common,q_mean=float(qs.mean()),q_ensemble_std=float(qs.std(0,unbiased=False).mean()),
                        q_categorical_entropy=float(qh.mean()),q_edge_probability=float(qe.mean()),
                        bellman_abs_mean=float(residual.abs().mean()),bellman_bias=float(residual.mean()),
                        bootstrap_actor_entropy=float(info['entropy'].mean()),discount=discount,
                        terminal_mask='all zero: real nonterminal HalfCheetah; truncation bootstraps',
                        empirical_calibration='NOT_MEASURED',target_network='saved_target_Q',
                        residual_definition='Q_online - (reward + discount * min_of_two_saved_target_Q_heads)'))
                    if method=='frozen':
                        base_features[e]=z.detach().cpu().clone()
                        base_diagnostics[e]=(pred.detach().cpu().clone(),qs.detach().cpu().clone())
                    if method=='ttt':
                        torch.testing.assert_close(pred.cpu(),base_diagnostics[e][0],rtol=0,atol=0)
                        torch.testing.assert_close(qs.cpu(),base_diagnostics[e][1],rtol=0,atol=0)
                    enc=model._encoder.state_dict()
                    disp=sum(float((v.cpu()-nominal['_encoder.'+name]).square().sum()) for name,v in enc.items())
                    norm=sum(float(nominal['_encoder.'+name].square().sum()) for name in enc)
                    components={}
                    for component in ('_dynamics','_reward','_Qs','_pi'):
                        square=sum(float((v.cpu()-nominal[component+'.'+name]).square().sum())
                                   for name,v in getattr(model,component).state_dict().items() if torch.is_tensor(v))
                        components[component[1:]+'_parameter_l2']=square**.5
                    drift.append(dict(**common,encoder_parameter_l2=disp**.5,encoder_relative_l2=(disp/max(norm,1e-30))**.5,
                        **components,
                        feature_mse_vs_frozen=float((z.cpu()-base_features[e]).square().mean()),feature_mean=float(z.mean()),feature_std=float(z.std()),
                        interpretation='coordinate movement, not prediction accuracy or information loss'))
                torch.manual_seed(20261112+s)
                z=model.encode(tensors(initial[None]),None).repeat(32,1);total=torch.zeros(32,device=a.device)
                for h,acts in enumerate(tapes):
                    total+=discount**h*tdmath.two_hot_inv(model.reward(z,tensors(acts),None),cfg).flatten()
                    z=model.next(z,tensors(acts),None)
                action,_=model.pi(z,None)
                terminal=discount**cfg.horizon*model.Q(z,action,None,return_type='avg').flatten()
                scores=total+terminal;full_rank=rankdata(-scores.cpu().numpy());reward_rank=rankdata(-total.cpu().numpy())
                if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite planning score')
                for i in range(32):
                    plans.append(dict(parameter=p,severity=k,adaptation_seed=s,method=method,candidate=i,
                        observation_bank=data[0][1],initial_timestep=0,action_tape_start=int(starts[i]),
                        candidate_origin='Frozen executed action segments transplanted to common initial observation',
                        reward_sum=float(total[i]),discounted_terminal_q=float(terminal[i]),total_score=float(scores[i]),
                        predicted_rank=float(full_rank[i]),reward_only_rank=float(reward_rank[i]),
                        actual_ranking='ACTUAL_RANKING_NOT_MEASURED'))
            if state_hash(model)!=before:raise ValueError('Model state changed during inference')
            del model
        del joint,ttt
        print(f'FORWARD PASS {p} k={k:g} seed={s}',flush=True)
    return reward_rows,qrows,drift,plans
