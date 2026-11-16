"""Migrated historical computations; numerical formulas preserved."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from experiments.nsgym.analysis.control_q.config import METHODS, PAIRS, TASKS
from q_planning_core import Config, distributions, candidate_bank, state_hash, score
from experiments.nsgym.analysis.control_q.metrics import decomposition, numpy, pack_metrics, ranking_metrics, read, require, stable_seed
from experiments.nsgym.analysis.control_q.models import load_models


def select_bank(banks, task, family, k, evals, horizon, states):
    obs=[];actions=[];tapes=[];terminal=[];initial=[];starts=[];imidx=[]
    for e in evals:
        p,_=banks[(task,family,k,e)]
        with np.load(p) as d:
            o=d['observations'];a=d['actions'];T=len(a)
            require(len(o)==T+1 and T== (1000 if task=='halfcheetah' else 50),'Unexpected bank episode shape')
            require(np.isfinite(o).all() and np.isfinite(a).all() and np.abs(a).max()<=1.000001,'Invalid bank values')
            ix=np.linspace(0,T-1,16,dtype=int)
            st=np.linspace(0,T-horizon-1,states,dtype=int)
            require(len(set(st))==states and np.all(st+horizon<T),'Terminal action out of bounds')
            obs.extend(o[ix]);actions.extend(a[ix]);imidx.extend(ix)
            initial.extend(o[st]);tapes.extend([a[s:s+horizon] for s in st]);terminal.extend(a[st+horizon]);starts.extend(st)
    return dict(obs=np.asarray(obs),action=np.asarray(actions),initial=np.asarray(initial),tapes=np.asarray(tapes),
                terminal_action=np.asarray(terminal),starts=np.asarray(starts),immediate_steps=np.asarray(imidx))


def infer_job(root, job, cfg_dict, banks, src, args, out):
    import torch
    from common import math as native_math
    cfg=Config(**cfg_dict);task=job['task'];family=job['family'];k=job['severity'];seed=job['adaptation_seed']
    identity={n:job[n] for n in ('task','family','severity','adaptation_seed')}
    gamma=min(max(1-cfg.discount_denom/cfg.episode_length,cfg.discount_min),cfg.discount_max)
    H=cfg.horizon;evals=TASKS[task]['evaluation'][:2] if args.smoke else TASKS[task]['evaluation']
    B=len(evals);S=args.states;C=args.candidates
    data=select_bank(banks,task,family,k,evals,H,S)
    models,hashes=load_models(root,job,cfg_dict,src,args.device)
    before={m:state_hash(model) for m,model in models.items()}
    tensor=lambda x:torch.as_tensor(x,device=args.device,dtype=torch.float32)
    packs={};terms={};arrays=dict(evaluation_seeds=np.asarray(evals),starts=data['starts'],
        immediate_steps=data['immediate_steps'],symlog_support=np.linspace(cfg.vmin,cfg.vmax,cfg.num_bins),
        initial_observations=data['initial'],executed_tapes=data['tapes'],terminal_actions=data['terminal_action'])
    with torch.inference_mode():
        for m,model in models.items():
            packs[m]=numpy(distributions(model.Q(model.encode(tensor(data['obs']),None),tensor(data['action']),None,return_type='all'),cfg))
            z=model.encode(tensor(data['initial']),None)
            for h in range(H):z=model.next(z,tensor(data['tapes'][:,h]),None)
            arrays[m+'_tape_terminal_latent']=z.cpu().numpy()
            terms[m]=numpy(distributions(model.Q(z,tensor(data['terminal_action']),None,return_type='all'),cfg))
            if m=='DQ':terms['DQ_F']=numpy(distributions(models['F'].Q(z,tensor(data['terminal_action']),None,return_type='all'),cfg))
        for key in ('probability','scalar'):
            require(np.allclose(packs['F'][key],packs['D'][key],rtol=0,atol=1e-7),'F=D immediate invariant failed')
        decoder_error=0.
        for kind,values in [('immediate',packs),('terminal',terms)]:
            for m,pack in values.items():
                p=pack['probability'];require(np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-6),'Invalid probabilities')
                # Independent reconstruction in symlog space, head by head.
                expected=np.sign((p*arrays['symlog_support']).sum(-1))*np.expm1(np.abs((p*arrays['symlog_support']).sum(-1)))
                require(np.allclose(expected,pack['scalar'],rtol=3e-5,atol=1e-5),'Native per-head decoder mismatch')
                decoder_error=max(decoder_error,float(np.max(np.abs(expected-pack['scalar']))))
                for key,val in pack.items():arrays[f'{m}_{kind}_{key}']=val
        rows={'immediate':[],'terminal':[],'effects':[],'rankings':[],'ranking_summary':[],'components':[]}
        for kind,values,n in [('immediate',packs,16),('terminal',terms,S)]:
            for m,pack in values.items():
                metrics=pack_metrics(pack,values['F'])
                for j,e in enumerate(evals):
                    rows[kind].append(dict(**identity,evaluation_seed=e,method=m,samples=n,horizon=H,discount=gamma,
                        **{name:float(x.reshape(B,n)[j].mean()) for name,x in metrics.items()}))
        effects=decomposition({m:p['scalar'][:2].mean(0) for m,p in terms.items()},gamma**H)
        for name,val in effects.items():arrays['effect_'+name]=val
        for j,e in enumerate(evals):
            rows['effects'].append(dict(**identity,evaluation_seed=e,samples=S,head_pair='0,1',discount_power=gamma**H,
                **{name:float(val.reshape(B,S)[j].mean()) for name,val in effects.items()}))
        # Generated exogenous first-iteration-style candidates, never historical MPPI.
        candidates=np.stack([candidate_bank(H,C,cfg.action_dim,stable_seed(task,family,k,e,int(st),'candidates'),cfg.max_std).transpose(1,0,2)
                            for e,st in zip(np.repeat(evals,S),data['starts'])])
        candidate_hash=hashlib.sha256(candidates.tobytes()).hexdigest()
        arrays['candidates']=candidates;arrays['candidate_ids']=np.arange(C)
        acts=tensor(candidates.reshape(B*S*C,H,cfg.action_dim));terminal=tensor(np.repeat(data['terminal_action'],C,axis=0))
        initial=tensor(np.repeat(data['initial'],C,axis=0));scores={};rewards={};terminal_scores={};native={}
        for m,model in models.items():
            z=model.encode(initial,None);reward=torch.zeros(B*S*C,device=args.device)
            for h in range(H):
                reward+=gamma**h*native_math.two_hot_inv(model.reward(z,acts[:,h],None),cfg).flatten()
                z=model.next(z,acts[:,h],None)
            q=native_math.two_hot_inv(model.Q(z,terminal,None,return_type='all'),cfg).squeeze(-1)[:2].mean(0)
            rewards[m]=reward.cpu().numpy().reshape(B*S,C);terminal_scores[m]=(gamma**H*q).cpu().numpy().reshape(B*S,C)
            scores[m]=rewards[m]+terminal_scores[m]
            if m=='DQ':
                swap=native_math.two_hot_inv(models['F'].Q(z,terminal,None,return_type='all'),cfg).squeeze(-1)[:2].mean(0)
                rewards['DQ_F']=rewards[m];terminal_scores['DQ_F']=(gamma**H*swap).cpu().numpy().reshape(B*S,C)
                scores['DQ_F']=rewards['DQ_F']+terminal_scores['DQ_F']
            native[m]=[]
            for rep in range(args.sensitivity_repeats):
                devices=[torch.device(args.device).index or 0] if args.device.startswith('cuda') else []
                with torch.random.fork_rng(devices=devices):
                    values=[];terminal_actions=[]
                    # Native planner samples one pair per planning state/call,
                    # not one pair shared over the entire observation bank.
                    for ix,(e,st) in enumerate(zip(np.repeat(evals,S),data['starts'])):
                        torch.manual_seed(stable_seed(task,family,k,int(e),int(st),'native',rep))
                        sl=slice(ix*C,(ix+1)*C)
                        action,_=model.pi(z[sl],None)
                        value=model.Q(z[sl],action,None,return_type='avg').flatten()
                        values.append((reward[sl]+gamma**H*value).cpu().numpy())
                        terminal_actions.append(action.cpu().numpy())
                    native[m].append(np.asarray(values))
                    arrays[f'{m}_native_terminal_action_{rep}']=np.asarray(terminal_actions)
            # Independently compare first state to unchanged native _estimate_value.
            native_seed=stable_seed(task,family,k,'native-validation')
            captured=score(model,tensor(data['initial'][0]),tensor(candidates[0].transpose(1,0,2)),cfg,gamma,native_seed)
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(native_seed)
                ap,_=model.pi(z[:C],None)
                expected=reward[:C]+gamma**H*model.Q(z[:C],ap,None,return_type='avg').flatten()
            torch.testing.assert_close(expected,captured['score'],rtol=2e-5,atol=2e-4)
            for name,value in [('scores',scores[m]),('rewards',rewards[m]),('terminal_scores',terminal_scores[m]),('native_scores',np.asarray(native[m]))]:
                arrays[m+'_'+name]=value
        for name,value in [('scores',scores['DQ_F']),('rewards',rewards['DQ_F']),('terminal_scores',terminal_scores['DQ_F'])]: arrays['DQ_F_'+name]=value
        require(hashlib.sha256(candidates.tobytes()).hexdigest()==candidate_hash,'Candidate input arrays changed')
        arrays['candidate_input_sha256']=np.asarray(candidate_hash)
        top=min(5,C)
        for m,values in scores.items():
            for j,e in enumerate(evals):
                sl=slice(j*S,(j+1)*S);best=values[sl].argmax(-1);ii=np.arange(S)+j*S
                rows['components'].append(dict(**identity,evaluation_seed=e,method=m,
                    selected_reward=float(rewards[m][ii,best].mean()),selected_terminal=float(terminal_scores[m][ii,best].mean()),
                    mean_candidate_reward=float(rewards[m][sl].mean()),mean_candidate_terminal=float(terminal_scores[m][sl].mean()),
                    margin=float(np.mean(np.sort(values[sl],axis=-1)[:,-1]-np.sort(values[sl],axis=-1)[:,-2]))))
                for s in range(S):
                    ix=j*S+s;order=np.argsort(-values[ix],kind='stable');winner=int(order[0])
                    rows['rankings'].append(dict(**identity,evaluation_seed=e,timestep=int(data['starts'][ix]),method=m,
                        protocol='controlled_fixed_action_pair_0_1',top1=winner,top_k=json.dumps(order[:top].tolist()),
                        top1_first_action=json.dumps(candidates[ix,winner,0].tolist()),reward_sum=float(rewards[m][ix,winner]),
                        terminal_contribution=float(terminal_scores[m][ix,winner]),score=float(values[ix,winner]),
                        margin=float(values[ix,order[0]]-values[ix,order[1]])))
        for protocol,all_scores,pairs in [('controlled', {m:[s] for m,s in scores.items()},PAIRS),('native_terminal_sensitivity',native,PAIRS[:4])]:
            for a,b in pairs:
                for j,e in enumerate(evals):
                    metrics=[]
                    for aa,bb in zip(all_scores[a],all_scores[b]):
                        for ix in range(j*S,(j+1)*S):metrics.append(ranking_metrics(aa[ix],bb[ix],candidates[ix,:,0],top))
                    frame=pd.DataFrame(metrics)
                    rows['ranking_summary'].append(dict(**identity,evaluation_seed=e,comparison=f'{a} → {b}',protocol=protocol,
                        states=S,repeats=len(all_scores[a]),undefined_spearman=int(frame.spearman.isna().sum()),
                        **{col:float(frame[col].mean()) for col in frame.columns}))
        for name,val in arrays.items():
            if np.asarray(val).dtype.kind in 'fc':require(np.isfinite(val).all(),f'Nonfinite array {name}')
        require(all(before[m]==state_hash(model) for m,model in models.items()),'Inference changed model state')
    stem=f'{task}-{family}-k{k:g}-seed{seed}'
    np.savez_compressed(out/'q_arrays'/f'{stem}.npz',**arrays)
    validation=dict(**identity,status='PASS',F_D_immediate_invariant=True,shared_encoder=True,probability_normalization=True,
        native_decoder=True,max_native_decoder_roundoff=decoder_error,
        additive_max_error=float(np.abs(effects['reconstruction_error']).max()),candidate_identity=True,
        native_estimate_value_parity=True,model_hashes_unchanged=True,component_hashes=hashes,
        terminal_indexing=True,finite_arrays=True,environment_transitions=0,optimizer_steps=0)
    return rows,validation


def cached_immediate(root, job, cfg, src, args, out):
    """Restricted archive fallback; never label a one-state terminal example as a population."""
    import torch
    from common import math as native_math
    task=job['task'];family=job['family'];k=job['severity'];seed=job['adaptation_seed']
    evals=TASKS[task]['evaluation'][:2] if args.smoke else TASKS[task]['evaluation']
    cached={m:[] for m in METHODS};steps=[]
    if task=='reacher':
        cache=root/'unified-analysis/control-prediction-q-corrected-v3'
        require(read(src.track(cache/'q_inference/completion.json','cache provenance'))['status']=='PASS','Unverified Reacher Q cache')
        require(read(src.track(cache/'manifest.json','cache provenance'))['checkpoint_sha256']==TASKS[task]['sha'],'Cache nominal origin')
        p=src.track(cache/f'q_inference/{family}-k{k:g}-seed{seed}.npz','cache immediate')
        with np.load(p) as data:
            require(np.array_equal(data['evaluation_seed'],np.repeat(TASKS[task]['evaluation'],16)),'Cache evaluation identities')
            require(np.allclose(data['symlog_support'],np.linspace(cfg['vmin'],cfg['vmax'],cfg['num_bins'])),'Cache support')
            steps=data['timestep'][:len(evals)*16]
            for m in METHODS:cached[m]=data[m+'_probability'][:,:len(evals)*16].copy()
    else:
        cache=root/'q-planning-mechanism-750k-v1'
        require(read(src.track(cache/'provenance/immediate.json','cache provenance'))['status']=='PASS','Unverified HalfCheetah Q cache')
        for e in evals:
            p=src.track(cache/f'immediate/{TASKS[task]["families"][family]}-k{k:g}-seed{seed}-eval{e}.npz','cache immediate')
            with np.load(p) as data:
                require(np.allclose(data['symlog_support'],np.linspace(cfg['vmin'],cfg['vmax'],cfg['num_bins'])),'Cache support')
                require(len(data['timestep'])==16,'Unexpected cache state count')
                steps.extend(data['timestep'])
                for m in METHODS:cached[m].append(data[m+'_probability'].copy())
        cached={m:np.concatenate(v,axis=1) for m,v in cached.items()}
    packs={};arrays={'symlog_support':np.linspace(cfg['vmin'],cfg['vmax'],cfg['num_bins']),
                    'evaluation_seeds':np.asarray(evals),'immediate_steps':np.asarray(steps)}
    for m,p in cached.items():
        require(np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-6),'Invalid cached probabilities')
        q=native_math.two_hot_inv(torch.as_tensor(np.log(np.maximum(p,1e-30))),Config(**cfg)).squeeze(-1).numpy()
        packs[m]=dict(probability=p,scalar=q,entropy=-(p*np.log(np.maximum(p,1e-30))).sum(-1))
        for n,v in packs[m].items():arrays[f'{m}_immediate_{n}']=v
    require(np.allclose(cached['F'],cached['D'],rtol=0,atol=1e-7),'Cached invariant failed')
    result=[];identity={n:job[n] for n in ('task','family','severity','adaptation_seed')}
    for m,p in packs.items():
        metrics=pack_metrics(p,packs['F'])
        for j,e in enumerate(evals):result.append(dict(**identity,evaluation_seed=e,method=m,samples=16,source='verified_existing_probability_archive',
            **{n:float(v.reshape(len(evals),16)[j].mean()) for n,v in metrics.items()}))
    np.savez_compressed(out/'q_arrays'/f'{task}-{family}-k{k:g}-seed{seed}.npz',**arrays)
    return result
