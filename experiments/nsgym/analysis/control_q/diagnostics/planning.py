"""Migrated historical computations; numerical formulas preserved."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
import pandas as pd
from experiments.nsgym.analysis.control_q.config import METHODS, PAIRS, SCRIPTS, TASKS
from q_planning_core import Config, distributions, state_hash, score
from experiments.nsgym.analysis.control_q.metrics import centered_components, pack_metrics, ranking_metrics, read, require, stable_seed, write
from experiments.nsgym.analysis.control_q.models import Sources, inventory, load_models
from experiments.nsgym.analysis.control_q.metrics import aggregate


class RandomTape:
    """Replay actual native random draws, checking call type, shape and ordering."""
    def __init__(self, events=None):
        self.events=[] if events is None else events
        self.replay=events is not None;self.index=0
    def __enter__(self):
        import torch
        from common import math
        self.stack=ExitStack()
        for name in ('randn','randn_like','randperm'):
            original=getattr(torch,name)
            def call(*args,_name=name,_original=original,**kwargs):
                # Draw normally as well to preserve native global RNG evolution.
                value=_original(*args,**kwargs)
                if self.replay:
                    event=self.events[self.index]
                    require(event['kind']==_name and event['value'].shape==value.shape,'Random tape protocol mismatch')
                    value=event['value'].clone().to(value.device)
                else:self.events.append(dict(kind=_name,value=value.detach().clone()))
                self.index+=1
                return value
            self.stack.enter_context(patch.object(torch,name,call))
        original=math.gumbel_softmax_sample
        def gumbel(p,temperature=1.,dim=0):
            if self.replay:
                event=self.events[self.index];require(event['kind']=='gumbel' and tuple(p.shape)==event['shape'],'Gumbel replay protocol')
                torch.set_rng_state(event['rng'])
            else:self.events.append(dict(kind='gumbel',shape=tuple(p.shape),rng=torch.get_rng_state().clone()))
            self.index+=1
            return original(p,temperature,dim)
        self.stack.enter_context(patch.object(math,'gumbel_softmax_sample',gumbel))
        return self
    def __exit__(self,*args):
        self.stack.close()
        if args[0] is None and self.replay:require(self.index==len(self.events),'Unused random tape events')


class PlanTrace:
    def __init__(self,cfg,warm,t0):
        import torch
        self.cfg=cfg;self.rows=[];self.last={}
        self.mean=torch.zeros(cfg.horizon,cfg.action_dim)
        self.std=torch.full_like(self.mean,cfg.max_std)
        if not t0:self.mean[:-1]=warm[1:]
    def encoded(self,z):self.z=z.detach().clone()
    def value(self,reward,terminal):self.last.update(reward=reward.detach().clone(),terminal=terminal.detach().clone())
    def iteration(self,actions,values,elite,npi):
        import torch
        score=torch.exp(self.cfg.temperature*(values[elite]-values[elite].max(0).values));score=score/score.sum(0)
        mean=(score.unsqueeze(0)*actions[:,elite]).sum(1)/(score.sum(0)+1e-9)
        std=((score.unsqueeze(0)*(actions[:,elite]-mean.unsqueeze(1))**2).sum(1)/(score.sum(0)+1e-9)).sqrt().clamp(self.cfg.min_std,self.cfg.max_std)
        row=dict(mean_before=self.mean.clone(),std_before=self.std.clone(),candidates=actions.detach().clone(),
                 policy_prior=actions[:,:npi].detach().clone(),gaussian_candidates=actions[:,npi:].detach().clone(),
                 score=values.detach().clone(),elite=elite.detach().clone(),weight=score.detach().clone(),mean=mean,std=std,**self.last)
        self.rows.append(row);self.mean=mean;self.std=std;self.last={}
    def selected(self,idx,action):self.selected_id=idx.detach().clone();self.action=action.detach().clone()


def native_plan(model,cfg,gamma,obs,warm,t0,seed,tape=None,instrument=False):
    import torch
    from tdmpc2 import TDMPC2
    trace=PlanTrace(cfg,warm,t0) if instrument else None
    class Instrument:
        def __getattr__(self,name):return getattr(model,name)
        def Q(self,z,a,task,**kwargs):
            result=model.Q(z,a,task,**kwargs)
            trace.last.update(terminal_action=a.detach().clone(),terminal_q=result.detach().clone(),
                              head_pair=randomness.events[randomness.index-1]['value'][:2].clone())
            return result
    obj=SimpleNamespace(cfg=cfg,device=torch.device('cpu'),model=Instrument() if instrument else model,
                        discount=gamma,_prev_mean=warm.clone(),_mechanism_trace=trace)
    obj._estimate_value=lambda z,a,t:TDMPC2._estimate_value(obj,z,a,t)
    with torch.inference_mode(),torch.random.fork_rng():
        torch.manual_seed(seed)
        with RandomTape(tape) as randomness:
            action=TDMPC2._plan(obj,obs[None],t0=t0,eval_mode=True,task=None)
    if trace:
        torch.testing.assert_close(trace.mean,obj._prev_mean,rtol=0,atol=0)
        torch.testing.assert_close(trace.action,action,rtol=0,atol=0)
    return action,obj._prev_mean,trace,randomness.events


def action_metrics(a,b,actions):
    r=ranking_metrics(a,b,actions,min(5,len(a)))
    delta=b-a;ca=a-np.median(a);cb=b-np.median(b)
    corr=float(np.corrcoef(ca,cb)[0,1]) if ca.std()>0 and cb.std()>0 else None
    return dict(**r,mean_shift=float(delta.mean()),mean_absolute_shift=float(np.abs(delta).mean()),
                action_dependent_sd=float(delta.std()),centered_pearson=corr,
                max_margin=float(np.ptp(b)),best_second_gap=float(np.sort(b)[-1]-np.sort(b)[-2]))


def shared_score(model,cfg,gamma,obs,row):
    import torch
    from common import math
    acts=row['candidates'];z=model.encode(obs[None],None).repeat(cfg.num_samples,1)
    reward=torch.zeros(cfg.num_samples,1)
    for h in range(cfg.horizon):
        reward+=gamma**h*math.two_hot_inv(model.reward(z,acts[h],None),cfg)
        z=model.next(z,acts[h],None)
    q=math.two_hot_inv(model.Q(z,row['terminal_action'],None,return_type='all'),cfg)[row['head_pair']].sum(0)/2
    return reward,gamma**cfg.horizon*q


def run(root,source,out,presentation_source=None):
    import torch
    from q_planning_core import ReadOnlyHybrid
    torch.set_num_threads(2);started=time.monotonic();src=Sources()
    require(not out.exists() and not out.with_suffix('.zip').exists(),'Output must be new')
    out.mkdir(parents=True)
    for folder in ['reports','tables','arrays','figures/main','figures/supplementary']:(out/folder).mkdir(parents=True)
    src.track(Path(__file__),'implementation')
    from experiments.nsgym.analysis.control_q.models import track_implementation
    track_implementation(src)
    done=read(src.track(source/'completion.json','v1 completion'))
    validation=read(src.track(source/'NUMERICAL_VALIDATION.json','v1 validation'))
    require(done['status']=='PASS' and validation['status']=='PASS','v1 incomplete')
    if presentation_source is None:
        matches=[p.parent for p in root.rglob('completion.json') if p.parent.name=='q-terminal-planning-presentation-v2']
        require(len(matches)==1,'Locate one completed v2 presentation source')
        v2=matches[0]
    else:
        v2=Path(presentation_source).resolve()
    require(read(src.track(v2/'completion.json','v2 completion'))['status']=='PASS','v2 incomplete')
    require(read(src.track(v2/'NUMERICAL_VALIDATION.json','v2 validation'))['status']=='PASS','v2 invalid')
    previous=read(src.track(source/'SOURCE_INPUTS.json','v1 provenance'))
    jobs,configs,banks,gaps=inventory(root,src,False)
    require(len(jobs)==108 and all(j['valid'] for j in jobs) and not gaps,'Incomplete source coverage')
    for p,record in src.files.items():
        if p in previous and Path(p).suffix!='.py':require(record['sha256']==previous[p]['sha256'],f'Changed v1 source {p}')
    for p in [SCRIPTS/'q_planning_core.py',SCRIPTS/'mechanism_forward.py',SCRIPTS.parents[2]/'tdmpc2/tdmpc2.py',SCRIPTS.parents[2]/'tdmpc2/common/world_model.py',SCRIPTS.parents[2]/'tdmpc2/common/math.py']:src.track(p,'native reference code')
    array_files={}
    for job in jobs:
        stem=f"{job['task']}-{job['family']}-k{job['severity']:g}-seed{job['adaptation_seed']}"
        array_files[stem]=src.track(source/'q_arrays'/(stem+'.npz'),'saved v1 arrays')
    report='''# Phase 0 source audit

All 108 streams have complete v1 inference and numerical validation. V2 canonical return comparison is complete. Nominal checkpoint, adaptation snapshot, manifests, bank indices and seed splits are validated by the existing inventory/loader. Shared sources must match v1 hashes.

- Saved arrays suffice: matched-behavior immediate categorical/scalar changes; terminal action tapes; 32-candidate terminal Q and score swaps; ranking and return links. These model calls are not repeated.
- New checkpoint inference required: immediate multi-action Q on the same 80 saved starting observations and 32 existing candidate first actions; native MPPI output/intermediate traces.
- New simulator interaction required for true candidate return, regret, Monte Carlo calibration or causal control attribution. None is performed here.

Native planning protocol is predeclared: all 36 condition labels and 3 adaptation seeds, first two held-out evaluation seeds only. The first seed uses the first saved start with t0=True (cold). The second seed uses the second saved start with t0=False and the same saved Frozen tape as an imposed initial previous mean for every model. This is a controlled warm initialization, not a historical MPPI cache. One replayed randomness tape per state; eval_mode=True (no exploration noise), native stochastic final elite selection retained. Native sample/elite/prior/iteration/horizon configuration is read separately for each task. This subset is not claimed to cover all 20 evaluation seeds; fixed-candidate analyses retain all 20.

Immediate multi-action Q uses all 20 evaluation seeds × 4 states × 32 existing first actions and all decoded heads. Native MPPI uses real learned policy priors, native stochastic terminal actions and native random head pairs. Fixed v1 scoring uses common terminal actions and heads 0,1. The protocols remain separate.
'''
    (out/'reports/PHASE0_SOURCE_AUDIT.md').write_text(report);(out/'reports/SOURCE_AUDIT.md').write_text(report)
    write(out/'protocol.json',dict(native_evaluation_seed_count=2,states_per_native_seed=1,native_indices=[0,5],
        action_distance_thresholds=[.01,.05,.1],native_configurations=configs,source=str(source),mode='CPU native unchanged _plan, eval_mode=True'))
    matched=[];conditional=[];planning=[];native=[];shared=[];checks=[];examples={};failures=[]
    for job in jobs:
        task=job['task'];family=job['family'];k=job['severity'];seed=job['adaptation_seed']
        identity={n:job[n] for n in ['task','family','severity','adaptation_seed']}
        stem=f'{task}-{family}-k{k:g}-seed{seed}'
        try:
            cfg=Config(**configs[task]);gamma=min(max(1-cfg.discount_denom/cfg.episode_length,cfg.discount_min),cfg.discount_max)
            a=np.load(array_files[stem]);arrays={};max_identity=0.
            require(a['candidates'].shape[:3]==(80,32,cfg.horizon),'Candidate shape')
            require(hashlib.sha256(a['candidates'].tobytes()).hexdigest()==str(a['candidate_input_sha256']),'Candidate hash')
            require(np.array_equal(a['F_immediate_probability'],a['D_immediate_probability']) and np.array_equal(a['F_immediate_scalar'],a['D_immediate_scalar']),'F=D immediate')
            for m in METHODS:
                pack={name:a[f'{m}_immediate_{name}'] for name in ['probability','scalar','entropy']}
                f={name:a[f'F_immediate_{name}'] for name in pack}
                p=pack['probability'];support=a['symlog_support'];expected=np.sign((p*support).sum(-1))*np.expm1(np.abs((p*support).sum(-1)))
                require(np.allclose(p.sum(-1),1,atol=1e-6) and np.allclose(expected,pack['scalar'],rtol=3e-5,atol=1e-5),'Saved decoder/probability parity')
                metrics=pack_metrics(pack,f)
                for j,e in enumerate(a['evaluation_seeds']):matched.append(dict(**identity,method=m,evaluation_seed=int(e),**{name:float(v.reshape(20,16)[j].mean()) for name,v in metrics.items()}))
            for phase,suffix in [('terminal_action_value','terminal_scores'),('fixed_total_score','scores')]:
                for left,right in PAIRS:
                    aa=a[left+'_'+suffix];bb=a[right+'_'+suffix]
                    if phase=='terminal_action_value':aa=aa/gamma**cfg.horizon;bb=bb/gamma**cfg.horizon
                    for i in range(80):
                        r=dict(**identity,evaluation_seed=int(a['evaluation_seeds'][i//4]),timestep=int(a['starts'][i]),phase=phase,comparison=f'{left} → {right}',**action_metrics(aa[i],bb[i],a['candidates'][i,:,0]))
                        (conditional if phase=='terminal_action_value' else planning).append(r)
            err=np.max(np.abs((a['DQ_scores']-a['DQ_F_scores'])-(a['DQ_terminal_scores']-a['DQ_F_terminal_scores'])))
            require(err<1e-3 and np.array_equal(a['DQ_rewards'],a['DQ_F_rewards']),'Fixed Q swap identity')
            max_identity=float(err)
            for m in [*METHODS,'DQ_F']:centered_components(a[m+'_rewards'],a[m+'_terminal_scores'],a[m+'_scores'])
            require(np.max(np.abs(a['effect_predictor_D']+a['effect_predictor_D_to_DQ']+a['effect_critic_DQ']-a['effect_total_DQ_minus_F']))<1e-8,'Terminal additive identity')
            models,hashes=load_models(root,job,configs[task],src,'cpu');before={m:state_hash(v) for m,v in models.items()}
            models['DQ_F']=ReadOnlyHybrid(models['DQ'],models['F'])
            obs=torch.as_tensor(a['initial_observations'],dtype=torch.float32);actions=torch.as_tensor(a['candidates'][:,:,0],dtype=torch.float32)
            with torch.inference_mode():
                for m in METHODS:
                    z=models[m].encode(obs,None).repeat_interleave(32,0)
                    pack=distributions(models[m].Q(z,actions.reshape(-1,cfg.action_dim),None,return_type='all'),cfg)
                    arrays[m+'_action_q']=pack['scalar'].numpy().reshape(cfg.num_q,80,32)
                    arrays[m+'_action_probability_example']=pack['probability'].numpy().reshape(cfg.num_q,80,32,cfg.num_bins)[:,0]
                    require(np.allclose(pack['probability'].sum(-1).numpy(),1,atol=1e-6),'New probability normalization')
                require(np.array_equal(arrays['F_action_q'],arrays['D_action_q']),'Multi-action immediate invariant')
                for left,right in PAIRS[:4]:
                    aa=arrays[left+'_action_q'].mean(0);bb=arrays[right+'_action_q'].mean(0)
                    for i in range(80):conditional.append(dict(**identity,evaluation_seed=int(a['evaluation_seeds'][i//4]),timestep=int(a['starts'][i]),phase='immediate_action_value',comparison=f'{left} → {right}',**action_metrics(aa[i],bb[i],a['candidates'][i,:,0])))
                for case,ix in enumerate([0,5]):
                    e=int(a['evaluation_seeds'][ix//4]);warm=torch.as_tensor(a['executed_tapes'][ix],dtype=torch.float32).clone();t0=case==0
                    if t0:warm.zero_()
                    rngseed=stable_seed(task,family,k,e,int(a['starts'][ix]),'native-replay')
                    tape=None;traces={};finals={}
                    for m in ['F','D','DQ_F','DQ','J']:
                        reference,refmean,_,events=native_plan(models[m],cfg,gamma,obs[ix],warm,t0,rngseed,tape,False)
                        if tape is None:tape=events
                        result,mean,trace,_=native_plan(models[m],cfg,gamma,obs[ix],warm,t0,rngseed,tape,True)
                        torch.testing.assert_close(result,reference,rtol=0,atol=0);torch.testing.assert_close(mean,refmean,rtol=0,atol=0)
                        traces[m]=trace;finals[m]=result
                        for name in trace.rows[0]:arrays[f'case{case}_{m}_{name}']=np.stack([r[name].numpy() for r in trace.rows])
                        arrays[f'case{case}_{m}_action']=result.numpy()
                    for n,event in enumerate(tape):arrays[f'case{case}_random_{n}_{event["kind"]}']=event.get('value',event.get('rng')).numpy()
                    for left,right in PAIRS:
                        for it,(l,r) in enumerate(zip(traces[left].rows,traces[right].rows)):
                            delta=(finals[right]-finals[left]).numpy();distance=float(np.linalg.norm(delta))
                            native.append(dict(**identity,evaluation_seed=e,timestep=int(a['starts'][ix]),initialization='cold' if t0 else 'imposed_warm',comparison=f'{left} → {right}',iteration=it,
                                mean_l2=float(torch.norm(l['mean']-r['mean'])),std_l2=float(torch.norm(l['std']-r['std'])),
                                elite_overlap=len(set(l['elite'].tolist())&set(r['elite'].tolist()))/cfg.num_elites,
                                top1_id_disagreement=int(l['score'].argmax()!=r['score'].argmax()),
                                candidates_diverged=int(not torch.equal(l['candidates'],r['candidates'])),candidate_l2=float(torch.norm(l['candidates']-r['candidates'])),
                                final_action_l2=distance,action_abs_by_dimension=json.dumps(np.abs(delta).tolist()),
                                above_001=float(distance>.01),above_005=float(distance>.05),above_010=float(distance>.1)))
                    # Shared proposals/terminal actions/head pair explicitly replay the DQ planner's cached tensors.
                    for it,row in enumerate(traces['DQ'].rows):
                        rd,qd=shared_score(models['DQ'],cfg,gamma,obs[ix],row)
                        rf,qf=shared_score(models['DQ_F'],cfg,gamma,obs[ix],row)
                        torch.testing.assert_close(rd,row['reward'],rtol=2e-5,atol=2e-4);torch.testing.assert_close(qd,row['terminal'],rtol=2e-5,atol=2e-4)
                        torch.testing.assert_close(rd,rf,rtol=0,atol=0)
                        aa=(rf+qf).flatten().numpy();bb=(rd+qd).flatten().numpy()
                        arrays[f'case{case}_shared_{it}_Fcritic']=aa;arrays[f'case{case}_shared_{it}_DQcritic']=bb
                        shared.append(dict(**identity,evaluation_seed=e,timestep=int(a['starts'][ix]),iteration=it,
                            **action_metrics(aa,bb,row['candidates'][0].numpy())))
                    # Iteration 0 proposals and learned prior must coincide in the controlled Q swap.
                    torch.testing.assert_close(traces['DQ_F'].rows[0]['candidates'],traces['DQ'].rows[0]['candidates'],rtol=0,atol=0)
                    for l,r in zip(traces['DQ_F'].rows,traces['DQ'].rows):torch.testing.assert_close(l['policy_prior'],r['policy_prior'],rtol=0,atol=0)
                require(all(state_hash(models[m])==h for m,h in before.items()),'Model state modified')
            for key,value in arrays.items():
                if np.asarray(value).dtype.kind in 'fc':require(np.isfinite(value).all(),f'Nonfinite trace {key}')
            np.savez_compressed(out/'arrays'/(stem+'.npz'),**arrays)
            if k==2 and seed==TASKS[task]['adaptation'][0]:examples[(task,family)]=(dict(a),arrays)
            a.close();checks.append(dict(**identity,status='PASS',native_parity=True,fixed_q_swap_max_error=max_identity,model_hashes_unchanged=True))
            print('PASS '+stem,flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc();failures.append(dict(**identity,error=str(exc)));break
    frames={name:pd.DataFrame(rows) for name,rows in [('matched',matched),('action',conditional),('fixed',planning),('native',native),('shared_native',shared)]}
    for name,df in frames.items():df.to_csv(out/'tables'/f'{name}_per_state.csv',index=False)
    if failures:
        write(out/'completion.json',dict(status='PARTIAL',completed=len(checks),failures=failures))
        write(out/'NUMERICAL_VALIDATION.json',dict(status='PARTIAL',checks=checks,failures=failures))
        raise RuntimeError('Planner diagnostic failed; PARTIAL retained')
    # Preserve the numerical stage without invoking any historical plot suite.
    common=['task','family','severity']
    specs={'matched':(['method'],['scalar_q','head_std','entropy','js_frozen','signed_delta','absolute_delta']),
           'action':(['phase','comparison'],['mean_shift','mean_absolute_shift','action_dependent_sd','spearman','argmax_disagreement']),
           'fixed':(['phase','comparison'],['mean_shift','action_dependent_sd','spearman','argmax_disagreement']),
           'native':(['comparison','iteration'],['mean_l2','std_l2','elite_overlap','top1_id_disagreement','candidate_l2','final_action_l2']),
           'shared_native':(['iteration'],['mean_shift','action_dependent_sd','spearman','argmax_disagreement'])}
    for name,(extra,metrics) in specs.items():
        frame=frames[name]
        group=common+extra+['adaptation_seed','evaluation_seed']
        values=frame.groupby(group,dropna=False)[metrics].mean().reset_index()
        aggregate(values,common+extra,metrics).to_csv(out/'tables'/f'{name}_summary.csv',index=False)
    # The canonical return/prediction table comes from verified held-out source.
    paired=pd.read_csv(src.track(v2/'tables/paired_heldout_returns.csv','verified paired returns'))
    raw=[]
    for task,folder,filename in [('halfcheetah','dynamics-q-750k-v1','four_method_paired_comparison.csv'),
                                 ('reacher','reacher-dynamics-q-v1','four_way_per_seed.csv')]:
        table=pd.read_csv(src.track(root/folder/'comparison'/filename,'canonical held-out results'))
        if task=='halfcheetah':
            table['family']=table.parameter.map({'bthigh_damping':'damping','torso_mass':'mass'})
            returns={'F':'frozen_return','D':'ttt_return','DQ':'dq_return','J':'joint_return'}
            mse={'F':'frozen_latent_mse','D':'ttt_latent_mse','DQ':'dq_latent_mse'}
        else:
            table=table[table.kind=='shifted']
            returns={m:v+'_episode_return' for m,v in [('F','Frozen'),('D','Dynamics-only'),('DQ','Dynamics+Q'),('J','Joint RL')]}
            mse={'F':'frozen_mse','D':'dynamics_mse','DQ':'dq_mse'}
        for method,column in returns.items():
            frame=table[['family','severity','adaptation_seed','evaluation_seed']].copy()
            frame['task']=task;frame['method']=method;frame['return']=table[column].to_numpy()
            frame['fixed_encoder_latent_mse']=table[mse[method]].to_numpy() if method in mse else np.nan
            matched=frame.merge(paired[paired.method==method],on=common+['method','adaptation_seed','evaluation_seed'],
                                validate='one_to_one',suffixes=('_raw','_reference'))
            require(len(matched)==len(frame) and np.allclose(matched.return_raw,matched.return_reference),
                    'Canonical paired return mismatch')
            raw.append(frame)
    evidence=pd.concat(raw,ignore_index=True)
    evidence.to_csv(out/'tables/actual_control_prediction_evidence.csv',index=False)
    aggregate(evidence,common+['method'],['return','fixed_encoder_latent_mse']).to_csv(
        out/'tables/control_prediction_summary.csv',index=False)
    require(not src.verify(),'Sources changed during numeric analysis')
    write(out/'SOURCE_INPUTS.json',src.files)
    write(out/'NUMERICAL_VALIDATION.json',dict(status='PASS',streams=len(checks),
        native_parity_comparisons=len(checks)*2*5,environment_interactions=0,optimizer_steps=0))
    write(out/'completion.json',dict(status='PASS',streams=len(checks),conditions=36,
        native_states_per_stream=2,figures=0,environment_interactions=0,optimizer_steps=0))
