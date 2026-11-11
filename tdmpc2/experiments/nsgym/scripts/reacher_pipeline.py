"""Reacher experiment primitives. No training or remote jobs on import."""
import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tdmpc2'))
DEFAULT = ROOT/'experiments/nsgym/configs/reacher_pipeline_v1.json'

def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def table(p, rr):
    if not rr:raise ValueError(f'No rows for {p}')
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rr)
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def config(path):
    c=read(path);splits=[c[k] for k in ('validation_seeds','calibration_seeds','adaptation_seeds','evaluation_seeds')]
    flat=sum(splits,[])
    assert len(set(flat))==len(flat)
    assert c['snapshot_budgets']==sorted(set(c['snapshot_budgets'])) and c['snapshot_budgets'][0]==0
    assert all(b%50==0 for b in c['snapshot_budgets'])
    return c

def conditions(c):
    return [dict(family=f,parameter=p,severity=k) for f,p in c['families'].items() for k in c['severities']]
def tag(cond):return f"{cond['family']}-k{cond['severity']:g}"
def env(cond):
    from envs.nsgym.reacher_severity import SeverityReacher
    return SeverityReacher(parameter=cond['parameter'],severity=cond['severity'])
def seeded(s):
    import torch
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def provenance(c):
    paths=list((ROOT/'experiments/nsgym/scripts').glob('*reacher*.py'))+[ROOT/'tdmpc2/envs/nsgym/reacher_severity.py',ROOT/'tdmpc2/common/decision_ttt.py',ROOT/'tdmpc2/common/stage3a.py',ROOT/'tdmpc2/tdmpc2.py',ROOT/'tdmpc2/common/world_model.py']
    return dict(config=c,config_fingerprint=fingerprint(c),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},versions={k:importlib.metadata.version(k) for k in ('gymnasium','mujoco','ns_gym','torch','numpy')})
def done(out,**extra):write(out/'completion.json',dict(status='PASS',**extra))
def gate(path,stage,c,selection=None):
    p=Path(path)
    assert read(p/'completion.json')['status']=='PASS',f'Incomplete gate: {p}'
    m=read(p/'manifest.json')
    assert m['stage']==stage and m['config_fingerprint']==fingerprint(c),f'Gate config/stage mismatch: {p}'
    if selection:assert m['checkpoint_sha256']==selection['model_sha256']
    for relative,value in m.get('source_hashes',{}).items():
        assert sha(ROOT/relative)==value, f'Source changed after gate: {relative}'
    return m

def selected(path,c):
    s=read(path);cp=Path(s['checkpoint']);assert cp.is_absolute()
    assert sha(cp/'model.pt')==s['model_sha256']
    assert s['pipeline_config_fingerprint']==fingerprint(c)
    return s

def load(cp,c):
    from training_support import load_checkpoint
    a,m=load_checkpoint(cp);cfg=m['config']
    assert cfg['task']=='nsgym-reacher' and cfg['environment_id']=='Reacher-v5'
    assert cfg['action_dim']==2 and cfg['obs_shape']['state']==[10]
    assert cfg['episode_length']==50 and cfg['nsgym']['mode']=='nominal' and cfg['nsgym']['mass_scale']==1
    for package in ('ns_gym','gymnasium','mujoco'):
        assert importlib.metadata.version(package)==m['provenance']['versions'][package],f'Version mismatch: {package}'
    for key in ('validation_seeds','calibration_seeds','adaptation_seeds','evaluation_seeds'):
        assert not set(c[key]) & set(cfg['eval_seeds'])
        assert all(not cfg['seed']<=s<=cfg['seed']+cfg['steps'] for s in c[key])
    def forbidden(*args,**kwargs):raise RuntimeError('Original RL updates forbidden')
    a.update=forbidden;a.optim.step=forbidden;a.pi_optim.step=forbidden;a.model.eval()
    return a,m

def error(a,o,ac,n):
    import torch
    with torch.no_grad():
        tensors=[torch.as_tensor(v,dtype=torch.float32,device=a.device) for v in (o,ac,n)]
        z=a.model.encode(tensors[0],None);target=a.model.encode(tensors[2],None)
        v=(a.model.next(z,tensors[1],None)-target).square().mean().item()
    if not np.isfinite(v):raise FloatingPointError('Nonfinite latent error')
    return v

def action(a,o,t,rng=None):
    import torch
    if a is None:return rng.uniform(-1,1,2).astype(np.float32)
    return a.act(torch.as_tensor(o,dtype=torch.float32),t0=t==0,eval_mode=True).detach().cpu().numpy()

def step_record(e,ac,n,r,term,trunc,info):
    p=e.physical();m=e.unwrapped.model;f=p['force'];limited=m.actuator_forcelimited.astype(bool);bounds=m.actuator_forcerange
    return dict(reward=r,reward_dist=float(info['reward_dist']),reward_ctrl=float(info['reward_ctrl']),
        distance=p['distance'],action_abs=float(np.abs(ac).mean()),action_saturation=float((np.abs(ac)>=.99).mean()),
        force_rms=float(np.sqrt(np.mean(f*f))),force_saturation=float(np.mean((f[limited]<=bounds[limited,0]+1e-6)|(f[limited]>=bounds[limited,1]-1e-6))) if limited.any() else None,
        force_limited_count=int(limited.sum()),actual_parameter=e.readback(),nominal_parameter=e.nominal_value,
        terminated=term,truncated=trunc,**{key:json.dumps(p[key].tolist()) for key in ('q','qvel','qacc','tip','tip_velocity','target','force','ctrl')})

def episode(a,cond,seed,out=None):
    from training_support import tree_hash,learned_state
    before=tree_hash(learned_state(a));seeded(seed);e=env(cond)
    try:
        o,_=e.reset(seed=seed);initial=e.physical()['distance'];obs=[o.copy()];acts=[];rr=[]
        for t in range(50):
            ac=action(a,o,t);n,r,term,trunc,info=e.step(ac)
            row=dict(**cond,seed=seed,t=t,latent_mse=error(a,o,ac,n),**step_record(e,ac,n,r,term,trunc,info))
            rr.append(row);obs.append(n.copy());acts.append(ac.copy());o=n
            if term or trunc:break
        assert len(rr)==50 and not term and trunc
        rec=dict(**cond,seed=seed,episode_return=sum(x['reward'] for x in rr),reward_per_step=np.mean([x['reward'] for x in rr]),
            reward_dist=sum(x['reward_dist'] for x in rr),reward_ctrl=sum(x['reward_ctrl'] for x in rr),
            distance_initial=initial,distance_mean=np.mean([x['distance'] for x in rr]),distance_final=rr[-1]['distance'],distance_min=min(x['distance'] for x in rr),
            latent_mse=np.mean([x['latent_mse'] for x in rr]),action_abs=np.mean([x['action_abs'] for x in rr]),action_saturation=np.mean([x['action_saturation'] for x in rr]),
            force_rms=np.mean([x['force_rms'] for x in rr]),force_saturation=np.mean([x['force_saturation'] for x in rr]) if rr[0]['force_limited_count'] else None,
            force_limited_count=rr[0]['force_limited_count'],actual_parameter=e.readback(),nominal_parameter=e.nominal_value,episode_length=len(rr),terminated=term,truncated=trunc)
        if out:
            filename=f'{tag(cond)}-seed{seed}.npz';np.savez_compressed(out/filename,observations=np.array(obs),actions=np.array(acts))
            rec['data_file']=filename;rec['data_sha256']=sha(out/filename)
        assert tree_hash(learned_state(a))==before
        return rec,rr
    finally:e.close()

def stats(values):
    x=np.asarray(values,dtype=float);assert len(x) and np.isfinite(x).all()
    rng=np.random.default_rng(20261111);ci=np.quantile(x[rng.integers(len(x),size=(4000,len(x)))].mean(1),[.025,.975]) if len(x)>1 else [None,None]
    q1,q3=np.quantile(x,[.25,.75]);iqr=q3-q1
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),std=float(x.std(ddof=1)) if len(x)>1 else None,ci_low=ci[0],ci_high=ci[1],outliers=int(((x<q1-1.5*iqr)|(x>q3+1.5*iqr)).sum()))

def inventory(c,out):
    from envs.nsgym.reacher_severity import SeverityReacher
    from ns_gym.wrappers.mujoco_env import param_look_up
    rr=[]
    for cond in conditions(c):
        e=env(cond)
        try:
            o,_=e.reset(seed=c['calibration_seeds'][0]);inertia=e.unwrapped.model.body_inertia.copy()
            for _ in range(3):e.step(np.array([.6,-.4]))
            e.reset(seed=c['calibration_seeds'][0]);e.check()
            assert np.all(inertia[1:]>0) and np.all(2*inertia.max(1)<=inertia.sum(1)+1e-15)
            body=cond['parameter'].split('_')[0];mass=cond['parameter'].endswith('mass')
            rr.append(dict(**cond,nominal=e.nominal_value,actual=e.readback(),unit='kg' if mass else 'N m s/rad',field='body_mass' if mass else 'dof_damping',
                valid=True,reason='',inertia_changed=False,reset='persistent fixed value before reset observation',observation_exposed=False,
                setter_source=inspect.getsource(param_look_up('ReacherEnv',cond['parameter'])[0][1]).strip(),
                dt=e.unwrapped.dt,frame_skip=e.unwrapped.frame_skip,body_or_joint=body,obs_dim=len(o),action_dim=2,horizon=50,
                actuator_gear=json.dumps(e.unwrapped.model.actuator_gear.tolist()),actuator_forcerange=json.dumps(e.unwrapped.model.actuator_forcerange.tolist()),actuator_forcelimited=json.dumps(e.unwrapped.model.actuator_forcelimited.tolist()),actuator_ctrlrange=json.dumps(e.unwrapped.model.actuator_ctrlrange.tolist()),
                reward_dist_weight=e.unwrapped._reward_dist_weight,reward_control_weight=e.unwrapped._reward_control_weight))
        finally:e.close()
    table(out/'reacher_physical_parameter_inventory.csv',rr)
    write(out/'reacher_physical_severity_grid.yaml',dict(families=c['families'],severities=c['severities'],mass_semantics=c['mass_semantics']))
    (out/'REACHER_PHYSICAL_PARAMETER_AUDIT.md').write_text('# Reacher physical parameter audit\n\nAll 24 runtime conditions passed positivity, inertia triangle and readback checks. Official setters and units are in the inventory CSV. Mass changes leave inertia unchanged: this is an abstract independent mass intervention, not homogeneous density scaling. All positive diagonal inertias remain mathematically valid, but fixed geometry/uniform-density material interpretation is not claimed.\n\nPolicy input is state only (10 floats); physical properties and change metadata are never passed to the agent. Fixed values persist across episode resets. Commands are normalized in [-1,1]^2; actual force limits are recorded separately.\n')

def probe(c,out,a=None):
    """Common nominal random action bank; all integration state includes time/ctrl/warmstart."""
    rr=[];bank=[]
    nominal=conditions(c)[0]|dict(severity=1.)
    for seed in c['calibration_seeds']:
        e=env(nominal);rng=np.random.default_rng(seed);o,_=e.reset(seed=seed)
        try:
            for t in range(20):
                ac=rng.uniform(-1,1,2).astype(np.float32)
                if t>=5:bank.append((seed,t,e.integration_state().copy(),o.copy(),ac.copy(),e.physical()))
                o,*_=e.step(ac)
        finally:e.close()
    raw=[]
    for cond in conditions(c):
        e=env(cond);base=env(cond|dict(severity=1.))
        try:
            for seed,t,state,o,ac,pre in bank:
                base.reset(seed=seed);e.reset(seed=seed)
                bo=base.restore(state);so=e.restore(state);np.testing.assert_allclose(bo,so,rtol=0,atol=1e-12)
                bn,br,bt,btr,bi=base.step(ac);bp=base.physical();bs=base.integration_state()
                base.reset(seed=seed);base.restore(state);repeat=base.step(ac)
                np.testing.assert_allclose(base.integration_state(),bs,rtol=0,atol=1e-12)
                np.testing.assert_allclose(repeat[0],bn,rtol=0,atol=1e-12)
                n,r,term,trunc,info=e.step(ac);p=e.physical()
                if cond['severity']==1:np.testing.assert_allclose(e.integration_state(),bs,rtol=0,atol=1e-12)
                ne=error(a,o,ac,bn) if a else None;se=error(a,o,ac,n) if a else None
                magnitude=float(np.abs(ac).mean());lo,hi=c['action_regime_thresholds']
                row=dict(**cond,seed=seed,bank_step=t,nominal_parameter=e.nominal_value,actual_parameter=e.readback(),
                    delta_q_rad=float(np.linalg.norm(p['q']-bp['q'])),delta_qvel_rad_s=float(np.linalg.norm(p['qvel']-bp['qvel'])),
                    delta_tip_m=float(np.linalg.norm(p['tip']-bp['tip'])),delta_accel_rad_s2=float(np.linalg.norm((p['qvel']-bp['qvel'])/e.unwrapped.dt)),
                    nominal_latent_mse=ne,shifted_latent_mse=se,delta_latent_mse=se-ne if a else None,latent_ratio=se/ne if a and ne>c['ratio_floor'] else None,
                    action_abs=magnitude,action_regime='low' if magnitude<lo else 'high' if magnitude>=hi else 'mid',
                    moving=bool(np.linalg.norm(pre['qvel'])>1e-6),numerical_valid=True,
                    initial_state=json.dumps(state.tolist()),action=json.dumps(ac.tolist()),next_observation=json.dumps(n.tolist()),nominal_next_observation=json.dumps(bn.tolist()),
                    reward_dist=info['reward_dist'],reward_ctrl=info['reward_ctrl'],
                    **{f'{prefix}_{key}':json.dumps(value[key].tolist()) for prefix,value in [('pre',pre),('nominal',bp),('shifted',p)] for key in ('q','qvel','qacc','tip','tip_velocity','target','force','ctrl')})
                rr.append(row);raw.append((o,ac,n,bn))
        finally:e.close();base.close()
    table(out/'reacher_physical_response.csv',rr)
    np.savez_compressed(out/'counterfactual_bank.npz',observations=np.array([x[0] for x in raw]),actions=np.array([x[1] for x in raw]),next_observations=np.array([x[2] for x in raw]),nominal_next_observations=np.array([x[3] for x in raw]))
    assert all(x['moving'] for x in rr)
    # Some nontrivial response is required for every nonnominal condition.
    for cond in conditions(c):
        subset=[r for r in rr if all(r[k]==v for k,v in cond.items())]
        if cond['severity']!=1:assert max(r['delta_qvel_rad_s'] for r in subset)>1e-10
    return rr

def discover(search,c,out,run=None):
    search=Path(search).resolve()
    if not search.is_dir():raise ValueError(f'Search root does not exist: {search}')
    candidates=[];errors=[]
    for p in search.rglob('metadata.json'):
        if not (p.parent/'model.pt').is_file():continue
        try:
            m=read(p)
            if m.get('config',{}).get('task')!='nsgym-reacher':continue
            if not (p.parent/'optimizers.pt').is_file():raise ValueError('Missing optimizers.pt')
            cfg=m['config'];assert cfg['nsgym']['mode']=='nominal' and cfg['nsgym']['mass_scale']==1
            assert sha(p.parent/'model.pt')==m['model_sha256']
            assert sha(p.parent/'optimizers.pt')==m['optimizer_sha256']
            candidates.append(dict(checkpoint=str(p.parent.resolve()),run=str(p.parent.parent.resolve()),step=m['environment_transitions'],metadata=m))
        except Exception as ex:errors.append(dict(path=str(p),error=str(ex)))
    runs=sorted(set(r['run'] for r in candidates))
    chosen=str(Path(run).resolve()) if run else runs[0] if len(runs)==1 else None
    available=sorted([r for r in candidates if r['run']==chosen],key=lambda r:r['step'])
    if len(available)>=3:
        indices=sorted(set([len(available)//3,2*len(available)//3,len(available)-1]))
        if len(indices)<3:indices=[0,len(available)//2,len(available)-1]
        validation=[available[i]['checkpoint'] for i in indices]
    else:validation=[]
    log_exists=bool(chosen and (Path(chosen)/'training.jsonl').is_file())
    if not candidates:reason=f'No valid nominal Reacher checkpoint metadata/model/optimizer triplets found under {search}. Search the full project root or locate the training run first.'
    elif chosen is None:reason=f'Multiple Reacher runs found under {search}; select one of the listed runs with --run.'
    elif len(available)<3:reason=f'Only {len(available)} valid checkpoints in {chosen}; at least three are required.'
    elif not log_exists:reason=f'Missing full training.jsonl in {chosen}; cannot complete the nominal convergence audit.'
    else:reason=None
    result=dict(search_root=str(search),runs=runs,selected_run=chosen,candidates=candidates,validation_checkpoints=validation,errors=errors,
        maximum_transitions=max((r['step'] for r in available),default=None),
        status='READY' if reason is None else 'BLOCKED',reason=reason)
    write(out/'discovery.json',result)
    if chosen and log_exists:training_curves(Path(chosen),out)
    return result

def training_curves(run,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=['consistency_loss','reward_loss','value_loss','total_loss','grad_norm','pi_grad_norm','pi_entropy','pi_scaled_entropy','pi_scale']
    bins={k:[] for k in metrics};rows=[];current=None;count=0;nonfinite=0;last={};first_update=None;previous_step=-1;update_mismatches=0
    metas=sorted(run.glob('checkpoint-*/metadata.json'))
    training_config=read(metas[-1])['config'] if metas else None
    def flush():
        for k,v in bins.items():
            if v:rows.append(dict(bin_end=(current+1)*50000,metric=k,n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),std=float(np.std(v))))
            v.clear()
    with (run/'training.jsonl').open() as f:
        for line in f:
            r=json.loads(line);step=int(r['environment_transitions']);assert step>=previous_step;previous_step=step
            if training_config:
                expected=step-1 if step>training_config['seed_steps'] else 0
                update_mismatches+=int(r.get('gradient_updates')!=expected)
            b=(step-1)//50000
            if current is not None and b!=current:flush()
            current=b;count+=1;last=r
            if first_update is None and r.get('gradient_updates',0)>0:first_update=r
            for k in metrics:
                v=r.get(k,r.get('metrics',{}).get(k))
                if isinstance(v,(int,float)):
                    if np.isfinite(v):bins[k].append(v)
                    else:nonfinite+=1
        if current is not None:flush()
    if rows:table(out/'training_bins.csv',rows)
    fig,axes=plt.subplots(5,2,figsize=(13,16))
    for ax,k in zip(axes.flat,metrics):
        rr=[x for x in rows if x['metric']==k];ax.plot([x['bin_end'] for x in rr],[x['mean'] for x in rr],label='mean');ax.plot([x['bin_end'] for x in rr],[x['median'] for x in rr],label='median');ax.set_title(k);ax.set_xlabel('environment transitions');ax.legend()
    ev=run/'evaluation.csv';evalrows=csvread(ev) if ev.exists() else []
    evalrows=[x for x in evalrows if str(x.get('random_actions','False')).lower()!='true' and str(x.get('zero_actions','False')).lower()!='true']
    evalmeans={}
    for row in evalrows:
        if 'environment_transitions' in row and 'episode_return' in row:evalmeans.setdefault(int(row['environment_transitions']),[]).append(float(row['episode_return']))
    evalmeans={k:float(np.mean(v)) for k,v in evalmeans.items()}
    ax=axes.flat[-1]
    if evalrows:
        xkey=next((k for k in ('environment_transitions','step') if k in evalrows[0]),None)
        ykey=next((k for k in ('episode_return','return_mean','mean_return') if k in evalrows[0]),None)
        if xkey and ykey:ax.scatter([float(r[xkey]) for r in evalrows],[float(r[ykey]) for r in evalrows],s=5)
    ax.set_title('Existing training evaluation (if available)');fig.tight_layout();fig.savefig(out/'reacher_training_curves.png');plt.close(fig)
    write(out/'training_audit.json',dict(rows=count,nonfinite_metric_values=nonfinite,first_logged_update=first_update,last_record=last,update_count_mismatches=update_mismatches,training_config=training_config,nominal_evaluation_means=evalmeans,final_minus_best_evaluation=(evalmeans[max(evalmeans)]-max(evalmeans.values())) if evalmeans else None,
        seed_exploration_semantics='step <= seed_steps random; first update at transition seed_steps+1 performs seed_steps updates; cumulative updates=transitions-1 thereafter',terminal_semantics='True termination only enters replay terminal flag. Reacher base never terminates; TimeLimit truncates at 50. TD target bootstraps through truncation.',
        caveat='Loss stability does not prove control convergence. Missing metrics remain missing. Sparse logging may omit the first actual update.'))

def nominal(c,discovery,out):
    d=read(discovery);assert d['status']=='READY';rr=[]
    for cp in d['validation_checkpoints']:
        a,m=load(cp,c)
        for seed in c['validation_seeds']:
            r,_=episode(a,dict(family='nominal',parameter='joint0_damping',severity=1.),seed)
            rr.append(dict(checkpoint=cp,training_step=m['environment_transitions'],**r))
        del a
    table(out/'reacher_checkpoint_validation.csv',rr)
    comparison=[]
    for cp in d['validation_checkpoints']:
        r=[x for x in rr if x['checkpoint']==cp];s=stats([x['episode_return'] for x in r])
        comparison.append(dict(checkpoint=cp,training_step=r[0]['training_step'],score=s['mean']-s['std']/np.sqrt(s['n']),**s))
    best=max(comparison,key=lambda x:(x['score'],-x['training_step']));m=read(Path(best['checkpoint'])/'metadata.json')
    selection=dict(checkpoint=best['checkpoint'],model_sha256=sha(Path(best['checkpoint'])/'model.pt'),training_step=best['training_step'],selection_criteria=c['selection_criterion'],validation_seeds=c['validation_seeds'],nominal_performance=best,configuration=m['config'],environment_config_fingerprint=fingerprint(m['config']),pipeline_config_fingerprint=fingerprint(c),training_provenance=m['provenance'])
    write(out/'selected_reacher_checkpoint.json',selection);table(out/'checkpoint_summary.csv',comparison)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.errorbar([x['training_step'] for x in comparison],[x['mean'] for x in comparison],yerr=[[x['mean']-x['ci_low'] for x in comparison],[x['ci_high']-x['mean'] for x in comparison]],fmt='o-');ax.set(xlabel='training transitions',ylabel='nominal return (seed bootstrap 95% CI)');fig.tight_layout();fig.savefig(out/'reacher_checkpoint_comparison.png');plt.close(fig)
    training_curves(Path(d['selected_run']),out)
    (out/'REACHER_NOMINAL_VALIDATION.md').write_text('# Nominal validation\n\nSelection uses only independent nominal seeds: '+c['selection_criterion']+'.\n\nSelected: '+best['checkpoint']+'\n\nAll candidates and per-seed reaching metrics are in the CSV. Compare training_audit.json and the plotted evaluation history for late regression; optimization stability alone is not convergence. No binary success threshold is defined. One training seed limits generalization.\n')
    return selection

def frozen(c,s,out):
    a,_=load(s['checkpoint'],c);rr=[];steps=[];nominal_cache={}
    for cond in conditions(c):
        for seed in c['evaluation_seeds']:
            if cond['severity']==1 and seed in nominal_cache:
                reference=env(cond)
                try:value=reference.nominal_value
                finally:reference.close()
                r,ss=nominal_cache[seed]
                r=dict(r,**cond,nominal_parameter=value,actual_parameter=value,duplicate_of=f'D0-k1-seed{seed}')
                ss=[dict(x,**cond,nominal_parameter=value,actual_parameter=value) for x in ss]
            else:
                r,ss=episode(a,cond,seed,out);r['duplicate_of']=''
                if cond['severity']==1:nominal_cache[seed]=(r,ss)
            rr.append(r);steps.extend(ss)
    table(out/'reacher_frozen_severity.csv',rr);table(out/'steps.csv',steps)
    return rr

def tiny():
    import torch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self._encoder=torch.nn.Linear(10,8);self._dynamics=torch.nn.Linear(10,8);self._pi=torch.nn.Linear(8,2);self._reward=torch.nn.Linear(8,1);self._Qs=torch.nn.Linear(8,2);self._target_Qs=torch.nn.Linear(8,2);self.register_buffer('norm',torch.ones(1))
        def encode(self,o,task=None):return self._encoder(o)
        def next(self,z,a,task=None):return self._dynamics(torch.cat([z,a],-1))
    class Agent:
        def __init__(self):
            self.model=Model();self.device=torch.device('cpu');self.optim=torch.optim.Adam(self.model.parameters());self.pi_optim=torch.optim.Adam(self.model._pi.parameters());self.scale=torch.nn.Identity()
        def act(self,o,**kwargs):
            with torch.no_grad():return self.model._pi(self.model.encode(o)).tanh()
    seeded(17);return Agent()

def adapter(a,c,method,seed):
    from common.stage3a import EpisodeAdapter
    from common.decision_ttt import DecisionAdapter
    settings=dict(mode='decision_ttt' if method=='decision' else 'frozen',update_granularity='decision' if method=='decision' else 'episode',consistency_steps=1,action_input='commanded',replay_source='fresh_shifted',sampling='uniform_without_replacement',capacity=c['capacity'],batch_size=c['batch_size'],warmup=c['warmup'],lr=c['lr'],updates_per_episode=c['updates_per_episode'],episodes=1,eval_every=1,adaptation_seeds=[seed],evaluation_seeds=c['evaluation_seeds'],update_every_decisions=c['update_every_decisions'],updates_per_decision=1)
    return (DecisionAdapter if method=='decision' else EpisodeAdapter)(a.model,settings,c['consistency_coef'],c['grad_clip'],seed=seed)

def auxiliary(a):return {k:getattr(a,k).state_dict() for k in ('optim','pi_optim','scale') if hasattr(a,k)}

def adapt_one(c,s,out,cond,method,seed,budgets,synthetic=False):
    import torch
    from training_support import tree_hash
    from common.stage3a import equal_state
    out.mkdir(exist_ok=False)
    a=tiny() if synthetic else load(s['checkpoint'],c)[0];ad=adapter(a,c,method,seed);seeded(seed)
    fixed=tree_hash(ad.fixed_state());aux=tree_hash(auxiliary(a));initial=tree_hash(a.model.state_dict());started=time.perf_counter();rr=[];episodes=[]
    identity=dict(**cond,method=method,adaptation_seed=seed)
    def save():
        ad.assert_fixed();assert tree_hash(auxiliary(a))==aux
        payload=dict(identity=identity,config=c,selection=s,synthetic=synthetic,declared_budgets=budgets,adapter=ad.state_dict(),model_hash=tree_hash(a.model.state_dict()),fixed_hash=fixed,aux_hash=aux,wall_seconds=time.perf_counter()-started)
        p=out/f'snapshot-{ad.transitions:07d}.pt';torch.save(payload,p);v=torch.load(p,weights_only=False,map_location='cpu');assert tree_hash(payload)==tree_hash(v)
        # Actual reload into a separate model and prediction equality.
        b=tiny() if synthetic else load(s['checkpoint'],c)[0];bd=adapter(b,c,method,seed);bd.load_state_dict(v['adapter']);assert tree_hash(b.model.state_dict())==payload['model_hash']
        o,ac,n=(tuple(x.numpy() for x in ad.replay[-1]) if ad.replay else (np.zeros(10),np.zeros(2),np.zeros(10)))
        assert error(a,o,ac,n)==error(b,o,ac,n)
        before=tree_hash(bd.state_dict())
        with bd.held_out(allow_in_episode=True):
            try:bd.observe(torch.zeros(10),torch.zeros(2),torch.zeros(10),1)
            except RuntimeError:pass
            else:raise AssertionError('Heldout replay contamination allowed')
            error(b,o,ac,n)
        assert tree_hash(bd.state_dict())==before
        del b,bd
    save();e=env(cond)
    try:
        for ep in range(budgets[-1]//50):
            o,_=e.reset(seed=seed if ep==0 else None);ad.begin_episode();total=0
            for t in range(50):
                ac=action(a,o,t);n,r,term,trunc,info=e.step(ac);mse=error(a,o,ac,n);before=ad.gradient_steps
                tensors=tuple(torch.tensor(v,dtype=torch.float32) for v in (o,ac,n))
                ad.observe(*tensors,1);assert equal_state(ad.replay[-1],tensors)
                if term or trunc:ad.end_episode()
                row=dict(**identity,episode=ep,t=t,transitions=ad.transitions,gradient_updates=ad.gradient_steps,updates_this_transition=ad.gradient_steps-before,preupdate_latent_mse=mse,loss=ad.last_metrics['consistency_loss'] if ad.gradient_steps>before else None,grad_norm=ad.last_metrics['predictor_gradient_norm'] if ad.gradient_steps>before else None,**step_record(e,ac,n,r,term,trunc,info));rr.append(row);total+=r;o=n
                if ad.transitions in budgets:save()
                if term or trunc:break
            assert not ad.in_episode
            episodes.append(dict(**identity,episode=ep,online_return=total))
        ad.assert_fixed();assert tree_hash(auxiliary(a))==aux
        assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in rr if r['loss'] is not None)
        assert (ad.gradient_steps==0)==(method=='frozen')
        assert (tree_hash(a.model.state_dict())==initial)==(method=='frozen')
        table(out/'online_steps.csv',rr);table(out/'online_episodes.csv',episodes)
        write(out/'manifest.json',dict(identity=identity,config=c,selection=s,synthetic=synthetic))
        done(out,metrics=ad.metrics(),wall_seconds=time.perf_counter()-started,fixed_hash=fixed,auxiliary_hash=aux,model_hash=tree_hash(a.model.state_dict()))
    finally:e.close()
    return read(out/'completion.json')


def evaluate_snapshot(c,s,snapshot,out,frozen_dir,kind):
    import torch
    from training_support import tree_hash
    payload=torch.load(snapshot,weights_only=False,map_location='cpu');assert not payload['synthetic']
    assert payload['selection']['model_sha256']==s['model_sha256'] and payload['config']==c
    identity=payload['identity'];a,_=load(s['checkpoint'],c);ad=adapter(a,c,identity['method'],identity['adaptation_seed']);assert tree_hash(ad.fixed_state())==payload['fixed_hash'];ad.load_state_dict(payload['adapter']);assert tree_hash(a.model.state_dict())==payload['model_hash']
    before=tree_hash(ad.state_dict());aux=tree_hash(auxiliary(a));cond={k:identity[k] for k in ('family','parameter','severity')};rr=[];steps=[]
    with ad.held_out(allow_in_episode=True):
        if kind=='predict':
            source=csvread(Path(frozen_dir)/'reacher_frozen_severity.csv');source=[r for r in source if r['family']==cond['family'] and float(r['severity'])==cond['severity']]
            assert sorted(int(r['seed']) for r in source)==sorted(c['evaluation_seeds'])
            for r in source:
                p=Path(frozen_dir)/r['data_file'];assert sha(p)==r['data_sha256']
                with np.load(p) as z:
                    errors=[error(a,o,ac,n) for o,ac,n in zip(z['observations'][:-1],z['actions'],z['observations'][1:])]
                    regimes=np.abs(z['actions']).mean(1)
                rr.append(dict(**identity,budget=ad.transitions,declared_final_budget=payload['declared_budgets'][-1],nominal_parameter=float(r['nominal_parameter']),actual_parameter=float(r['actual_parameter']),seed=int(r['seed']),latent_mse=float(np.mean(errors)),samples=len(errors),source_sha256=sha(p),low_action_n=int((regimes<c['action_regime_thresholds'][0]).sum()),mid_action_n=int(((regimes>=c['action_regime_thresholds'][0])&(regimes<c['action_regime_thresholds'][1])).sum()),high_action_n=int((regimes>=c['action_regime_thresholds'][1]).sum())))
        else:
            for label,condition in [('shifted',cond),('nominal',cond|dict(severity=1.))]:
                for seed in c['evaluation_seeds']:
                    r,ss=episode(a,condition,seed);rr.append(dict(**(r|{'family':identity['family'],'severity':identity['severity']}),kind=label,method=identity['method'],adaptation_seed=identity['adaptation_seed'],budget=ad.transitions,declared_final_budget=payload['declared_budgets'][-1],gradient_updates=ad.gradient_steps,adaptation_wall_seconds=payload['wall_seconds']));steps.extend([dict(x,kind=label,budget=ad.transitions,method=identity['method'],adaptation_seed=identity['adaptation_seed']) for x in ss])
    assert tree_hash(ad.state_dict())==before and tree_hash(auxiliary(a))==aux==payload['aux_hash']
    table(out/f'{kind}.csv',rr)
    if steps:table(out/'steps.csv',steps)
    done(out,snapshot=str(Path(snapshot).resolve()),snapshot_sha256=sha(snapshot),budget=ad.transitions,identity=identity,replay_evaluation_isolated=True)
