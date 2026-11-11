#!/usr/bin/env python3
"""NS-Gym dynamics-only TTT. Help/dry-run do not load CUDA or execute physics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT/'experiments/nsgym/configs/dynamics_only_ttt.json'
SHA = 'f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b'


def read(p): return json.loads(Path(p).read_text())
def js(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, allow_nan=False)
def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def rows(p, data):
    if not data: raise ValueError(f'Empty records: {p}')
    with Path(p).open('x', newline='') as f:
        w = csv.DictWriter(f, list(data[0])); w.writeheader(); w.writerows(data)


def dependencies():
    # Support the local nested checkout as well as the normal server layout.
    import research_utils
    for p in (ROOT/'tdmpc2', ROOT/'tdmpc2/tdmpc2'):
        if (p/'common/stage3a.py').is_file():
            sys.path.insert(0, str(p)); break
    global np, torch, EpisodeAdapter, DecisionAdapter, equal_state, tree_hash, SeverityHalfCheetah
    import numpy as np
    import torch
    from common.stage3a import EpisodeAdapter, equal_state
    from common.decision_ttt import DecisionAdapter
    from training_support import tree_hash
    from envs.nsgym.severity import SeverityHalfCheetah


def configuration(a):
    c = read(a.config)
    split = read(ROOT/'experiments/nsgym/configs/evaluation_seed_splits.yaml')
    grid = read(ROOT/'experiments/nsgym/configs/physical_severity_grid.yaml')
    c['adaptation_seeds'] = split[c['adaptation_stream_split']]
    c['evaluation_seeds'] = split[c['heldout_split']]
    c['conditions'] = [dict(parameter=p, severity=k) for ps in grid['families'].values() for p in ps for k in grid['severities']]
    assert not set(c['adaptation_seeds']) & set(c['evaluation_seeds'])
    if a.command == 'smoke':
        c.update(horizon=24, onset=8, episodes_per_stream=2, snapshot_budgets=[0,8,24,48],
                 batch_size=4, warmup=4, update_every_decisions=4, updates_per_episode=6,
                 adaptation_seeds=c['adaptation_seeds'][:1], evaluation_seeds=c['evaluation_seeds'][:1],
                 conditions=[dict(parameter=p, severity=1.5) for p in ('bthigh_damping','torso_mass')])
    assert c['snapshot_budgets'] == sorted(set(c['snapshot_budgets']))
    assert c['snapshot_budgets'][0] == 0
    assert c['snapshot_budgets'][-1] == c['horizon']*c['episodes_per_stream']
    assert c['capacity'] >= max(c['batch_size'], c['warmup']) > 0
    return c


def settings(c, method, seed):
    # Adapter validates the list of independently seeded streams. Runner owns
    # continuation resets within each stream; no per-episode seed arithmetic.
    return dict(mode={'frozen':'frozen','decision':'decision_ttt','episode':'dynamics_only'}[method],
        update_granularity='decision' if method=='decision' else 'episode', consistency_steps=1,
        action_input='commanded', replay_source='fresh_shifted', sampling='uniform_without_replacement',
        capacity=c['capacity'], batch_size=c['batch_size'], warmup=c['warmup'], lr=c['lr'],
        updates_per_episode=c['updates_per_episode'], episodes=1, eval_every=1,
        adaptation_seeds=[seed], evaluation_seeds=c['evaluation_seeds'],
        update_every_decisions=c['update_every_decisions'], updates_per_decision=1)


def make_adapter(agent,c,method,seed):
    cls = DecisionAdapter if method=='decision' else EpisodeAdapter
    return cls(agent.model, settings(c,method,seed),c['consistency_coef'],c['grad_clip'],seed=seed)


def seeded(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def tiny_agent():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._encoder=torch.nn.Linear(17,8)
            self._dynamics=torch.nn.Sequential(torch.nn.Linear(14,32),torch.nn.Tanh(),torch.nn.Linear(32,8))
            self._reward=torch.nn.Linear(8,1);self._pi=torch.nn.Linear(8,6)
            self._Qs=torch.nn.Linear(8,2);self._target_Qs=torch.nn.Linear(8,2)
            self.register_buffer('normalization',torch.ones(1))
        def encode(self,o,task=None): return self._encoder(o)
        def next(self,z,a,task=None): return self._dynamics(torch.cat([z,a],-1))
    class Agent:
        def __init__(self): self.model=Model(); self.device=torch.device('cpu')
        def act(self,o,t0=False,eval_mode=True):
            with torch.no_grad(): return self.model._pi(self.model.encode(o)).tanh()
    return Agent()


def load_agent(a):
    # Fixed seed makes every method's synthetic initial model exactly identical.
    seeded(17)
    if a.synthetic: return tiny_agent(), {'synthetic':True}
    from training_support import load_checkpoint
    cp=Path(a.checkpoint)
    if digest(cp/'model.pt') != SHA: raise ValueError('Only fixed primary 750k is allowed')
    agent,meta=load_checkpoint(cp)
    import importlib.metadata
    for package in ('ns_gym','gymnasium','mujoco'):
        if importlib.metadata.version(package)!=meta['provenance']['versions'][package]:raise ValueError(f'Environment version mismatch: {package}')
    def forbidden(*args,**kwargs):raise RuntimeError('Original RL optimizer/update is forbidden')
    agent.update=forbidden;agent.optim.step=forbidden;agent.pi_optim.step=forbidden
    contract=dict(task='nsgym-halfcheetah',action_dim=6,episode_length=1000,horizon=3,num_samples=512,iterations=6,mpc=True,episodic=False,seed=17)
    for k,v in contract.items():
        if meta['config'][k]!=v: raise ValueError(f'750k config mismatch: {k}')
    if meta['config']['obs_shape']['state'] not in ([17],(17,)): raise ValueError('Observation shape mismatch')
    return agent,meta


def source_hashes(a):
    import inspect
    from common import stage3a, decision_ttt
    paths=[Path(__file__),Path(a.config),Path(inspect.getfile(stage3a)),Path(inspect.getfile(decision_ttt)),Path(inspect.getfile(SeverityHalfCheetah))]
    core=Path(inspect.getfile(stage3a)).parents[1]
    paths += [core/'tdmpc2.py',core/'common/world_model.py',core/'envs/nsgym/halfcheetah.py',ROOT/'experiments/nsgym/configs/evaluation_seed_splits.yaml',ROOT/'experiments/nsgym/configs/physical_severity_grid.yaml']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def aux_state(agent):
    return {k:getattr(agent,k).state_dict() for k in ('optim','pi_optim','scale') if hasattr(agent,k)}


def environment(c,cond,schedule):
    return SeverityHalfCheetah(**cond,schedule=schedule,onset=c['onset'],max_episode_steps=c['horizon'])


def error(agent,o,a,n):
    d=agent.device
    with torch.no_grad():
        z=agent.model.encode(torch.as_tensor(o,dtype=torch.float32,device=d),None)
        target=agent.model.encode(torch.as_tensor(n,dtype=torch.float32,device=d),None)
        v=(agent.model.next(z,torch.as_tensor(a,dtype=torch.float32,device=d),None)-target).square().mean().item()
    if not np.isfinite(v): raise FloatingPointError('Nonfinite prediction error')
    return v


def rollout(agent,adapter,c,cond,schedule,seed,env,episode,training):
    o,_=env.reset(seed=seed if episode==0 or not training else None)
    if training: adapter.begin_episode()
    data=[]; observations=[o.copy()]; actions=[]
    for t in range(c['horizon']):
        begin=time.perf_counter()
        action=agent.act(torch.tensor(o,dtype=torch.float32),t0=t==0,eval_mode=True).detach().cpu().numpy()
        n,r,term,trunc,info=env.step(action)
        mse=error(agent,o,action,n) # pre-update error, observed transition only
        before=adapter.gradient_steps
        if training:
            adapter.observe(torch.tensor(o,dtype=torch.float32),torch.tensor(action,dtype=torch.float32),torch.tensor(n,dtype=torch.float32),1)
            # An environment transition has five simulator integration steps;
            # native_steps above counts Gym transitions, never control repeats.
            if not equal_state(adapter.replay[-1], tuple(torch.tensor(v,dtype=torch.float32) for v in (o,action,n))):
                raise AssertionError('Replay transition misalignment')
            if term or trunc: adapter.end_episode()
        force=env.unwrapped.data.actuator_force.copy(); model=env.unwrapped.model
        limited=model.actuator_forcelimited.astype(bool); bounds=model.actuator_forcerange
        force_sat=float(np.mean((force[limited]<=bounds[limited,0]+1e-6)|(force[limited]>=bounds[limited,1]-1e-6))) if limited.any() else None
        data.append(dict(episode=episode,seed=seed,t=t,aligned_step=t-c['onset'],
            adaptation_steps=adapter.transitions,gradient_updates=adapter.gradient_steps,
            updates_this_transition=adapter.gradient_steps-before,
            update_seconds=adapter.update_compute_time,transition_seconds=time.perf_counter()-begin,
            latent_mse_preupdate=mse,training_loss=adapter.last_metrics['consistency_loss'] if training and adapter.gradient_steps>before else None,
            gradient_norm=adapter.last_metrics['predictor_gradient_norm'] if training and adapter.gradient_steps>before else None,
            reward=r,velocity=float(info['x_velocity']),control_cost=-float(info['reward_ctrl']),
            action_abs=float(np.abs(action).mean()),action_rms=float(np.sqrt(np.mean(action**2))),
            action_component_saturation=float((np.abs(action)>=.99).mean()),
            actuator_force_rms=float(np.sqrt(np.mean(force**2))),actuator_force_saturation=force_sat,
            force_limited_count=int(limited.sum()),actual_parameter=env.readback(),nominal_parameter=env.nominal_value,
            terminated=term,truncated=trunc))
        observations.append(n.copy());actions.append(action.copy());o=n
        yield data[-1], observations, actions
        if term or trunc: break
    if training and adapter.in_episode: raise AssertionError('Expected completed episode')


def snapshot(out,agent,adapter,c,identity,aux_before):
    adapter.assert_fixed()
    assert tree_hash(aux_state(agent))==aux_before
    payload=dict(version=1,identity=identity,configuration=c,adapter=adapter.state_dict(),
        fixed_hash=tree_hash(adapter.fixed_state()),model_hash=tree_hash(agent.model.state_dict()),
        auxiliary_hash=aux_before,checkpoint_sha256=SHA if not identity['synthetic'] else None)
    path=out/f'snapshot-{adapter.transitions:07d}.pt'
    torch.save(payload,path)
    restored=torch.load(path,weights_only=False,map_location='cpu')
    assert tree_hash(payload)==tree_hash(restored), 'Snapshot serialization mismatch'
    return path


def run_one(a,c,out,cond,method,schedule,seed):
    out.mkdir(parents=True,exist_ok=False)
    agent,meta=load_agent(a);adapter=make_adapter(agent,c,method,seed);seeded(seed)
    initial=tree_hash(agent.model.state_dict());fixed=tree_hash(adapter.fixed_state());aux=tree_hash(aux_state(agent))
    identity=dict(**cond,method=method,schedule=schedule,adaptation_seed=seed,synthetic=a.synthetic)
    js(out/'manifest.json',dict(version=1,identity=identity,configuration=c,checkpoint_sha256=None if a.synthetic else SHA,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=source_hashes(a),
        versions={k:__import__('importlib.metadata',fromlist=['version']).version(k) for k in ('torch','numpy','gymnasium','mujoco','ns_gym')},
        checkpoint_metadata=meta,initial_model_hash=initial,initial_fixed_hash=fixed,initial_auxiliary_hash=aux,
        agent_inputs=['observation','action','next_observation'],snapshot_resume='evaluation_only'))
    snapshot(out,agent,adapter,c,identity,aux)
    step_rows=[];ep_rows=[]
    env=environment(c,cond,schedule)
    try:
        js(out/'actuator_contract.json',dict(forcelimited=env.unwrapped.model.actuator_forcelimited.tolist(),
            forcerange=env.unwrapped.model.actuator_forcerange.tolist(),gear=env.unwrapped.model.actuator_gear.tolist(),
            ctrlrange=env.unwrapped.model.actuator_ctrlrange.tolist(),dt=env.unwrapped.dt))
        for ep in range(c['episodes_per_stream']):
            episode_rows=[]
            for row,obs,acts in rollout(agent,adapter,c,cond,schedule,seed,env,ep,True):
                step_rows.append(row);episode_rows.append(row)
                if adapter.transitions in c['snapshot_budgets']:snapshot(out,agent,adapter,c,identity,aux)
            np.savez_compressed(out/f'online-{ep:03d}.npz',observations=np.array(obs),actions=np.array(acts))
            ep_rows.append(dict(episode=ep,adaptation_seed=seed,online_return=sum(x['reward'] for x in episode_rows),steps=len(episode_rows)))
        adapter.assert_fixed(); assert tree_hash(aux_state(agent))==aux
        if not a.synthetic:assert digest(Path(a.checkpoint)/'model.pt')==SHA
        assert (tree_hash(agent.model.state_dict())==initial)==(method=='frozen')
        rows(out/'online_steps.csv',step_rows);rows(out/'online_episodes.csv',ep_rows)
        js(out/'completion.json',dict(status='PASS',identity=identity,metrics=adapter.metrics(),
            initial_model_hash=initial,final_model_hash=tree_hash(agent.model.state_dict()),
            initial_fixed_hash=fixed,final_fixed_hash=tree_hash(adapter.fixed_state()),auxiliary_hash=aux,
            finite_losses=all(np.isfinite(x['training_loss']) for x in step_rows if x['training_loss'] is not None)))
    finally:env.close()


def evaluate_snapshot(a):
    payload=torch.load(a.snapshot,weights_only=False,map_location='cpu')
    c=payload['configuration'];identity=payload['identity']
    if identity['synthetic']!=a.synthetic: raise ValueError('Synthetic/real snapshot mismatch')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    agent,_=load_agent(a);adapter=make_adapter(agent,c,identity['method'],identity['adaptation_seed'])
    assert tree_hash(adapter.fixed_state())==payload['fixed_hash']
    adapter.load_state_dict(payload['adapter'])
    assert tree_hash(agent.model.state_dict())==payload['model_hash']
    before=tree_hash(adapter.state_dict());aux=tree_hash(aux_state(agent))
    assert aux==payload['auxiliary_hash'], 'Original optimizer/scale mismatch'
    cond={k:identity[k] for k in ('parameter','severity')};allrows=[];allsteps=[];fixedrows=[]
    # Evaluation is a separate process/model; the source run is never mutated.
    with adapter.held_out(allow_in_episode=True):
        try: adapter.observe(torch.zeros(17),torch.zeros(6),torch.zeros(17),1)
        except RuntimeError: pass
        else: raise AssertionError('Held-out replay write accepted')
        for kind,schedule in [('shifted',identity['schedule']),('nominal','nominal')]:
            for seed in c['evaluation_seeds']:
                seeded(seed);env=environment(c,cond,schedule)
                try:
                    rr=[x[0] for x in rollout(agent,adapter,c,cond,schedule,seed,env,0,False)]
                    allsteps.extend([dict(kind=kind,**r) for r in rr])
                    allrows.append(dict(kind=kind,seed=seed,episode_return=sum(r['reward'] for r in rr),
                        return_per_step=np.mean([r['reward'] for r in rr]),velocity=np.mean([r['velocity'] for r in rr]),
                        control_cost=sum(r['control_cost'] for r in rr),action_saturation=np.mean([r['action_component_saturation'] for r in rr]),
                        episode_length=len(rr),terminated=rr[-1]['terminated'],truncated=rr[-1]['truncated']))
                finally:env.close()
        if a.fixed_data:
            # Original frozen raw episodes, never online adaptation trajectories.
            episode_table=Path(a.fixed_data)/'episodes.csv'
            with episode_table.open() as f: source=list(csv.DictReader(f))
            reference=read(Path(a.fixed_data)/'manifest.json')
            assert reference['expected_checkpoint_sha256']==SHA
            assert read(Path(a.fixed_data)/'completion.json')['status']=='PASS'
            source=[r for r in source if r['parameter']==identity['parameter'] and float(r['severity'])==identity['severity'] and int(r['seed']) in c['evaluation_seeds'] and r['schedule']==identity['schedule']]
            if set(int(r['seed']) for r in source)!=set(c['evaluation_seeds']) or len(source)!=len(c['evaluation_seeds']):
                raise ValueError('Fixed data must contain exactly one episode for every held-out seed/condition')
            for r in source:
                # Existing collector names raw files via file field (resolved below).
                path=Path(a.fixed_data)/r['data_file']
                with np.load(path) as z:
                    values=[error(agent,o,ac,n) for o,ac,n in zip(z['observations'][:-1],z['actions'],z['observations'][1:])]
                fixedrows.append(dict(seed=int(r['seed']),latent_mse=float(np.mean(values)),samples=len(values),source=str(path),sha256=digest(path)))
    assert tree_hash(adapter.state_dict())==before and tree_hash(aux_state(agent))==aux
    rows(out/'heldout_episodes.csv',allrows);rows(out/'heldout_steps.csv',allsteps)
    if fixedrows:rows(out/'heldout_fixed_data.csv',fixedrows)
    js(out/'manifest.json',dict(identity=identity,checkpoint_sha256=payload['checkpoint_sha256'],snapshot=str(Path(a.snapshot).resolve()),snapshot_sha256=digest(a.snapshot),
        budget=adapter.transitions,gradient_updates=adapter.gradient_steps,update_seconds=adapter.update_compute_time,
        fixed_data_status='MEASURED' if fixedrows else 'NOT_MEASURED',configuration=c))
    js(out/'completion.json',dict(status='PASS',adapter_hash_before=before,adapter_hash_after=tree_hash(adapter.state_dict()),heldout_leakage=False))


def run_experiment(a,c):
    root=Path(a.output)
    if a.resume:
        if not root.is_dir():raise ValueError(f'Resume output does not exist: {root}')
        if (root/'completion.json').exists():raise ValueError(f'Run already complete: {root}')
    else:
        root.mkdir(parents=True,exist_ok=False)
    current_sources=source_hashes(a) if a.resume else None
    skipped=[];restarted=[]
    for cond in c['conditions']:
        for method in (c['methods'] if a.method=='all' else [a.method]):
            for seed in c['adaptation_seeds']:
                name=f"{cond['parameter']}-k{cond['severity']:g}-{method}-seed{seed}"
                target=root/name
                identity=dict(**cond,method=method,schedule=a.schedule,adaptation_seed=seed,synthetic=a.synthetic)
                if a.resume and target.exists():
                    manifest=read(target/'manifest.json') if (target/'manifest.json').is_file() else None
                    if manifest is not None:
                        if manifest.get('identity')!=identity or manifest.get('configuration')!=c or manifest.get('checkpoint_sha256')!=(None if a.synthetic else SHA):
                            raise ValueError(f'Existing condition identity/configuration differs: {target}')
                        # This change adds resume bookkeeping only. Every algorithm,
                        # environment and configuration source must still match.
                        old_sources=manifest.get('source_hashes',{})
                        for path,value in current_sources.items():
                            if path.endswith('/scripts/run_dynamics_only_ttt.py'):continue
                            if old_sources.get(path)!=value:raise ValueError(f'Source changed for {target}: {path}')
                    completion=target/'completion.json'
                    if completion.is_file() and read(completion).get('status')=='PASS':
                        done=read(completion)
                        if manifest is None or done.get('identity')!=identity or done['metrics']['adaptation_transitions']!=c['horizon']*c['episodes_per_stream']:
                            raise ValueError(f'Invalid completed condition: {target}')
                        if not (target/'online_steps.csv').is_file() or not (target/'online_episodes.csv').is_file():
                            raise ValueError(f'Completed condition files missing: {target}')
                        skipped.append(name);print(f'SKIP PASS {name}',flush=True);continue
                    # Preserve partial data outside RUNS so aggregation cannot
                    # accidentally ingest an interrupted duplicate.
                    archive=root.parent.parent/'interrupted'
                    archive.mkdir(parents=True,exist_ok=True)
                    saved=archive/f'{root.name}-{name}-{uuid4().hex[:8]}'
                    target.rename(saved)
                    restarted.append(dict(condition=name,archived=str(saved)))
                    print(f'ARCHIVED PARTIAL {target} -> {saved}',flush=True)
                run_one(a,c,target,cond,method,a.schedule,seed)
                print(f'PASS {name}',flush=True)
    js(root/'completion.json',dict(status='PASS',synthetic=a.synthetic,configuration=c,
        resumed=a.resume,skipped_completed=skipped,restarted_partial=restarted))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['smoke','run','evaluate'])
    p.add_argument('--config',default=str(DEFAULT));p.add_argument('--checkpoint',default=str(ROOT/'results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000'))
    p.add_argument('--output',required=True);p.add_argument('--method',choices=['all','frozen','decision','episode'],default='all')
    p.add_argument('--schedule',choices=['fixed','abrupt'],default='fixed')
    p.add_argument('--synthetic',action='store_true',help='CPU fixture only; smoke or evaluation of synthetic snapshots')
    p.add_argument('--snapshot');p.add_argument('--fixed-data',help='Original Frozen stage directory containing episodes.csv and raw NPZ')
    p.add_argument('--resume',action='store_true',help='Resume an incomplete run, preserving PASS conditions and archiving partial ones')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.synthetic and a.command=='run':p.error('Synthetic is restricted to smoke')
    if a.resume and a.command!='run':p.error('--resume is supported only for run')
    if a.command=='evaluate' and not a.snapshot:p.error('--snapshot required')
    c=configuration(a)
    if a.dry_run:
        print(json.dumps(dict(command=a.command,configuration=c,synthetic=a.synthetic),indent=2));return
    dependencies()
    if a.command=='evaluate':evaluate_snapshot(a);return
    run_experiment(a,c)

if __name__=='__main__':main()
