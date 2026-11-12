#!/usr/bin/env python3
"""Independent Full Joint RL adaptation and explicit Reacher held-out evaluation."""
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

import run_dynamics_only_ttt as base

COMPONENTS = ('_encoder', '_dynamics', '_reward', '_Qs', '_pi', '_target_Qs_params')
REACHER_CONFIG = base.ROOT/'experiments/nsgym/configs/reacher_pipeline_v1.json'


def reacher_config():
    c=base.read(REACHER_CONFIG)
    seeds=sum((c[k] for k in ('validation_seeds','calibration_seeds','adaptation_seeds','evaluation_seeds')),[])
    if len(seeds)!=len(set(seeds)):raise ValueError('Reacher seed splits overlap')
    if c['snapshot_budgets'][-1]!=10000:raise ValueError('Unexpected Reacher final budget')
    return c


def fingerprint(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def protocol(task='halfcheetah'):
    if task == 'reacher':
        c = reacher_config()
        return dict(method='full_joint_rl', task='reacher', schedule='fixed',
                    conditions=[dict(family=f,parameter=p,severity=k) for f,p in c['families'].items() for k in c['severities']],
                    adaptation_seeds=c['adaptation_seeds'],
                    evaluation_seeds=c['evaluation_seeds'], horizon=50,
                    transitions=10000, replay_capacity=10200, first_update=75,
                    update_every=25, expected_updates=398,
                    optimizer='restore_selected_reacher_Adam_states_and_scale',
                    replay='fresh_completed_training_episodes_only_original_SliceSampler',
                    action_eval_mode=False, random_seed_exploration=False,
                    automatic_heldout_evaluation=False)
    if task != 'halfcheetah':raise ValueError(f'Unsupported task: {task}')
    c = base.configuration(argparse.Namespace(config=base.DEFAULT, command='run'))
    return dict(method='full_joint_rl', schedule='fixed', conditions=c['conditions'],
                adaptation_seeds=c['adaptation_seeds'], evaluation_seeds=c['evaluation_seeds'],
                horizon=1000, transitions=3000, replay_capacity=3003,
                first_update=1000, updates_per_transition_after_first_episode=1,
                expected_updates=2001, optimizer='restore_both_750k_Adam_states_and_scale',
                replay='fresh_completed_training_episodes_only_original_SliceSampler',
                action_eval_mode=False, random_seed_exploration=False,
                automatic_heldout_evaluation=False)


def gpu_status(index, minimum_mib=8192, allow_shared=False):
    """Check free memory and, unless explicitly shared, GPU occupancy."""
    query = subprocess.check_output(['nvidia-smi', f'--id={index}',
        '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid, free, utilization = [x.strip() for x in query.strip().split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    other = []; own = False
    for line in processes.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields)==2 and fields[0]==uuid:
            if int(fields[1])==os.getpid():own=True
            else:other.append(int(fields[1]))
    occupied = bool(other) or (not own and int(utilization)>10)
    if int(free)<minimum_mib or (occupied and not allow_shared):
        raise RuntimeError(f'GPU {index} unavailable: other PIDs={other}, free={free} MiB, utilization={utilization}%. No training started on this GPU.')
    return dict(index=index, uuid=uuid, free_mib=int(free), utilization_percent=int(utilization),
                other_pids=other, allow_shared_gpu=allow_shared, minimum_free_mib=minimum_mib)


def sources(task='halfcheetah'):
    paths = [Path(__file__), Path(base.__file__), base.DEFAULT,
             base.ROOT/'tdmpc2/tdmpc2.py', base.ROOT/'tdmpc2/common/buffer.py',
             base.ROOT/'tdmpc2/common/world_model.py',
             base.ROOT/'experiments/nsgym/scripts/training_support.py',
             base.ROOT/'tdmpc2/envs/nsgym/severity.py',
             base.ROOT/'experiments/nsgym/configs/physical_severity_grid.yaml',
             base.ROOT/'experiments/nsgym/configs/evaluation_seed_splits.yaml']
    if task == 'reacher':
        paths = [Path(__file__), base.ROOT/'experiments/nsgym/scripts/reacher_pipeline.py',
                 base.ROOT/'experiments/nsgym/configs/reacher_pipeline_v1.json',
                 base.ROOT/'tdmpc2/envs/nsgym/reacher_severity.py',
                 base.ROOT/'tdmpc2/envs/nsgym/reacher.py',
                 base.ROOT/'experiments/nsgym/scripts/training_support.py',
                 base.ROOT/'tdmpc2/tdmpc2.py', base.ROOT/'tdmpc2/common/buffer.py',
                 base.ROOT/'tdmpc2/common/world_model.py']
    return {str(p.relative_to(base.ROOT)):base.digest(p) for p in paths}


def checkpoint_identity(a, c):
    if c.get('task') == 'reacher':
        if a.selection is None:raise ValueError('Reacher requires --selection from nominal validation')
        selected = base.read(a.selection)
        if Path(selected['checkpoint']).resolve() != a.checkpoint.resolve():
            raise ValueError('Checkpoint differs from nominal Reacher selection')
        if selected['pipeline_config_fingerprint'] != fingerprint(reacher_config()):
            raise ValueError('Reacher selection pipeline config mismatch')
        if base.digest(a.checkpoint/'model.pt') != selected['model_sha256']:
            raise ValueError('Reacher checkpoint hash mismatch')
        return selected['model_sha256']
    if base.digest(a.checkpoint/'model.pt') != base.SHA:
        raise ValueError('Wrong nominal HalfCheetah checkpoint')
    return base.SHA


def should_update(transition, c, replay_episodes):
    if not replay_episodes or transition < c['first_update']:
        return False
    return c.get('update_every', 1) == 1 or transition % c['update_every'] == 0


def expected_updates(budget, c):
    cadence=c.get('update_every',1)
    if budget < c['first_update']:return 0
    return budget//cadence-(c['first_update']-1)//cadence


def optimizer_steps(optimizer):
    return sorted({int(s['step'].item()) for s in optimizer.state.values() if 'step' in s})


def enable_training(agent):
    # load_checkpoint intentionally freezes everything for evaluation.
    for p in agent.model.parameters():
        p.requires_grad_(False)
    for optimizer in (agent.optim, agent.pi_optim):
        for group in optimizer.param_groups:
            for p in group['params']:
                p.requires_grad_(True)


def run_stream(a, c, identity, target, budget):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from common.buffer import Buffer
    from training_support import load_checkpoint, tree_hash

    class CheckedBuffer(Buffer):
        sample_contract = None

        def sample(self):
            result = super().sample()
            obs, action, reward, terminated, task = result
            expected = (self.cfg.horizon, self.cfg.batch_size)
            if obs.shape[:2] != (expected[0]+1, expected[1]) or action.shape[:2] != expected:
                raise ValueError('Original replay sample shape mismatch')
            for value in (obs, action, reward, terminated):
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite replay sample (dummy row or invalid transition)')
            if terminated.any():
                raise ValueError('Unexpected terminal in continuing-task replay')
            self.sample_contract = dict(obs_shape=list(obs.shape), action_shape=list(action.shape),
                                        reward_shape=list(reward.shape), device=str(obs.device))
            return result

    target.mkdir()
    started = time.perf_counter()
    agent, meta = load_checkpoint(a.checkpoint)
    task=c.get('task','halfcheetah')
    checkpoint_sha=checkpoint_identity(a,c)
    if meta['model_sha256'] != checkpoint_sha:
        raise ValueError('Loaded checkpoint identity mismatch')
    expected_config=dict(task='nsgym-reacher' if task=='reacher' else 'nsgym-halfcheetah',
                         episodic=False,horizon=3,batch_size=256,
                         episode_length=c['horizon'],action_dim=2 if task=='reacher' else 6,
                         num_samples=512,iterations=6,mpc=True)
    if task=='reacher':
        selected=base.read(a.selection)
        if (meta['config']['environment_id']!='Reacher-v5' or
                meta['config']['obs_shape']['state']!=[10] or
                selected['environment_config_fingerprint']!=fingerprint(meta['config'])):
            raise ValueError('Unexpected Reacher nominal configuration')
        for seeds in (c['adaptation_seeds'],c['evaluation_seeds']):
            if set(seeds)&set(meta['config']['eval_seeds']):
                raise ValueError('Reacher train-evaluation seeds overlap adaptation or evaluation')
            if any(meta['config']['seed']<=s<=meta['config']['seed']+meta['config']['steps'] for s in seeds):
                raise ValueError('Reacher training seeds overlap adaptation or evaluation')
    for key, expected in expected_config.items():
        if meta['config'][key] != expected:
            raise ValueError(f'Unexpected checkpoint {key}')
    for package in ('ns_gym', 'gymnasium', 'mujoco'):
        if importlib.metadata.version(package) != meta['provenance']['versions'][package]:
            raise ValueError(f'Environment version mismatch: {package}')
    enable_training(agent)
    base.seeded(identity['adaptation_seed'])
    before = {k:tree_hash(getattr(agent.model,k).state_dict()) for k in COMPONENTS}
    initial_optim = [optimizer_steps(o) for o in (agent.optim,agent.pi_optim)]
    cfg = deepcopy(agent.cfg)
    cfg.buffer_size = cfg.steps = c['replay_capacity']
    replay = CheckedBuffer(cfg)
    if task=='reacher':
        from envs.nsgym.reacher_severity import SeverityReacher
        env=SeverityReacher(parameter=identity['parameter'],severity=identity['severity'],
                            max_episode_steps=c['horizon'])
    else:
        env=base.SeverityHalfCheetah(parameter=identity['parameter'], severity=identity['severity'],
                                    schedule='fixed', max_episode_steps=c['horizon'])
    manifest = dict(identity=identity, protocol=c, checkpoint=str(Path(a.checkpoint).resolve()),
                    checkpoint_sha256=checkpoint_sha, optimizer_sha256=meta['optimizer_sha256'],
                    checkpoint_config=meta['config'], source_hashes=sources(task),
                    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
                    optimizer_initial_steps=initial_optim,
                    optimizer_groups=[[dict(lr=g['lr'],params=len(g['params'])) for g in o.param_groups]
                                      for o in (agent.optim,agent.pi_optim)],
                    nominal_parameter=env.nominal_value, evaluation_seed_replay_access=False,
                    nominal_selection=str(a.selection.resolve()) if a.selection else None)
    base.js(target/'manifest.json',manifest)
    torch.cuda.reset_peak_memory_stats()
    updates = 0; update_seconds = 0.; episode = 0; episode_return = 0.; completed = []

    def td(obs, action=None, reward=float('nan'), terminated=float('nan')):
        return TensorDict(dict(obs=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0),
            action=torch.full((1,expected_config['action_dim']),float('nan')) if action is None else torch.as_tensor(action,dtype=torch.float32).unsqueeze(0),
            reward=torch.tensor([reward],dtype=torch.float32),
            terminated=torch.tensor([terminated],dtype=torch.float32)),batch_size=[1])

    try:
        obs,_ = env.reset(seed=identity['adaptation_seed']); trajectory=[td(obs)]
        with (target/'training.jsonl').open('x') as log:
            for transition in range(1,budget+1):
                torch.compiler.cudagraph_mark_step_begin()
                action=agent.act(torch.as_tensor(obs,dtype=torch.float32),t0=len(trajectory)==1,
                                 eval_mode=False).detach().cpu().numpy()
                nxt,reward,terminated,truncated,info=env.step(action)
                if terminated or not np.isfinite(nxt).all() or not np.isfinite(reward) or not np.isfinite(action).all():
                    raise ValueError('Invalid training transition')
                trajectory.append(td(nxt,action,reward,terminated)); episode_return+=reward
                if truncated:
                    if len(trajectory)!=c['horizon']+1:
                        raise ValueError('Unexpected episode horizon')
                    replay.add(torch.cat(trajectory)); episode+=1
                    completed.append(dict(episode=episode,return_=episode_return,length=c['horizon']))
                metrics={}
                if should_update(transition,c,replay.num_eps):
                    torch.cuda.synchronize();begin=time.perf_counter()
                    raw=agent.update(replay)
                    metrics={k:float(v.item()) if isinstance(v,torch.Tensor) else float(v) for k,v in raw.items()}
                    if not all(math.isfinite(v) for v in metrics.values()):
                        raise ValueError('Nonfinite joint training loss or diagnostics')
                    torch.cuda.synchronize();update_seconds+=time.perf_counter()-begin;updates+=1
                log.write(json.dumps(dict(transition=transition,updates=updates,episode=episode,
                    replay_episodes=replay.num_eps,reward=reward,actual_parameter=env.readback(),
                    terminated=bool(terminated),truncated=bool(truncated),metrics=metrics),allow_nan=False)+'\n')
                if transition%100==0:
                    log.flush();print(f'{identity}: {transition}/{budget} transitions, {updates} updates',flush=True)
                obs=nxt
                if truncated and transition<budget:
                    obs,_=env.reset();trajectory=[td(obs)];episode_return=0.
        if updates!=expected_updates(budget,c):
            raise ValueError('Unexpected update count')
        after={k:tree_hash(getattr(agent.model,k).state_dict()) for k in COMPONENTS}
        changed={k:before[k]!=after[k] for k in COMPONENTS}
        if not all(changed.values()):
            raise ValueError(f'Full Joint RL component did not change: {changed}')
        for p in agent.model.parameters():
            if not torch.isfinite(p).all():raise ValueError('Nonfinite final parameter')
        final_optim=[optimizer_steps(o) for o in (agent.optim,agent.pi_optim)]
        for initial,final in zip(initial_optim,final_optim):
            if final!=sorted({v+updates for v in initial}):
                raise ValueError('Adam step counters did not advance as expected')
        if base.digest(a.checkpoint/'model.pt')!=checkpoint_sha or base.digest(a.checkpoint/'optimizers.pt')!=meta['optimizer_sha256']:
            raise ValueError('Source checkpoint changed during training')
        payload=dict(model=agent.model.state_dict(),optim=agent.optim.state_dict(),
            pi_optim=agent.pi_optim.state_dict(),scale=agent.scale.state_dict(),identity=identity,
            checkpoint_sha256=checkpoint_sha,configuration=meta['config'],protocol=c,
            transitions=budget,updates=updates,resume='evaluation_only_no_replay_or_environment_state')
        snapshot=target/f'snapshot-{budget:07d}.pt';torch.save(payload,snapshot)
        base.js(target/'completion.json',dict(status='PASS',transitions=budget,updates=updates,
            optimizer_step_calls=2*updates,completed_episodes=episode,replay_rows=len(replay._buffer),
            replay_sample_contract=replay.sample_contract,components_changed=changed,
            optimizer_initial_steps=initial_optim,optimizer_final_steps=final_optim,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            update_seconds=update_seconds,wall_seconds=time.perf_counter()-started,
            snapshot_sha256=base.digest(snapshot),heldout_evaluation='NOT_RUN'))
    finally:
        env.close()


def evaluation_sources(training_manifest):
    """Training sources must match; the runner itself gained this evaluation command."""
    recorded=training_manifest['source_hashes']
    runner=str(Path(__file__).relative_to(base.ROOT))
    for relative, expected in recorded.items():
        if relative != runner and base.digest(base.ROOT/relative)!=expected:
            raise ValueError(f'Training dependency changed after adaptation: {relative}')
    return dict(training_runner_sha256=recorded[runner],evaluation_runner_sha256=base.digest(__file__),
                unchanged_training_dependencies=True)


def evaluate_reacher_one(a,c,snapshot,target,selection,frozen_rows):
    import numpy as np
    import torch
    from training_support import tree_hash, learned_state
    import reacher_pipeline as reacher

    child=base.read(snapshot.parent/'completion.json')
    if child['status']!='PASS' or child['snapshot_sha256']!=base.digest(snapshot):
        raise ValueError(f'Incomplete or changed Joint RL snapshot: {snapshot}')
    payload=torch.load(snapshot,map_location='cpu',weights_only=False)
    identity=payload['identity']
    cond={k:identity[k] for k in ('family','parameter','severity')}
    expected=dict(**cond,adaptation_seed=identity['adaptation_seed'],method='full_joint_rl',schedule='fixed')
    expected_name=f'{cond["parameter"]}-k{cond["severity"]:g}-full-joint-seed{identity["adaptation_seed"]}'
    if (identity!=expected or payload['protocol']!=c or
            snapshot.parent.name!=expected_name or
            payload['checkpoint_sha256']!=selection['model_sha256'] or
            payload['configuration']!=selection['configuration'] or
            payload['transitions']!=c['transitions'] or
            payload['updates']!=c['expected_updates'] or
            child['updates']!=c['expected_updates']):
        raise ValueError('Joint RL snapshot identity, checkpoint or training budget mismatch')
    agent,meta=reacher.load(a.checkpoint,reacher.config(REACHER_CONFIG))
    if meta['model_sha256']!=selection['model_sha256']:
        raise ValueError('Loaded Reacher checkpoint differs from nominal selection')
    agent.model.load_state_dict(payload['model'])
    agent.optim.load_state_dict(payload['optim'])
    agent.pi_optim.load_state_dict(payload['pi_optim'])
    agent.scale.load_state_dict(payload['scale'])
    agent.model.eval()
    before=tree_hash(learned_state(agent))
    if before!=tree_hash({key:payload[key] for key in ('model','optim','pi_optim','scale')}):
        raise ValueError('Joint RL snapshot did not reload exactly')
    target.mkdir()
    matching=[x for x in frozen_rows if x['family']==cond['family'] and float(x['severity'])==cond['severity']]
    if sorted(int(x['seed']) for x in matching)!=sorted(c['evaluation_seeds']):
        raise ValueError('Frozen held-out bank does not match evaluation seeds')
    frozen_shift={int(x['seed']):x for x in matching}
    frozen_nominal={int(x['seed']):x for x in frozen_rows if x['family']=='D0' and float(x['severity'])==1.}
    if sorted(frozen_nominal)!=sorted(c['evaluation_seeds']):
        raise ValueError('Nominal Frozen held-out bank is incomplete')
    prediction=[];control=[];steps=[]
    for row in matching:
        path=Path(a.frozen)/row['data_file']
        if base.digest(path)!=row['data_sha256']:
            raise ValueError(f'Frozen held-out trajectory changed: {path}')
        with np.load(path) as data:
            errors=[reacher.error(agent,o,ac,n) for o,ac,n in zip(
                data['observations'][:-1],data['actions'],data['observations'][1:])]
        prediction.append(dict(**identity,budget=c['transitions'],seed=int(row['seed']),
            latent_mse=float(np.mean(errors)),frozen_latent_mse=float(row['latent_mse']),
            delta_heldout_latent_mse=float(np.mean(errors))-float(row['latent_mse']),
            samples=len(errors),nominal_parameter=float(row['nominal_parameter']),
            actual_parameter=float(row['actual_parameter']),source_sha256=row['data_sha256']))
    for label,condition in [('shifted',cond),('nominal',cond|dict(severity=1.))]:
        for seed in c['evaluation_seeds']:
            record,step_rows=reacher.episode(agent,condition,seed)
            baseline=frozen_shift[seed] if label=='shifted' else frozen_nominal[seed]
            control.append(dict(**(record|dict(family=cond['family'],severity=cond['severity'])),
                kind=label,method='full_joint_rl',adaptation_seed=identity['adaptation_seed'],
                budget=c['transitions'],declared_final_budget=c['transitions'],
                gradient_updates=payload['updates'],
                paired_return_difference=float(record['episode_return'])-float(baseline['episode_return']),
                frozen_episode_return=float(baseline['episode_return'])))
            steps.extend([dict(x,kind=label,method='full_joint_rl',
                               adaptation_seed=identity['adaptation_seed'],budget=c['transitions'])
                          for x in step_rows])
    if tree_hash(learned_state(agent))!=before:
        raise ValueError('Joint RL model or optimizers changed during held-out evaluation')
    if base.digest(snapshot)!=child['snapshot_sha256']:
        raise ValueError('Joint RL snapshot changed during held-out evaluation')
    if (base.digest(a.checkpoint/'model.pt')!=selection['model_sha256'] or
            base.digest(a.checkpoint/'optimizers.pt')!=meta['optimizer_sha256']):
        raise ValueError('Nominal checkpoint changed during held-out evaluation')
    reacher.table(target/'predict.csv',prediction)
    reacher.table(target/'control.csv',control)
    reacher.table(target/'steps.csv',steps)
    base.js(target/'completion.json',dict(status='PASS',snapshot=str(snapshot.resolve()),
        snapshot_sha256=child['snapshot_sha256'],identity=identity,
        evaluation_seeds=c['evaluation_seeds'],model_unchanged=True,
        heldout_evaluation='PASS'))
    return control,prediction


def evaluate_reacher(a,c,checkpoint_sha):
    import numpy as np
    import reacher_pipeline as reacher
    selection=base.read(a.selection)
    if a.frozen is None or (a.runs is None)==(a.snapshot is None):
        raise ValueError('Reacher evaluate requires --frozen and exactly one of --runs or --snapshot')
    reacher.gate(a.frozen,'frozen',reacher.config(REACHER_CONFIG),selection)
    frozen_rows=reacher.csvread(a.frozen/'reacher_frozen_severity.csv')
    if a.runs:
        training=base.read(a.runs/'manifest.json')
        done=base.read(a.runs/'completion.json')
        snapshots=sorted(a.runs.glob('*/snapshot-0010000.pt'))
        expected_names={f'{x["parameter"]}-k{x["severity"]:g}-full-joint-seed{seed}'
                        for x in c['conditions'] for seed in c['adaptation_seeds']}
        if (done['status']!='PASS' or set(done['completed'])!=expected_names or
                {x.parent.name for x in snapshots}!=expected_names):
            raise ValueError('Full Reacher Joint RL training matrix is incomplete')
    else:
        snapshots=[a.snapshot]
        training=base.read(a.snapshot.parent.parent/'manifest.json')
    if (training['command']!='run' or training['protocol']!=c or
            training['checkpoint_sha256']!=checkpoint_sha):
        raise ValueError('Joint RL training manifest mismatch')
    source_audit=evaluation_sources(training)
    if a.output.exists():raise FileExistsError('Use a new evaluation output directory')
    a.output.mkdir(parents=True)
    base.js(a.output/'manifest.json',dict(command='evaluate',task='reacher',protocol=c,
        checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=checkpoint_sha,
        selection=str(a.selection.resolve()),frozen=str(a.frozen.resolve()),
        training=str((a.runs or a.snapshot.parent.parent).resolve()),
        snapshot_count=len(snapshots),evaluation_seeds=c['evaluation_seeds'],
        source_audit=source_audit))
    controls=[];predictions=[];completed=[]
    try:
        for snapshot in snapshots:
            cr,pr=evaluate_reacher_one(a,c,snapshot,a.output/snapshot.parent.name,
                                       selection,frozen_rows)
            controls.extend(cr);predictions.extend(pr);completed.append(snapshot.parent.name)
            print('PASS',snapshot.parent.name,flush=True)
        if len({(x['family'],x['severity'],x['adaptation_seed'],x['seed'],x['kind']) for x in controls})!=len(controls):
            raise ValueError('Duplicate held-out control evaluation')
        if len({(x['family'],x['severity'],x['adaptation_seed'],x['seed']) for x in predictions})!=len(predictions):
            raise ValueError('Duplicate held-out prediction evaluation')
        reacher.table(a.output/'reacher_joint_control.csv',controls)
        reacher.table(a.output/'reacher_joint_prediction.csv',predictions)
        summaries=[]
        for cond in c['conditions']:
            for kind in ('shifted','nominal'):
                for metric in ('episode_return','paired_return_difference','distance_final','latent_mse'):
                    per=[float(np.mean([float(x[metric]) for x in controls if
                          x['family']==cond['family'] and x['severity']==cond['severity'] and
                          x['adaptation_seed']==seed and x['kind']==kind]))
                         for seed in c['adaptation_seeds'] if any(x['family']==cond['family'] and
                          x['severity']==cond['severity'] and x['adaptation_seed']==seed and
                          x['kind']==kind for x in controls)]
                    if per:summaries.append(dict(**cond,kind=kind,metric=metric,**reacher.stats(per)))
        if summaries:reacher.table(a.output/'reacher_joint_seed_summary.csv',summaries)
        (a.output/'REACHER_JOINT_RL_EVALUATION.md').write_text(
            '# Reacher Full Joint RL held-out evaluation\n\n'
            f'{len(snapshots)} final 10k snapshots evaluated on the fixed 20-seed held-out bank. '
            'Each adapted model is evaluated under its fixed shift and nominal physics with updates disabled. '
            'Prediction uses hashed Frozen trajectories only. Per-seed paired return differences use the '
            'corresponding Frozen condition and one canonical nominal Frozen bank; repeated k=1 labels are '
            'not independent nominal samples. Summary statistics first average evaluation seeds within '
            'each adaptation replicate, then use the three adaptation seeds as the sampling unit. '
            'Training used eval_mode=False actions; dynamics-only training used eval_mode=True actions, '
            'so online trajectories and compute are not matched. This evaluation does not select a '
            'checkpoint or severity based on shifted return.\n')
        base.js(a.output/'completion.json',dict(status='PASS',completed=completed,
            snapshots=len(snapshots),control_rows=len(controls),prediction_rows=len(predictions),
            evaluation_seeds=c['evaluation_seeds'],heldout_evaluation='PASS'))
    except Exception as exc:
        base.js(a.output/'failure.json',dict(error=type(exc).__name__,message=str(exc),
            completed=completed,time=time.time()))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','gpu-check','smoke','run','status','evaluate'])
    p.add_argument('--task',choices=['halfcheetah','reacher'],default='halfcheetah')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--selection',type=Path,help='Required for Reacher: nominal selected_reacher_checkpoint.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--gpu',type=int,default=0,help='Physical nvidia-smi GPU index')
    p.add_argument('--allow-shared-gpu',action='store_true',
                   help='Allow concurrent compute processes/high utilization; retain free-memory checks')
    p.add_argument('--smoke-result',type=Path,help='PASS smoke output from this runner revision')
    p.add_argument('--runs',type=Path,help='PASS Full Joint RL run directory for held-out evaluation')
    p.add_argument('--snapshot',type=Path,help='One completed final Joint RL snapshot for evaluation pilot')
    p.add_argument('--frozen',type=Path,help='PASS Reacher Frozen stage with held-out trajectories')
    a=p.parse_args();c=protocol(a.task)
    if a.task=='reacher' and a.command in ('smoke','run','evaluate') and a.selection is None:
        p.error('--selection is required for Reacher smoke/run/evaluate')
    if a.command=='evaluate' and a.task!='reacher':p.error('Held-out evaluation currently supports Reacher only')
    if a.command=='plan':
        smoke_budget=100 if a.task=='reacher' else 1004
        smoke_condition=(next(x for x in c['conditions'] if x['family']=='D0' and x['severity']==1.5)
                         if a.task=='reacher' else dict(parameter='bthigh_damping',severity=2.))
        streams=len(c['conditions'])*len(c['adaptation_seeds'])
        print(json.dumps(dict(protocol=c,streams=streams,total_transitions=streams*c['transitions'],
            total_updates=streams*c['expected_updates'],
            smoke=dict(condition=smoke_condition,
                       transitions=smoke_budget,updates=expected_updates(smoke_budget,c))),indent=2));return
    if a.command=='gpu-check':
        print(json.dumps(gpu_status(a.gpu,allow_shared=a.allow_shared_gpu),indent=2));return
    if a.output is None:p.error('--output is required')
    if a.command=='status':
        done=list(a.output.glob('*/completion.json'))
        print(json.dumps(dict(pass_streams=sum(base.read(f).get('status')=='PASS' for f in done),
            batch=base.read(a.output/'completion.json') if (a.output/'completion.json').exists() else 'INCOMPLETE'),indent=2));return
    if a.checkpoint is None:p.error('--checkpoint is required')
    if a.output.exists():raise FileExistsError('Use a new versioned output directory')
    checkpoint_sha=checkpoint_identity(a,c)
    if a.command=='evaluate':
        gpu=gpu_status(a.gpu,8192,a.allow_shared_gpu)
        os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
        base.dependencies()
        evaluate_reacher(a,c,checkpoint_sha)
        return
    minimum=8192
    if a.command=='run':
        if a.smoke_result is None:p.error('run requires --smoke-result from a real CUDA smoke')
        smoke=base.read(a.smoke_result/'completion.json')
        sm=base.read(a.smoke_result/'manifest.json')
        if (smoke['status']!='PASS' or sm['command']!='smoke' or
                sm['source_hashes']!=sources(a.task) or sm['protocol']!=c or
                sm['checkpoint_sha256']!=checkpoint_sha):
            raise ValueError('Matching successful smoke is required')
        minimum=max(minimum,math.ceil(smoke['peak_reserved_mib']*1.5+2048))
    gpu=gpu_status(a.gpu,minimum,a.allow_shared_gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    base.dependencies()
    a.output.mkdir(parents=True)
    base.js(a.output/'manifest.json',dict(command=a.command,protocol=c,gpu=gpu,source_hashes=sources(a.task),
        checkpoint_sha256=checkpoint_sha,checkpoint=str(a.checkpoint.resolve()),
        selection=str(a.selection.resolve()) if a.selection else None,heldout_evaluation='NOT_RUN'))
    conditions=c['conditions'] if a.command=='run' else [
        next(x for x in c['conditions'] if x['family']=='D0' and x['severity']==1.5)
        if a.task=='reacher' else dict(parameter='bthigh_damping',severity=2.)]
    seeds=c['adaptation_seeds'] if a.command=='run' else c['adaptation_seeds'][:1]
    passed=[];peak=0.
    for cond in conditions:
        for seed in seeds:
            gpu_status(a.gpu,minimum,a.allow_shared_gpu)
            identity=dict(**cond,adaptation_seed=seed,method='full_joint_rl',schedule='fixed')
            name=f'{cond["parameter"]}-k{cond["severity"]:g}-full-joint-seed{seed}'
            try:
                run_stream(a,c,identity,a.output/name,c['transitions'] if a.command=='run' else 100 if a.task=='reacher' else 1004)
            except Exception as exc:
                base.js(a.output/'completion.json',dict(status='PARTIAL',completed=passed,
                    failed=dict(identity=identity,error=repr(exc)),heldout_evaluation='NOT_RUN'))
                raise
            gc.collect();base.torch.cuda.empty_cache()
            passed.append(name);peak=max(peak,base.read(a.output/name/'completion.json')['peak_reserved_mib'])
            print('PASS',name,flush=True)
    base.js(a.output/'completion.json',dict(status='PASS',completed=passed,peak_reserved_mib=peak,
        heldout_evaluation='NOT_RUN'))


if __name__=='__main__':main()
