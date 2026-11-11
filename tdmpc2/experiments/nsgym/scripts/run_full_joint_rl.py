#!/usr/bin/env python3
"""Independent Full Joint RL adaptation; never starts held-out evaluation."""
import argparse
from copy import deepcopy
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

import run_dynamics_only_ttt as base

COMPONENTS = ('_encoder', '_dynamics', '_reward', '_Qs', '_pi', '_target_Qs_params')


def protocol():
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


def sources():
    paths = [Path(__file__), Path(base.__file__), base.DEFAULT,
             base.ROOT/'tdmpc2/tdmpc2.py', base.ROOT/'tdmpc2/common/buffer.py',
             base.ROOT/'tdmpc2/common/world_model.py',
             base.ROOT/'experiments/nsgym/scripts/training_support.py',
             base.ROOT/'tdmpc2/envs/nsgym/severity.py',
             base.ROOT/'experiments/nsgym/configs/physical_severity_grid.yaml',
             base.ROOT/'experiments/nsgym/configs/evaluation_seed_splits.yaml']
    return {str(p.relative_to(base.ROOT)):base.digest(p) for p in paths}


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
                raise ValueError('Unexpected HalfCheetah terminal in continuing-task replay')
            self.sample_contract = dict(obs_shape=list(obs.shape), action_shape=list(action.shape),
                                        reward_shape=list(reward.shape), device=str(obs.device))
            return result

    target.mkdir()
    started = time.perf_counter()
    agent, meta = load_checkpoint(a.checkpoint)
    if meta['model_sha256'] != base.SHA:
        raise ValueError('Only nominal 750k primary checkpoint is allowed')
    for key, expected in dict(task='nsgym-halfcheetah', episodic=False, horizon=3,
                              batch_size=256, episode_length=1000, action_dim=6,
                              num_samples=512,iterations=6,mpc=True).items():
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
    env = base.SeverityHalfCheetah(parameter=identity['parameter'], severity=identity['severity'],
                                 schedule='fixed', max_episode_steps=1000)
    manifest = dict(identity=identity, protocol=c, checkpoint=str(Path(a.checkpoint).resolve()),
                    checkpoint_sha256=base.SHA, optimizer_sha256=meta['optimizer_sha256'],
                    checkpoint_config=meta['config'], source_hashes=sources(),
                    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
                    optimizer_initial_steps=initial_optim,
                    optimizer_groups=[[dict(lr=g['lr'],params=len(g['params'])) for g in o.param_groups]
                                      for o in (agent.optim,agent.pi_optim)],
                    nominal_parameter=env.nominal_value, evaluation_seed_replay_access=False)
    base.js(target/'manifest.json',manifest)
    torch.cuda.reset_peak_memory_stats()
    updates = 0; update_seconds = 0.; episode = 0; episode_return = 0.; completed = []

    def td(obs, action=None, reward=float('nan'), terminated=float('nan')):
        return TensorDict(dict(obs=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0),
            action=torch.full((1,6),float('nan')) if action is None else torch.as_tensor(action,dtype=torch.float32).unsqueeze(0),
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
                    if len(trajectory)!=1001:
                        raise ValueError('Unexpected episode horizon')
                    replay.add(torch.cat(trajectory)); episode+=1
                    completed.append(dict(episode=episode,return_=episode_return,length=1000))
                metrics={}
                if replay.num_eps:
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
        if updates!=max(0,budget-999):
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
        if base.digest(a.checkpoint/'model.pt')!=base.SHA or base.digest(a.checkpoint/'optimizers.pt')!=meta['optimizer_sha256']:
            raise ValueError('Source checkpoint changed during training')
        payload=dict(model=agent.model.state_dict(),optim=agent.optim.state_dict(),
            pi_optim=agent.pi_optim.state_dict(),scale=agent.scale.state_dict(),identity=identity,
            checkpoint_sha256=base.SHA,configuration=meta['config'],protocol=c,
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','gpu-check','smoke','run','status'])
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--gpu',type=int,default=0,help='Physical nvidia-smi GPU index')
    p.add_argument('--allow-shared-gpu',action='store_true',
                   help='Allow concurrent compute processes/high utilization; retain free-memory checks')
    p.add_argument('--smoke-result',type=Path,help='PASS smoke output from this runner revision')
    a=p.parse_args();c=protocol()
    if a.command=='plan':
        print(json.dumps(dict(protocol=c,streams=36,total_transitions=108000,total_updates=72036,
            smoke=dict(parameter='bthigh_damping',severity=2.,transitions=1004,updates=5)),indent=2));return
    if a.command=='gpu-check':
        print(json.dumps(gpu_status(a.gpu,allow_shared=a.allow_shared_gpu),indent=2));return
    if a.output is None:p.error('--output is required')
    if a.command=='status':
        done=list(a.output.glob('*/completion.json'))
        print(json.dumps(dict(pass_streams=sum(base.read(f).get('status')=='PASS' for f in done),
            batch=base.read(a.output/'completion.json') if (a.output/'completion.json').exists() else 'INCOMPLETE'),indent=2));return
    if a.checkpoint is None:p.error('--checkpoint is required')
    if a.output.exists():raise FileExistsError('Use a new versioned output directory')
    if base.digest(a.checkpoint/'model.pt')!=base.SHA:raise ValueError('Wrong nominal checkpoint')
    minimum=8192
    if a.command=='run':
        if a.smoke_result is None:p.error('run requires --smoke-result from a real CUDA smoke')
        smoke=base.read(a.smoke_result/'completion.json')
        sm=base.read(a.smoke_result/'manifest.json')
        if smoke['status']!='PASS' or sm['command']!='smoke' or sm['source_hashes']!=sources() or sm['protocol']!=c:
            raise ValueError('Matching successful smoke is required')
        minimum=max(minimum,math.ceil(smoke['peak_reserved_mib']*1.5+2048))
    gpu=gpu_status(a.gpu,minimum,a.allow_shared_gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    base.dependencies()
    a.output.mkdir(parents=True)
    base.js(a.output/'manifest.json',dict(command=a.command,protocol=c,gpu=gpu,source_hashes=sources(),
        checkpoint_sha256=base.SHA,checkpoint=str(a.checkpoint.resolve()),heldout_evaluation='NOT_RUN'))
    conditions=c['conditions'] if a.command=='run' else [dict(parameter='bthigh_damping',severity=2.)]
    seeds=c['adaptation_seeds'] if a.command=='run' else c['adaptation_seeds'][:1]
    passed=[];peak=0.
    for cond in conditions:
        for seed in seeds:
            gpu_status(a.gpu,minimum,a.allow_shared_gpu)
            identity=dict(**cond,adaptation_seed=seed,method='full_joint_rl',schedule='fixed')
            name=f'{cond["parameter"]}-k{cond["severity"]:g}-full-joint-seed{seed}'
            try:
                run_stream(a,c,identity,a.output/name,3000 if a.command=='run' else 1004)
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
