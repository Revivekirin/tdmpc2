"""Configuration, evaluation and checkpoint IO around unchanged TD-MPC2."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

from research_utils import ROOT, metadata, write_json, digest, bootstrap
import numpy as np
import torch
from omegaconf import OmegaConf
from common import MODEL_SIZE
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from envs.nsgym import make_env


def require_cuda(device):
    if device!='cuda:0':raise ValueError('Original agent, RunningScale and replay require cuda:0; no CPU substitution is implemented')
    if not torch.cuda.is_available():raise RuntimeError('BLOCKED: CUDA is unavailable. Run the unchanged TD-MPC2 training/evaluation on a CUDA host.')


def config_from_file(path, smoke=False):
    base=OmegaConf.to_container(OmegaConf.load(ROOT/'tdmpc2/config.yaml'),resolve=False)
    base={k:v for k,v in base.items() if k!='defaults' and v!='???'}
    override=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
    base.update(override)
    if smoke:
        base.update(steps=35,seed_steps=16,seed_exploration_transitions=17,batch_size=8,buffer_size=256,eval_freq=35,
                    eval_episodes=2,checkpoint_freq=35,model_size=1,compile=False,
                    num_samples=32,num_elites=8,num_pi_trajs=4,iterations=2)
        base['nsgym']=dict(mode='nominal',mass_scale=1.,max_episode_steps=8)
        base['eval_seeds']=base['eval_seeds'][:2]
    base.update(MODEL_SIZE[base['model_size']])
    base.update(task_dim=0,tasks=[base['task']],bin_size=(base['vmax']-base['vmin'])/(base['num_bins']-1),
                smoke=smoke)
    validate_config(base)
    return base


def validate_config(c):
    supported={'nsgym-halfcheetah':'HalfCheetah-v5','nsgym-pusher':'Pusher-v5'}
    if supported.get(c['task'])!=c['environment_id']:raise ValueError('Unsupported NS-Gym task/environment pair')
    if c['nsgym']['mode']!='nominal' or c['nsgym']['mass_scale']!=1:raise ValueError('Training must be nominal')
    if c['device']!='cuda:0':raise ValueError('Original implementation requires cuda:0')
    if c['seed_exploration_transitions']!=c['seed_steps']+1:raise ValueError('Original zero-based seed threshold requires seed_exploration_transitions = seed_steps + 1')
    if c['updates_per_transition']!=1:raise ValueError('Preserve the original update ratio of 1 after seed pretraining')
    for key in ['steps','seed_steps','batch_size','buffer_size','eval_freq','eval_episodes','checkpoint_freq','horizon']:
        if not isinstance(c[key],int) or c[key]<=0:raise ValueError(f'{key} must be a positive integer')
    if c['seed_steps']<c['nsgym']['max_episode_steps'] or c['steps']<=c['seed_steps']:
        raise ValueError('Need a complete seed episode before updates and a post-seed budget')
    if c['nsgym']['max_episode_steps']<=c['horizon']:raise ValueError('Episode length must exceed replay horizon')
    if c['num_pi_trajs']>c['num_samples'] or c['num_elites']>c['num_samples']:raise ValueError('Invalid MPC sample counts')
    seeds=c['eval_seeds']
    if len(seeds)!=c['eval_episodes'] or len(set(seeds))!=len(seeds):raise ValueError('Distinct explicit evaluation seeds required')
    # Training resets use seed + episode index. Conservatively exclude the full budget range.
    if any(c['seed']<=s<=c['seed']+c['steps'] for s in seeds):
        raise ValueError('Evaluation seeds overlap possible training reset seeds')


def build(config):
    cfg=cfg_to_dataclass(OmegaConf.create(config))
    env=make_env(cfg)
    cfg.seed_steps=config['seed_steps'] # Adapter convenience default must not replace the declared budget.
    return cfg,env


def tree_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            x=v.detach().cpu().contiguous();h.update(str((x.dtype,x.shape)).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def learned_state(agent):
    return dict(model=agent.model.state_dict(),optim=agent.optim.state_dict(),pi_optim=agent.pi_optim.state_dict(),scale=agent.scale.state_dict())


def evaluate(agent, config, seeds, scale=1., random_actions=False, fixed=False, zero_actions=False):
    """Separate environment; independent seed per episode; preserve training RNG."""
    if random_actions and zero_actions:raise ValueError('Choose one baseline action mode')
    rng_state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    before=tree_hash(learned_state(agent)) if agent is not None else None
    records=[]; started=time.perf_counter()
    was_training=agent.model.training if agent is not None else False
    planning_state=agent._prev_mean.detach().clone() if agent is not None else None
    try:
        if agent is not None:agent.model.eval()
        for seed in seeds:
            set_seed(seed)
            c=deepcopy(config);c['seed']=seed;c['nsgym']=dict(mode='fixed' if fixed or scale!=1 else 'nominal',mass_scale=float(scale),max_episode_steps=config['nsgym']['max_episode_steps'])
            logs=[];cfg=cfg_to_dataclass(OmegaConf.create(c));env=make_env(cfg,recorder=logs.append)
            try:
                obs=env.reset(seed=seed);actions=[];vel=[];run=[];ctrl=[];rewards=[];terminated=False
                is_pusher=config['task']=='nsgym-pusher'
                goal_distances=[];near=[]
                initial_goal_distance=float(np.linalg.norm(obs.cpu().numpy()[-6:-3]-obs.cpu().numpy()[-3:])) if is_pusher else None
                rng=np.random.default_rng(seed)
                with torch.no_grad():
                    for t in range(cfg.episode_length):
                        action=(torch.zeros(cfg.action_dim,dtype=torch.float32) if zero_actions else
                                torch.tensor(rng.uniform(-1,1,cfg.action_dim),dtype=torch.float32) if random_actions else
                                agent.act(obs,t0=t==0,eval_mode=True))
                        obs,reward,done,info=env.step(action)
                        if not torch.isfinite(obs).all() or not torch.isfinite(reward):raise ValueError('Non-finite evaluation rollout')
                        actions.append(action.cpu().numpy());rewards.append(float(reward))
                        if is_pusher:
                            state=obs.cpu().numpy()
                            goal_distances.append(float(np.linalg.norm(state[-6:-3]-state[-3:])))
                            run.append(float(info['reward_dist']))
                            near.append(float(info['reward_near']))
                        else:
                            vel.append(float(info['x_velocity']));run.append(float(info['reward_run']))
                        ctrl.append(float(info['reward_ctrl']))
                        terminated=bool(info['terminated'])
                        if done:break
                mass_key='link_mass' if is_pusher else 'torso_mass'
                target=logs[0][mass_key]*scale
                assert all(abs(r[mass_key]-target)<1e-10 for r in logs if r['event']=='step')
                a=np.array(actions)
                record=dict(seed=int(seed),mass_scale=float(scale),mass_parameter='r_shoulder_pan_link_mass' if is_pusher else 'torso_mass',
                    mass_value=float(logs[1][mass_key]),mass_readback_min=min(r[mass_key] for r in logs[1:]),mass_readback_max=max(r[mass_key] for r in logs[1:]),shift_transition=0 if fixed or scale!=1 else None,
                    episode_return=sum(rewards),episode_length=len(rewards),reward_control=sum(ctrl),control_effort=float(np.square(a).sum()),
                    action_abs_mean=float(np.abs(a).mean()),action_std=float(a.std()),
                    action_change_rms=float(np.sqrt(np.mean(np.diff(a,axis=0)**2))) if len(a)>1 else 0.,
                    terminated=terminated,truncated=bool(info['truncated']),random_actions=random_actions,
                    zero_actions=zero_actions)
                if is_pusher:
                    record.update(goal_distance_initial=initial_goal_distance,
                                  goal_distance_final=goal_distances[-1],
                                  goal_progress=initial_goal_distance-goal_distances[-1],
                                  goal_distance_mean=float(np.mean(goal_distances)),
                                  reward_dist=sum(run),reward_near=sum(near))
                else:
                    record.update(torso_mass=record['mass_value'],forward_velocity=float(np.mean(vel)),
                                  velocity_std=float(np.std(vel)),reward_run=sum(run))
                records.append(record)
            finally:env.close()
        if agent is not None:assert tree_hash(learned_state(agent))==before,'Learned/optimizer state changed during evaluation'
    finally:
        random.setstate(rng_state[0]);np.random.set_state(rng_state[1]);torch.set_rng_state(rng_state[2])
        if rng_state[3]:torch.cuda.set_rng_state_all(rng_state[3])
        if agent is not None:
            agent.model.train(was_training)
            agent._prev_mean.copy_(planning_state)
    return records,time.perf_counter()-started


def checkpoint(agent, cfg, directory, transitions, updates, provenance):
    target=Path(directory)/f'checkpoint-{transitions:09d}'
    target.mkdir(exist_ok=False)
    agent.save(target/'model.pt')
    torch.save(dict(optim=agent.optim.state_dict(),pi_optim=agent.pi_optim.state_dict(),scale=agent.scale.state_dict()),target/'optimizers.pt')
    effective=vars(cfg).copy()
    write_json(target/'metadata.json',dict(config=effective,environment_transitions=transitions,
        optimizer_updates=updates,optimizer_steps_per_update=2,model_sha256=digest(target/'model.pt'),
        optimizer_sha256=digest(target/'optimizers.pt'),provenance=provenance,
        continuation='Optimizer states saved; exact resume is NOT supported (replay and environment/RNG state not checkpointed).'))
    from tdmpc2 import TDMPC2
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    fresh=TDMPC2(cfg_to_dataclass(OmegaConf.create(effective)))
    fresh.load(target/'model.pt')
    saved=torch.load(target/'optimizers.pt',map_location=fresh.device,weights_only=False)
    fresh.optim.load_state_dict(saved['optim']);fresh.pi_optim.load_state_dict(saved['pi_optim']);fresh.scale.load_state_dict(saved['scale'])
    assert tree_hash(learned_state(fresh))==tree_hash(learned_state(agent))
    assert all(torch.isfinite(p).all() for p in fresh.model.parameters())
    del fresh
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    return target


def load_checkpoint(path):
    path=Path(path);meta=json.loads((path/'metadata.json').read_text())
    assert digest(path/'model.pt')==meta['model_sha256'],'Checkpoint identity mismatch'
    assert digest(path/'optimizers.pt')==meta['optimizer_sha256'],'Optimizer checkpoint identity mismatch'
    require_cuda(meta['config']['device'])
    from tdmpc2 import TDMPC2
    cfg,env=build(meta['config']);env.close()
    if cfg.action_dim!=meta['config']['action_dim'] or json.loads(json.dumps(cfg.obs_shape))!=meta['config']['obs_shape']:raise ValueError('Checkpoint dimensions mismatch')
    agent=TDMPC2(cfg);agent.load(path/'model.pt')
    saved=torch.load(path/'optimizers.pt',map_location=agent.device,weights_only=False)
    agent.optim.load_state_dict(saved['optim']);agent.pi_optim.load_state_dict(saved['pi_optim']);agent.scale.load_state_dict(saved['scale'])
    for p in agent.model.parameters():p.requires_grad_(False)
    return agent,meta
