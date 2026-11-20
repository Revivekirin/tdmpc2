"""Controlled old-loop versus canonical-runner execution and Joint update parity."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'tdmpc2'))
sys.path.insert(0, str(PROJECT/'experiments/nsgym/scripts'))

from common.stage3a import equal_state
from research.dynamics_shift.joint import JointAdaptation
from research.dynamics_shift.runner import AdaptationRunner
from research.dynamics_shift.provenance import (
    PRE_RUNNER_MIGRATION_SHA256, RUNNER_MIGRATION_SHA256, compatible_sources)


class Environment:
    def __init__(self, horizon):
        self.horizon, self.position, self.resets, self.closed = horizon, 0, [], False

    def reset(self, seed=None):
        self.resets.append(seed)
        self.position = 0
        return np.array([float(len(self.resets))], dtype=np.float32), {}

    def step(self, action):
        self.position += 1
        nxt = np.array([float(len(self.resets)) + self.position/10], dtype=np.float32)
        return nxt, float(action[0]) + self.position, False, self.position == self.horizon, {}

    def close(self):
        self.closed = True


class Agent:
    def __init__(self):
        self.weight = torch.tensor(1.)
        self.optimizer_steps = 0

    def act(self, obs, *, t0, eval_mode):
        assert not eval_mode
        return np.array([float(torch.rand(())) + float(obs[0]) + int(t0)], dtype=np.float32)

    def update(self, replay):
        selected = replay.sample()
        self.weight += selected
        self.optimizer_steps += 1
        return {'loss': self.weight.clone()}


class Replay:
    def __init__(self):
        self.episodes = []
        self.num_eps = 0
        self.sampled = []

    def add(self, episode):
        self.episodes.append(deepcopy(episode))
        self.num_eps += 1

    def sample(self):
        index = random.randrange(self.num_eps)
        self.sampled.append(index)
        return torch.tensor(sum(row[2] for row in self.episodes[index]), dtype=torch.float32)


def execute_old(budget, cfg):
    """Literal controlled form of the historical Joint stream loop."""
    env, agent, replay = Environment(cfg['horizon']), Agent(), Replay()
    obs, _ = env.reset(seed=71)
    trajectory = [(float(obs[0]), None, 0.)]
    records, checkpoints, updates = [], [], 0
    for transition in range(1, budget+1):
        action = agent.act(obs, t0=len(trajectory)==1, eval_mode=False)
        nxt, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((float(nxt[0]), float(action[0]), reward))
        if truncated:
            replay.add(trajectory)
        metrics = {}
        if replay.num_eps and transition >= cfg['first_update'] and (
                cfg['update_every']==1 or transition % cfg['update_every']==0):
            metrics = agent.update(replay)
            updates += 1
        records.append((transition, float(action[0]), float(nxt[0]), reward,
                        replay.num_eps, updates, float(metrics['loss']) if metrics else None))
        if transition in (cfg['horizon'], budget):
            checkpoints.append((transition, deepcopy(replay.episodes), agent.weight.clone()))
        obs = nxt
        if truncated and transition < budget:
            obs, _ = env.reset()
            trajectory = [(float(obs[0]), None, 0.)]
    state = (torch.get_rng_state().clone(), random.getstate())
    env.close()
    return env, agent, replay, records, checkpoints, state


def execute_new(budget, cfg):
    env, agent, replay = Environment(cfg['horizon']), Agent(), Replay()
    strategy = JointAdaptation(agent, replay, cfg)
    trajectory = []
    records, checkpoints = [], []

    def reset(obs, _episode):
        nonlocal trajectory
        trajectory = [(float(obs[0]), None, 0.)]

    def observe(event):
        trajectory.append((float(event.next_observation[0]), float(event.action[0]), event.reward))
        if event.truncated:
            replay.add(trajectory)
        return strategy.observe()

    runner = AdaptationRunner(env,
        lambda obs, _step, _episode: agent.act(obs, t0=len(trajectory)==1, eval_mode=False),
        observe, horizon=cfg['horizon'], initial_seed=71, on_reset=reset,
        reset_on=lambda event:bool(event.truncated))

    def record(event, metrics):
        transition = event.index+1
        records.append((transition, float(event.action[0]), float(event.next_observation[0]),
                        event.reward, replay.num_eps, strategy.updates,
                        float(metrics['loss']) if metrics else None))

    def checkpoint(event, _metrics):
        transition = event.index+1
        if transition in (cfg['horizon'], budget):
            checkpoints.append((transition, deepcopy(replay.episodes), agent.weight.clone()))

    runner.run(budget, record, checkpoint=checkpoint)
    state = (torch.get_rng_state().clone(), random.getstate())
    assert strategy.transitions == budget
    assert strategy.updates == strategy.expected_updates(budget)
    return env, agent, replay, records, checkpoints, state


@pytest.mark.parametrize('configuration,budget,expected', [
    ({'horizon':4,'first_update':4,'update_every':1}, 11, 8),
    ({'horizon':3,'first_update':5,'update_every':2}, 12, 4),
])
def test_old_and_new_joint_stream_event_rng_replay_and_checkpoint_parity(configuration,budget,expected):
    torch.manual_seed(500);random.seed(501)
    old = execute_old(budget, configuration)
    torch.manual_seed(500);random.seed(501)
    new = execute_new(budget, configuration)
    assert old[0].resets == new[0].resets
    assert old[0].closed and new[0].closed
    assert old[2].episodes == new[2].episodes
    assert old[2].sampled == new[2].sampled
    assert old[3] == new[3]
    assert equal_state(old[4], new[4])
    assert old[1].optimizer_steps == new[1].optimizer_steps == expected
    assert torch.equal(old[1].weight, new[1].weight)
    assert torch.equal(old[5][0], new[5][0]) and old[5][1] == new[5][1]


def test_runner_error_closes_environment_before_completion():
    env = Environment(2)
    runner = AdaptationRunner(env, lambda *_:np.array([0.],dtype=np.float32),
                              lambda event:None,horizon=2,initial_seed=1)
    with pytest.raises(RuntimeError, match='record'):
        runner.run(3, lambda *_:(_ for _ in ()).throw(RuntimeError('record')),
                   complete=lambda:pytest.fail('completion after error'))
    assert env.closed and env.resets == [1]


def test_historical_runner_source_provenance_is_narrowly_compatible():
    current = {path:hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()
               for path in PRE_RUNNER_MIGRATION_SHA256}
    assert current == RUNNER_MIGRATION_SHA256
    assert compatible_sources(PRE_RUNNER_MIGRATION_SHA256,current)
    assert not compatible_sources({**PRE_RUNNER_MIGRATION_SHA256,
        'tdmpc2/tdmpc2.py':'old'}, {**current,'tdmpc2/tdmpc2.py':'changed'})
    changed = dict(current)
    key = next(iter(changed))
    changed[key] = 'future-edit'
    assert not compatible_sources(PRE_RUNNER_MIGRATION_SHA256,changed)


def test_reacher_resume_manifest_uses_its_own_source_set():
    import run_dynamics_q as dq_runner
    key = 'experiments/nsgym/scripts/run_reacher_dynamics_q.py'
    with TemporaryDirectory() as temporary:
        output = Path(temporary)
        old = {'command':'run','source_hashes':{key:PRE_RUNNER_MIGRATION_SHA256[key]}}
        current = {'command':'run','source_hashes':{key:RUNNER_MIGRATION_SHA256[key]}}
        (output/'manifest.json').write_text(json.dumps(old))
        assert dq_runner.open_stage(output,current,resume=True) == output


def test_joint_one_original_update_matches_canonical_strategy_on_fixed_batch():
    """Exercise TDMPC2.update, including actor/target, with the same saved batch."""
    from types import SimpleNamespace
    from common.scale import RunningScale
    from tdmpc2 import TDMPC2
    from common.world_model import WorldModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(722)
        cfg = SimpleNamespace(multitask=False, episodic=False, obs='state',
            obs_shape={'state':(17,)}, num_enc_layers=2, enc_dim=16,
            latent_dim=16, mlp_dim=16, action_dim=6, task_dim=0, simnorm_dim=8,
            num_q=5, num_bins=101, vmin=-10., vmax=10., bin_size=.2,
            dropout=.01, log_std_min=-10., log_std_max=2., tau=.01,
            horizon=3, batch_size=4, rho=.5, consistency_coef=20.,
            reward_coef=.1, termination_coef=0., value_coef=.1,
            grad_clip_norm=20., entropy_coef=1e-4)
        agent = TDMPC2.__new__(TDMPC2)
        torch.nn.Module.__init__(agent)
        agent.cfg, agent.device = cfg, torch.device('cpu')
        agent.model = WorldModel(cfg)
        agent.optim = torch.optim.Adam([
            {'params':agent.model._encoder.parameters()},
            {'params':agent.model._dynamics.parameters()},
            {'params':agent.model._reward.parameters()},
            {'params':agent.model._Qs.parameters()}],lr=3e-4)
        agent.pi_optim = torch.optim.Adam(agent.model._pi.parameters(),lr=3e-4,eps=1e-5)
        scale = RunningScale.__new__(RunningScale)
        torch.nn.Module.__init__(scale)
        scale.cfg = cfg
        scale.value = torch.nn.Buffer(torch.ones(1))
        scale._percentiles = torch.nn.Buffer(torch.tensor([5.,95.]))
        agent.scale, agent.discount = scale, .995
        generator = torch.Generator().manual_seed(731)
        batch = (torch.randn(4,4,17,generator=generator),
                 torch.randn(3,4,6,generator=generator).tanh(),
                 torch.randn(3,4,1,generator=generator),
                 torch.zeros(3,4,1),None)

        class SavedBatch:
            num_eps = 1
            def __init__(self):self.samples=0
            def sample(self):
                self.samples += 1
                return batch

        replay = SavedBatch()
        initial_model = deepcopy(agent.model.state_dict())
        initial_scale = deepcopy(agent.scale.state_dict())
        torch.manual_seed(740)
        initial_rng = torch.get_rng_state().clone()
        old_metrics = agent.update(replay)
        old_model = deepcopy(agent.model.state_dict())
        old_optim = deepcopy(agent.optim.state_dict())
        old_pi_optim = deepcopy(agent.pi_optim.state_dict())
        old_scale = deepcopy(agent.scale.state_dict())
        old_rng = torch.get_rng_state().clone()
        assert replay.samples == 1

        agent.model.load_state_dict(initial_model)
        agent.optim = torch.optim.Adam([
            {'params':agent.model._encoder.parameters()},
            {'params':agent.model._dynamics.parameters()},
            {'params':agent.model._reward.parameters()},
            {'params':agent.model._Qs.parameters()}],lr=3e-4)
        agent.pi_optim = torch.optim.Adam(agent.model._pi.parameters(),lr=3e-4,eps=1e-5)
        agent.scale.load_state_dict(initial_scale)
        torch.set_rng_state(initial_rng)
        strategy = JointAdaptation(agent,replay,{'first_update':1,'update_every':1})
        new_metrics = strategy.observe()
        assert strategy.transitions == strategy.updates == 1
        assert replay.samples == 2
        assert equal_state(old_metrics,new_metrics)
        assert equal_state(old_model,agent.model.state_dict())
        assert equal_state(old_optim,agent.optim.state_dict())
        assert equal_state(old_pi_optim,agent.pi_optim.state_dict())
        assert equal_state(old_scale,agent.scale.state_dict())
        assert torch.equal(old_rng,torch.get_rng_state())
        assert not equal_state(initial_model,old_model)


def test_old_and_new_dynamics_q_collection_preserve_real_update_and_rng():
    """Compare the pre-migration 75-step loop with event-driven collection."""
    import run_dynamics_q as dq_runner
    from common.dynamics_q import DynamicsQAdapter

    class DQEnvironment:
        def __init__(self):
            self.observation = np.linspace(-.2,.2,17,dtype=np.float32)
            self.resets = []
        def reset(self,seed=None):
            self.resets.append(seed)
            return self.observation.copy(),{}
        def step(self,action):
            self.observation = self.observation + float(np.mean(action))*.0001 + .0003
            return self.observation.copy(),.15,False,False,{}

    def execute(canonical):
        torch.manual_seed(810)
        random.seed(811)
        env = DQEnvironment()
        agent = dq_runner.cpu_agent()
        config, _ = dq_runner.configuration()
        identity = dict(parameter='bthigh_damping',severity=.5,
                        adaptation_seed=42000001,method='dynamics_q_118',schedule='fixed')
        adapter = DynamicsQAdapter(agent,config,identity)
        records = []

        def act(_obs,_step,_episode):
            return torch.randn(6).tanh().numpy()

        def observe(event):
            return adapter.observe(event.observation,event.action,event.next_observation,
                                   event.reward,event.terminated,event.truncated,
                                   event.episode,event.episode_step,3.,6.)

        if canonical:
            runner = AdaptationRunner(env,act,observe,horizon=config['horizon'],initial_seed=41)
            for event,metrics in runner.iter_steps(75):
                records.append((event.index, event.action.copy(),
                                event.next_observation.copy(), deepcopy(metrics)))
        else:
            obs,_ = env.reset(seed=41)
            for step in range(75):
                ep,t=divmod(step,config['horizon'])
                action=act(obs,t,ep)
                nxt,reward,term,trunc,_=env.step(action)
                metrics=adapter.observe(obs,action,nxt,reward,term,trunc,ep,t,3.,6.)
                records.append((step,action.copy(),nxt.copy(),deepcopy(metrics)))
                obs=nxt
        state=adapter.state_dict()
        state.pop('update_seconds')
        return (agent.model.state_dict(),adapter.dynamics_optimizer.state_dict(),
                adapter.q_optimizer.state_dict(),state,adapter.last_sample_steps,
                torch.get_rng_state().clone(),random.getstate(),records,env.resets)

    old,new=execute(False),execute(True)
    for index in (0,1,2,3,4):
        assert equal_state(old[index],new[index])
    assert torch.equal(old[5],new[5]) and old[6]==new[6]
    assert old[8]==new[8]==[41]
    assert len(old[7])==len(new[7])==75
    for left,right in zip(old[7],new[7]):
        assert left[0]==right[0]
        np.testing.assert_array_equal(left[1],right[1])
        np.testing.assert_array_equal(left[2],right[2])
        assert equal_state(left[3],right[3])
    assert tuple(old[3][key] for key in ('transitions','dynamics_steps','q_steps','target_steps')) == (75,1,1,1)


@pytest.mark.parametrize('decision_updates', [False, True])
def test_old_and_new_episode_rollout_preserve_actions_boundaries_and_records(monkeypatch,decision_updates):
    import run_dynamics_only_ttt as original_runner
    monkeypatch.setattr(original_runner,'torch',torch,raising=False)
    monkeypatch.setattr(original_runner,'np',np,raising=False)
    monkeypatch.setattr(original_runner,'equal_state',equal_state,raising=False)
    monkeypatch.setattr(original_runner,'error',lambda _a,o,ac,n:float(np.square(n-o).mean()),raising=False)

    class PhysicalEnvironment:
        nominal_value = 6.
        def __init__(self):
            self.unwrapped = self
            self.data = type('Data',(),{})()
            self.model = type('Physical',(),{
                'actuator_forcelimited':np.array([1]),
                'actuator_forcerange':np.array([[-1.,1.]])})()
            self.resets=[]
            self.position=0
        def reset(self,seed=None):
            self.resets.append(seed);self.position=0
            return np.array([.1,.2],dtype=np.float32),{}
        def step(self,action):
            self.position+=1
            self.data.actuator_force=np.array([float(action[0])])
            nxt=np.array([.1+self.position*.01,.2+float(action[0])*.01],dtype=np.float32)
            return nxt,1.,False,self.position==4,dict(x_velocity=.3,reward_ctrl=-.1)
        def readback(self):return 3.

    class EpisodeAgent:
        def act(self,obs,*,t0,eval_mode):
            assert eval_mode
            return (torch.rand(1)+obs[:1]+int(t0)).tanh()

    class EpisodeStrategy:
        def __init__(self):
            self.replay=[]
            self.transitions=self.gradient_steps=0
            self.update_compute_time=0.
            self.last_metrics=dict(consistency_loss=None,predictor_gradient_norm=None)
            self.in_episode=False
        def begin_episode(self):self.in_episode=True
        def observe(self,o,ac,n,_native):
            self.replay.append(tuple(x.clone() for x in (o,ac,n)))
            self.transitions+=1
            if decision_updates and self.transitions%2==0:
                self.gradient_steps+=1
                self.last_metrics=dict(consistency_loss=.2,predictor_gradient_norm=.3)
        def end_episode(self):
            self.in_episode=False
            if not decision_updates:
                self.gradient_steps+=1
                self.last_metrics=dict(consistency_loss=.2,predictor_gradient_norm=.3)

    def execute(canonical):
        torch.manual_seed(920)
        env,agent,adapter=PhysicalEnvironment(),EpisodeAgent(),EpisodeStrategy()
        c={'horizon':4,'onset':2}
        if canonical:
            rows=list(original_runner.rollout(agent,adapter,c,{},'fixed',19,env,0,True))
            records=[row for row,_,_ in rows]
            observations,actions=rows[-1][1:]
        else:
            o,_=env.reset(seed=19);adapter.begin_episode()
            records=[];observations=[o.copy()];actions=[]
            for t in range(4):
                begin=0.
                action=agent.act(torch.tensor(o,dtype=torch.float32),t0=t==0,eval_mode=True).detach().cpu().numpy()
                n,reward,term,trunc,info=env.step(action)
                mse=float(np.square(n-o).mean())
                before=adapter.gradient_steps
                adapter.observe(torch.tensor(o,dtype=torch.float32),torch.tensor(action,dtype=torch.float32),
                                torch.tensor(n,dtype=torch.float32),1)
                if term or trunc:adapter.end_episode()
                force=env.unwrapped.data.actuator_force.copy();model=env.unwrapped.model
                limited=model.actuator_forcelimited.astype(bool);bounds=model.actuator_forcerange
                force_sat=float(np.mean((force[limited]<=bounds[limited,0]+1e-6)|
                                        (force[limited]>=bounds[limited,1]-1e-6))) if limited.any() else None
                records.append(dict(episode=0,seed=19,t=t,aligned_step=t-c['onset'],
                    adaptation_steps=adapter.transitions,gradient_updates=adapter.gradient_steps,
                    updates_this_transition=adapter.gradient_steps-before,
                    update_seconds=adapter.update_compute_time,transition_seconds=begin,
                    latent_mse_preupdate=mse,
                    training_loss=adapter.last_metrics['consistency_loss'] if adapter.gradient_steps>before else None,
                    gradient_norm=adapter.last_metrics['predictor_gradient_norm'] if adapter.gradient_steps>before else None,
                    reward=reward,velocity=float(info['x_velocity']),control_cost=-float(info['reward_ctrl']),
                    action_abs=float(np.abs(action).mean()),action_rms=float(np.sqrt(np.mean(action**2))),
                    action_component_saturation=float((np.abs(action)>=.99).mean()),
                    actuator_force_rms=float(np.sqrt(np.mean(force**2))),actuator_force_saturation=force_sat,
                    force_limited_count=int(limited.sum()),actual_parameter=env.readback(),nominal_parameter=env.nominal_value,
                    terminated=term,truncated=trunc))
                observations.append(n.copy());actions.append(action.copy());o=n
                if term or trunc:break
        for row in records:row.pop('transition_seconds')
        return env.resets,records,observations,actions,adapter,torch.get_rng_state().clone()

    old,new=execute(False),execute(True)
    assert old[0]==new[0]==[19]
    assert old[1]==new[1]
    for index in (2,3):
        for left,right in zip(old[index],new[index]):np.testing.assert_array_equal(left,right)
    assert equal_state(old[4].replay,new[4].replay)
    assert (old[4].transitions,old[4].gradient_steps)==(new[4].transitions,new[4].gradient_steps)
    assert torch.equal(old[5],new[5])
