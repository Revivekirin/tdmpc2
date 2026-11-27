"""CPU golden observations of unchanged research and native planning code.

Record once from the audited source with --record; ordinary test runs only read
the committed fixture. No simulator, CUDA, training job, or historical output.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import platform
import random
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
sys.path.insert(0, str(PROJECT / "experiments/nsgym/scripts"))
sys.path.insert(0, str(PROJECT / "tests"))

from common.dynamics_q import DynamicsQAdapter, expected_updates
from common import math as categorical_math
from common.planner_trace import PlannerTrace
from common.stage3a import EpisodeAdapter, equal_state
import run_dynamics_q as dq_runner
from test_planner_instrumentation import agent as planner_agent, load_planner

CPU_GOLDEN = PROJECT / "tests/fixtures/research_numerical_baselines.json"
if platform.system() == 'Darwin' and torch.__version__ == '2.14.0':
    GOLDEN = CPU_GOLDEN
else:
    runtime = f'{platform.system().lower()}_{platform.machine().lower()}_torch_{torch.__version__}'
    runtime = runtime.replace('.', '_').replace('+', '_')
    GOLDEN = PROJECT / 'tests/fixtures' / f'research_numerical_baselines_{runtime}.json'


def digest_tensor(x):
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(str((x.dtype, tuple(x.shape))).encode() +
                          x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_record():
    state = torch.get_rng_state()
    return dict(torch_cpu_base64=base64.b64encode(state.numpy().tobytes()).decode(),
                numpy_state_base64=base64.b64encode(pickle.dumps(np.random.get_state())).decode(),
                python_state_base64=base64.b64encode(pickle.dumps(random.getstate())).decode(),
                numpy_sha256=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                python_sha256=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())


def optimizer_summary(optimizer):
    return dict(param_groups=len(optimizer.param_groups),
                parameter_count=sum(len(g['params']) for g in optimizer.param_groups),
                state_entries=len(optimizer.state),
                steps=sorted({int(s['step']) for s in optimizer.state.values()}))


def dynamics_only_observation():
    torch.manual_seed(20261119)
    random.seed(20261119)
    np.random.seed(20261119)
    torch.set_num_threads(1)
    model = dq_runner.cpu_agent().model
    settings = dict(mode='dynamics_only', update_granularity='episode', consistency_steps=1,
                    action_input='commanded', replay_source='fresh_shifted',
                    sampling='uniform_without_replacement', capacity=8, batch_size=4,
                    warmup=4, lr=3e-4, updates_per_episode=1, episodes=1, eval_every=1,
                    adaptation_seeds=[41], evaluation_seeds=[42])
    adapter = EpisodeAdapter(model, settings, consistency_coef=1., grad_clip_norm=20., seed=41)
    inputs = []
    generator = torch.Generator().manual_seed(9001)
    before_rng = rng_record()
    adapter.begin_episode()
    for _ in range(4):
        obs = torch.randn(17, generator=generator)
        action = torch.randn(6, generator=generator).tanh()
        nxt = torch.randn(17, generator=generator)
        inputs.append([obs, action, nxt])
        adapter.observe(obs, action, nxt, 1)
    replay_before = adapter.rng.getstate()
    loss_before = float(adapter.loss(list(adapter.replay)).detach())
    old = deepcopy(model._dynamics.state_dict())
    metrics = adapter.end_episode()
    adapter.assert_fixed()
    drift = float(torch.sqrt(sum((v - old[k]).square().sum() for k, v in model._dynamics.state_dict().items())))
    predicted = model.next(model.encode(inputs[0][0][None], None), inputs[0][1][None], None)
    return dict(seed=20261119, input_generator_seed=9001,
                input_hashes=[[digest_tensor(x) for x in row] for row in inputs],
                rng_before=before_rng, rng_after=rng_record(),
                replay_rng_before_base64=base64.b64encode(pickle.dumps(replay_before)).decode(),
                replay_rng_after_base64=base64.b64encode(pickle.dumps(adapter.rng.getstate())).decode(),
                replay_rng_before_sha256=hashlib.sha256(repr(replay_before).encode()).hexdigest(),
                replay_rng_after_sha256=hashlib.sha256(repr(adapter.rng.getstate()).encode()).hexdigest(),
                loss_before=loss_before, loss_after=float(adapter.loss(list(adapter.replay)).detach()),
                metrics=metrics, predictor_change_l2=drift,
                predictor_output_hash=digest_tensor(predicted),
                predictor_output=predicted.detach().tolist(),
                optimizer=optimizer_summary(adapter.optimizer),
                counters=list(adapter.state_dict()['counters']), fixed_parameters_unchanged=True)


def dynamics_q_observation():
    torch.manual_seed(20261119)
    random.seed(20261119)
    np.random.seed(20261119)
    torch.set_num_threads(1)
    cfg, _ = dq_runner.configuration()
    identity = dict(parameter='bthigh_damping', severity=2., adaptation_seed=42000001,
                    method='dynamics_q_118', schedule='fixed')
    agent = dq_runner.cpu_agent()
    adapter = DynamicsQAdapter(agent, cfg, identity)
    generator = torch.Generator().manual_seed(9002)
    before_rng = rng_record()
    first_input = None
    preupdate = None
    target_before = None
    for step in range(75):
        obs = torch.randn(17, generator=generator)
        action = torch.randn(6, generator=generator).tanh()
        nxt = torch.randn(17, generator=generator)
        if step == 0:
            first_input = [digest_tensor(x) for x in (obs, action, nxt)]
        adapter.observe(obs, action, nxt, float(step % 7), False, False,
                        0, step, 12., 6.)
        if step == 63:
            replay = list(adapter.replay)
            x = torch.stack([r['observation'] for r in replay])
            u = torch.stack([r['action'] for r in replay])
            y = torch.stack([r['next_observation'] for r in replay])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(777)
                dynamics_loss, critic_loss, logits, target = adapter.losses(replay)
                torch.manual_seed(777)
                with torch.no_grad():
                    z0 = agent.model.encode(x, None)
                    z1 = agent.model.encode(y, None)
                    independent_target = agent._td_target(z1,
                        torch.tensor([[r['reward']] for r in replay]),
                        torch.zeros(64,1), None)
                    independent_mse = (agent.model.next(z0,u,None)-z1).square().mean()
                    projected = categorical_math.two_hot(target,agent.cfg)
                    independent_ce = torch.stack([
                        -(projected*head.log_softmax(-1)).sum(-1).mean()
                        for head in logits]).mean()
                torch.testing.assert_close(target, independent_target)
                torch.testing.assert_close(dynamics_loss, independent_mse)
                torch.testing.assert_close(critic_loss, independent_ce)
            preupdate = dict(replay_latent_hash=digest_tensor(z0),
                             actual_next_latent_hash=digest_tensor(z1),
                             td_target_hash=digest_tensor(target),
                             q_logits_hash=digest_tensor(logits),
                             replay_latent_first=z0[0].tolist(),
                             actual_next_latent_first=z1[0].tolist(),
                             td_target_first=target[0].tolist(),
                             q_logits_first=logits[0,0].tolist(),
                             dynamics_mse=float(dynamics_loss.detach()),
                             categorical_cross_entropy=float(critic_loss.detach()),
                             independent_target_equal=True,
                             independent_mse_equal=True,
                             independent_cross_entropy_equal=True)
        if step == 73:
            target_before = deepcopy(agent.model._target_Qs_params.state_dict())
    batch = list(adapter.replay)
    obs = torch.stack([r['observation'] for r in batch])
    action = torch.stack([r['action'] for r in batch])
    nxt = torch.stack([r['next_observation'] for r in batch])
    with torch.no_grad():
        z, next_z = agent.model.encode(obs, None), agent.model.encode(nxt, None)
        prediction = agent.model.next(z, action, None)
    audit = {r['component']:r['changed'] for r in adapter.parameter_audit()}
    drift = {r['component']:r['drift_l2'] for r in adapter.parameter_audit()}
    online = agent.model._detach_Qs_params.state_dict()
    for key,value in agent.model._target_Qs_params.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value,target_before[key].lerp(online[key],agent.cfg.tau))
    state = adapter.state_dict()
    return dict(seed=20261119, input_generator_seed=9002, first_input_hashes=first_input,
                replay_observation_hash=digest_tensor(obs), replay_action_hash=digest_tensor(action),
                replay_next_observation_hash=digest_tensor(nxt),
                encoded_latent_hash=digest_tensor(z), actual_next_latent_hash=digest_tensor(next_z),
                predicted_next_latent_hash=digest_tensor(prediction),
                predicted_next_latent_first=prediction[0].tolist(),
                rng_before=before_rng, rng_after=rng_record(),
                replay_rng_base64=base64.b64encode(pickle.dumps(adapter.rng.getstate())).decode(),
                replay_rng_sha256=hashlib.sha256(repr(adapter.rng.getstate()).encode()).hexdigest(),
                sampled_global_steps=adapter.last_sample_steps,
                preupdate=preupdate, metrics=adapter.last_metrics,
                parameter_changes=audit, parameter_drift_l2=drift,
                dynamics_optimizer=optimizer_summary(adapter.dynamics_optimizer),
                q_optimizer=optimizer_summary(adapter.q_optimizer),
                target_updates=adapter.target_steps, target_ema_equal=True,
                counters=[adapter.transitions, adapter.dynamics_steps, adapter.q_steps],
                target_state_hashes={k:digest_tensor(v) for k,v in
                                     agent.model._target_Qs_params.state_dict().items() if isinstance(v,torch.Tensor)},
                snapshot_keys=sorted(state), expected_3000_updates=expected_updates(3000),
                expected_update_events=list(range(75,3001,25)))


def planning_observation():
    torch.manual_seed(20261119)
    random.seed(20261119)
    np.random.seed(20261119)
    torch.set_num_threads(1)
    planner = planner_agent(load_planner())
    trace = PlannerTrace()
    planner._mechanism_trace = trace
    before_rng = rng_record()
    action = planner.act(torch.tensor([.1, -.2]), t0=True, eval_mode=True)
    iterations = []
    for row in trace.iterations:
        values = torch.tensor(row['actual_planner_score'])
        actions = torch.tensor(row['actions'])
        elite = torch.tensor(row['elite_ids'], dtype=torch.long)
        selected_values = values[elite]
        weight = torch.softmax(planner.cfg.temperature*(selected_values-selected_values.max()), 0)
        selected = actions[:, elite]
        mean = (weight[None,:,None]*selected).sum(1)/(weight.sum()+1e-9)
        std = ((weight[None,:,None]*(selected-mean[:,None])**2).sum(1)/(weight.sum()+1e-9)).sqrt()
        std = std.clamp(planner.cfg.min_std, planner.cfg.max_std)
        iterations.append(dict(policy_prior_hash=digest_tensor(actions[:,:planner.cfg.num_pi_trajs]),
                               gaussian_candidates_hash=digest_tensor(actions[:,planner.cfg.num_pi_trajs:]),
                               predicted_reward_hash=digest_tensor(torch.tensor(row['predicted_reward_sum'])),
                               terminal_q_hash=digest_tensor(torch.tensor(row['terminal_q_contribution'])),
                               total_scores=[float(x) for x in values], elite_indices=elite.tolist(),
                               gaussian_mean=mean.tolist(), gaussian_std=std.tolist()))
    torch.testing.assert_close(torch.tensor(iterations[-1]['gaussian_mean']), planner._prev_mean)
    assert trace.selection['selected_candidate_id'] in iterations[-1]['elite_indices']
    return dict(seed=20261119, observation_hash=digest_tensor(torch.tensor([.1,-.2])),
                encoded_latent=trace.encoded_z.tolist(), rng_before=before_rng,
                rng_after=rng_record(), iterations=iterations,
                warm_start_mean=planner._prev_mean.tolist(),
                selected_candidate_id=trace.selection['selected_candidate_id'],
                selected_action=action.tolist())


def observation_bundle():
    return dict(schema=1, baseline_head='4659c66', upstream='332f19e',
                versions=dict(torch=torch.__version__.split('+')[0], numpy=np.__version__),
                source_sha256={str(p.relative_to(PROJECT)):digest_file(p) for p in
                    [PROJECT/'tdmpc2/common/stage3a.py', PROJECT/'tdmpc2/common/dynamics_q.py',
                     PROJECT/'tdmpc2/common/world_model.py', PROJECT/'tdmpc2/tdmpc2.py',
                     PROJECT/'tdmpc2/common/planner_trace.py']},
                dynamics_only=dynamics_only_observation(),
                dynamics_q=dynamics_q_observation(), planning=planning_observation())


def compare(expected, actual, path='fixture'):
    if isinstance(expected, dict):
        assert set(expected)==set(actual), path
        for key in expected: compare(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, list):
        assert len(expected)==len(actual), path
        for i,(a,b) in enumerate(zip(expected,actual)):compare(a,b,f'{path}[{i}]')
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=2e-6, abs=2e-7), path
    else:
        assert actual == expected, path


def test_golden_research_numerical_baselines():
    assert GOLDEN.exists(), f'Missing platform-specific golden fixture: {GOLDEN}; record it before refactoring'
    expected=json.loads(GOLDEN.read_text())
    assert expected['versions']==dict(torch=torch.__version__.split('+')[0],numpy=np.__version__)
    actual=observation_bundle()
    # The fixture records the pre-migration source paths. Compare all numerical,
    # RNG and checkpoint observations; inspect current source provenance separately.
    compare({k:v for k,v in expected.items() if k!='source_sha256'},
            {k:v for k,v in actual.items() if k!='source_sha256'})
    migrated={'tdmpc2/common/stage3a.py','tdmpc2/common/dynamics_q.py'}
    assert expected['source_sha256']==json.loads(CPU_GOLDEN.read_text())['source_sha256']
    for path, digest in expected['source_sha256'].items():
        if path in migrated:
            assert actual['source_sha256'][path]!=digest, f'{path} was expected to become a compatibility module'
        else:
            assert actual['source_sha256'][path]==digest, f'Unmigrated source changed: {path}'


def test_migrated_adaptation_ownership_and_historical_provenance():
    from research.dynamics_shift.dynamics import DynamicsAdaptation
    from research.dynamics_shift.dynamics_critic import DynamicsCriticAdaptation
    from research.dynamics_shift.decision import DecisionAdapter
    from common.decision_ttt import DecisionAdapter as LegacyDecisionAdapter
    assert EpisodeAdapter is DynamicsAdaptation
    assert DynamicsQAdapter is DynamicsCriticAdaptation
    assert LegacyDecisionAdapter is DecisionAdapter
    assert DecisionAdapter.loss is DynamicsAdaptation.loss
    assert DecisionAdapter._optimizer_steps is DynamicsAdaptation._optimizer_steps
    original=json.loads(CPU_GOLDEN.read_text())['source_sha256']
    assert original['tdmpc2/common/stage3a.py']=='47b7e36f892d63f867db7881da19e2edc53dd338317380e5da22d97b4855a3cc'
    assert original['tdmpc2/common/dynamics_q.py']=='59141b63b306fc2092ef674f61e6f7290b531e151a3bd3176c3b609a70817225'


def test_dynamics_q_schedule_and_ownership_contract():
    assert expected_updates(64)==0
    assert list(range(75,3001,25))[-1]==3000
    assert len(range(75,3001,25))==118==expected_updates(3000)
    result=json.loads(CPU_GOLDEN.read_text())['dynamics_q']
    assert result['counters']==[75,1,1]
    assert result['parameter_changes']==dict(encoder=False,dynamics=True,reward=False,
                                             q=True,target_q=True,actor=False)
    assert result['dynamics_optimizer']['steps']==result['q_optimizer']['steps']==[1]
    assert result['target_updates']==1
    assert len(result['sampled_global_steps'])==64
    assert len(set(result['sampled_global_steps']))==64
    assert max(result['sampled_global_steps'])<=75


if __name__ == '__main__' and '--record' in sys.argv:
    assert not GOLDEN.exists(), 'Refuse to overwrite an existing golden fixture'
    result = observation_bundle()
    reference = json.loads(CPU_GOLDEN.read_text())
    assert result['baseline_head'] == reference['baseline_head']
    assert result['source_sha256'] == reference['source_sha256'], 'Source differs from audited baseline'
    GOLDEN.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(GOLDEN)
