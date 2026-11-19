"""Opt-in, read-only CUDA checks against the historical NS-Gym artifacts.

Run on the compatible server. Local CPU collection skips these tests; a missing
historical source on a CUDA host is a failure, never a synthetic replacement.
"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'tdmpc2'))
sys.path.insert(0, str(PROJECT / 'experiments/nsgym/scripts'))
from common import math as categorical_math
from common.dynamics_q import DynamicsQAdapter
from common.planner_trace import PlannerTrace
import evaluate_joint_final as joint
import evaluate_ttt_final as decision
import run_dynamics_only_ttt as dynamics_only
import run_dynamics_q as dynamics_q

RESULTS = next((p for p in (PROJECT/'results', PROJECT.parent/'results') if
                (p/'nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000/model.pt').is_file()),
               PROJECT/'results')
NOMINAL = RESULTS/'nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000'
DECISION = RESULTS/'nsgym/dynamics-only-750k-v1/runs/fixed-decision/bthigh_damping-k0.5-decision-seed42000001'
DYNAMICS_Q = RESULTS/'nsgym/dynamics-q-750k-v1/runs/fixed-decision/bthigh_damping-k0.5-dynamics-q-seed42000001'
JOINT = RESULTS/'nsgym/full-joint-750k-v1/runs-fixed/bthigh_damping-k0.5-full-joint-seed42000001'
FROZEN = RESULTS/'nsgym/physical-severity-750k-v1/frozen'
LATENT = RESULTS/'nsgym/physical-severity-750k-v1/heldout-latent/latent_errors.csv'
BANK = FROZEN/'condition-00-seed-43000001.npz'


def require_server(*paths):
    if not torch.cuda.is_available():
        pytest.skip('BLOCKED: compatible CUDA host required')
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        pytest.fail('BLOCKED: historical source missing: ' + ', '.join(missing))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def components(agent):
    from training_support import tree_hash
    model = agent.model
    return {name: tree_hash(getattr(model, name).state_dict()) for name in
            ('_encoder', '_dynamics', '_reward', '_Qs', '_target_Qs', '_pi')}


def bank_arrays():
    index = json.loads((FROZEN/'dataset_index.json').read_text())
    entry = next(row for row in index if row['file'] == BANK.name)
    assert sha256(BANK) == entry['sha256']
    with np.load(BANK) as data:
        obs = np.asarray(data['observations'], dtype=np.float32)
        actions = np.asarray(data['actions'], dtype=np.float32)
        rewards = np.asarray(data['rewards'], dtype=np.float32)
        actual = np.asarray(data['actual_parameter'])
    assert obs.shape == (1001, 17) and actions.shape == (1000, 6)
    assert rewards.shape == actual.shape == (1000,)
    return obs, actions, rewards, actual


def test_cuda_nominal_production_load_saved_bank_and_native_planning():
    require_server(NOMINAL/'model.pt', NOMINAL/'optimizers.pt', NOMINAL/'metadata.json',
                   FROZEN/'dataset_index.json', BANK, LATENT)
    import training_support
    agent, metadata = training_support.load_checkpoint(NOMINAL)
    agent.model.eval()
    assert agent.device == torch.device('cuda:0')
    assert metadata['model_sha256'] == sha256(NOMINAL/'model.pt')
    assert metadata['optimizer_sha256'] == sha256(NOMINAL/'optimizers.pt')
    source = torch.load(NOMINAL/'model.pt', map_location='cpu', weights_only=False)['model']
    loaded = agent.model.state_dict()
    assert set(source) == set(loaded)
    assert all(source[key].shape == loaded[key].shape for key in source)
    assert training_support.tree_hash(source) == training_support.tree_hash(loaded)
    before = components(agent)
    obs, actions, _, _ = bank_arrays()
    with torch.no_grad():
        o = torch.as_tensor(obs[:64], device=agent.device)
        n = torch.as_tensor(obs[1:65], device=agent.device)
        a = torch.as_tensor(actions[:64], device=agent.device)
        z = agent.model.encode(o, None)
        prediction = agent.model.next(z, a, None)
        target = agent.model.encode(n, None)
        assert torch.isfinite(prediction).all()
        assert torch.equal(prediction, agent.model.next(z, a, None))
        mse_first = float((prediction[0]-target[0]).square().mean())
    with LATENT.open(newline='') as stream:
        reference = next(row for row in csv.DictReader(stream)
                         if row['source_file'] == BANK.name and row['horizon'] == '1'
                         and row['source_step'] == '0')
    expected = float(reference['latent_mse'])
    assert mse_first == pytest.approx(expected, rel=5e-4, abs=2e-7)
    trace = PlannerTrace()
    agent._mechanism_trace = trace
    torch.manual_seed(42000001)
    torch.cuda.manual_seed_all(42000001)
    action = agent.act(torch.as_tensor(obs[0]), t0=True, eval_mode=True)
    assert action.shape == (6,) and torch.isfinite(action).all()
    assert (action.abs() <= 1).all()
    assert trace.iterations and trace.selection
    for row in trace.iterations:
        score = np.asarray(row['actual_planner_score'])
        reward = np.asarray(row['predicted_reward_sum'])
        terminal = np.asarray(row['terminal_q_contribution'])
        np.testing.assert_allclose(score, reward+terminal, rtol=0, atol=1e-5)
        assert set(row['elite_ids']).issubset(set(range(len(score))))
    assert trace.selection['selected_candidate_id'] in trace.iterations[-1]['elite_ids']
    assert components(agent) == before
    with TemporaryDirectory(prefix='tdmpc2-cuda-golden-') as temporary:
        path = Path(temporary)/'model.pt'
        agent.save(path)
        agent.load(path)
        assert components(agent) == before
    print(json.dumps(dict(check='nominal_bank_planning', gpu=torch.cuda.get_device_name(0),
                          torch=torch.__version__, bank_sha256=sha256(BANK),
                          first_step_latent_mse=mse_first, historical_mse=expected,
                          absolute_deviation=abs(mse_first-expected),
                          component_hashes=before, selected_action=action.cpu().tolist()), sort_keys=True))


def test_cuda_historical_adaptation_production_loaders():
    require_server(NOMINAL/'metadata.json', DECISION/'snapshot-0003000.pt',
                   DECISION/'manifest.json', DECISION/'completion.json',
                   DYNAMICS_Q/'snapshot-0003000.pt', JOINT/'snapshot-0003000.pt',
                   JOINT/'manifest.json', JOINT/'completion.json')
    import training_support
    ttt_config = dynamics_only.configuration(SimpleNamespace(config=dynamics_only.DEFAULT, command='run'))
    decision_snapshot = DECISION/'snapshot-0003000.pt'
    agent, adapter, payload = decision.load_adapted(
        SimpleNamespace(checkpoint=NOMINAL), ttt_config, DECISION,
        dict(snapshot_sha256=sha256(decision_snapshot)))
    assert payload['model_hash'] == training_support.tree_hash(agent.model.state_dict())
    assert adapter.gradient_steps == 118
    adapter.assert_fixed()
    decision_hashes = components(agent)
    del agent, adapter, payload

    config, _ = dynamics_q.configuration()
    dq_snapshot = DYNAMICS_Q/'snapshot-0003000.pt'
    identity = dict(parameter='bthigh_damping', severity=.5, adaptation_seed=42000001,
                    method='dynamics_q_118', schedule='fixed')
    item = dict(identity, snapshot=str(dq_snapshot), snapshot_sha256=sha256(dq_snapshot))
    agent, adapter = dynamics_q.load_final(SimpleNamespace(checkpoint=NOMINAL), config, item)
    assert adapter.transitions == 3000 and adapter.dynamics_steps == adapter.q_steps == 118
    assert adapter.target_steps == 118
    adapter.assert_fixed()
    dq_hashes = components(agent)
    del agent, adapter

    joint_snapshot = JOINT/'snapshot-0003000.pt'
    joint_payload = torch.load(joint_snapshot, map_location='cpu', weights_only=False)
    agent, joint_hash = joint.load_agent(SimpleNamespace(checkpoint=NOMINAL),
        dict(path=JOINT, snapshot_sha256=sha256(joint_snapshot)), joint_payload['protocol'])
    assert joint_hash == training_support.tree_hash(training_support.learned_state(agent))
    assert joint_payload['resume'] == 'evaluation_only_no_replay_or_environment_state'
    assert joint_payload['transitions'] == 3000 and joint_payload['updates'] == 2001
    assert components(agent)
    print(json.dumps(dict(check='historical_adapted_loaders', decision_components=decision_hashes,
                          dynamics_q_components=dq_hashes, joint_state_hash=joint_hash,
                          exact_continuation='UNSUPPORTED: replay, environment and RNG state absent'), sort_keys=True))


def test_cuda_dynamics_q_real_bank_update_and_cpu_parity():
    require_server(NOMINAL/'metadata.json', BANK, FROZEN/'dataset_index.json')
    obs, actions, rewards, actual = bank_arrays()
    agent, _ = dynamics_only.load_agent(SimpleNamespace(checkpoint=NOMINAL, synthetic=False))
    config, _ = dynamics_q.configuration()
    identity = dict(parameter='bthigh_damping', severity=.5, adaptation_seed=42000001,
                    method='dynamics_q_118', schedule='fixed')
    adapter = DynamicsQAdapter(agent, config, identity)
    before = components(agent)
    target_before = None
    for step in range(75):
        if step == 74:
            target_before = deepcopy(agent.model._target_Qs_params.state_dict())
        adapter.observe(torch.as_tensor(obs[step]), torch.as_tensor(actions[step]),
                        torch.as_tensor(obs[step+1]), float(rewards[step]), False, False,
                        0, step, float(actual[step]), 6.)
    assert (adapter.transitions, adapter.dynamics_steps, adapter.q_steps, adapter.target_steps) == (75, 1, 1, 1)
    assert len(set(adapter.last_sample_steps)) == 64
    after = components(agent)
    assert all(before[key] == after[key] for key in ('_encoder', '_reward', '_pi'))
    assert before['_dynamics'] != after['_dynamics'] and before['_Qs'] != after['_Qs']
    assert before['_target_Qs'] != after['_target_Qs']
    dyn_ids = {id(p) for group in adapter.dynamics_optimizer.param_groups for p in group['params']}
    q_ids = {id(p) for group in adapter.q_optimizer.param_groups for p in group['params']}
    assert dyn_ids and q_ids and dyn_ids.isdisjoint(q_ids)
    assert {int(s['step']) for s in adapter.dynamics_optimizer.state.values()} == {1}
    assert {int(s['step']) for s in adapter.q_optimizer.state.values()} == {1}
    for key, value in agent.model._target_Qs_params.state_dict().items():
        torch.testing.assert_close(value, target_before[key].lerp(
            agent.model._detach_Qs_params.state_dict()[key], agent.cfg.tau))
    batch = list(adapter.replay)[:64]
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        dynamics_loss, q_loss, logits, target = adapter.losses(batch)
        projected = categorical_math.two_hot(target, agent.cfg)
        independent_ce = torch.stack([
            -(projected * head.log_softmax(-1)).sum(-1).mean() for head in logits]).mean()
        torch.testing.assert_close(q_loss, independent_ce)
        observations = torch.stack([r['observation'] for r in batch]).to(agent.device)
        next_observations = torch.stack([r['next_observation'] for r in batch]).to(agent.device)
        actions_batch = torch.stack([r['action'] for r in batch]).to(agent.device)
        z = agent.model.encode(observations, None)
        next_z = agent.model.encode(next_observations, None)
        torch.testing.assert_close(dynamics_loss, (agent.model.next(z, actions_batch, None)-next_z).square().mean())
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        rewards_batch = torch.tensor([[r['reward']] for r in batch], device=agent.device)
        terminals_batch = torch.zeros_like(rewards_batch)
        with torch.no_grad():
            independent_target = agent._td_target(next_z, rewards_batch, terminals_batch, None)
        torch.testing.assert_close(target, independent_target)
    print(json.dumps(dict(check='real_bank_dynamics_q', dynamics_loss=float(dynamics_loss),
                          categorical_q_loss=float(q_loss), target_mean=float(target.mean()),
                          gradient_norms=adapter.last_metrics,
                          component_hashes_before=before, component_hashes_after=after), sort_keys=True))


def test_cuda_cpu_fixed_tensor_numerical_parity():
    require_server()
    torch.manual_seed(20261119)
    cpu = dynamics_q.cpu_agent().model.eval()
    gpu = deepcopy(cpu).to('cuda:0').eval()
    obs = torch.linspace(-1, 1, 32*17).reshape(32, 17)
    action = torch.linspace(-.8, .8, 32*6).reshape(32, 6)
    with torch.no_grad():
        z_cpu = cpu.encode(obs, None)
        pred_cpu = cpu.next(z_cpu, action, None)
        logits_cpu = cpu.Q(z_cpu, action, None, return_type='all')
        z_gpu = gpu.encode(obs.cuda(), None)
        pred_gpu = gpu.next(z_gpu, action.cuda(), None)
        logits_gpu = gpu.Q(z_gpu, action.cuda(), None, return_type='all')
        labels_cpu = torch.linspace(-2, 2, 32).reshape(32, 1)
        labels_gpu = labels_cpu.cuda()
        ce_cpu = torch.stack([categorical_math.soft_ce(head, labels_cpu, cpu.cfg).mean()
                              for head in logits_cpu]).mean()
        ce_gpu = torch.stack([categorical_math.soft_ce(head, labels_gpu, gpu.cfg).mean()
                              for head in logits_gpu]).mean()
    deviations = {}
    for name, expected, actual in [('encoded_latent', z_cpu, z_gpu),
                                   ('predicted_next_latent', pred_cpu, pred_gpu),
                                   ('categorical_q_logits', logits_cpu, logits_gpu),
                                   ('categorical_q_cross_entropy', ce_cpu, ce_gpu)]:
        actual_cpu = actual.cpu()
        deviations[name] = float((expected-actual_cpu).abs().max())
        torch.testing.assert_close(actual_cpu, expected, rtol=5e-4, atol=2e-5)
    print(json.dumps(dict(check='fixed_tensor_cpu_cuda_parity', dtype='float32',
                          rtol=5e-4, atol=2e-5, max_absolute_deviation=deviations), sort_keys=True))
