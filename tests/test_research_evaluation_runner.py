"""Controlled legacy-loop parity and negative contracts for evaluation orchestration."""

import hashlib
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))

from research.dynamics_shift.evaluation import (
    EvaluationCallbacks, EvaluationConfig, EvaluationRunner, aligned_transitions,
    critic_intervention_controllers, evaluate_fixed_bank, require_bank,
)


class Environment:
    def __init__(self, *, horizon=4, boundary="truncated"):
        self.horizon, self.boundary = horizon, boundary
        self.index = 0
        self.closed = False
        self.reset_seed = None

    def reset(self, seed):
        self.reset_seed = seed
        self.index = 0
        return np.array([seed % 7], dtype=np.float32), {}

    def step(self, action):
        self.index += 1
        obs = np.array([self.index + float(action[0])], dtype=np.float32)
        return obs, float(action[0]) + self.index, self.boundary == "terminated" and self.index == self.horizon, self.boundary == "truncated" and self.index == self.horizon, {"index": self.index}

    def close(self):
        self.closed = True


class Controller:
    def __init__(self):
        self.weight = torch.tensor(2.)
        self.optimizer_steps = 0
        self.replay_writes = 0
        self.calls = []

    def act(self, observation, index):
        self.calls.append((float(observation[0]), index))
        return np.array([float(torch.rand(())) + float(np.random.random()) + random.random()], dtype=np.float32)

    def guard(self):
        assert self.weight.item() == 2.
        assert self.optimizer_steps == self.replay_writes == 0


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    return (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())


def run_legacy(seed, horizon, controller, boundary):
    seeded(seed)
    env = Environment(horizon=horizon, boundary=boundary)
    observation, _ = env.reset(seed=seed)
    initial = observation.copy()
    rows = []
    total = 0.
    for index in range(horizon):
        action = controller.act(observation, index)
        next_observation, reward, terminated, truncated, info = env.step(action)
        rows.append((index, observation.copy(), action.copy(), next_observation.copy(), reward, terminated, truncated, info))
        total += float(reward)
        observation = next_observation
        if terminated or truncated:
            break
    controller.guard()
    env.close()
    return initial, rows, total, rng_state()


@pytest.mark.parametrize("kind", ["frozen_control", "adapted_control", "nominal_retention"])
@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_episode_actions_records_rng_and_immutability_match_legacy(kind, boundary):
    seed = 43000001
    old = Controller()
    initial, expected, total, old_rng = run_legacy(seed, 4, old, boundary)
    controller = Controller()
    envs = []
    recorded = []

    def make_env():
        env = Environment(horizon=4, boundary=boundary)
        envs.append(env)
        return env

    callbacks = EvaluationCallbacks(
        seed=seeded, create_environment=make_env,
        reset=lambda env, value: env.reset(seed=value)[0],
        action=controller.act,
        step=lambda env, action: env.step(action),
        record_step=lambda env, event: recorded.append(event),
        verify_unchanged=controller.guard,
    )
    result = EvaluationRunner().run(EvaluationConfig(seed, 4), callbacks)
    assert np.array_equal(result.initial_observation, initial)
    assert result.total_reward == total
    assert len(result.steps) == len(expected) == len(recorded)
    for event, row in zip(result.steps, expected):
        assert event.index == row[0]
        assert np.array_equal(event.observation, row[1])
        assert np.array_equal(event.action, row[2])
        assert np.array_equal(event.next_observation, row[3])
        assert (event.reward, event.terminated, event.truncated, event.info) == row[4:]
    assert controller.calls == old.calls
    assert rng_state()[0] == old_rng[0]
    assert np.array_equal(rng_state()[1], old_rng[1])
    assert torch.equal(rng_state()[2], old_rng[2])
    assert envs[0].reset_seed == seed and envs[0].closed


def test_early_boundary_stops_and_closes_environment():
    env = Environment(horizon=2, boundary="terminated")
    callbacks = EvaluationCallbacks(seed=seeded, create_environment=lambda: env,
        reset=lambda obj, seed: obj.reset(seed)[0], action=lambda obs, index: np.array([0.], dtype=np.float32),
        step=lambda obj, action: obj.step(action), record_step=lambda obj, event: None,
        verify_unchanged=lambda: None)
    result = EvaluationRunner().run(EvaluationConfig(5, 9), callbacks)
    assert len(result.steps) == 2 and result.steps[-1].terminated
    assert env.closed


def test_guard_failure_closes_environment():
    env = Environment()
    callbacks = EvaluationCallbacks(seed=seeded, create_environment=lambda: env,
        reset=lambda obj, seed: obj.reset(seed)[0], action=lambda obs, index: np.array([0.], dtype=np.float32),
        step=lambda obj, action: obj.step(action), record_step=lambda obj, event: None,
        verify_unchanged=lambda: (_ for _ in ()).throw(RuntimeError("model changed")))
    with pytest.raises(RuntimeError, match="model changed"):
        EvaluationRunner().run(EvaluationConfig(5, 4), callbacks)
    assert env.closed


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError, match="horizon"):
        EvaluationConfig(5, 0)


def test_fixed_bank_hash_alignment_and_missing_source(tmp_path):
    bank = tmp_path / "transitions.npz"
    np.savez(bank, observations=np.arange(5, dtype=np.float32), actions=np.arange(4, dtype=np.float32))
    sha = hashlib.sha256(bank.read_bytes()).hexdigest()
    assert require_bank(bank, sha) == bank
    with np.load(bank, allow_pickle=False) as data:
        rows = aligned_transitions(data["observations"], data["actions"])
    assert [(float(o), float(a), float(n)) for o, a, n in rows] == [(0., 0., 1.), (1., 1., 2.), (2., 2., 3.), (3., 3., 4.)]
    with pytest.raises(ValueError, match="identity mismatch"):
        require_bank(bank, "0" * 64)
    with pytest.raises(FileNotFoundError):
        require_bank(tmp_path / "absent.npz", sha)
    with pytest.raises(ValueError, match="one more observation"):
        aligned_transitions(np.arange(3), np.arange(3))


def test_fixed_bank_prediction_matches_direct_rows_and_rejects_boundaries(tmp_path):
    bank = tmp_path / "fixed.npz"
    observations = np.arange(5, dtype=np.float32)
    actions = np.array([.5, .25, .75, 1.], dtype=np.float32)
    np.savez(bank, observations=observations, actions=actions,
             terminated=np.array([False, False, False, False]),
             truncated=np.array([False, False, False, True]))
    sha = hashlib.sha256(bank.read_bytes()).hexdigest()
    score = lambda obs, act, next_obs: np.square(obs + act - next_obs)
    expected = float(np.mean([float((o+a-n)**2) for o, a, n in aligned_transitions(observations, actions)]))
    result = evaluate_fixed_bank(bank, sha, score, batch_size=2)
    assert result.bank_sha256 == sha and result.samples == 4
    assert result.latent_mse == expected
    with pytest.raises(ValueError, match="Batch size"):
        evaluate_fixed_bank(bank, sha, score, batch_size=0)
    np.savez(bank, observations=observations, actions=actions,
             terminated=np.array([False, True, False, False]))
    changed = hashlib.sha256(bank.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unaligned terminated"):
        evaluate_fixed_bank(bank, changed, score)


def test_critic_intervention_identity_and_freeze():
    class Model:
        def __init__(self, hashes):
            self.training = False
            self.hashes = hashes
        def parameters(self):
            return [torch.nn.Parameter(torch.tensor(0.), requires_grad=False)]

    class Composer:
        def compose(self, kind, source):
            assert kind in ("dynamics_only", "dynamics_q")
            return source
        def controlled_q_swap(self, source):
            hashes = dict(source.component_hashes, online_q="nominal")
            return SimpleNamespace(model=Model(hashes), component_hashes=hashes)

    from types import SimpleNamespace
    a_hashes = dict(encoder="e", dynamics="d0", reward="r", target_q="t", actor="p", normalization="n", online_q="nominal")
    c_hashes = dict(a_hashes, dynamics="d1", online_q="adapted")
    a = SimpleNamespace(model=Model(a_hashes), component_hashes=a_hashes)
    c = SimpleNamespace(model=Model(c_hashes), component_hashes=c_hashes)
    # Mock only the tensor-hash calculation; the production composer has its
    # own full-state isolation tests against historical checkpoints.
    from unittest.mock import patch
    with patch("research.dynamics_shift.checkpoints.model_hashes", side_effect=lambda model: model.hashes):
        x, y, z = critic_intervention_controllers(Composer(), a, c)
        assert x.component_hashes["dynamics"] == "d0"
        assert y.component_hashes["dynamics"] == z.component_hashes["dynamics"] == "d1"
        assert y.component_hashes["online_q"] == "nominal"
        assert z.component_hashes["online_q"] == "adapted"
        c.component_hashes = dict(c_hashes, dynamics="different")
        with pytest.raises(ValueError):
            critic_intervention_controllers(Composer(), a, c)


def test_joint_production_evaluator_uses_runner_and_preserves_schema():
    from types import SimpleNamespace
    from unittest.mock import patch
    sys.path.insert(0, str(PROJECT / "experiments/nsgym/scripts"))
    import evaluate_joint_final as joint

    class PhysicsEnvironment:
        nominal_value = 2.
        def __init__(self):
            self.index = 0
            self.closed = False
            self.unwrapped = SimpleNamespace(
                data=SimpleNamespace(actuator_force=np.array([1., 2.])),
                model=SimpleNamespace(actuator_forcelimited=np.array([False, False]),
                                      actuator_forcerange=np.array([[-3., 3.], [-3., 3.]])))
        def reset(self, seed):
            self.index = 0
            return np.zeros(17, dtype=np.float32), {}
        def step(self, action):
            self.index += 1
            return np.full(17, self.index, dtype=np.float32), float(self.index), False, self.index == 3, dict(x_velocity=2., reward_ctrl=-.5)
        def readback(self):
            return 3.
        def close(self):
            self.closed = True

    class JointAgent:
        def act(self, observation, *, t0, eval_mode):
            assert eval_mode
            return torch.full((6,), .25 if t0 else .5)

    envs = []
    def make_environment(*args):
        env = PhysicsEnvironment()
        envs.append(env)
        return env

    check = dict(parameter="bthigh_damping", severity=1.5, adaptation_seed=7)
    frozen = {"episodes": {("bthigh_damping", 1.5, 19): dict(episode_return=9., forward_velocity=1., control_cost=2.)}}
    with patch.object(joint, "load_agent", return_value=(JointAgent(), "unchanged")), \
         patch.object(joint, "assert_unchanged") as unchanged, \
         patch.object(joint.ttt, "seeded") as seed_fn, \
         patch.object(joint.ttt, "environment", side_effect=make_environment):
        rows = joint.evaluate_one(SimpleNamespace(), {"evaluation_seeds": [19], "horizon": 3},
                                  frozen, check, "shifted")
    assert len(rows) == 1
    row = rows[0]
    assert row["evaluation_seed"] == 19
    assert row["joint_return"] == 6.
    assert row["joint_control_cost"] == 1.5
    assert row["joint_reward_run"] == 7.5
    assert row["episode_length"] == 3
    assert row["terminated"] is False and row["truncated"] is True
    assert row["nominal_parameter"] == 2.
    assert row["actual_parameter_min"] == row["actual_parameter_max"] == 3.
    assert row["joint_action_abs"] == (0.25 + 0.5 + 0.5) / 3
    assert all(env.closed for env in envs)
    seed_fn.assert_called_once_with(19)
    unchanged.assert_called()
