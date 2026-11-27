"""Autoregressive dynamics objective and real Dynamics+Q adapter contracts."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
sys.path.insert(0, str(PROJECT / "experiments/nsgym/scripts"))

from research.dynamics_shift.audit import equal_state
from research.dynamics_shift.dynamics_critic import DynamicsCriticAdaptation
from research.dynamics_shift.objectives import (
    DynamicsObjectiveConfig, MultiStepDynamicsConsistency, ObjectiveHooks, one_step_dynamics_mse,
)
from research.dynamics_shift.replay import AdaptationReplayBuffer


class LinearDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.next_layer = torch.nn.Linear(3, 2)

    def next(self, latent, action, task):
        assert task is None
        return self.next_layer(torch.cat((latent, action), -1))


def test_horizon_one_matches_original_loss_gradients_and_adam_step():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        old = LinearDynamics()
        new = deepcopy(old)
        z = torch.tensor([[.2, -.4], [.5, .1]])
        actions = torch.tensor([[[.3], [-.2]]])
        targets = torch.tensor([[[.1, .6], [-.4, .2]]])
        original = one_step_dynamics_mse(old, z, actions[0], targets[0])
        multi = MultiStepDynamicsConsistency(1, .5, "horizon")(
            new, z, actions, targets)
        torch.testing.assert_close(multi, original, rtol=0, atol=0)
        original.backward()
        multi.backward()
        for before, after in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(after.grad, before.grad, rtol=0, atol=0)
        old_optimizer = torch.optim.Adam(old.parameters(), lr=1e-4)
        new_optimizer = torch.optim.Adam(new.parameters(), lr=1e-4)
        old_optimizer.step()
        new_optimizer.step()
        assert equal_state(old.state_dict(), new.state_dict())
        assert equal_state(old_optimizer.state_dict(), new_optimizer.state_dict())


def test_rollout_is_autoregressive_action_aligned_and_temporally_weighted():
    class Trace:
        def __init__(self):
            self.inputs = []

        def next(self, latent, action, task):
            self.inputs.append((latent.detach().clone(), action.detach().clone()))
            return latent * 2 + action

    model = Trace()
    z = torch.tensor([[1.]])
    actions = torch.tensor([[[3.]], [[5.]], [[7.]]])
    targets = torch.zeros(3, 1, 1)
    loss = MultiStepDynamicsConsistency(3, .5, "horizon")(
        model, z, actions, targets)
    assert [float(item[0]) for item, _ in model.inputs] == [1., 5., 15.]
    assert [float(item[0]) for _, item in model.inputs] == [3., 5., 7.]
    assert float(loss) == pytest.approx((5**2 + .5 * 15**2 + .25 * 37**2) / 3)


def test_later_loss_backpropagates_through_earlier_dynamics_and_not_encoder_targets():
    class ScalarDynamics(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))

        def next(self, latent, action, task):
            return self.weight * latent + action

    model = ScalarDynamics()
    initial = torch.ones(1, 1, requires_grad=True)
    targets = torch.zeros(2, 1, 1, requires_grad=True)
    loss = MultiStepDynamicsConsistency(2, .5, "horizon")(
        model, initial, torch.zeros(2, 1, 1), targets)
    loss.backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor(10.), rtol=0, atol=0)
    assert initial.grad is None and targets.grad is None


def test_valid_mask_normalization_and_invalid_configurations():
    class Add:
        def next(self, latent, action, task):
            return latent + action

    actions = torch.tensor([[[1.], [2.]], [[3.], [4.]]])
    target = torch.zeros(2, 2, 1)
    mask = torch.tensor([[True, True], [True, False]])
    horizon = MultiStepDynamicsConsistency(2, .5, "horizon")(
        Add(), torch.zeros(2, 1), actions, target, mask)
    valid = MultiStepDynamicsConsistency(2, .5, "valid_steps")(
        Add(), torch.zeros(2, 1), actions, target, mask)
    numerator = 1.**2 + 2.**2 + .5 * 4.**2
    assert float(horizon) == pytest.approx(numerator / 4)
    assert float(valid) == pytest.approx(numerator / 3)
    with pytest.raises(ValueError, match="valid prediction"):
        MultiStepDynamicsConsistency(2, .5, "valid_steps")(
            Add(), torch.zeros(2, 1), actions, target, torch.zeros_like(mask))
    defaults = DynamicsObjectiveConfig.from_mapping({})
    assert defaults.as_dict() == dict(dynamics_objective="one_step_dynamics_consistency",
                                     critic_objective="replay_encoded_critic", rollout_horizon=1,
                                     temporal_discount=1., normalization_mode="horizon")
    settings = dict(dynamics_objective="multi_step_imagined_dynamics_consistency",
                    critic_objective="replay_encoded_critic", rollout_horizon=3,
                    temporal_discount=.5, normalization_mode="valid_steps")
    assert DynamicsObjectiveConfig.from_mapping(settings).as_dict() == settings
    with pytest.raises(ValueError):
        DynamicsObjectiveConfig.from_mapping({"rollout_horizon": 3})
    for change in (dict(dynamics_objective="unknown"), dict(critic_objective="imagined_multi_position_critic"),
                   dict(rollout_horizon=0),
                   dict(temporal_discount=0), dict(temporal_discount=float("nan")),
                   dict(normalization_mode="unknown")):
        with pytest.raises(ValueError):
            DynamicsObjectiveConfig.from_mapping({**settings, **change})
    with pytest.raises(ValueError, match="episode horizon"):
        DynamicsCriticAdaptation(object(), dict(settings, horizon=2), {})
    with pytest.raises(ValueError, match="Custom objective"):
        DynamicsCriticAdaptation(object(), dict(settings, horizon=10), {}, objectives=ObjectiveHooks())


def test_episode_windows_end_at_boundary_and_never_cross_it():
    def row(episode, step, global_step, *, terminated=False, truncated=False):
        return dict(episode=episode, episode_step=step, global_step=global_step,
                    terminated=terminated, truncated=truncated)

    replay = AdaptationReplayBuffer([
        row(0, 0, 1), row(0, 1, 2, truncated=True), row(1, 0, 3),
        row(1, 1, 4, terminated=True), row(2, 0, 5), row(2, 2, 7),
    ], maxlen=6)
    assert [[r["global_step"] for r in window] for window in replay.contiguous_sequences(2)] == [
        [1, 2], [3, 4]]
    assert replay.contiguous_sequences(3) == []


def test_adapter_sequence_shortage_and_terminal_window_with_minimal_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._encoder = torch.nn.Linear(2, 2)
            self._dynamics = torch.nn.Linear(3, 2)
            self._Qs = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(2)])
            self._detach_Qs_params = self._Qs
            self._target_Qs_params = deepcopy(self._Qs)
            self._reward = torch.nn.Linear(3, 1)
            self._pi = torch.nn.Linear(2, 1)

        def encode(self, observation, task):
            return self._encoder(observation)

        def next(self, latent, action, task):
            return self._dynamics(torch.cat((latent, action), -1))

        def Q(self, latent, action, task, return_type):
            features = torch.cat((latent, action), -1)
            return torch.stack([head(features) for head in self._Qs])

        def soft_update_target_Q(self):
            for target, online in zip(self._target_Qs_params.parameters(), self._Qs.parameters()):
                target.lerp_(online, .1)

    class Agent:
        def __init__(self):
            self.model = Model()
            self.discount = .9
            self.cfg = SimpleNamespace(num_bins=3, num_q=2, vmin=-1., vmax=1.,
                                       bin_size=1., tau=.1)

        def _td_target(self, next_z, reward, terminal, task):
            return reward

    settings = dict(dynamics_objective="multi_step_imagined_dynamics_consistency",
                    critic_objective="replay_encoded_critic", rollout_horizon=3,
                    temporal_discount=.5, normalization_mode="valid_steps",
                    horizon=4, batch_size=2, warmup=2, capacity=10, update_every=1,
                    betas=[.9, .999], eps=1e-8, dynamics_lr=1e-4, q_lr=3e-4,
                    dynamics_grad_clip=10., q_grad_clip=20., consistency_coef=1.,
                    value_coef=.1, evaluation_seeds=[101])
    identity = dict(adaptation_seed=1, severity=.5, parameter="damping")
    agent = Agent()
    adapter = DynamicsCriticAdaptation(agent, settings, identity)
    initial = torch.tensor([.1, -.2])
    for step in range(4):
        observation = initial + step * .01
        next_observation = initial + (step + 1) * .01
        adapter.observe(observation, torch.tensor([.2]), next_observation,
                        .1, False, step == 3, 0, step, 3., 6.)
    assert adapter.sequence_skips == 2
    assert adapter.dynamics_steps == adapter.q_steps == adapter.target_steps == 1
    assert len(adapter.last_sequence_steps) == 2
    assert sorted(window[-1] for window in adapter.last_sequence_steps) == [3, 4]
    assert adapter.state_dict()["objective_configuration"] == adapter.objective_config.as_dict()
    incompatible = DynamicsCriticAdaptation(Agent(), dict(settings, temporal_discount=.8), identity)
    with pytest.raises(ValueError, match="configuration mismatch"):
        incompatible.load_state_dict(adapter.state_dict())


def _real_rows(count, discount):
    observation = torch.linspace(-.2, .2, 17)
    rows = []
    for step in range(count):
        action = (torch.linspace(-.5, .5, 6) + step * .001).tanh()
        next_observation = observation + .001 + action.mean() * .0001
        rows.append(dict(observation=observation.clone(), action=action.clone(),
                         next_observation=next_observation.clone(), reward=.1 * (step % 7),
                         discount=float(discount), terminated=False, truncated=False,
                         episode=0, episode_step=step, global_step=step + 1))
        observation = next_observation
    return rows


def _real_adapters(rollout_horizon):
    pytest.importorskip("tensordict")
    import run_dynamics_q as runner
    from common.dynamics_q import DynamicsQAdapter

    config, _ = runner.configuration()
    config = dict(config)
    multi_config = dict(config, dynamics_objective="multi_step_imagined_dynamics_consistency",
                        critic_objective="replay_encoded_critic", rollout_horizon=rollout_horizon,
                        temporal_discount=.5, normalization_mode="horizon")
    identity = dict(parameter="bthigh_damping", severity=.5, adaptation_seed=42000001,
                    method="dynamics_q_118", schedule="fixed")
    torch.manual_seed(817)
    baseline_agent = runner.cpu_agent()
    torch.manual_seed(817)
    multi_agent = runner.cpu_agent()
    assert equal_state(baseline_agent.model.state_dict(), multi_agent.model.state_dict())
    baseline = DynamicsQAdapter(baseline_agent, config, identity)
    multi = DynamicsQAdapter(multi_agent, multi_config, identity)
    rows = _real_rows(75, baseline_agent.discount)
    baseline.replay.extend(deepcopy(rows))
    multi.replay.extend(deepcopy(rows))
    baseline.transitions = multi.transitions = 75
    return baseline, multi, baseline_agent, multi_agent, rows


def test_real_model_h1_adapter_and_checkpoint_parity():
    pytest.importorskip("tensordict")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            baseline, multi, baseline_agent, multi_agent, _ = _real_adapters(1)
            torch.manual_seed(202)
            original = baseline.update()
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(202)
            current = multi.update()
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert original == current
            assert equal_state(baseline_agent.model.state_dict(), multi_agent.model.state_dict())
            assert equal_state(baseline.dynamics_optimizer.state_dict(), multi.dynamics_optimizer.state_dict())
            assert equal_state(baseline.q_optimizer.state_dict(), multi.q_optimizer.state_dict())
            assert baseline.last_sample_steps == [row[0] for row in multi.last_sequence_steps]
            assert "objective_configuration" not in baseline.state_dict()
            assert multi.state_dict()["objective_configuration"] == multi.objective_config.as_dict()
            fresh = type(multi)(multi_agent, multi.config, multi.identity)
            fresh.load_state_dict(multi.state_dict())
            assert equal_state(fresh.state_dict(), multi.state_dict())
            with pytest.raises(ValueError, match="multi-step"):
                baseline.load_state_dict(multi.state_dict())
    finally:
        torch.set_num_threads(old_threads)


def test_real_model_h3_critic_ema_and_frozen_components_remain_isolated():
    pytest.importorskip("tensordict")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            baseline, multi, baseline_agent, multi_agent, rows = _real_adapters(3)
            before = deepcopy(multi.fixed_state())
            target_before = deepcopy(multi_agent.model._target_Qs_params.state_dict())
            windows = multi.replay.contiguous_sequences(3)[:2]
            initial_z, sequence_actions, targets, valid = multi._sequence_tensors(windows)
            assert initial_z.shape == (2, multi_agent.cfg.latent_dim)
            assert sequence_actions.shape == (3, 2, multi_agent.cfg.action_dim)
            assert targets.shape == (3, 2, multi_agent.cfg.latent_dim)
            assert valid.all() and not targets.requires_grad and not initial_z.requires_grad
            with torch.no_grad():
                for step in range(3):
                    expected_target = multi_agent.model.encode(torch.stack([
                        window[step]["next_observation"] for window in windows]), None)
                    torch.testing.assert_close(targets[step], expected_target, rtol=0, atol=0)
            unrelated = deepcopy(windows)
            unrelated[0][1]["observation"] += 1
            with pytest.raises(ValueError, match="boundary or missing step"):
                multi._sequence_tensors(unrelated)
            torch.manual_seed(204)
            original = baseline.update()
            torch.manual_seed(204)
            current = multi.update()
            assert current["q_loss"] == original["q_loss"]
            assert equal_state(baseline.q_optimizer.state_dict(), multi.q_optimizer.state_dict())
            assert equal_state(baseline_agent.model._target_Qs_params.state_dict(),
                               multi_agent.model._target_Qs_params.state_dict())
            assert equal_state(multi.fixed_state(), before)
            assert (multi.dynamics_steps, multi.q_steps, multi.target_steps) == (1, 1, 1)
            assert all(len(steps) == 3 and steps[1] == steps[0] + 1 and steps[2] == steps[1] + 1
                       for steps in multi.last_sequence_steps)
            for key, value in multi_agent.model._target_Qs_params.state_dict().items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, target_before[key].lerp(
                        multi_agent.model._detach_Qs_params.state_dict()[key], multi_agent.cfg.tau),
                        rtol=0, atol=0)
            with multi.held_out():
                with pytest.raises(RuntimeError, match="replay"):
                    multi.observe(rows[0]["observation"], rows[0]["action"],
                                  rows[0]["next_observation"], 0., False, False,
                                  0, 0, 3., 6.)
    finally:
        torch.set_num_threads(old_threads)


def test_real_adapter_defers_updates_until_short_episode_windows_exist():
    pytest.importorskip("tensordict")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            _, multi, _, _, _ = _real_adapters(2)
            multi.replay.clear()
            multi.transitions = 0
            multi.config["horizon"] = 2
            initial = torch.linspace(-.2, .2, 17)
            for index in range(150):
                step = index % 2
                observation = initial + step * .001
                next_observation = initial + (step + 1) * .001
                multi.observe(observation, torch.zeros(6), next_observation,
                              .1, False, step == 1, index // 2, step, 3., 6.)
            assert multi.sequence_skips == 3
            assert multi.dynamics_steps == multi.q_steps == multi.target_steps == 1
            assert all(window[-1] == window[0] + 1 for window in multi.last_sequence_steps)
            state = multi.state_dict()
            assert state["sequence_skips"] == 3
    finally:
        torch.set_num_threads(old_threads)
