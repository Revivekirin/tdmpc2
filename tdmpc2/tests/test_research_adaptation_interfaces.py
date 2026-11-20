"""Composition checks; numerical behavior is guarded by the immutable goldens."""

from pathlib import Path
import random
import sys
from copy import deepcopy

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'tdmpc2'))
sys.path.insert(0, str(PROJECT/'experiments/nsgym/scripts'))

from common.decision_ttt import DecisionAdapter
from common.dynamics_q import DynamicsQAdapter
from common.stage3a import EpisodeAdapter, equal_state
from research.dynamics_shift import AdaptationConfig, compose_adaptation
from test_stage3a import Model, settings


@pytest.mark.parametrize('method,mode,implementation', [
    ('frozen', 'frozen', EpisodeAdapter),
    ('dynamics_only', 'dynamics_only', EpisodeAdapter),
    ('decision_ttt', 'decision_ttt', DecisionAdapter),
])
def test_episode_composition_reuses_original_object_and_state(method, mode, implementation):
    original = settings(mode)
    if mode == 'decision_ttt':
        original.update(update_granularity='decision', update_every_decisions=4,
                        updates_per_decision=1)
    with torch.random.fork_rng():
        torch.manual_seed(87)
        model = Model()
        cpu_rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        binding = compose_adaptation(AdaptationConfig(method, original), model=model,
                                     consistency_coef=20., grad_clip_norm=20., seed=11)
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert random.getstate() == python_rng
    strategy = binding.strategy
    assert binding.runner is None and strategy is not None
    assert type(strategy.implementation) is implementation
    assert strategy.settings is original
    assert strategy.replay is strategy.implementation.replay
    assert strategy.rng is strategy.implementation.rng
    assert equal_state(strategy.state_dict(), strategy.implementation.state_dict())
    strategy.assert_fixed()


def test_dynamics_q_composition_reuses_optimizer_and_replay_ownership():
    import run_dynamics_q as runner
    with torch.random.fork_rng():
        torch.manual_seed(13)
        agent = runner.cpu_agent()
        config, _ = runner.configuration()
        identity = dict(parameter='bthigh_damping', severity=.5,
                        adaptation_seed=42000001, method='dynamics_q_118', schedule='fixed')
        cpu_rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        binding = compose_adaptation(AdaptationConfig('dynamics_q', config, identity), agent=agent)
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert random.getstate() == python_rng
    strategy = binding.strategy
    assert strategy is not None and type(strategy.implementation) is DynamicsQAdapter
    assert strategy.replay is strategy.implementation.replay
    assert strategy.rng is strategy.implementation.rng
    assert strategy.dynamics_optimizer is strategy.implementation.dynamics_optimizer
    assert strategy.q_optimizer is strategy.implementation.q_optimizer
    assert strategy.state_dict().keys() == strategy.implementation.state_dict().keys()
    strategy.assert_fixed()


def test_joint_runner_delegates_original_arguments_without_constructing_algorithm():
    calls = []

    def production_run(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    binding = compose_adaptation(AdaptationConfig('joint'), run_stream=production_run)
    assert binding.strategy is None and binding.runner is not None
    assert binding.runner.implementation is production_run
    marker = object()
    result = binding.runner.run(marker, budget=3000)
    assert calls == [((marker,), {'budget': 3000})]
    assert result is not None


def test_factory_rejects_method_configuration_mismatch():
    with pytest.raises(ValueError, match='mode differ'):
        compose_adaptation(AdaptationConfig('decision_ttt', settings('dynamics_only')),
                           model=Model(), consistency_coef=1., grad_clip_norm=20.)
    with pytest.raises(ValueError, match='existing production run_stream'):
        compose_adaptation(AdaptationConfig('joint'))


def test_decision_uses_episode_learning_objective_and_optimizer_step():
    assert DecisionAdapter.loss is EpisodeAdapter.loss
    assert DecisionAdapter._optimizer_steps is EpisodeAdapter._optimizer_steps
    assert DecisionAdapter.observe is not EpisodeAdapter.observe
    assert DecisionAdapter.update is not EpisodeAdapter.update


def test_dynamics_q_delegate_matches_direct_update_and_replay_rng():
    import run_dynamics_q as runner
    config, _ = runner.configuration()
    identity = dict(parameter='bthigh_damping', severity=.5,
                    adaptation_seed=42000001, method='dynamics_q_118', schedule='fixed')
    with torch.random.fork_rng():
        torch.manual_seed(917)
        direct_agent = runner.cpu_agent()
        torch.manual_seed(917)
        wrapped_agent = runner.cpu_agent()
        direct = DynamicsQAdapter(direct_agent, config, identity)
        wrapped = compose_adaptation(AdaptationConfig('dynamics_q', config, identity),
                                     agent=wrapped_agent).strategy
        assert wrapped is not None
        inputs = []
        for step in range(75):
            obs = torch.linspace(-1, 1, 17) + step*.001
            action = torch.linspace(-.5, .5, 6)
            inputs.append((obs, action, obs+.002, .01*step, False, False,
                           0, step, 3., 6.))
        torch.manual_seed(521)
        for transition in inputs:
            direct.observe(*transition)
        torch.manual_seed(521)
        for transition in inputs:
            wrapped.observe(*transition)
    assert equal_state(direct_agent.model.state_dict(), wrapped_agent.model.state_dict())
    assert direct.last_sample_steps == wrapped.last_sample_steps
    assert direct.rng.getstate() == wrapped.rng.getstate()
    assert direct.last_metrics == wrapped.last_metrics
    direct_state, wrapped_state = deepcopy(direct.state_dict()), deepcopy(wrapped.state_dict())
    direct_state.pop('update_seconds')
    wrapped_state.pop('update_seconds')
    assert equal_state(direct_state, wrapped_state)
    assert (direct.transitions, direct.dynamics_steps, direct.q_steps, direct.target_steps) == (75, 1, 1, 1)
