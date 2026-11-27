"""Real MuJoCo tests: no mocked simulator or expensive policy training."""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tdmpc2'))
from envs.nsgym import HalfCheetah, make_env


def rollout(env, seed=17, steps=30):
    observation, info = env.reset(seed=seed)
    states = [observation.copy()]
    rewards = []
    rng = np.random.default_rng(92)
    for action in rng.uniform(-0.8, 0.8, (steps, 6)):
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(observation.copy())
        rewards.append(reward)
    return np.array(states), np.array(rewards)


def test_nominal_matches_unwrapped_gymnasium():
    wrapped = HalfCheetah()
    raw = gym.make('HalfCheetah-v5').unwrapped
    try:
        a, ar = rollout(wrapped)
        b, br = rollout(raw)
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
        np.testing.assert_allclose(ar, br, atol=1e-10, rtol=0)
    finally:
        wrapped.close()
        raw.close()


def test_reported_reward_components_match_scalar_reward():
    env = HalfCheetah()
    try:
        env.reset(seed=17)
        _, reward, _, _, info = env.step(np.full(6, .2))
        assert info['reward_run'] + info['reward_ctrl'] == pytest.approx(reward)
        assert info['reward_run'] == pytest.approx(info['x_velocity'])
    finally:
        env.close()


@pytest.mark.parametrize('mode', ['nominal', 'fixed', 'abrupt', 'reversible'])
def test_timing_reset_and_reproducibility(mode):
    records = []
    env = HalfCheetah(mode=mode, switch_step=3, restore_step=7,
                     max_episode_steps=12, recorder=records.append)
    try:
        first = rollout(env, steps=10)
        mass = records[0]['torso_mass']
        expected = [mass * (1.5 if mode == 'fixed' or
                    (mode == 'abrupt' and i >= 3) or
                    (mode == 'reversible' and 3 <= i < 7) else 1) for i in range(10)]
        np.testing.assert_array_equal([r['torso_mass'] for r in records[1:]], expected)
        assert [r['ns_time'] for r in records[1:]] == list(range(1, 11))
        assert [r['transition_index'] for r in records[1:]] == list(range(10))
        records.clear()
        second = rollout(env, steps=10)
        assert records[0]['torso_mass'] == mass
        np.testing.assert_array_equal([r['torso_mass'] for r in records[1:]], expected)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        # Reset without explicit seed must still restart physics and scheduler.
        records.clear()
        env.reset()
        assert records[0]['torso_mass'] == mass
        for _ in range(10):
            env.step(np.zeros(6))
        np.testing.assert_array_equal([r['torso_mass'] for r in records[1:]], expected)
    finally:
        env.close()


def test_mass_shift_changes_real_transitions_and_restores_model_constants():
    nominal = HalfCheetah()
    shifted = HalfCheetah(mode='reversible', mass_scale=2, switch_step=3, restore_step=7)
    try:
        a, _ = rollout(nominal)
        b, _ = rollout(shifted)
        np.testing.assert_allclose(a[:4], b[:4], atol=1e-10, rtol=0)
        assert np.max(np.abs(a[4:8] - b[4:8])) > 1e-5
        # Restoration restores the parameter/model, not the already-diverged trajectory.
        np.testing.assert_array_equal(nominal.unwrapped.model.body_mass,
                                      shifted.unwrapped.model.body_mass)
        np.testing.assert_allclose(nominal.unwrapped.model.body_subtreemass,
                                   shifted.unwrapped.model.body_subtreemass, atol=1e-12, rtol=0)
        # Same full integration state + restored physics yields the same next transition.
        import mujoco
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(nominal.unwrapped.model, spec))
        mujoco.mj_getState(nominal.unwrapped.model, nominal.unwrapped.data, state, spec)
        mujoco.mj_setState(shifted.unwrapped.model, shifted.unwrapped.data, state, spec)
        mujoco.mj_forward(shifted.unwrapped.model, shifted.unwrapped.data)
        aa = nominal.step(np.full(6, .2))
        bb = shifted.step(np.full(6, .2))
        np.testing.assert_allclose(aa[0], bb[0], atol=1e-10, rtol=0)
        assert aa[1] == pytest.approx(bb[1], abs=1e-10)
    finally:
        nominal.close()
        shifted.close()


def test_agent_contract_seed_and_episode_end():
    records = []
    env = HalfCheetah(mode='fixed', max_episode_steps=2, recorder=records.append, seed=34)
    try:
        with pytest.raises(gym.error.ResetNeeded):
            env.step(np.zeros(6))
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert obs.shape == (17,)
        a = env.action_space.sample()
        obs2, _ = env.reset(seed=34)
        np.testing.assert_array_equal(obs, obs2)
        np.testing.assert_array_equal(a, env.action_space.sample())
        for i in range(2):
            obs, reward, terminated, truncated, info = env.step(np.zeros(6))
            assert type(reward) is float
            assert env.observation_space.contains(obs)
            assert set(info) <= {'x_position', 'x_velocity', 'reward_run', 'reward_ctrl'}
            assert not terminated
            assert truncated == (i == 1)
        assert records[-1]['torso_mass'] > records[0]['torso_mass']
        with pytest.raises(gym.error.ResetNeeded):
            env.step(np.zeros(6))
        env.reset()
        assert env.step(np.zeros(6))[3] is False
    finally:
        env.close()


def test_tdmpc2_tensor_contract_and_adaptation_imports():
    import torch
    from omegaconf import OmegaConf
    from common.stage3a import EpisodeAdapter
    from common.decision_ttt import DecisionAdapter
    from common.shift_protocol import make_protocol
    assert EpisodeAdapter and DecisionAdapter and make_protocol
    cfg = OmegaConf.create(dict(task='nsgym-halfcheetah', seed=17, obs='state', multitask=False,
                                nsgym=dict(max_episode_steps=2)))
    env = make_env(cfg)
    try:
        obs = env.reset()
        assert obs.dtype == torch.float32 and tuple(obs.shape) == (17,)
        assert cfg.action_dim == 6 and cfg.episode_length == 2
        assert cfg.obs_shape.state == [17]
        for i in range(2):
            obs, reward, done, info = env.step(torch.zeros(6))
            assert reward.ndim == 0 and reward.dtype == torch.float32
            assert done == (i == 1)
            assert not bool(info['terminated'])
            assert info['truncated'] == (i == 1)
            assert set(info) <= {'x_position','x_velocity','reward_run','reward_ctrl',
                                 'terminated','truncated','success'}
    finally:
        env.close()


@pytest.mark.parametrize('kwargs', [dict(mode='unknown'), dict(mass_scale=0),
    dict(mass_scale=float('nan')), dict(mass_scale=float('inf')), dict(mass_scale=1e-15),
    dict(switch_step=-1), dict(switch_step=1.5), dict(max_episode_steps=0),
    dict(mode='reversible', switch_step=5, restore_step=5),
    dict(mode='abrupt', switch_step=1000), dict(mode='reversible', restore_step=1000)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        HalfCheetah(**kwargs)


def test_invalid_actions():
    env = HalfCheetah()
    try:
        env.reset()
        for action in (np.zeros(5), np.full(6, np.nan), np.full(6, 2)):
            with pytest.raises(ValueError):
                env.step(action)
    finally:
        env.close()
