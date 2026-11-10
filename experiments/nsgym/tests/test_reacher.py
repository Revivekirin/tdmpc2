"""Real Reacher-v5 nominal integration and diagnostic checks."""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tdmpc2'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from envs.nsgym import Reacher
from training_support import build, config_from_file, evaluate


def test_nominal_matches_gymnasium_and_hides_change_metadata():
    wrapped = Reacher(seed=17)
    raw = gym.make('Reacher-v5')
    try:
        a, info = wrapped.reset(seed=17)
        b, _ = raw.reset(seed=17)
        np.testing.assert_allclose(a, b, atol=0, rtol=0)
        assert info == {}
        assert wrapped.observation_space.shape == (10,)
        assert wrapped.action_space.shape == (2,)
        for t in range(20):
            action = np.array([0.2 * np.sin(t), 0.2 * np.cos(t)], dtype=np.float32)
            obs_a, reward_a, term_a, trunc_a, info_a = wrapped.step(action)
            obs_b, reward_b, term_b, trunc_b, info_b = raw.step(action)
            np.testing.assert_allclose(obs_a, obs_b, atol=1e-10, rtol=0)
            assert abs(reward_a - reward_b) < 1e-10
            assert (term_a, trunc_a) == (term_b, trunc_b)
            assert set(info_a) == {'reward_dist', 'reward_ctrl'}
            assert abs(sum(info_a.values()) - reward_a) < 1e-8
            assert abs(np.linalg.norm(obs_a[-2:]) + info_a['reward_dist']) < 1e-8
    finally:
        wrapped.close()
        raw.close()


def test_nominal_only_and_episode_boundary():
    with pytest.raises(ValueError, match='nominal physics only'):
        Reacher(mode='fixed')
    env = Reacher(seed=17, max_episode_steps=8)
    try:
        env.reset(seed=17)
        for _ in range(7):
            _, _, terminated, truncated, _ = env.step(np.zeros(2))
            assert not terminated and not truncated
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        assert not terminated and truncated
        with pytest.raises(gym.error.ResetNeeded):
            env.step(np.zeros(2))
    finally:
        env.close()


def test_training_factory_and_random_evaluation():
    config = config_from_file(Path('experiments/nsgym/configs/train_reacher_pilot.yaml'), smoke=True)
    cfg, env = build(config)
    try:
        assert cfg.obs_shape == {'state': (10,)}
        assert cfg.action_dim == 2
        assert cfg.episode_length == 8
        assert env.reset(seed=29).shape == (10,)
    finally:
        env.close()
    rows, _ = evaluate(None, config, config['eval_seeds'], random_actions=True)
    zero, _ = evaluate(None, config, config['eval_seeds'], zero_actions=True)
    assert len(rows) == len(zero) == 2
    assert all(row['episode_length'] == 8 and row['truncated'] for row in rows)
    assert all(abs(row['episode_return'] - row['reward_dist'] - row['reward_control']) < 1e-6 for row in rows)
    assert all(row['zero_actions'] and row['control_effort'] == 0 for row in zero)
    assert all(abs(row['goal_distance_initial'] - row['goal_distance_final'] - row['goal_progress']) < 1e-8
               for row in rows + zero)
