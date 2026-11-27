"""Real Pusher-v5 integration checks; no model training required."""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tdmpc2'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from envs.nsgym import Pusher
from training_support import config_from_file, build, evaluate


def test_nominal_matches_gymnasium_with_normalized_actions():
    wrapped = Pusher(seed=17)
    raw = gym.make('Pusher-v5')
    try:
        a, info = wrapped.reset(seed=17)
        b, _ = raw.reset(seed=17)
        np.testing.assert_allclose(a, b, atol=0, rtol=0)
        assert info == {}
        assert wrapped.observation_space.shape == (23,)
        assert wrapped.action_space.shape == (7,)
        for t in range(20):
            action = np.full(7, 0.2 * np.sin(t), dtype=np.float32)
            obs_a, reward_a, term_a, trunc_a, info_a = wrapped.step(action)
            obs_b, reward_b, term_b, trunc_b, info_b = raw.step(2 * action)
            np.testing.assert_allclose(obs_a, obs_b, atol=1e-10, rtol=0)
            assert abs(reward_a - reward_b) < 1e-10
            assert (term_a, trunc_a) == (term_b, trunc_b)
            assert set(info_a) == {'reward_dist', 'reward_ctrl', 'reward_near'}
            assert abs(sum(info_a.values()) - reward_a) < 1e-8
    finally:
        wrapped.close()
        raw.close()


@pytest.mark.parametrize('mode,expected', [
    ('fixed', [1.5] * 8),
    ('abrupt', [1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5]),
    ('reversible', [1, 1, 1, 1.5, 1.5, 1.5, 1, 1]),
])
def test_schedule_mass_readback_and_reset(mode, expected):
    logs = []
    env = Pusher(mode=mode, mass_scale=1.5, switch_step=3, restore_step=6,
                 max_episode_steps=8, seed=17, recorder=logs.append)
    try:
        env.reset(seed=17)
        nominal = logs[0]['link_mass']
        for _ in range(8):
            _, _, _, _, info = env.step(np.full(7, 0.25, dtype=np.float32))
            assert 'Ground Truth Env Change' not in info
        np.testing.assert_allclose([r['link_mass'] for r in logs[1:]],
                                   nominal * np.array(expected), atol=1e-10, rtol=0)
        with pytest.raises(gym.error.ResetNeeded):
            env.step(np.zeros(7))
        env.reset(seed=17)
        assert logs[-1]['link_mass'] == nominal
        env.step(np.zeros(7))
        assert abs(logs[-1]['link_mass'] - nominal * expected[0]) < 1e-10
    finally:
        env.close()


def test_shift_changes_simulator_trajectory():
    nominal = Pusher(seed=17)
    shifted = Pusher(seed=17, mode='fixed', mass_scale=1.5)
    try:
        nominal.reset(seed=17)
        shifted.reset(seed=17)
        action = np.full(7, 0.7, dtype=np.float32)
        observations = []
        for _ in range(20):
            observations = [nominal.step(action)[0], shifted.step(action)[0]]
        assert np.max(np.abs(observations[0] - observations[1])) > 1e-6
        assert shifted.unwrapped.model.body('r_shoulder_pan_link').mass[0] > nominal.unwrapped.model.body('r_shoulder_pan_link').mass[0]
    finally:
        nominal.close()
        shifted.close()


def test_training_factory_and_random_evaluation():
    config = config_from_file(Path('experiments/nsgym/configs/train_pusher_pilot.yaml'), smoke=True)
    cfg, env = build(config)
    try:
        assert cfg.obs_shape == {'state': (23,)}
        assert cfg.action_dim == 7
        assert cfg.episode_length == 8
        assert env.reset(seed=23).shape == (23,)
    finally:
        env.close()
    rows, _ = evaluate(None, config, config['eval_seeds'], random_actions=True)
    zero, _ = evaluate(None, config, config['eval_seeds'], zero_actions=True)
    assert len(rows) == 2
    assert all(r['episode_length'] == 8 and r['mass_parameter'] == Pusher.PARAMETER for r in rows)
    assert all(abs(r['episode_return'] - r['reward_dist'] - r['reward_near'] - r['reward_control']) < 1e-4 for r in rows)
    assert all(r['zero_actions'] and r['control_effort'] == 0 for r in zero)
    assert all(abs(r['goal_distance_initial'] - r['goal_distance_final'] - r['goal_progress']) < 1e-8 for r in rows + zero)
