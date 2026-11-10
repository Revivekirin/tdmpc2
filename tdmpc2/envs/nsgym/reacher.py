"""Nominal Reacher-v5 state-only adapter for the TD-MPC2 NS-Gym pipeline."""
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from ns_gym.wrappers import MujocoWrapper


class Reacher(gym.Wrapper):
    """Expose nominal Reacher without NS-Gym change metadata to the agent."""

    def __init__(self, *, mode='nominal', mass_scale=1.0, seed=0,
                 max_episode_steps=50, recorder: Callable[[dict], None] | None = None,
                 render_mode=None):
        if mode != 'nominal' or mass_scale != 1.0:
            raise ValueError('Reacher adapter currently supports nominal physics only')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if (isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int)
                or max_episode_steps <= 0):
            raise ValueError('max_episode_steps must be a positive integer')
        raw = gym.make('Reacher-v5', max_episode_steps=max_episode_steps,
                       render_mode=render_mode)
        if (raw.observation_space.shape != (10,) or raw.action_space.shape != (2,)
                or not np.all(raw.action_space.low == -1)
                or not np.all(raw.action_space.high == 1)):
            raw.close()
            raise ValueError('Unexpected Reacher-v5 observation or action space')
        try:
            wrapped = MujocoWrapper(raw, {}, change_notification=False,
                                    delta_change_notification=False,
                                    scalar_reward=True, persistent_params=False)
        except Exception:
            raw.close()
            raise
        super().__init__(wrapped)
        self.observation_space = raw.observation_space
        self.action_space = raw.action_space
        self.max_episode_steps = max_episode_steps
        self._initial_seed = seed
        self._first_reset = True
        self._needs_reset = True
        self._episode = -1
        self._step = 0
        self._recorder = recorder

    def reset(self, *, seed=None, options=None):
        if seed is None and self._first_reset:
            seed = self._initial_seed
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.action_space.seed(seed)
        self._first_reset = False
        self._needs_reset = False
        self._episode += 1
        self._step = 0
        if self._recorder is not None:
            self._recorder(dict(event='reset', episode=self._episode, seed=seed))
        return np.asarray(obs['state']).copy(), self._public_info(info)

    @staticmethod
    def _public_info(info):
        return {key: info[key] for key in ('reward_dist', 'reward_ctrl') if key in info}

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded('Call reset before stepping or after episode end')
        action = np.asarray(action)
        if (action.shape != self.action_space.shape or not np.all(np.isfinite(action))
                or np.any(action < -1) or np.any(action > 1)):
            raise ValueError('Expected a finite Reacher action in [-1, 1] with shape (2,)')
        obs, reward, terminated, truncated, info = self.env.step(action.copy())
        if self._recorder is not None:
            self._recorder(dict(event='step', episode=self._episode,
                                transition_index=self._step))
        self._step += 1
        self._needs_reset = bool(terminated or truncated)
        return (np.asarray(obs['state']).copy(), float(reward), bool(terminated),
                bool(truncated), self._public_info(info))

    def close(self):
        self._needs_reset = True
        return self.env.close()
