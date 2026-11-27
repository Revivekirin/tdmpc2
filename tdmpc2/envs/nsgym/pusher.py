"""NS-Gym Pusher-v5 adapter with a normalized TD-MPC2 action interface.

Only the 23-dimensional physical observation is exposed to the agent. NS-Gym
change notifications and physical readbacks stay outside the policy interface.
"""
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from ns_gym.schedulers import DiscreteScheduler
from ns_gym.update_functions import StepWiseUpdate
from ns_gym.wrappers import MujocoWrapper


class Pusher(gym.Wrapper):
    """Change the shoulder pan link mass using the official NS-Gym wrapper."""

    PARAMETER = 'r_shoulder_pan_link_mass'

    def __init__(self, *, mode='nominal', mass_scale=1.5, switch_step=30,
                 restore_step=60, seed=0, max_episode_steps=100,
                 recorder: Callable[[dict], None] | None = None,
                 render_mode=None):
        if mode not in {'nominal', 'fixed', 'abrupt', 'reversible'}:
            raise ValueError(f'Unknown NS-Gym mode: {mode}')
        if not np.isfinite(mass_scale) or mass_scale <= 0:
            raise ValueError('mass_scale must be finite and positive')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
               for v in (switch_step, restore_step, seed, max_episode_steps)):
            raise ValueError('Steps and seed must be nonnegative integers')
        if max_episode_steps == 0:
            raise ValueError('max_episode_steps must be positive')
        if mode in {'abrupt', 'reversible'} and switch_step >= max_episode_steps:
            raise ValueError('switch_step must occur before truncation')
        if mode == 'reversible' and not switch_step < restore_step < max_episode_steps:
            raise ValueError('Require switch_step < restore_step < max_episode_steps')

        raw = gym.make('Pusher-v5', max_episode_steps=max_episode_steps,
                       render_mode=render_mode)
        self._nominal_mass = float(raw.unwrapped.model.body('r_shoulder_pan_link').mass[0])
        shifted = self._nominal_mass * mass_scale
        if not np.isfinite(shifted) or shifted <= 1e-6:
            raw.close()
            raise ValueError('Scaled link mass must exceed 1e-6 kg')
        changes = {}
        if mode != 'nominal':
            times = [0 if mode == 'fixed' else switch_step]
            values = [shifted]
            if mode == 'reversible':
                times.append(restore_step)
                values.append(self._nominal_mass)
            changes[self.PARAMETER] = StepWiseUpdate(DiscreteScheduler(set(times)), values)
        try:
            wrapped = MujocoWrapper(raw, changes, change_notification=False,
                                    delta_change_notification=False,
                                    scalar_reward=True, persistent_params=False)
        except Exception:
            raw.close()
            raise
        super().__init__(wrapped)
        self.observation_space = raw.observation_space
        # TD-MPC2 plans in [-1, 1]. Pusher's native torque bounds are [-2, 2].
        if not (np.all(raw.action_space.low == -2) and np.all(raw.action_space.high == 2)):
            wrapped.close()
            raise ValueError('Expected Pusher-v5 native action bounds [-2, 2]')
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=raw.action_space.shape,
                                           dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self._initial_seed = seed
        self._first_reset = True
        self._needs_reset = True
        self._episode = -1
        self._step = 0
        self._mode = mode
        self._recorder = recorder

    @staticmethod
    def _state(observation):
        return np.asarray(observation['state']).copy()

    @staticmethod
    def _public_info(info):
        return {key: info[key] for key in ('reward_dist', 'reward_ctrl', 'reward_near')
                if key in info}

    def _record(self, event, transition_index, seed=None):
        if self._recorder is None:
            return
        model = self.unwrapped.model
        self._recorder(dict(event=event, episode=self._episode,
                            transition_index=transition_index, seed=seed,
                            mode=self._mode, ns_time=int(self.env.t),
                            link_mass=float(model.body('r_shoulder_pan_link').mass[0])))

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
        self._record('reset', None, seed)
        return self._state(obs), self._public_info(info)

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded('Call reset before stepping or after episode end')
        action = np.asarray(action)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError('Expected a finite normalized Pusher action with shape (7,)')
        if np.any(action < -1) or np.any(action > 1):
            raise ValueError('Normalized Pusher action must be in [-1, 1]')
        native = action.astype(np.float32) * 2
        obs, reward, terminated, truncated, info = self.env.step(native)
        self._record('step', self._step)
        self._step += 1
        self._needs_reset = bool(terminated or truncated)
        return (self._state(obs), float(reward), bool(terminated), bool(truncated),
                self._public_info(info))

    def close(self):
        self._needs_reset = True
        return self.env.close()
