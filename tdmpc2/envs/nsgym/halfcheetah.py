"""HalfCheetah research adapter for the pinned official NS-Gym API.

Agent API: standard Gymnasium, state-only observation, scalar reward, allowlisted
info. Physical readbacks go exclusively to the experiment-owned recorder.
Schedules count Gymnasium transitions (five MuJoCo integration steps each),
starting at zero BEFORE the action; every reset restarts the schedule.
"""
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from ns_gym.schedulers import DiscreteScheduler
from ns_gym.update_functions import StepWiseUpdate
from ns_gym.wrappers import MujocoWrapper


class HalfCheetah(gym.Wrapper):
    """Nominal/fixed/abrupt/reversible shifts of torso mass in kilograms.

    Fixed applies on transition 0; abrupt on switch_step; reversible restores
    the original mass on restore_step. Reset itself restores nominal physics.
    No agent receives schedule time, parameter values or change notifications.
    """

    def __init__(self, *, mode='nominal', mass_scale=1.5, switch_step=100,
                 restore_step=200, seed=0, max_episode_steps=1000,
                 recorder: Callable[[dict], None] | None = None,
                 render_mode=None):
        if mode not in {'nominal', 'fixed', 'abrupt', 'reversible'}:
            raise ValueError(f'Unknown NS-Gym mode: {mode}')
        if not np.isfinite(mass_scale) or mass_scale <= 0:
            raise ValueError('mass_scale must be finite and positive')
        for name, value in [('switch_step', switch_step), ('restore_step', restore_step),
                            ('max_episode_steps', max_episode_steps), ('seed', seed)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if max_episode_steps == 0:
            raise ValueError('max_episode_steps must be positive')
        if mode in {'abrupt', 'reversible'} and switch_step >= max_episode_steps:
            raise ValueError('switch_step must occur before episode truncation')
        if mode == 'reversible' and not switch_step < restore_step < max_episode_steps:
            raise ValueError('Require switch_step < restore_step < max_episode_steps')
        raw = gym.make('HalfCheetah-v5', max_episode_steps=max_episode_steps,
                       render_mode=render_mode)
        self._nominal_mass = float(raw.unwrapped.model.body('torso').mass[0])
        shifted = self._nominal_mass * mass_scale
        if shifted <= 1e-6 or not np.isfinite(shifted):
            raw.close()
            raise ValueError('Scaled torso mass must be finite and greater than 1e-6 kg')
        changes = {}
        if mode != 'nominal':
            times = [0 if mode == 'fixed' else switch_step]
            values = [shifted]
            if mode == 'reversible':
                times.append(restore_step)
                values.append(self._nominal_mass)
            changes['torso_mass'] = StepWiseUpdate(DiscreteScheduler(set(times)), values)
        try:
            wrapped = MujocoWrapper(raw, changes, change_notification=False,
                                    delta_change_notification=False,
                                    scalar_reward=True, persistent_params=False)
        except Exception:
            raw.close()
            raise
        super().__init__(wrapped)
        # NS-Gym advertises a Dict that includes privileged metadata. Discard it.
        self.observation_space = raw.observation_space
        self.max_episode_steps = max_episode_steps
        self._initial_seed = seed
        self._first_reset = True
        self._needs_reset = True
        self._episode = -1
        self._step = 0
        self._recorder = recorder
        self._mode = mode

    @staticmethod
    def _state(observation):
        return np.asarray(observation['state']).copy()

    @staticmethod
    def _public_info(info):
        # Upstream always includes Ground Truth Env Change/Delta Change in info,
        # even when notifications are disabled. Never pass that dictionary on.
        return {key: info[key] for key in ('x_position', 'x_velocity', 'reward_run',
                                          'reward_ctrl') if key in info}

    def _record(self, event, transition_index, seed=None):
        if self._recorder is None:
            return
        model = self.unwrapped.model
        self._recorder(dict(event=event, episode=self._episode,
                            transition_index=transition_index, seed=seed,
                            mode=self._mode, ns_time=int(self.env.t),
                            torso_mass=float(model.body('torso').mass[0]),
                            gravity=model.opt.gravity.copy().tolist(),
                            body_mass=model.body_mass.copy().tolist(),
                            dof_damping=model.dof_damping.copy().tolist(),
                            geom_friction=model.geom_friction.copy().tolist()))

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
            raise ValueError('Expected a finite HalfCheetah action with shape (6,)')
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError('Action outside HalfCheetah action bounds')
        obs, reward, terminated, truncated, info = self.env.step(action.copy())
        # scalar_reward=True is official; explicit float prevents tensor/dataclass
        # rewards accidentally propagating into TD-MPC2.
        reward = float(reward)
        self._record('step', self._step)
        self._step += 1
        self._needs_reset = bool(terminated or truncated)
        return self._state(obs), reward, bool(terminated), bool(truncated), self._public_info(info)

    def close(self):
        self._needs_reset = True
        return self.env.close()
