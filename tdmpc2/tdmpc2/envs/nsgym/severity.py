"""Experiment-only physical schedules through official NS-Gym setters/updates."""
import numpy as np
from ns_gym.schedulers import DiscreteScheduler
from ns_gym.update_functions import StepWiseUpdate
from ns_gym.wrappers import MujocoWrapper
from ns_gym.wrappers.mujoco_env import param_look_up
from .halfcheetah import HalfCheetah


def multiplier(schedule, t, severity, onset, duration, restore, period):
    if schedule == 'nominal':
        return 1.
    if schedule == 'fixed':
        return severity
    if schedule == 'abrupt':
        return severity if t >= onset else 1.
    if schedule == 'gradual':
        return 1 + (severity - 1) * np.clip((t - onset) / duration, 0, 1)
    if schedule == 'reversible':
        return severity if onset <= t < restore else 1.
    if schedule == 'repeated':
        return severity if t >= onset and ((t - onset) // period) % 2 == 0 else 1.
    raise ValueError(schedule)


class SeverityHalfCheetah(HalfCheetah):
    """A single official scalar parameter; existing training adapter is unchanged."""
    def __init__(self, *, parameter, severity, schedule, onset=250, duration=250,
                 restore=750, period=250, max_episode_steps=1000, seed=0):
        if not np.isfinite(severity) or severity <= 0:
            raise ValueError('Severity must be finite and positive')
        if onset < 0 or duration <= 0 or period <= 0 or restore <= onset:
            raise ValueError('Invalid schedule timing')
        super().__init__(mode='nominal', mass_scale=1., max_episode_steps=max_episode_steps, seed=seed)
        self.parameter = parameter
        self.schedule = schedule
        self.severity = severity
        self.timing = dict(onset=onset, duration=duration, restore=restore, period=period)
        self.getter = param_look_up('HalfCheetahEnv', parameter)[0][0]
        self.nominal_value = float(self.getter(self.unwrapped))
        if ('mass' in parameter and self.nominal_value * severity <= 1e-6) or self.nominal_value <= 0:
            self.close()
            raise ValueError('This positive-multiplier protocol requires positive scalar nominal/target values')
        if parameter.endswith('_damping'):
            joint = 'ffoot' if parameter == 'ffeet_damping' else parameter.removesuffix('_damping')
            model = self.unwrapped.model
            if model.joint(joint).id != model.jnt_dofadr[model.joint(joint).id]:
                raise ValueError('Upstream joint-id damping index differs from DOF address')
        self.expected = [float(multiplier(schedule, t, severity, **self.timing)) * self.nominal_value
                         for t in range(max_episode_steps)]
        times = [t for t, v in enumerate(self.expected)
                 if (t == 0 and v != self.nominal_value) or (t > 0 and v != self.expected[t-1])]
        changes = {parameter: StepWiseUpdate(DiscreteScheduler(set(times)), [self.expected[t] for t in times])} if times else {}
        # Replace only the nominal NS wrapper, retaining its original Gymnasium env.
        self.env = MujocoWrapper(self.env.env, changes, change_notification=False,
                                 delta_change_notification=False, scalar_reward=True, persistent_params=False)

    def readback(self):
        return float(self.getter(self.unwrapped))

    def scheduler_state(self):
        fn = self.env.tunable_params.get(self.parameter)
        return dict(ns_time=int(self.env.t), remaining_values=len(fn.param_list) if fn else 0,
                    episode_local_step=self._step, parameter=self.parameter, value=self.readback())

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        np.testing.assert_allclose(self.readback(), self.nominal_value, rtol=0, atol=1e-12)
        return result

    def step(self, action):
        t = self._step
        result = super().step(action)
        np.testing.assert_allclose(self.readback(), self.expected[t], rtol=0, atol=1e-10)
        if not np.isfinite(result[0]).all() or not np.isfinite(result[1]):
            raise FloatingPointError('Nonfinite physical rollout')
        return result
