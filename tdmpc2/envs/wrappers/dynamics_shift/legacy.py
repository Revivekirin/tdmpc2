"""Hidden action gains, measured in zero-based native environment steps."""
import numpy as np


class DynamicsShiftWrapper:
    """Legacy four-value API; resets never rewind the global clock.

    Repeated mode alternates nominal/shifted blocks of ``period`` native steps.
    Sinusoidal mode uses ``period`` as a full cycle. The final action component
    is the gripper for ManiSkill2's supported Panda arm controllers.
    """

    def __init__(self, env, mode='nominal', gain=0.7, switch_step=100,
                 period=200, amplitude=0.3, clock='global', preserve_gripper=True):
        if mode not in {'nominal', 'fixed', 'abrupt', 'repeated', 'sinusoidal'}:
            raise ValueError(f'Unknown shift mode: {mode}')
        if clock not in {'global', 'episode'}:
            raise ValueError(f'Unknown shift clock: {clock}')
        if not np.isfinite(gain) or gain < 0 or not np.isfinite(amplitude) or not 0 <= amplitude <= 1:
            raise ValueError('Gain must be finite/nonnegative and amplitude in [0, 1].')
        if int(switch_step) != switch_step or switch_step < 0 or int(period) != period or period <= 0:
            raise ValueError('switch_step must be a nonnegative integer; period a positive integer.')
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.low = np.asarray(self.action_space.low)
        self.high = np.asarray(self.action_space.high)
        if (self.low.ndim != 1 or self.low.shape != self.high.shape
                or not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high))
                or not np.all(self.low < self.high)):
            raise ValueError('Expected a finite vector Box action space with low < high.')
        self.mode, self.gain, self.clock = mode, float(gain), clock
        self.switch_step, self.period, self.amplitude = int(switch_step), int(period), float(amplitude)
        self.preserve_gripper = preserve_gripper
        self.global_step = self.episode_step = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.episode_step = 0
        return result

    def applied_gain(self):
        t = self.global_step if self.clock == 'global' else self.episode_step
        if self.mode == 'fixed':
            return self.gain
        if self.mode == 'abrupt':
            return 1.0 if t < self.switch_step else self.gain
        if self.mode == 'repeated':
            return 1.0 if (t // self.period) % 2 == 0 else self.gain
        if self.mode == 'sinusoidal':
            return 1.0 + self.amplitude * np.sin(2 * np.pi * t / self.period)
        return 1.0

    def step(self, action):
        commanded = np.asarray(action).copy()
        if commanded.shape != self.low.shape or not np.all(np.isfinite(commanded)):
            raise ValueError('Action must be finite and match the native action shape.')
        gain = self.applied_gain()
        executed = commanded.copy()
        arm = slice(None, -1) if self.preserve_gripper else slice(None)
        executed[arm] = commanded[arm] * gain
        executed = np.clip(executed, self.low, self.high)
        obs, reward, done, info = self.env.step(executed)
        info = dict(info)
        info.update(shift_gain=float(gain), shift_global_step=self.global_step,
                    shift_episode_step=self.episode_step,
                    shift_commanded_action=commanded, shift_executed_action=executed.copy())
        self.global_step += 1
        self.episode_step += 1
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()
