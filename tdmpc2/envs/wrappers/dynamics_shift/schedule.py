"""Parameter-independent schedules. Values may be scalars, vectors or matrices."""
from copy import deepcopy
import numpy as np

CLOCKS = {'global_native_step', 'episode_native_step', 'global_decision_step',
          'episode_decision_step', 'episode_index'}
TYPES = {'nominal', 'fixed', 'abrupt', 'repeated', 'sinusoidal', 'linear_drift',
         'random_piecewise', 'sequence'}
DISCRETE = {'nominal', 'fixed', 'abrupt', 'repeated', 'random_piecewise', 'sequence'}


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


class Schedule:
    def __init__(self, config, nominal, shifted):
        self.config = deepcopy(config)
        self.kind = config.get('type', 'nominal')
        self.clock = config.get('clock', 'global_native_step')
        self.reset_on_episode = config.get('reset_on_episode', False)
        if self.kind not in TYPES or self.clock not in CLOCKS:
            raise ValueError('Unsupported schedule type or clock.')
        if self.clock.startswith('episode_') and self.clock != 'episode_index' and not self.reset_on_episode:
            raise ValueError('Episode clocks require reset_on_episode=true.')
        if self.clock.startswith('global_') and self.reset_on_episode:
            raise ValueError('Global clocks require reset_on_episode=false; use an episode clock.')
        if self.clock == 'episode_index' and self.reset_on_episode:
            raise ValueError('episode_index cannot reset on each episode.')
        self.nominal, self.shifted = np.asarray(nominal, dtype=float), np.asarray(shifted, dtype=float)
        if self.nominal.shape != self.shifted.shape or not np.all(np.isfinite([self.nominal, self.shifted])):
            raise ValueError('Nominal and shifted parameters must have identical finite shapes.')
        self.interval = integer(config.get('interval', 200), 'interval', 1)
        self.switch = integer(config.get('switch_step', 100), 'switch_step')
        self.duration = integer(config.get('duration', 200), 'duration', 1)
        self.amplitude = float(config.get('amplitude', 1.0))
        if not np.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError('amplitude must be finite and nonnegative.')
        self.seed = integer(config.get('seed', 42), 'seed')
        self.choices = config.get('values')
        self.segments = config.get('segments', [])
        if self.kind == 'random_piecewise' and not self.choices:
            raise ValueError('random_piecewise requires a nonempty values list.')
        if self.kind == 'sequence' and not self.segments:
            raise ValueError('sequence requires segments with duration and value.')
        for segment in self.segments:
            integer(segment['duration'], 'segment.duration', 1)
        for value in self.possible_values():
            if np.asarray(value).shape != self.nominal.shape or not np.all(np.isfinite(value)):
                raise ValueError('Schedule values must match parameter shape and be finite.')
        self.rng = np.random.default_rng(self.seed)
        self.random_index, self.random_value = -1, None

    def possible_values(self):
        if self.kind == 'nominal':
            return [self.nominal]
        if self.kind == 'sinusoidal':
            delta = self.amplitude * (self.shifted - self.nominal)
            return [self.nominal - delta, self.nominal + delta]
        if self.kind == 'random_piecewise':
            return self.choices or []
        if self.kind == 'sequence':
            return [s['value'] for s in self.segments]
        return [self.nominal, self.shifted]

    def reset(self):
        if self.reset_on_episode:
            self.rng = np.random.default_rng(self.seed)
            self.random_index, self.random_value = -1, None

    def value(self, t):
        t = integer(t, 'schedule time')
        segment = 0
        if self.kind == 'nominal':
            value = self.nominal
        elif self.kind == 'fixed':
            value = self.shifted
        elif self.kind == 'abrupt':
            segment = int(t >= self.switch)
            value = self.shifted if segment else self.nominal
        elif self.kind == 'repeated':
            segment = t // self.interval
            value = self.shifted if segment % 2 else self.nominal
        elif self.kind == 'sinusoidal':
            segment = t // self.interval
            value = self.nominal + self.amplitude * (self.shifted - self.nominal) * np.sin(2*np.pi*t/self.interval)
        elif self.kind == 'linear_drift':
            weight = np.clip((t - self.switch) / self.duration, 0, 1)
            segment = 0 if t < self.switch else (1 if t < self.switch + self.duration else 2)
            value = self.nominal + weight * (self.shifted - self.nominal)
        elif self.kind == 'random_piecewise':
            segment = t // self.interval
            if segment < self.random_index:
                raise ValueError('Schedule clock moved backwards; restore/reset its state first.')
            while self.random_index < segment:
                self.random_value = deepcopy(self.choices[int(self.rng.integers(len(self.choices)))])
                self.random_index += 1
            value = self.random_value
        else:
            end = 0
            for segment, item in enumerate(self.segments):
                end += item['duration']
                if t < end:
                    break
            value = self.segments[segment]['value']  # Hold final segment indefinitely.
        value = np.asarray(value, dtype=float)
        return (float(value) if value.ndim == 0 else value.copy()), segment

    def state_dict(self):
        return deepcopy(dict(rng=self.rng.bit_generator.state, random_index=self.random_index,
                             random_value=self.random_value))

    def load_state_dict(self, state):
        self.rng.bit_generator.state = deepcopy(state['rng'])
        self.random_index = state['random_index']
        self.random_value = deepcopy(state['random_value'])
