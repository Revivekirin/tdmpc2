"""Gain, bias, coupling and bounded native-action delay."""
from collections import deque
from copy import deepcopy
import numpy as np
from .base import ShiftWrapper
from .schedule import DISCRETE, integer


class ActionShiftWrapper(ShiftWrapper):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.low, self.high = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
        if (self.low.ndim != 1 or self.low.shape != self.high.shape or self.low.size < 2
                or not np.all(np.isfinite([self.low, self.high])) or not np.all(self.low < self.high)):
            raise ValueError('Expected finite vector Box bounds with low < high.')
        target = self.parameter.get('target', 'arm')
        if target not in {'arm', 'all'}:
            raise ValueError('Action target must be arm or all.')
        self.indices = slice(None, -1) if target == 'arm' else slice(None)
        self.dim = self.low.size - (target == 'arm')
        self.kind = self.parameter['type']
        shape = np.asarray(self.parameter['nominal']).shape
        if self.kind == 'action_coupling':
            if shape != (self.dim, self.dim) or not np.array_equal(self.parameter['nominal'], np.eye(self.dim)):
                raise ValueError('Coupling nominal must be the identity matrix matching target dimensions.')
        elif self.kind in {'action_gain', 'action_bias'}:
            if shape not in {(), (self.dim,)}:
                raise ValueError('Gain/bias must be a scalar or target-sized vector.')
            neutral = 1 if self.kind == 'action_gain' else 0
            if not np.all(np.asarray(self.parameter['nominal']) == neutral):
                raise ValueError('Gain nominal must be 1; bias nominal must be 0.')
        elif self.kind != 'action_delay':
            raise ValueError(f'Unknown action parameter {self.kind}.')
        self.queue = deque()
        self.max_delay = 0
        if self.kind == 'action_delay':
            if self.schedule.kind not in DISCRETE:
                raise ValueError('Integer action delay does not support continuous schedules.')
            if shape or self.parameter['nominal'] != 0:
                raise ValueError('Delay nominal must be scalar zero.')
            self.max_delay = max(integer(v, 'delay') for v in self.schedule.possible_values())
            if self.max_delay > 10000:
                raise ValueError('Delay exceeds the 10000 native-step queue limit.')
        if self.kind == 'action_gain' and any(np.any(np.asarray(v) < 0) for v in self.schedule.possible_values()):
            raise ValueError('Gain schedule may not become negative.')
        initial = np.asarray(self.parameter.get('initial_action', 0.), dtype=self.low.dtype)
        self.initial_action = np.broadcast_to(initial, self.low.shape).copy()
        if not np.all(np.isfinite(self.initial_action)) or np.any(self.initial_action < self.low) or np.any(self.initial_action > self.high):
            raise ValueError('Delay initial_action must be finite and within native action bounds.')
        self.reset_queue_on_episode = self.parameter.get('reset_queue_on_episode', True)

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        if self.reset_queue_on_episode:
            self.queue.clear()
        return obs

    def transform(self, action, value):
        executed = action.astype(np.result_type(action.dtype, self.low.dtype), copy=True)
        if self.kind == 'action_gain':
            executed[self.indices] *= value
        elif self.kind == 'action_bias':
            executed[self.indices] += value
        elif self.kind == 'action_coupling':
            executed[self.indices] = np.asarray(value) @ action[self.indices]
        else:
            self.queue.append(action.copy())
            while len(self.queue) > self.max_delay + 1:
                self.queue.popleft()
            delay = integer(value, 'delay')
            old = self.queue[-delay-1] if len(self.queue) > delay else self.initial_action
            executed[self.indices] = old[self.indices]
        return np.clip(executed, self.low, self.high)

    def state_dict(self):
        return dict(super().state_dict(), queue=[a.copy() for a in self.queue])

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.queue = deque(deepcopy(state['queue']))
