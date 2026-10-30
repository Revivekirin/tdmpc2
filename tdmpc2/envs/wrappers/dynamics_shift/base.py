"""Native-step wrapper and diagnostic boundary events, independent of learning."""
from copy import deepcopy
import numpy as np
from .schedule import Schedule


class ShiftWrapper:
    mid_episode_supported = True

    def __init__(self, env, config):
        self.env, self.config = env, deepcopy(config)
        self.action_space, self.observation_space = env.action_space, env.observation_space
        self.parameter = config['parameter']
        self.schedule = Schedule(config['schedule'], self.parameter['nominal'], self.parameter['shifted'])
        self.global_step = self.episode_step = 0
        self.decision_step = self.episode_decision_step = 0
        self.episode_id = -1
        self.change_count = 0
        self.previous_parameter = self.previous_segment = None
        self.events = []
        self.boundary_callbacks = []
        self._decision_open = False
        self._last = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def clock_time(self):
        return {'global_native_step': self.global_step, 'episode_native_step': self.episode_step,
                'global_decision_step': self.decision_step, 'episode_decision_step': self.episode_decision_step,
                'episode_index': max(0, self.episode_id)}[self.schedule.clock]

    def begin_decision(self):
        if self._decision_open:
            raise RuntimeError('A decision is already open.')
        self._decision_open = True

    def end_decision(self):
        if not self._decision_open:
            raise RuntimeError('No decision is open.')
        self._decision_open = False
        self.decision_step += 1
        self.episode_decision_step += 1

    def reset(self, *args, **kwargs):
        if self._decision_open:
            raise RuntimeError('Cannot reset during an action repeat.')
        obs = self.env.reset(*args, **kwargs)
        self.episode_id += 1
        self.episode_step = self.episode_decision_step = 0
        self.schedule.reset()
        return obs

    def _record_parameter(self, value, segment):
        changed = self.previous_parameter is not None and not np.array_equal(value, self.previous_parameter)
        boundary = self.previous_segment is None or segment != self.previous_segment
        # A discrete fixed/episode-reset parameter change also establishes a boundary.
        discrete = self.schedule.kind not in {'sinusoidal', 'linear_drift'}
        if changed:
            self.change_count += 1
        if boundary or (discrete and changed):
            event = dict(episode_id=self.episode_id, native_step=self.global_step,
                         global_step=self.global_step, episode_step=self.episode_step,
                         decision_step=self.decision_step, episode_decision_step=self.episode_decision_step,
                         shift_segment_id=segment, shift_change_count=self.change_count,
                         applied_parameter=deepcopy(value), previous_parameter=deepcopy(self.previous_parameter),
                         schedule_time=self.clock_time(), clock=self.schedule.clock)
            self.events.append(event)
            for callback in self.boundary_callbacks:
                callback(deepcopy(event))
        self.previous_parameter, self.previous_segment = deepcopy(value), segment

    def transform(self, action, value):
        raise NotImplementedError

    def current_parameter(self):
        return self.schedule.value(self.clock_time())

    def step(self, action):
        if self.episode_id < 0:
            raise RuntimeError('Call reset before step.')
        standalone = not self._decision_open
        commanded = np.asarray(action).copy()
        if commanded.shape != self.action_space.shape or not np.all(np.isfinite(commanded)):
            raise ValueError('Action must be finite and match native Box shape.')
        if standalone:
            self.begin_decision()
        value, segment = self.current_parameter()
        self._record_parameter(value, segment)
        executed = self.transform(commanded, value)
        obs, reward, done, info = self.env.step(executed)
        record = dict(episode_id=self.episode_id, native_step=self.global_step, global_step=self.global_step,
                      episode_step=self.episode_step, decision_step=self.decision_step,
                      episode_decision_step=self.episode_decision_step, shift_type=self.parameter['type'],
                      schedule_type=self.schedule.kind, clock=self.schedule.clock, schedule_time=self.clock_time(),
                      shift_segment_id=segment, shift_change_count=self.change_count,
                      nominal_parameter=deepcopy(self.parameter['nominal']), applied_parameter=deepcopy(value),
                      commanded_action=commanded, executed_action=executed.copy(),
                      reward=float(reward), done=bool(done), mid_episode_supported=self.mid_episode_supported,
                      global_native_step=self.global_step, episode_native_step=self.episode_step,
                      policy_decision_step=self.decision_step, regime_id=segment,
                      intended_parameter=deepcopy(value), success=bool(info.get('success', False)))
        self._last = record
        info = dict(info, shift_record=record)
        self.global_step += 1
        self.episode_step += 1
        if standalone:
            self.end_decision()
        return obs, reward, done, info

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def state_dict(self):
        names = ('global_step', 'episode_step', 'decision_step', 'episode_decision_step', 'episode_id',
                 'change_count', 'previous_parameter', 'previous_segment', 'events', '_decision_open', '_last')
        return deepcopy(dict(version=1, config=self.config, counters={n: getattr(self, n) for n in names},
                             schedule=self.schedule.state_dict()))

    def load_state_dict(self, state):
        if state['version'] != 1 or state['config'] != self.config:
            raise ValueError('Shift snapshot version/config does not match.')
        self.schedule.load_state_dict(state['schedule'])
        for name, value in state['counters'].items():
            setattr(self, name, deepcopy(value))

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()
