"""Future learning adapters consume commanded transitions, never hidden actions."""
from dataclasses import dataclass
import numpy as np

ADAPTATION_COMPONENTS = {
    'frozen': (), 'dynamics_only': ('dynamics',), 'value_only': ('Q',),
    'dynamics_value': ('dynamics', 'Q'),
    'dynamics_value_policy': ('dynamics', 'Q', 'policy_prior'),
    'full': ('encoder', 'dynamics', 'reward', 'Q', 'policy_prior'),
}


@dataclass
class Transition:
    observation: object
    commanded_action: object
    reward: float
    next_observation: object
    done: bool
    diagnostics: dict


class FrozenProtocol:
    def observe(self, transition):
        """No replay, gradient updates or model mutation in this implementation."""
        return None


def evaluation_seeds(seed, count, explicit=None):
    values = list(explicit) if explicit is not None else list(range(seed, seed + count))
    if count <= 0 or len(values) != count:
        raise ValueError('Supply exactly one evaluation seed per episode.')
    if any(isinstance(s, bool) or int(s) != s or not 0 <= s < 2**32 for s in values):
        raise ValueError('Episode seeds must be integers in [0, 2**32).')
    return [int(s) for s in values]


def make_protocol(mode):
    if mode not in ADAPTATION_COMPONENTS:
        raise ValueError(f'Unknown adaptation protocol: {mode}.')
    if mode != 'frozen':
        raise NotImplementedError(f'{mode} is declared only; learning is not implemented in this phase.')
    return FrozenProtocol()


def validate_probe_holdout(schedule, split, reserved):
    if split not in {'train', 'eval', 'probe'}:
        raise ValueError('benchmark_split must be train, eval or probe.')
    if split == 'probe':
        return
    candidates = [np.asarray(v) for v in schedule.possible_values()]
    for value in reserved:
        value = np.asarray(value)
        if value.shape != schedule.nominal.shape:
            raise ValueError('Probe parameter shape does not match the shift.')
        if schedule.kind in {'sinusoidal', 'linear_drift'}:
            low, high = np.minimum(candidates[0], candidates[1]), np.maximum(candidates[0], candidates[1])
            overlaps = np.all((low <= value) & (value <= high))
        else:
            overlaps = any(np.array_equal(value, v) for v in candidates)
        if overlaps:
            raise ValueError('A reserved probe parameter occurs in the train/eval schedule.')
