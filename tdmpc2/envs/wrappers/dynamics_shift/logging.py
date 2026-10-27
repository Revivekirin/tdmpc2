"""Serializable shift state plus an explicit, capability-checked simulator snapshot.

Exact dynamics continuation also needs controller state, environment RNGs and outer
wrapper counters. Agent/planner and global torch RNG state belong to the caller.
"""
from copy import deepcopy
import numpy as np
from .base import ShiftWrapper


def find_shift(env):
    while env is not None:
        if isinstance(env, ShiftWrapper):
            return env
        env = getattr(env, 'env', None)
    raise ValueError('No benchmark shift wrapper found.')


def capture_environment(env):
    shift = find_shift(env)
    if shift._decision_open:
        raise ValueError('Full environment snapshots require a completed decision.')
    native = shift.unwrapped
    controller = getattr(getattr(native, 'agent', None), 'controller', None)
    if not callable(getattr(controller, 'get_state', None)) or not callable(getattr(controller, 'set_state', None)):
        raise NotImplementedError('Exact simulator resume requires controller get_state/set_state support.')
    counters, wrapper = [], env
    while wrapper is not None:
        fields = {key: deepcopy(wrapper.__dict__[key]) for key in ('_t', '_elapsed_steps') if key in wrapper.__dict__}
        counters.append(fields)
        wrapper = wrapper.__dict__.get('env')
    rngs = {key: deepcopy(getattr(native, key).get_state()) for key in ('_main_rng', '_episode_rng')
            if hasattr(native, key)}
    return dict(version=1, shift=shift.state_dict(), simulator=np.asarray(native.get_state()).copy(),
                controller=deepcopy(controller.get_state()), rngs=rngs, wrapper_counters=counters,
                episode_seed=getattr(native, '_episode_seed', None), main_seed=getattr(native, '_main_seed', None))


def restore_environment(env, state):
    if state['version'] != 1:
        raise ValueError('Unsupported environment snapshot version.')
    shift = find_shift(env)
    # Destination must be freshly reset for the same task/seed/assets before restore.
    shift.load_state_dict(state['shift'])
    native = shift.unwrapped
    native.set_state(state['simulator'])
    native.agent.controller.set_state(deepcopy(state['controller']))
    for key, value in state['rngs'].items():
        getattr(native, key).set_state(value)
    native._episode_seed, native._main_seed = state['episode_seed'], state['main_seed']
    wrapper = env
    for fields in state['wrapper_counters']:
        if wrapper is None:
            raise ValueError('Wrapper stack differs from saved environment.')
        for key, value in fields.items():
            setattr(wrapper, key, deepcopy(value))
        wrapper = wrapper.__dict__.get('env')
    if wrapper is not None:
        raise ValueError('Wrapper stack differs from saved environment.')


def encode_state(value):
    """JSON-safe lossless ndarray/tuple serialization; no pickle loading."""
    if isinstance(value, np.ndarray):
        return {'__ndarray__': value.tolist(), 'dtype': str(value.dtype), 'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {'__tuple__': [encode_state(v) for v in value]}
    if isinstance(value, dict):
        return {k: encode_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode_state(v) for v in value]
    return value


def decode_state(value):
    if isinstance(value, dict):
        if '__ndarray__' in value:
            return np.asarray(value['__ndarray__'], dtype=value['dtype']).reshape(value['shape'])
        if '__tuple__' in value:
            return tuple(decode_state(v) for v in value['__tuple__'])
        return {k: decode_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_state(v) for v in value]
    return value
