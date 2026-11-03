"""Read-only native-step pairing evidence for frozen benchmark runs."""
import hashlib
import json
import random
import numpy as np


FIELDS = ('commanded_action', 'executed_action', 'delay_queue_before',
          'delay_queue_after', 'observation', 'simulator', 'controller', 'rng')


def rng_digest():
    """Capture global policy RNGs without drawing from them."""
    return hashlib.sha256(json.dumps(rng_hashes(), sort_keys=True).encode()).hexdigest()


def rng_hashes():
    """Separate Python, NumPy, Torch CPU and each CUDA generator digest."""
    import torch
    from envs.wrappers.dynamics_shift.logging import encode_state
    state = dict(python=repr(random.getstate()), numpy=encode_state(np.random.get_state()),
                 torch_cpu=torch.get_rng_state().tolist(),
                 torch_cuda=[value.tolist() for value in torch.cuda.get_rng_state_all()])
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return dict(python=digest(state['python']), numpy=digest(state['numpy']),
                torch_cpu=digest(state['torch_cpu']),
                torch_cuda=[digest(value) for value in state['torch_cuda']])


def native_record(env, observation, shift_record, queue_before):
    from common.benchmark_metrics import initial_state
    from envs.wrappers.dynamics_shift.logging import encode_state
    state = initial_state(env, observation)
    return dict(native_step=shift_record['episode_native_step'],
                decision_step=shift_record['policy_decision_step'],
                parameter=encode_state(shift_record['applied_parameter']),
                commanded_action=encode_state(shift_record['commanded_action']),
                executed_action=encode_state(shift_record['executed_action']),
                delay_queue_before=encode_state(queue_before),
                delay_queue_after=encode_state(list(env.queue)) if hasattr(env, 'queue') else None,
                observation=state['observation'],
                **{key: state.get(key) for key in ('simulator', 'controller', 'rng')},
                unavailable=state['unavailable'], policy_rng_after_step=rng_digest())


def equal(a, b, atol):
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k], atol) for k in a)
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b) and all(equal(x, y, atol) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    if isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return a == b
    try:
        return np.shape(a) == np.shape(b) and bool(np.allclose(a, b, rtol=0, atol=atol, equal_nan=True))
    except (TypeError, ValueError):
        return a == b


def compare_prefix(reference, shifted, switch_step, atol, compare_policy_rng=True):
    """Compare exposed components only; never certify hidden simulator state."""
    n = min(len(reference), len(shifted), switch_step)
    first = None
    first_behavioral = None
    nonqueue_mismatch = False
    behavioral = False
    for index in range(n):
        a, b = reference[index], shifted[index]
        step = a['native_step']
        differences = [field for field in FIELDS if a.get(field) is None or b.get(field) is None
                       or not equal(a[field], b[field], atol)]
        if not equal(a['parameter'], b['parameter'], atol):
            differences.insert(0, 'parameter')
        if compare_policy_rng and a.get('policy_rng_after_step') != b.get('policy_rng_after_step'):
            differences.append('policy_rng_after_step')
        if differences:
            observed = any(field in differences for field in ('commanded_action', 'executed_action', 'observation', 'parameter'))
            if first is None:
                first = (step, differences)
            if observed:
                behavioral = True
                first_behavioral = step
                break
            nonqueue_mismatch |= any(field not in ('delay_queue_before', 'delay_queue_after') for field in differences)
    incomplete = len(reference) != len(shifted) and n < switch_step
    unavailable = any(row.get('unavailable') for row in reference[:n] + shifted[:n])
    return dict(pre_shift_equivalence='FAIL' if behavioral else ('UNVERIFIED' if incomplete or unavailable or nonqueue_mismatch else 'PASS_OBSERVED_PREFIX'),
                first_divergence_step=first[0] if first else None,
                first_divergence_fields=first[1] if first else [],
                first_behavioral_divergence_step=first_behavioral,
                pairing_validity='INVALID_PRE_SHIFT' if behavioral else 'UNVERIFIED',
                compared_native_steps=index + 1 if behavioral else n,
                simulator_full_state='UNVERIFIED')
