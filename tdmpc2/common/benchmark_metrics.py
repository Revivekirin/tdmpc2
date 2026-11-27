"""Measured exposure, distinct from regime boundaries and action effects."""
import numpy as np


def exposure(records, atol=1e-8):
    changed = [not np.allclose(r['applied_parameter'], r['nominal_parameter'], rtol=0, atol=atol) for r in records]
    values = np.asarray([r['applied_parameter'] for r in records])
    transitions = sum(a['regime_id'] != b['regime_id'] for a, b in zip(records, records[1:]))
    # Count sign changes across nominal, ignoring exact/tolerance-level nominal samples.
    signs = []
    for r in records:
        delta = np.asarray(r['applied_parameter']) - np.asarray(r['nominal_parameter'])
        if delta.ndim == 0 and abs(float(delta)) > atol:
            signs.append(int(np.sign(delta)))
    return dict(native_length=len(records), shifted_native_steps=sum(changed),
                exposure_fraction=sum(changed)/len(records) if records else 0.,
                first_shift_native_step=next((r['global_native_step'] for r, c in zip(records, changed) if c), None),
                shift_transition_count=transitions,
                applied_parameter_min=values.min(axis=0).tolist() if records else None,
                applied_parameter_max=values.max(axis=0).tolist() if records else None,
                nominal_crossing_count=sum(a != b for a,b in zip(signs,signs[1:])),
                nominal_native_steps=len(records)-sum(changed),
                intended_applied_match=all(np.allclose(r['intended_parameter'],r['applied_parameter'],rtol=0,atol=atol) for r in records))


def initial_state(env, observation):
    """Capture each available component independently; never infer completeness."""
    from envs.wrappers.dynamics_shift.logging import find_shift, encode_state
    native = find_shift(env).unwrapped
    obs = observation.detach().cpu().numpy() if hasattr(observation, 'detach') else observation
    result = dict(observation=encode_state(np.asarray(obs).copy()), unavailable={})
    getters = dict(simulator=lambda: native.get_state(),
                   controller=lambda: native.agent.controller.get_state(),
                   rng=lambda: {k: getattr(native,k).get_state() for k in ('_main_rng','_episode_rng')})
    for key, getter in getters.items():
        try:
            result[key] = encode_state(getter())
        except (AttributeError, NotImplementedError, TypeError, RuntimeError) as exc:
            result['unavailable'][key] = str(exc)
    return result


def compare_initial(a, b, atol):
    from envs.wrappers.dynamics_shift.logging import decode_state
    def equal(x,y):
        if isinstance(x,dict) and isinstance(y,dict):
            return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)) and isinstance(y,(tuple,list)):
            return len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
        if x is None or y is None: return x is y
        if isinstance(x,str) or isinstance(y,str): return x == y
        return np.shape(x)==np.shape(y) and bool(np.allclose(x,y,rtol=0,atol=atol))
    result={k: ('PASS' if equal(decode_state(a[k]),decode_state(b[k])) else 'FAIL')
            if k in a and k in b else 'UNVERIFIED' for k in ('observation','simulator','controller','rng')}
    result['full_state'] = 'UNVERIFIED'
    result['scope'] = 'Compared exposed state only; simulator internals/contact caches and all RNG sources are not certified.'
    return result
