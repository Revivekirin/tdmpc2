"""Order exposed pairing observations by the first possible divergence boundary.

This is a diagnostic classifier, not a full-state determinism certificate.
"""
import numpy as np
from common.pairing_validation import equal


def _array(value):
    if isinstance(value, dict) and '__ndarray__' in value:
        return np.asarray(value['__ndarray__'], dtype=value['dtype'])
    try:
        array = np.asarray(value)
        return array if np.issubdtype(array.dtype, np.number) else None
    except (TypeError, ValueError):
        return None


def _detail(boundary, left, right, atol, decision=None, native_step=None):
    a, b = _array(left), _array(right)
    result = dict(boundary=boundary, decision=decision, native_step=native_step,
                  exact_equal=equal(left, right, 0), within_tolerance=equal(left, right, atol),
                  atol=atol)
    if a is not None and b is not None and a.shape == b.shape:
        delta = np.abs(a.astype(float) - b.astype(float))
        finite = np.isfinite(delta)
        result['shape'] = list(a.shape)
        result['max_abs_difference'] = float(delta[finite].max()) if finite.any() else None
        mismatch = np.argwhere((delta > atol) | ~finite)
        if delta.ndim == 0 and bool((delta > atol) | ~finite):
            index = ()
        elif mismatch.size:
            index = tuple(int(i) for i in mismatch[0])
        else:
            index = None
        if index is not None:
            result['first_numeric_index'] = list(index)
            result['first_numeric_left'] = float(a[index])
            result['first_numeric_right'] = float(b[index])
        elif a.size <= 128:
            result['raw_left'], result['raw_right'] = a.tolist(), b.tolist()
    else:
        result['raw_left'], result['raw_right'] = left, right
    return result


def compare_runs(reference, candidate, atol, switch_step):
    """Return first exact and tolerance-level differences in execution order."""
    events = []
    def add(boundary, left, right, decision=None, native_step=None, expected_structural=False):
        if not equal(left, right, 0):
            event = _detail(boundary, left, right, atol, decision, native_step)
            event['expected_structural'] = expected_structural
            events.append(event)

    for field in ('observation', 'simulator', 'controller', 'rng'):
        add('initial_' + field, reference['initial'].get(field), candidate['initial'].get(field))
    add('initial_policy_rng', reference.get('initial_policy_rng'), candidate.get('initial_policy_rng'))
    a_by_decision = {}
    b_by_decision = {}
    for row in reference['native']:
        if row['native_step'] < switch_step:
            a_by_decision.setdefault(row.get('decision_step', row['native_step'] // 2), []).append(row)
    for row in candidate['native']:
        if row['native_step'] < switch_step:
            b_by_decision.setdefault(row.get('decision_step', row['native_step'] // 2), []).append(row)
    for decision, (left, right) in enumerate(zip(reference['decisions'], candidate['decisions'])):
        add('planner_warm_start_before', left.get('planner_warm_start_before'),
            right.get('planner_warm_start_before'), decision,
            expected_structural=(decision == 0))
        add('policy_rng_before', left.get('rng_components_before'), right.get('rng_components_before'), decision)
        lp, rp = left.get('planner'), right.get('planner')
        if lp is not None and rp is not None:
            add('encoder_output', lp.get('encoded'), rp.get('encoded'), decision)
            add('planner_iteration_count', len(lp['iterations']), len(rp['iterations']), decision)
            for iteration, (li, ri) in enumerate(zip(lp['iterations'], rp['iterations'])):
                prefix = f'iteration_{iteration}_'
                add(prefix + 'candidate_actions', li['actions'], ri['actions'], decision)
                add(prefix + 'predicted_reward', li['reward'], ri['reward'], decision)
                add(prefix + 'terminal_q', li['terminal_q'], ri['terminal_q'], decision)
                add(prefix + 'total_score', li['score'], ri['score'], decision)
                add(prefix + 'elite_ids', li['elite_ids'], ri['elite_ids'], decision)
            add('elite_selection', (lp.get('selection') or {}).get('selected_candidate_id'),
                (rp.get('selection') or {}).get('selected_candidate_id'), decision)
        add('selected_action', left['action'], right['action'], decision)
        add('planner_warm_start_after', left.get('planner_warm_start_after'),
            right.get('planner_warm_start_after'), decision)
        add('policy_rng_after', left.get('rng_components_after'), right.get('rng_components_after'), decision)
        for native_left, native_right in zip(a_by_decision.get(decision, []), b_by_decision.get(decision, [])):
            step = native_left['native_step']
            for field in ('parameter', 'commanded_action', 'executed_action',
                          'observation', 'simulator', 'controller', 'rng',
                          'delay_queue_before', 'delay_queue_after', 'policy_rng_after_step'):
                add('native_' + field, native_left.get(field), native_right.get(field),
                    decision, step, expected_structural=field.startswith('delay_queue_'))
        add('native_step_count', len(a_by_decision.get(decision, [])),
            len(b_by_decision.get(decision, [])), decision)
    add('decision_count', len(reference['decisions']), len(candidate['decisions']))
    add('native_record_count', len(reference['native']), len(candidate['native']))
    substantive = [event for event in events if not event['expected_structural']]
    beyond = [event for event in substantive if not event['within_tolerance']]
    return dict(first_exact_divergence=events[0] if events else None,
                first_substantive_exact_divergence=substantive[0] if substantive else None,
                first_tolerance_divergence=beyond[0] if beyond else None,
                comparison_scope='exposed_state_and_instrumented_planner_only',
                simulator_full_state='UNVERIFIED',
                compared_decisions=min(len(reference['decisions']), len(candidate['decisions'])),
                compared_native_steps=min(len(reference['native']), len(candidate['native']), switch_step))


def divergence_context(reference, candidate, result):
    """Keep the neighboring exposed transitions and compact planner hashes."""
    event = (result['first_tolerance_divergence'] or
             result['first_substantive_exact_divergence'] or
             result['first_exact_divergence'])
    if event is None:
        return None
    import hashlib
    decision = event['decision']
    native_step = event['native_step']
    if native_step is None and decision is not None:
        native_step = 2 * decision

    def snapshot(run):
        neighbors = [row for row in run['native']
                     if native_step is not None and abs(row['native_step'] - native_step) <= 1]
        decision_row = run['decisions'][decision] if decision is not None and decision < len(run['decisions']) else None
        if decision_row is None:
            compact_decision = None
        else:
            planner = decision_row.get('planner')
            compact_decision = {key: decision_row.get(key) for key in
                                ('decision', 'action', 'rng_components_before', 'rng_components_after',
                                 'first_iteration_candidates_sha256',
                                 'planner_warm_start_before', 'planner_warm_start_after')}
            if planner is not None:
                compact_decision['encoder_output'] = planner.get('encoded')
                compact_decision['iterations'] = [dict(
                    candidate_sha256=hashlib.sha256(np.ascontiguousarray(row['actions']).tobytes()).hexdigest(),
                    reward_sha256=hashlib.sha256(np.ascontiguousarray(row['reward']).tobytes()).hexdigest(),
                    terminal_q_sha256=hashlib.sha256(np.ascontiguousarray(row['terminal_q']).tobytes()).hexdigest(),
                    score_sha256=hashlib.sha256(np.ascontiguousarray(row['score']).tobytes()).hexdigest(),
                    elite_ids=row['elite_ids']) for row in planner['iterations']]
                compact_decision['selection'] = planner.get('selection')
        return dict(initial=run['initial'] if event['boundary'].startswith('initial_') else None,
                    initial_policy_rng=run.get('initial_policy_rng'),
                    native_neighbors=neighbors, decision=compact_decision)

    return dict(event=event, reference=snapshot(reference), candidate=snapshot(candidate))
