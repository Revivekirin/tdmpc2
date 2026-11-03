"""Synthetic contract tests; checkpoint-backed evidence comes from validate_pairing.py."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tdmpc2'))
sys.path.insert(0, str(ROOT / 'scripts'))
from common.pairing_validation import compare_prefix
from validate_pairing import compare_decisions
from benchmark_imports import package
from test_dynamics_shift import Native


def row(step, value=0.):
    return dict(native_step=step, parameter=0., commanded_action=[value] * 4,
                executed_action=[value] * 4, delay_queue_before=[], delay_queue_after=[],
                observation=[value] * 4, simulator=[value] * 4, controller={'target': value},
                rng={'seed': 7}, unavailable={}, policy_rng_after_step='same')


class PairingValidationTests(unittest.TestCase):
    def test_first_observed_divergence(self):
        reference = [row(i) for i in range(10)]
        shifted = deepcopy(reference)
        shifted[2]['delay_queue_after'] = [[0.] * 4]
        shifted[6]['observation'][0] = .1
        result = compare_prefix(reference, shifted, 10, 1e-6)
        self.assertEqual(result['pre_shift_equivalence'], 'FAIL')
        self.assertEqual(result['first_divergence_step'], 2)
        self.assertEqual(result['first_behavioral_divergence_step'], 6)
        self.assertEqual(result['pairing_validity'], 'INVALID_PRE_SHIFT')

    def test_queue_capacity_is_not_behavioral_divergence(self):
        reference = [row(i) for i in range(10)]
        shifted = deepcopy(reference)
        shifted[2]['delay_queue_after'] = [[0.] * 4]
        result = compare_prefix(reference, shifted, 10, 1e-6)
        self.assertEqual(result['pre_shift_equivalence'], 'PASS_OBSERVED_PREFIX')
        self.assertEqual(result['first_divergence_fields'], ['delay_queue_after'])
        self.assertEqual(result['pairing_validity'], 'UNVERIFIED')

    def test_missing_or_different_exposed_state_is_unverified(self):
        a = [row(0)]
        b = deepcopy(a)
        b[0]['simulator'] = None
        self.assertEqual(compare_prefix(a, b, 10, 0)['pre_shift_equivalence'], 'UNVERIFIED')

    def test_open_loop_does_not_compare_unused_policy_rng(self):
        a = [row(0)]
        b = deepcopy(a)
        b[0]['policy_rng_after_step'] = 'replay-does-not-plan'
        self.assertEqual(compare_prefix(a, b, 10, 0, compare_policy_rng=False)['pre_shift_equivalence'],
                         'PASS_OBSERVED_PREFIX')

    def test_candidate_rng_and_analysis_action_checks(self):
        a = [dict(action=[0.1], rng_before='x', rng_after='y', first_iteration_candidates_sha256='a')]
        b = deepcopy(a)
        b[0]['first_iteration_candidates_sha256'] = 'b'
        self.assertEqual(compare_decisions(a, b, 0)['fields'], ['first_iteration_candidates_sha256'])
        self.assertEqual(compare_decisions(a, b, 0, ('action',))['status'], 'PASS_OBSERVED_PREFIX')

    def test_inert_delay_queue_with_same_open_loop_actions(self):
        def config(kind):
            return dict(enabled=True, application='step',
                        parameter=dict(type='action_delay', target='arm', nominal=0, shifted=2),
                        schedule=dict(type=kind, clock='episode_native_step', reset_on_episode=True, switch_step=10))
        nominal = package.ActionShiftWrapper(Native(), config('nominal'))
        abrupt = package.ActionShiftWrapper(Native(), config('abrupt'))
        nominal.reset(seed=7)
        abrupt.reset(seed=7)
        for step, action in enumerate(np.random.default_rng(7).uniform(-1, 1, (10, 4)).astype(np.float32)):
            a = nominal.step(action)
            b = abrupt.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[3]['shift_record']['executed_action'], b[3]['shift_record']['executed_action'])
            self.assertEqual(b[3]['shift_record']['applied_parameter'], 0.)
        self.assertEqual(len(nominal.queue), 1)
        self.assertEqual(len(abrupt.queue), 3)

    def test_native_observer_does_not_change_wrapper_output(self):
        cfg = dict(parameter=dict(type='action_delay', target='arm', nominal=0, shifted=2),
                   schedule=dict(type='abrupt', clock='episode_native_step', reset_on_episode=True, switch_step=10))
        baseline = package.ActionShiftWrapper(Native(), cfg)
        observed = package.ActionShiftWrapper(Native(), cfg)
        seen = []
        observed.pairing_observer = lambda obs, record, queue: seen.append((record['episode_step'], len(queue)))
        baseline.reset(seed=7)
        observed.reset(seed=7)
        for action in np.random.default_rng(7).uniform(-1, 1, (12, 4)).astype(np.float32):
            a = baseline.step(action)
            b = observed.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[3]['shift_record']['executed_action'], b[3]['shift_record']['executed_action'])
        self.assertEqual(len(seen), 12)


if __name__ == '__main__':
    unittest.main()
