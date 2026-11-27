"""Hydra compatibility checks require the evaluation config dependencies."""
import unittest
try:
    from test_benchmark_v2 import canonical
except ImportError:
    canonical = None


@unittest.skipIf(canonical is None, 'Hydra/OmegaConf are unavailable')
class MechanismConfigTests(unittest.TestCase):
    def test_default_disabled(self):
        self.assertFalse(canonical().analysis.enabled)
    def test_composition(self):
        for evaluation in ('standard','paired'):
            cfg=canonical('analysis=frozen_mechanism','compile=false',f'evaluation={evaluation}',
                          'dynamics_shift=action_delay','schedule=fixed','dynamics_shift.parameter.shifted=2')
            self.assertEqual(list(cfg.analysis.rollout_horizons),[1,3,5])
            self.assertEqual(cfg.dynamics_shift.parameter.shifted,2)
    def test_reject_unsupported(self):
        with self.assertRaises(ValueError):
            canonical('analysis=frozen_mechanism','compile=true')
        with self.assertRaises(ValueError):
            canonical('analysis=frozen_mechanism','compile=false','analysis.q_head_mode=avg')
