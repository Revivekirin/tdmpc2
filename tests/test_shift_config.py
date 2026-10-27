"""Real Hydra composition test, skipped when Hydra is unavailable."""
import importlib.util
import unittest
from benchmark_imports import ROOT, package
from test_dynamics_shift import Native
from test_shift_physics import PhysicsNative


@unittest.skipUnless(importlib.util.find_spec('hydra') is not None, 'requires hydra-core to validate CLI composition')
class ConfigTests(unittest.TestCase):
    def test_every_preset_composes_and_constructs(self):
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(ROOT/'tdmpc2'),version_base=None):
            for preset in ['nominal','action_gain','action_bias','action_coupling','action_delay','object_mass','contact_friction','action_gain_sequence']:
                cfg=compose(config_name='config',overrides=[f'configs/dynamics_shift@dynamics_shift={preset}',
                                                          'dynamics_shift.enabled=true'])
                shift=OmegaConf.to_container(cfg.dynamics_shift,resolve=True)
                native=PhysicsNative() if preset in {'object_mass','contact_friction'} else Native()
                if preset=='contact_friction':
                    # Fixture has named surfaces; real integration validates @plane.
                    shift['parameter']['surfaces']=['ground']
                env=package.make_shift(native,shift,task='pick-cube')
                env.reset(seed=12)
