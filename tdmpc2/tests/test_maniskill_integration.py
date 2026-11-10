"""Real simulator checks. Run with TD_MPC2_TEST_MANISKILL=1 on a SAPIEN host."""
import os
import unittest
import numpy as np
from test_dynamics_shift import Shift


@unittest.skipUnless(os.environ.get('TD_MPC2_TEST_MANISKILL') == '1', 'requires ManiSkill2 0.4.1 and a SAPIEN-capable host')
class SimulatorTests(unittest.TestCase):
    def test_nominal_and_actual_shift(self):
        import gym
        import mani_skill2.envs
        raw = gym.make('PickCube-v0', obs_mode='state', control_mode='pd_ee_delta_pos')
        wrapped = Shift(raw)
        try:
            action = np.array([.4, -.3, .2, -.5], dtype=np.float32)
            raw.reset(seed=17)
            state = raw.unwrapped.get_state().copy()
            baseline = raw.step(action)
            raw.reset(seed=17)
            raw.unwrapped.set_state(state.copy())
            nominal = wrapped.step(action)
            np.testing.assert_allclose(baseline[0], nominal[0], atol=1e-6, rtol=0)
            self.assertAlmostEqual(baseline[1], nominal[1], places=6)
            self.assertEqual(baseline[2], nominal[2])
            self.assertEqual(baseline[3]['success'], nominal[3]['success'])
            raw.reset(seed=17)
            raw.unwrapped.set_state(state.copy())
            wrapped.mode, wrapped.gain = 'fixed', .5
            shifted = wrapped.step(action)
            self.assertGreater(np.max(np.abs(nominal[0] - shifted[0])), 1e-7)
            np.testing.assert_allclose(shifted[3]['shift_executed_action'], [.2, -.15, .1, -.5])
        finally:
            raw.close()


@unittest.skipUnless(os.environ.get('TD_MPC2_TEST_MANISKILL') == '1', 'requires ManiSkill2 0.4.1 and a SAPIEN-capable host')
class BenchmarkSimulatorTests(unittest.TestCase):
    def make_native(self):
        import gym
        import mani_skill2.envs
        return gym.make('PickCube-v0', obs_mode='state', control_mode='pd_ee_delta_pos')

    def test_all_action_shifts_from_same_state(self):
        from benchmark_imports import package
        from test_shift_benchmark import config
        cases = [('action_gain',1.,.5), ('action_bias',0.,.2), ('action_delay',0,2),
                 ('action_coupling',np.eye(3).tolist(),[[1,.5,0],[0,1,0],[0,0,1]])]
        for kind, nominal, shifted in cases:
            raw=self.make_native()
            try:
                env=package.ActionShiftWrapper(raw,config(kind,'nominal',nominal,shifted))
                env.reset(seed=17)
                state=raw.unwrapped.get_state().copy()
                actions=np.tile(np.array([.4,-.3,.2,-.5],dtype=np.float32),(6,1))
                baseline=[raw.step(a) for a in actions]
                env.reset(seed=17); raw.unwrapped.set_state(state.copy())
                wrapped=[env.step(a) for a in actions]
                for original, identity in zip(baseline,wrapped):
                    np.testing.assert_allclose(original[0],identity[0],rtol=0,atol=1e-6)
                    self.assertAlmostEqual(original[1],identity[1],places=6)
                    self.assertEqual(original[3]['success'],identity[3]['success'])
                changed=package.ActionShiftWrapper(raw,config(kind,'fixed',nominal,shifted))
                changed.reset(seed=17); raw.unwrapped.set_state(state.copy())
                result=[changed.step(a) for a in actions]
                difference=max(np.max(np.abs(a[0]-b[0])) for a,b in zip(baseline,result))
                self.assertGreater(difference,1e-7,kind)
            finally:
                raw.close()

    def test_physics_readback_restore_and_contact_response(self):
        from benchmark_imports import package
        from test_shift_benchmark import config
        for kind in ['object_mass','contact_friction']:
            raw=self.make_native()
            try:
                cfg=config(kind,'fixed',1.,.5); cfg['application']='reset'
                cfg['parameter'].update(actor='obj',surfaces=['@plane'])
                env=package.make_shift(raw,cfg,task='pick-cube')
                env.reset(seed=17)
                state=raw.unwrapped.get_state().copy()
                def rollout():
                    positions=[]
                    for _ in range(20):
                        # Identical lateral force excites mass/friction; this is a physics
                        # sensitivity test, not an agent-performance measurement.
                        raw.unwrapped.obj.add_force_torque(np.array([1.,0,0]),np.zeros(3))
                        env.step(np.zeros(4,dtype=np.float32))
                        positions.append(raw.unwrapped.obj.get_pose().p.copy())
                    return np.array(positions)
                shifted=rollout()
                env.backend.restore()
                if kind=='object_mass':
                    self.assertAlmostEqual(raw.unwrapped.obj.get_mass(),env.backend.baseline['mass'],places=6)
                else:
                    for shape, original, values in env.backend.materials:
                        self.assertAlmostEqual(shape.get_physical_material().get_static_friction(),values[0],places=6)
                # Restore initial controller, task state and native step counters.
                env.reset(seed=17)
                env.backend.restore()
                raw.unwrapped.set_state(state.copy())
                nominal=rollout()
                self.assertGreater(np.max(np.abs(nominal-shifted)),1e-7,kind)
                env.reset(seed=17,reconfigure=True)
                self.assertAlmostEqual(env.backend.factor,.5)
            finally:
                raw.close()
