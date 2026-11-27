import ast
import importlib.util
from pathlib import Path
import types
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shift', ROOT / 'tdmpc2/envs/wrappers/dynamics_shift/legacy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Shift = module.DynamicsShiftWrapper


class Native:
    def __init__(self):
        self.action_space = types.SimpleNamespace(low=np.full(4, -1., dtype=np.float32), high=np.ones(4, dtype=np.float32), shape=(4,), dtype=np.float32)
        self.observation_space = self.action_space
        self.unwrapped = self
        self.state = np.zeros(4)

    def reset(self, **kwargs):
        self.kwargs = kwargs
        self.state = np.zeros(4)
        return self.state.copy()

    def step(self, action):
        self.state += action
        return self.state.copy(), float(self.state.sum()), False, {'success': False}

    def render(self, **kwargs):
        return kwargs


class ShiftTests(unittest.TestCase):
    def test_nominal_equivalence(self):
        raw, wrapped = Native(), Shift(Native())
        np.testing.assert_array_equal(raw.reset(), wrapped.reset())
        rng = np.random.default_rng(3)
        for action in rng.uniform(-1, 1, (30, 4)).astype(np.float32):
            a, b = raw.step(action), wrapped.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:3], b[1:3])
            self.assertEqual(a[3]['success'], b[3]['success'])

    def test_fixed_clip_hidden(self):
        env = Shift(Native(), mode='fixed', gain=2)
        action = np.array([.8, -.8, .2, .3], dtype=np.float32)
        obs, _, _, info = env.step(action)
        np.testing.assert_allclose(obs, [1, -1, .4, .3])
        np.testing.assert_array_equal(action, info['shift_commanded_action'])
        self.assertEqual(obs.shape, (4,))
        self.assertIs(env.unwrapped, env.env)
        self.assertEqual(env.render(mode='x'), {'mode': 'x'})

    def test_abrupt_reset_clocks(self):
        for clock, expected in [('global', .7), ('episode', 1.)]:
            env = Shift(Native(), mode='abrupt', switch_step=2, clock=clock)
            gains = [env.step(np.ones(4))[3]['shift_gain'] for _ in range(3)]
            self.assertEqual(gains, [1., 1., .7])
            env.reset(seed=13, options={'reconfigure': True})
            self.assertEqual(env.env.kwargs['seed'], 13)
            info = env.step(np.ones(4))[3]
            self.assertEqual((info['shift_global_step'], info['shift_episode_step']), (3, 0))
            self.assertEqual(info['shift_gain'], expected)

    def test_other_schedules(self):
        env = Shift(Native(), mode='repeated', period=2)
        self.assertEqual([env.step(np.zeros(4))[3]['shift_gain'] for _ in range(5)], [1, 1, .7, .7, 1])
        env = Shift(Native(), mode='sinusoidal', period=4)
        np.testing.assert_allclose([env.step(np.zeros(4))[3]['shift_gain'] for _ in range(5)], [1, 1.3, 1, .7, 1])

    def test_repeat_uses_actual_wrapper(self):
        tree = ast.parse((ROOT / 'tdmpc2/envs/maniskill.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManiSkillWrapper')
        fakegym = types.SimpleNamespace(Env=object, spaces=types.SimpleNamespace(Box=lambda **kw: kw))
        scope = {'gym': fakegym, 'np': np, 'ShiftWrapper': type('Unused', (), {})}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'maniskill.py', 'exec'), scope)
        native = Shift(Native(), mode='abrupt', switch_step=100, clock='episode')
        env = scope['ManiSkillWrapper'](native, None)
        env.reset(seed=9)
        for i in range(51):
            _, reward, _, info = env.step(np.ones(4, dtype=np.float32) * .1)
            records = info['shift_native_steps']
            self.assertEqual([r['shift_global_step'] for r in records], [2*i, 2*i+1])
            self.assertEqual([r['shift_gain'] for r in records], [1., 1.] if i < 50 else [.7, .7])
            self.assertAlmostEqual(reward, sum(r['reward'] for r in records))
        native.switch_step = 103
        self.assertEqual([r['shift_gain'] for r in env.step(np.zeros(4))[3]['shift_native_steps']], [1., .7])

    def test_invalid(self):
        for kwargs in [dict(mode='oops'), dict(clock='bad'), dict(period=0), dict(gain=float('nan')), dict(amplitude=2)]:
            with self.assertRaises(ValueError):
                Shift(Native(), **kwargs)
        with self.assertRaises(ValueError):
            Shift(Native()).step(np.zeros(3))


if __name__ == '__main__':
    unittest.main()
