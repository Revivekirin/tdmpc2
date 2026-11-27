"""Execute actual production planner methods against an independent pre-change oracle.
CPU toy world isolates planner semantics; checkpoint/simulator coverage is separate.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tdmpc2'))
from common import math
from common.planner_trace import PlannerTrace
from common.mechanism_analysis import preserve_rng, model_checksum


def load_planner():
    tree = ast.parse((ROOT/'tdmpc2/tdmpc2.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('act','_plan','_estimate_value')]
    scope = dict(torch=torch, math=math)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'production_planner', 'exec'), scope)
    return scope['TDMPC2']


spec = importlib.util.spec_from_file_location('original_planner', ROOT/'tests/fixtures/original_planner.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class ToyWorld(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight = torch.nn.Parameter(torch.ones(1))
    def encode(self, obs, task): return obs*self.weight
    def next(self, z, a, task): return z + .2*a
    def reward(self, z, a, task):
        v = (z+a).sum(-1, keepdim=True)
        return torch.cat([-v, v*.1, v], -1)
    def pi(self, z, task):
        return (z*.3 + .1*torch.randn_like(z)).tanh(), {'mean': (z*.3).tanh()}
    def termination(self, z, task): return (z.sum(-1, keepdim=True) > .3).float()
    def Q(self, z, a, task, return_type='avg'):
        out = torch.stack([self.reward(z, a, task)+torch.tensor([i*.1,0.,-i*.1]) for i in range(5)])
        if return_type=='all': return out
        ids = torch.randperm(5)[:2]
        return math.two_hot_inv(out[ids], self.cfg).mean(0)


def agent(cls, episodic=False):
    a = cls()
    a.cfg = types.SimpleNamespace(horizon=3, action_dim=2, num_samples=12, num_pi_trajs=3,
        max_std=1., min_std=.05, iterations=3, num_elites=4, temperature=.5,
        multitask=False, episodic=episodic, num_bins=3, vmin=-1., vmax=1., mpc=True)
    a.model = ToyWorld(a.cfg).eval()
    a.device = torch.device('cpu')
    a.discount = .97
    a._prev_mean = torch.zeros(3,2)
    a.plan = a._plan
    return a


class PlannerTests(unittest.TestCase):
    def test_original_equivalence_rng_warm_start_and_termination(self):
        for episodic in (False, True):
            for tracing in (False, True):
                for eval_mode in (False, True):
                    original, current = agent(oracle.OriginalPlanner, episodic), agent(load_planner(), episodic)
                    digest = model_checksum(current.model)
                    for step in range(4):
                        torch.manual_seed(100+step)
                        expected = original.act(torch.tensor([.1,-.2]), t0=step==0, eval_mode=eval_mode)
                        expected_rng = torch.random.get_rng_state().clone()
                        current._mechanism_trace = PlannerTrace() if tracing else None
                        torch.manual_seed(100+step)
                        actual = current.act(torch.tensor([.1,-.2]), t0=step==0, eval_mode=eval_mode)
                        self.assertTrue(torch.equal(expected, actual))
                        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
                        self.assertTrue(torch.equal(original._prev_mean, current._prev_mean))
                        if tracing:
                            for row in current._mechanism_trace.iterations:
                                np.testing.assert_array_equal(row['actual_planner_score'], row['predicted_reward_sum']+row['terminal_q_contribution'])
                            final = current._mechanism_trace.iterations[-1]
                            self.assertIn(current._mechanism_trace.selection['selected_candidate_id'], final['elite_ids'])
                    self.assertEqual(digest, model_checksum(current.model))

    def test_actor_eval_mean_vs_stochastic(self):
        a = agent(load_planner()); a.cfg.mpc = False
        obs = torch.tensor([.1,-.2])
        self.assertTrue(torch.equal(a.act(obs, eval_mode=True), (obs*.3).tanh()))
        self.assertFalse(torch.equal(a.act(obs), (obs*.3).tanh()))

    def test_rng_preserved_even_on_exception(self):
        import random
        torch.manual_seed(3); np.random.seed(4); random.seed(5)
        state = torch.random.get_rng_state().clone()
        expected_np = np.random.get_state(); expected_py = random.getstate()
        with self.assertRaises(RuntimeError):
            with preserve_rng():
                torch.rand(20); np.random.rand(20); random.random()
                raise RuntimeError('test')
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        np.testing.assert_array_equal(expected_np[1], np.random.get_state()[1])
        self.assertEqual(expected_py, random.getstate())
