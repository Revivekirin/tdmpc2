"""CPU integration coverage for held-out runner and Hydra composition."""
from copy import deepcopy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from hydra import compose, initialize_config_dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_stage3a import Model, settings, data
from common.stage3a import EpisodeAdapter, equal_state
from adapt_shift import evaluate
from common.benchmark_config import normalize


class Stage3ARunnerTest(unittest.TestCase):
    def test_hydra_methods(self):
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'tdmpc2'), version_base=None):
            for mode in ['frozen', 'dynamics_only']:
                cfg = compose(config_name='stage3a', overrides=[f'stage3a.mode={mode}',
                    'configs/dynamics_shift@dynamics_shift=action_delay', 'configs/schedule@schedule=fixed'])
                cfg = normalize(cfg)
                self.assertEqual(cfg.stage3a.mode, mode)
                self.assertEqual(cfg.schedule.type, 'fixed')

    def test_evaluation_isolation_and_budget(self):
        torch.manual_seed(7)
        model = Model()
        adapter = EpisodeAdapter(model, settings(), 20, 20)
        adapter.begin_episode()
        for transition in data():
            adapter.observe(*transition, 2)
        adapter.end_episode()
        class Agent:
            device = torch.device('cpu')
            def __init__(self):
                self.model = model
                self._prev_mean = torch.ones(3, 1)
            def act(self, obs, **kwargs):
                self._prev_mean.add_(1)
                self.model._Qs(torch.randn(2, 2))
                return torch.zeros(1)
        environments = []
        class Env:
            def __init__(self):
                self.closed = False
                self.seeds = []
            def reset(self, seed):
                self.seeds.append(seed)
                self.t = 0
                return torch.zeros(2)
            def step(self, action):
                self.t += 1
                return torch.ones(2), 1., self.t == 3, dict(success=1, shift_native_steps=[{}, {}])
            def close(self):
                self.closed = True
        def factory(cfg):
            env = Env()
            environments.append(env)
            return env
        agent = Agent()
        before = deepcopy(adapter.optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        cfg = SimpleNamespace(schedule={'type':'fixed'}, stage3a=settings(), horizon=3)
        result = evaluate(agent, adapter, cfg, factory, deepcopy(model._dynamics))
        self.assertTrue(equal_state(before, adapter.optimizer.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(agent._prev_mean, torch.ones(3, 1)))
        self.assertEqual(adapter.transitions, 16)
        self.assertEqual(adapter.native_steps, 32)
        for condition in ['shifted', 'nominal']:
            self.assertEqual(result[condition]['native_steps'], 12)
            self.assertEqual(result[condition]['policy_decisions'], 6)
            self.assertEqual(result[condition]['planner_q_input_moments']['count'], 12)
        self.assertTrue(all(e.closed and e.seeds == [20,21] for e in environments))
        self.assertEqual(len(model._Qs._forward_pre_hooks), 0)


if __name__ == '__main__':
    unittest.main()
