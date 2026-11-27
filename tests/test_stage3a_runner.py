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
from adapt_shift import evaluate, mse_statistics
from common.benchmark_config import normalize


class Stage3ARunnerTest(unittest.TestCase):
    def test_mse_statistics(self):
        result = mse_statistics([0., 2., 4.])
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['mean'], 2.)
        self.assertAlmostEqual(result['variance'], 8 / 3)
        self.assertAlmostEqual(result['std'], np.sqrt(8 / 3))
        self.assertAlmostEqual(result['p95'], 3.8)
        self.assertEqual(result['ddof'], 0)
        self.assertEqual(mse_statistics([2.])['std'], 0.)
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                mse_statistics(values)

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
                self.length = 2 if seed == 20 else 3
                return torch.zeros(2)
            def step(self, action):
                self.t += 1
                return torch.ones(2), 1., self.t == self.length, dict(success=1, shift_native_steps=[{}, {}])
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
            self.assertEqual(result[condition]['native_steps'], 10)
            self.assertEqual(result[condition]['policy_decisions'], 5)
            self.assertEqual(result[condition]['planner_q_input_moments']['count'], 10)
            stats = result[condition]['held_out_latent_prediction_stats']
            per_episode = [e['latent_prediction_mse'] for e in result[condition]['episodes']]
            self.assertEqual([e['count'] for e in per_episode], [2, 3])
            self.assertEqual(stats['transition']['count'], 5)
            self.assertEqual(stats['episode_means']['count'], 2)
            means = [e['mean'] for e in per_episode]
            self.assertAlmostEqual(stats['episode_means']['mean'], np.mean(means))
            self.assertAlmostEqual(stats['episode_means']['variance'], np.var(means))
            self.assertAlmostEqual(stats['transition']['mean'], np.average(means, weights=[2, 3]))
            self.assertEqual(result[condition]['held_out_latent_prediction_error'], stats['transition']['mean'])
        self.assertTrue(all(e.closed and e.seeds == [20,21] for e in environments))
        self.assertEqual(len(model._Qs._forward_pre_hooks), 0)
        # Budget checkpoints may occur in an active episode; evaluation must
        # leave it active without triggering an update or changing replay.
        adapter.begin_episode()
        cfg.stage3a['native_budgets'] = [5]
        evaluate(agent, adapter, cfg, factory, deepcopy(model._dynamics))
        self.assertTrue(adapter.in_episode)
        self.assertEqual(adapter.transitions, 16)
        self.assertTrue(equal_state(before, adapter.optimizer.state_dict()))


if __name__ == '__main__':
    unittest.main()
