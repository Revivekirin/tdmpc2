"""Exact native-budget boundaries and immutable fixed-data comparisons on CPU."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_stage3a import Model, settings, data
from common.stage3a import EpisodeAdapter, NativeBudgetMonitor, NativeBudgetReached, equal_state
from adapt_shift import fixed_prediction_metrics, collect_fixed_data


class MatchedTests(unittest.TestCase):
    def test_exact_mid_repeat_budget_and_no_partial_update(self):
        adapter = EpisodeAdapter(Model(), settings(), 20, 20)
        checkpoints = []
        def checkpoint(budget):
            with adapter.held_out(allow_in_episode=True):
                with self.assertRaises(RuntimeError):
                    adapter.update()
                checkpoints.append((budget, adapter.transitions, adapter.episodes))
        monitor = NativeBudgetMonitor([1, 2, 5], checkpoint)
        adapter.begin_episode()
        actual = 0
        with self.assertRaises(NativeBudgetReached):
            for decision in range(20):
                # Same observer timing as ShiftWrapper: after native env.step,
                # before repeat returns and before replay insertion.
                for repeat in range(2):
                    actual += 1
                    monitor(None, {'policy_decision_step': decision}, None)
                adapter.observe(*data(1)[0], 2)
        self.assertEqual(actual, 5)
        self.assertEqual(checkpoints, [(1, 0, 0), (2, 0, 0), (5, 2, 0)])
        self.assertEqual(monitor.decisions_started, 3)
        self.assertEqual(adapter.native_steps, 4)
        self.assertEqual(adapter.gradient_steps, 0)
        self.assertTrue(adapter.in_episode)

    def test_fixed_data_is_same_for_both_predictors(self):
        torch.manual_seed(3)
        model = Model()
        agent = SimpleNamespace(model=model, device=torch.device('cpu'))
        baseline = deepcopy(model._dynamics)
        transitions = data(4, seed=97)
        dataset = dict(zip(('observation', 'action', 'next_observation'),
                           [torch.stack(x) for x in zip(*transitions)]))
        dataset.update(episode_ids=torch.tensor([0, 1, 1, 1]), episodes=[{}, {}])
        before = deepcopy(dataset)
        result = fixed_prediction_metrics(agent, baseline, dataset)
        self.assertEqual(result['paired_transition_delta']['mean'], 0)
        with torch.no_grad():
            model._dynamics[0].bias[0].add_(1)
        result = fixed_prediction_metrics(agent, baseline, dataset)
        self.assertNotEqual(result['paired_transition_delta']['mean'], 0)
        self.assertTrue(equal_state(before, dataset))
        obs, action, target_obs = (dataset[k] for k in ('observation', 'action', 'next_observation'))
        with torch.no_grad():
            expected = (model.next(model.encode(obs, None), action, None)-model.encode(target_obs, None)).square().mean(-1)
        self.assertAlmostEqual(result['current']['transition']['mean'], expected.mean().item(), places=7)
        self.assertAlmostEqual(result['current']['episode_means']['mean'],
                               ((expected[0]+expected[1:].mean())/2).item(), places=7)

    def test_fixed_collection_separate_and_no_overwrite(self):
        model = Model()
        adapter = EpisodeAdapter(model, settings(), 20, 20)
        class Agent:
            def act(self, obs, **kwargs):
                return torch.zeros(1)
        class Env:
            def reset(self, seed):
                self.t = 0
                return torch.ones(2) * seed
            def step(self, action):
                self.t += 1
                return torch.ones(2), 0., self.t == 2, {'shift_native_steps': [{}, {}], 'success': 1}
            def close(self):
                pass
        cfg = SimpleNamespace(stage3a={'fixed_data_seeds': [100, 101]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixed.pt'
            collect_fixed_data(Agent(), adapter, cfg, lambda _: Env(), path, {'checkpoint_sha256': 'fixture'})
            dataset = torch.load(path, weights_only=False)
            self.assertEqual(dataset['collection_native_steps'], 8)
            self.assertEqual(dataset['collection_policy_decisions'], 4)
            self.assertEqual(adapter.transitions, 0)
            self.assertEqual(adapter.gradient_steps, 0)
            with self.assertRaises(FileExistsError):
                collect_fixed_data(Agent(), adapter, cfg, lambda _: Env(), path, {})


if __name__ == '__main__':
    unittest.main()
