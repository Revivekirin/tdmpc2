"""CPU protocol smoke tests, deliberately not evidence of real-environment gains."""
import copy
import json
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tdmpc2'))
from common.stage3a import EpisodeAdapter, equal_state, validate


def settings(mode='dynamics_only'):
    return dict(mode=mode, update_granularity='episode', consistency_steps=1,
                action_input='commanded', replay_source='fresh_shifted',
                sampling='uniform_without_replacement', capacity=64, batch_size=8,
                warmup=16, updates_per_episode=10, lr=.03, episodes=3, eval_every=1,
                adaptation_seeds=[10,11,12], evaluation_seeds=[20,21])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = nn.Linear(2, 2)
        self._dynamics = nn.Sequential(nn.Linear(3, 2), nn.Softmax(dim=-1))
        self._reward = nn.Linear(2, 1)
        self._Qs = nn.Linear(2, 1)
        self._pi = nn.Linear(2, 1)

    def encode(self, x, task):
        return torch.softmax(self._encoder(x), dim=-1)

    def next(self, z, a, task):
        return self._dynamics(torch.cat([z, a], dim=-1))


def data(n=16, seed=47):
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, 2, generator=generator)
    actions = torch.randn(n, 1, generator=generator).clamp(-1, 1)
    next_obs = obs + actions * torch.tensor([.7, -.3])
    return list(zip(obs, actions, next_obs))


class Stage3ATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = Model()
        self.adapter = EpisodeAdapter(self.model, settings(), 20., 20.)

    def test_episode_updates_and_heldout(self):
        adapter = self.adapter
        initial = copy.deepcopy(self.model._dynamics.state_dict())
        adapter.begin_episode()
        for transition in data():
            adapter.observe(*transition, 2)
        self.assertTrue(equal_state(initial, self.model._dynamics.state_dict()))
        with self.assertRaises(RuntimeError):
            adapter.update()
        adapter.end_episode()
        self.assertFalse(equal_state(initial, self.model._dynamics.state_dict()))
        self.assertEqual(adapter.gradient_steps, 10)
        self.assertEqual(adapter.native_steps, 32)
        self.assertEqual(adapter.transitions, 16)
        optimizer = copy.deepcopy(adapter.optimizer.state_dict())
        with adapter.held_out():
            self.assertTrue(torch.isfinite(adapter.loss(data())))
            with self.assertRaises(RuntimeError):
                adapter.observe(*data(1)[0], 2)
            with self.assertRaises(RuntimeError):
                adapter.update()
        self.assertTrue(equal_state(optimizer, adapter.optimizer.state_dict()))
        adapter.assert_fixed()
        self.assertTrue(all(p.grad is None for p in self.model._encoder.parameters()))

    def test_warmup_capacity_and_persistence(self):
        for n in [8, 8, 64]:
            self.adapter.begin_episode()
            for transition in data(n):
                self.adapter.observe(*transition, 2)
            self.adapter.end_episode()
            if n == 8 and self.adapter.episodes == 1:
                self.assertEqual(self.adapter.gradient_steps, 0)
        self.assertEqual(len(self.adapter.replay), 64)
        self.assertEqual(self.adapter.transitions, 80)
        self.assertEqual(self.adapter.gradient_steps, 20)
        self.assertTrue(all(v['step'] == 20 for v in self.adapter.optimizer.state.values()))

    def test_frozen_same_initial_checkpoint(self):
        frozen = copy.deepcopy(self.model)
        initial = copy.deepcopy(frozen.state_dict())
        adapter = EpisodeAdapter(frozen, settings('frozen'), 20., 20.)
        adapter.begin_episode()
        for transition in data():
            adapter.observe(*transition, 2)
        adapter.end_episode()
        self.assertTrue(equal_state(initial, frozen.state_dict()))
        self.assertIsNone(adapter.optimizer)
        self.assertEqual(adapter.gradient_steps, 0)

    def test_exact_one_step_loss(self):
        batch = data()
        obs, action, next_obs = map(torch.stack, zip(*batch))
        target = self.model.encode(next_obs, None).detach()
        predicted = self.model.next(self.model.encode(obs, None).detach(), action, None)
        self.assertTrue(torch.equal(self.adapter.loss(batch), (predicted-target).square().mean()))

    def test_reject_invalid_protocols(self):
        for key, value in [('mode', 'joint_rl'), ('mode', 'dynamics_value'),
                           ('consistency_steps', 3), ('update_granularity', 'step'),
                           ('action_input', 'executed'), ('replay_source', 'pretrained'),
                           ('evaluation_seeds', [10]), ('capacity', 4)]:
            s = settings()
            s[key] = value
            with self.assertRaises((ValueError, NotImplementedError)):
                validate(s)

    def test_detect_evaluation_mutation(self):
        with self.assertRaisesRegex(AssertionError, 'Evaluation mutated model'):
            with self.adapter.held_out():
                next(self.model._dynamics.parameters()).add_(1)

    def test_short_synthetic_adaptation(self):
        held_out = data(32, seed=97)
        before = self.adapter.loss(held_out).item()
        rows = []
        for _ in range(3):
            self.adapter.begin_episode()
            for transition in data():
                self.adapter.observe(*transition, 2)
            self.adapter.end_episode()
            with self.adapter.held_out():
                error = self.adapter.loss(held_out).item()
            rows.append(dict(**self.adapter.metrics(), synthetic_heldout_mse=error))
        self.assertLess(error, before)
        print(json.dumps(dict(test='synthetic CPU adaptation, not TD-MPC2 environment evaluation',
                              initial_mse=before, checkpoints=rows)))


if __name__ == '__main__':
    unittest.main()
