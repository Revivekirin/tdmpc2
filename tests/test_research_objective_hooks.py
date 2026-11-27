"""Default objective hooks preserve the current one-step Dynamics+Q equations."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
sys.path.insert(0, str(PROJECT / "experiments/nsgym/scripts"))
from common import math
from research.dynamics_shift.objectives import (
    DEFAULT_OBJECTIVES, ObjectiveHooks, one_step_dynamics_mse,
    replay_encoded_critic_loss,
)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamics = torch.nn.Linear(4, 2)
        self.critic = torch.nn.Linear(4, 6)

    def next(self, latent, action, task):
        assert task is None
        return self.dynamics(torch.cat((latent, action), -1))

    def Q(self, latent, action, task, return_type):
        assert task is None and return_type == 'all'
        return self.critic(torch.cat((latent, action), -1)).reshape(2, len(latent), 3)


def test_default_hooks_match_prechange_equations_and_detach_q_latents():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(14)
        model = SmallModel()
        z = torch.tensor([[0.1, -0.2], [0.3, 0.4]], requires_grad=True)
        action = torch.tensor([[0.2, 0.5], [-0.4, 0.1]])
        next_z = torch.tensor([[0.3, 0.1], [0.0, -0.2]])
        target = torch.tensor([[0.2], [-0.1]])
        cfg = SimpleNamespace(num_bins=3, vmin=-1.0, vmax=1.0, bin_size=1.0)
        old_dynamics = F.mse_loss(model.next(z, action, None), next_z)
        old_logits = model.Q(z.detach(), action, None, return_type='all')
        old_q = torch.stack([math.soft_ce(head, target, cfg).mean() for head in old_logits]).mean()
        hooks = ObjectiveHooks()
        assert hooks == DEFAULT_OBJECTIVES
        dynamics = hooks.dynamics(model, z, action, next_z)
        q, logits = hooks.critic(model, z, action, target, cfg)
        torch.testing.assert_close(dynamics, old_dynamics, rtol=0, atol=0)
        torch.testing.assert_close(q, old_q, rtol=0, atol=0)
        torch.testing.assert_close(logits, old_logits, rtol=0, atol=0)
        q.backward()
        assert z.grad is None
        assert all(parameter.grad is None for parameter in model.dynamics.parameters())
        assert any(parameter.grad is not None for parameter in model.critic.parameters())
        assert hooks.dynamics is one_step_dynamics_mse
        assert hooks.critic is replay_encoded_critic_loss


def test_adapter_accepts_hooks_without_changing_default_outputs_or_rng():
    pytest.importorskip('tensordict')
    import run_dynamics_q as runner
    from common.dynamics_q import DynamicsQAdapter
    from research.dynamics_shift.dynamics_critic import DynamicsCriticAdaptation

    config, _ = runner.configuration()
    identity = dict(parameter='bthigh_damping', severity=.5,
                    adaptation_seed=42000001, method='dynamics_q_118', schedule='fixed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        baseline = DynamicsQAdapter(runner.cpu_agent(), config, identity)
        torch.manual_seed(123)
        calls = []

        def dynamics(model, z, action, next_z):
            calls.append('dynamics')
            return one_step_dynamics_mse(model, z, action, next_z)

        def critic(model, z, action, target, cfg):
            calls.append('critic')
            return replay_encoded_critic_loss(model, z, action, target, cfg)

        hooked = DynamicsCriticAdaptation(runner.cpu_agent(), config, identity,
                                          objectives=ObjectiveHooks(dynamics, critic))
        assert DynamicsQAdapter is DynamicsCriticAdaptation
        assert baseline.objectives is DEFAULT_OBJECTIVES
        batch = [dict(observation=torch.linspace(-.2, .2, 17) + i * .001,
                      action=torch.linspace(-.5, .5, 6),
                      next_observation=torch.linspace(-.1, .3, 17) + i * .001,
                      reward=.1 * i, terminated=False, discount=.995)
                 for i in range(4)]
        torch.manual_seed(818)
        expected = baseline.losses(batch)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(818)
        actual = hooked.losses(batch)
        assert calls == ['dynamics', 'critic']
        assert torch.equal(torch.get_rng_state(), expected_rng)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
