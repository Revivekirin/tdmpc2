"""One real-transition Dynamics+Q update against independent optimizer equations."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
sys.path.insert(0, str(PROJECT / "experiments/nsgym/scripts"))
from common import math as categorical
from research.dynamics_shift.audit import equal_state


def test_one_update_parameters_optimizers_ema_and_held_out_state():
    pytest.importorskip("tensordict")
    import run_dynamics_q as runner
    from common.dynamics_q import DynamicsQAdapter

    config, _ = runner.configuration()
    identity = dict(parameter="bthigh_damping", severity=0.5,
                    adaptation_seed=42000001, method="dynamics_q_118", schedule="fixed")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(817)
            actual_agent = runner.cpu_agent()
            torch.manual_seed(817)
            expected_agent = runner.cpu_agent()
            assert equal_state(actual_agent.model.state_dict(), expected_agent.model.state_dict())
            actual = DynamicsQAdapter(actual_agent, config, identity)
            expected = DynamicsQAdapter(expected_agent, config, identity)
            fixed_before = deepcopy(actual.fixed_state())
            target_before = deepcopy(actual_agent.model._target_Qs_params.state_dict())
            generator = torch.Generator().manual_seed(991)
            inputs = []
            for step in range(75):
                observation = torch.randn(17, generator=generator)
                action = torch.randn(6, generator=generator).tanh()
                next_observation = torch.randn(17, generator=generator)
                inputs.append((observation, action, next_observation, float(step % 7),
                               False, False, 0, step, 3.0, 6.0))
            for transition in inputs[:74]:
                actual.observe(*transition)
            assert actual.dynamics_steps == actual.q_steps == actual.target_steps == 0
            expected.replay.extend(deepcopy(actual.replay))
            expected.transitions = 74
            expected.rng.setstate(actual.rng.getstate())
            rng_before = torch.get_rng_state().clone()

            actual.observe(*inputs[74])
            rng_after = torch.get_rng_state().clone()
            expected.replay.append(deepcopy(actual.replay[-1]))
            expected.transitions = 75
            torch.set_rng_state(rng_before)
            batch = expected.replay.sample(expected.rng, config["batch_size"])
            assert [record["global_step"] for record in batch] == actual.last_sample_steps
            obs, action, nxt = [torch.stack([row[key] for row in batch])
                                for key in ("observation", "action", "next_observation")]
            reward = torch.tensor([[row["reward"]] for row in batch])
            terminal = torch.zeros_like(reward)
            model = expected_agent.model
            model._Qs.train(True)
            with torch.no_grad():
                z, next_z = model.encode(obs, None), model.encode(nxt, None)
                target = expected_agent._td_target(next_z, reward, terminal, None)
            dynamics_loss = F.mse_loss(model.next(z, action, None), next_z)
            logits = model.Q(z.detach(), action, None, return_type="all")
            q_loss = torch.stack([categorical.soft_ce(head, target, expected_agent.cfg).mean()
                                  for head in logits]).mean()
            expected.dynamics_optimizer.zero_grad(set_to_none=True)
            expected.q_optimizer.zero_grad(set_to_none=True)
            (config["consistency_coef"] * dynamics_loss).backward()
            assert all(parameter.grad is None for parameter in expected.q_params)
            torch.nn.utils.clip_grad_norm_(expected.dynamics_params, config["dynamics_grad_clip"],
                                           error_if_nonfinite=True)
            expected.dynamics_optimizer.step()
            expected.dynamics_optimizer.zero_grad(set_to_none=True)
            (config["value_coef"] * q_loss).backward()
            assert all(parameter.grad is None for parameter in expected.dynamics_params)
            torch.nn.utils.clip_grad_norm_(expected.q_params, config["q_grad_clip"],
                                           error_if_nonfinite=True)
            expected.q_optimizer.step()
            expected.q_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.soft_update_target_Q()
            model.eval()
            assert torch.equal(torch.get_rng_state(), rng_after)
            assert equal_state(actual_agent.model.state_dict(), expected_agent.model.state_dict())
            assert equal_state(actual.dynamics_optimizer.state_dict(),
                               expected.dynamics_optimizer.state_dict())
            assert equal_state(actual.q_optimizer.state_dict(), expected.q_optimizer.state_dict())
            assert actual.last_metrics["dynamics_loss"] == pytest.approx(float(dynamics_loss), rel=1e-6)
            assert actual.last_metrics["q_loss"] == pytest.approx(float(q_loss), rel=1e-6)
            assert equal_state(actual.fixed_state(), fixed_before)
            for key, value in actual_agent.model._target_Qs_params.state_dict().items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, target_before[key].lerp(
                        actual_agent.model._detach_Qs_params.state_dict()[key], actual_agent.cfg.tau),
                        rtol=0, atol=0)
            assert (actual.dynamics_steps, actual.q_steps, actual.target_steps) == (1, 1, 1)
            held_model, held_adapter = deepcopy(actual_agent.model.state_dict()), actual.state_dict()
            with actual.held_out():
                with pytest.raises(RuntimeError, match="replay"):
                    actual.observe(*inputs[0])
                with pytest.raises(RuntimeError, match="update"):
                    actual.update()
            assert equal_state(held_model, actual_agent.model.state_dict())
            assert equal_state(held_adapter, actual.state_dict())
    finally:
        torch.set_num_threads(old_threads)
