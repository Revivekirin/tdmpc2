"""Read-only forward parity on the verified nominal checkpoint."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
from common import math as categorical


def original_method(name):
    tree = ast.parse((PROJECT / "tdmpc2/tdmpc2.py").read_text())
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TDMPC2")
    method = next(node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"torch": torch}
    exec(compile(module, str(PROJECT / "tdmpc2/tdmpc2.py"), "exec"), scope)
    return scope[name]


@pytest.fixture(scope="module")
def nominal():
    pytest.importorskip("tensordict")
    from research.dynamics_shift.checkpoints import load_checkpoint

    relative = Path("nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000")
    path = next((root / relative for root in (PROJECT.parent / "results", PROJECT / "results")
                 if (root / relative / "model.pt").is_file()), None)
    if path is None:
        pytest.skip("UNVERIFIED: nominal 750k source unavailable")
    golden = json.loads((PROJECT / "tests/fixtures/research_golden_references.json").read_text())
    with torch.random.fork_rng(devices=[]):
        loaded = load_checkpoint(path, expected_task="nsgym-halfcheetah")
    assert loaded.descriptor.nominal_sha256 == golden["nominal_checkpoint_sha256"]
    loaded.model.eval()
    return loaded.model


def assert_float32_finite(value, shape):
    assert value.shape == shape
    assert value.dtype == torch.float32
    assert torch.isfinite(value).all()


def test_nominal_encoder_dynamics_reward_and_critic_forward(nominal):
    model = nominal
    obs = torch.linspace(-0.3, 0.4, 3 * 17).reshape(3, 17)
    action = torch.linspace(-0.5, 0.5, 3 * 6).reshape(3, 6)
    with torch.no_grad():
        z = model.encode(obs, None)
        direct_z = model._encoder["state"](obs)
        next_z = model.next(z, action, None)
        direct_next_z = model._dynamics(torch.cat((z, action), dim=-1))
        reward = model.reward(z, action, None)
        direct_reward = model._reward(torch.cat((z, action), dim=-1))
        logits = model.Q(z, action, None, return_type="all")
        direct_logits = model._Qs(torch.cat((z, action), dim=-1))
        scalar_heads = categorical.two_hot_inv(logits, model.cfg)
    assert_float32_finite(z, (3, model.cfg.latent_dim))
    assert_float32_finite(next_z, z.shape)
    assert_float32_finite(reward, (3, model.cfg.num_bins))
    assert_float32_finite(logits, (model.cfg.num_q, 3, model.cfg.num_bins))
    assert_float32_finite(scalar_heads, (model.cfg.num_q, 3, 1))
    for actual, direct in ((z, direct_z), (next_z, direct_next_z),
                           (reward, direct_reward), (logits, direct_logits)):
        torch.testing.assert_close(actual, direct, rtol=0, atol=0)

    for mode in ("min", "avg"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(612)
            heads = torch.randperm(model.cfg.num_q)[:2]
            values = scalar_heads[heads]
            expected = values.min(0).values if mode == "min" else values.sum(0) / 2
            torch.manual_seed(612)
            observed = model.Q(z, action, None, return_type=mode)
        assert_float32_finite(observed, (3, 1))
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_nominal_actor_eval_mean_and_controlled_stochastic_sample(nominal):
    model = nominal
    observation = torch.linspace(-0.2, 0.2, 17)
    with torch.no_grad():
        z = model.encode(observation[None], None)
        raw_mean, raw_std = model._pi(z).chunk(2, dim=-1)
        expected_mean = raw_mean.tanh()
        expected_log_std = categorical.log_std(raw_std, model.log_std_min, model.log_std_dif)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(113)
            expected_action = (raw_mean + torch.randn_like(raw_mean) * expected_log_std.exp()).tanh()
            torch.manual_seed(113)
            sampled, info = model.pi(z, None)
        assert_float32_finite(sampled, (1, model.cfg.action_dim))
        torch.testing.assert_close(sampled, expected_action, rtol=0, atol=0)
        torch.testing.assert_close(info["mean"], expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(info["log_std"], expected_log_std, rtol=0, atol=0)

        # The original agent's eval branch selects the policy mean after sampling.
        agent = SimpleNamespace(model=model, device=torch.device("cpu"), cfg=SimpleNamespace(mpc=False))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(113)
            eval_action = original_method("act")(agent, observation, eval_mode=True)
        torch.testing.assert_close(eval_action, expected_mean[0], rtol=0, atol=0)
