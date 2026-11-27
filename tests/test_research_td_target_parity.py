"""Exercise the original one-step target and categorical projection on CPU."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
from common import math as categorical


def original_td_target():
    tree = ast.parse((PROJECT / "tdmpc2/tdmpc2.py").read_text())
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TDMPC2")
    method = next(node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name == "_td_target")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"torch": torch}
    exec(compile(module, str(PROJECT / "tdmpc2/tdmpc2.py"), "exec"), scope)
    return scope["_td_target"]


class TargetWorld:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []

    def encode(self, observation, task):
        self.calls.append(("encode", observation.clone()))
        return observation * 0.25 + 0.1

    def pi(self, latent, task):
        self.calls.append(("pi", latent.clone()))
        return latent[:, :2].tanh() + torch.randn(latent.shape[0], 2) * 0.01, {}

    def Q(self, latent, action, task, return_type="min", target=False):
        self.calls.append(("Q", latent.clone(), action.clone(), return_type, target))
        assert target and return_type == "min"
        logits = torch.stack([torch.cat((latent[:, :1] + action[:, :1] + i * 0.1,
                                        latent[:, 1:2] - action[:, 1:2],
                                        -latent[:, :1] + i * 0.05), dim=-1).squeeze(1)
                              for i in range(self.cfg.num_q)])
        heads = torch.randperm(self.cfg.num_q)[:2]
        return categorical.two_hot_inv(logits[heads], self.cfg).min(0).values


@pytest.mark.parametrize("terminal", [False, True])
def test_original_target_uses_real_next_encoding_and_one_bootstrap(terminal):
    cfg = SimpleNamespace(num_q=4, num_bins=3, vmin=-1.0, vmax=1.0, bin_size=1.0,
                          multitask=False)
    model = TargetWorld(cfg)
    agent = SimpleNamespace(model=model, discount=0.93, cfg=cfg)
    real_next = torch.tensor([[0.4, -0.3], [-0.2, 0.6]], dtype=torch.float32)
    reward = torch.tensor([[0.2], [-0.4]], dtype=torch.float32)
    done = torch.tensor([[float(terminal)], [0.0]], dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1403)
        with torch.no_grad():
            next_z = model.encode(real_next, None)
            observed = original_td_target()(agent, next_z, reward, done, None)
        actual_rng = torch.get_rng_state().clone()
        torch.manual_seed(1403)
        manual_z = real_next * 0.25 + 0.1
        manual_action = manual_z[:, :2].tanh() + torch.randn(2, 2) * 0.01
        logits = torch.stack([torch.stack((manual_z[:, 0] + manual_action[:, 0] + i * 0.1,
                                           manual_z[:, 1] - manual_action[:, 1],
                                           -manual_z[:, 0] + i * 0.05), -1)
                              for i in range(cfg.num_q)])
        heads = torch.randperm(cfg.num_q)[:2]
        decoded = categorical.two_hot_inv(logits[heads], cfg).min(0).values
        expected = reward + 0.93 * (1 - done) * decoded
        assert torch.equal(torch.get_rng_state(), actual_rng)
    assert [item[0] for item in model.calls] == ["encode", "pi", "Q"]
    assert torch.equal(model.calls[1][1], manual_z)
    assert model.calls[2][-2:] == ("min", True)
    assert observed.shape == reward.shape and observed.dtype == reward.dtype
    assert torch.isfinite(observed).all()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    if terminal:
        assert observed[0].item() == reward[0].item()


def test_original_categorical_projection_and_soft_cross_entropy():
    cfg = SimpleNamespace(num_bins=5, vmin=-2.0, vmax=2.0, bin_size=1.0)
    target = torch.tensor([[0.0], [0.6], [-0.7]], dtype=torch.float32)
    logits = torch.tensor([[0.2, -0.3, 0.1, 0.4, -0.1],
                           [-0.4, 0.2, 0.3, 0.1, 0.0],
                           [0.7, -0.1, 0.1, -0.5, 0.2]], dtype=torch.float32)
    projected = categorical.two_hot(target, cfg)
    assert projected.shape == logits.shape and projected.dtype == logits.dtype
    torch.testing.assert_close(projected.sum(-1), torch.ones(3), rtol=0, atol=1e-7)
    expected = -(projected * F.log_softmax(logits, dim=-1)).sum(-1, keepdim=True)
    observed = categorical.soft_ce(logits, target, cfg)
    assert torch.isfinite(observed).all()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
