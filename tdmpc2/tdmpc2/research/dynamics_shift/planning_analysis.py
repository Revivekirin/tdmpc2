"""Read-only analysis of native TD-MPC2 planning and fixed candidate banks.

The native planner remains the owner of candidate sampling, Q-head selection,
elite updates, and final action sampling. Historical analysis modules remain
available with their original source hashes and output schemas.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FixedCandidateAnalysis:
    candidate_sha256: str
    encoded_latent: Any
    latent_trajectory: tuple[Any, ...]
    reward_logits: tuple[Any, ...]
    reward_sum: Any
    terminal_latent: Any
    terminal_action: Any
    terminal_logits: Any
    terminal_q: Any
    terminal_contribution: Any
    total_score: Any
    elite_indices: Any
    elite_weights: Any
    effective_elites: float
    q_head_indices: Any


def candidate_sha256(actions: Any) -> str:
    """Hash candidate tensor bytes and shape; no implicit bank generation."""
    import torch

    value = torch.as_tensor(actions).detach().cpu().contiguous()
    digest = sha256()
    digest.update(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite(value: Any, label: str) -> None:
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite {label}")


def score_fixed_candidates(model: Any, observation: Any, actions: Any, cfg: Any,
                           discount: float, seed: int, *,
                           expected_candidate_sha256: str | None = None) -> FixedCandidateAnalysis:
    """Call the unchanged native `_estimate_value` on an exogenous action bank.

    The Q and reward wrappers only copy tensors. `randperm` is observed without
    replacing its draw, so the selected online Q heads and RNG consumption match
    the production implementation. The global Torch RNG is restored on return.
    This is model prediction, never an environment return or native MPPI action.
    """
    import torch
    from unittest.mock import patch
    from tdmpc2 import TDMPC2

    if cfg.multitask or cfg.episodic:
        raise ValueError("Fixed candidate analysis requires single-task non-episodic configuration")
    for name in ("action_dim", "latent_dim", "num_q", "num_bins", "vmin", "vmax", "obs_shape"):
        if getattr(cfg, name) != getattr(model.cfg, name):
            raise ValueError(f"Model/planner {name} mismatch")
    if not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("Invalid planning discount")
    if actions.ndim != 3 or actions.shape[2] != cfg.action_dim or not actions.shape[0] or not actions.shape[1]:
        raise ValueError("Candidate dimensions")
    if cfg.num_elites < 1 or cfg.num_elites > actions.shape[1]:
        raise ValueError("Invalid elite count")
    if observation.device != actions.device:
        raise ValueError("Observation and candidates must use the same device")
    _finite(observation, "observation")
    _finite(actions, "candidate actions")
    if bool((actions.abs() > 1).any()):
        raise ValueError("Candidate action bounds")
    identity = candidate_sha256(actions)
    if expected_candidate_sha256 is not None and identity != expected_candidate_sha256:
        raise ValueError("Candidate bank identity mismatch")

    local = copy(cfg)
    local.horizon, local.num_samples = actions.shape[:2]
    captured: dict[str, Any] = {"latent": [], "reward_logits": [], "head_indices": []}

    class Observer:
        def __getattr__(self, name: str) -> Any:
            return getattr(model, name)

        def reward(self, z: Any, action: Any, task: Any) -> Any:
            logits = model.reward(z, action, task)
            captured["reward_logits"].append(logits.detach().clone())
            return logits

        def next(self, z: Any, action: Any, task: Any) -> Any:
            latent = model.next(z, action, task)
            captured["latent"].append(latent.detach().clone())
            return latent

        def Q(self, z: Any, action: Any, task: Any, **kwargs: Any) -> Any:
            captured["terminal_latent"] = z.detach().clone()
            captured["terminal_action"] = action.detach().clone()
            value = model.Q(z, action, task, **kwargs)
            captured["terminal_q"] = value.detach().clone()
            captured["terminal_logits"] = model.Q(z, action, task, return_type="all").detach().clone()
            return value

    trace = SimpleNamespace(value=lambda reward, terminal: captured.update(
        reward=reward.detach().clone(), terminal=terminal.detach().clone()))
    native = SimpleNamespace(cfg=local, discount=discount, model=Observer(), _mechanism_trace=trace)
    original_randperm = torch.randperm

    def observe_randperm(*args: Any, **kwargs: Any) -> Any:
        value = original_randperm(*args, **kwargs)
        captured["head_indices"].append(value[:2].detach().clone())
        return value

    devices = [observation.device.index or 0] if observation.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(seed)
        z = model.encode(observation.reshape(1, -1), None).repeat(local.num_samples, 1)
        captured["encoded"] = z.detach().clone()
        with patch.object(torch, "randperm", observe_randperm):
            total = TDMPC2._estimate_value(native, z, actions, None)
        _finite(total, "native candidate score")
        if len(captured["latent"]) != local.horizon or len(captured["head_indices"]) != 1:
            raise ValueError("Unexpected native planning call sequence")
        torch.testing.assert_close(total, captured["reward"] + captured["terminal"], rtol=0, atol=0)
        score = total.flatten()
        elite = torch.topk(score, local.num_elites).indices
        weight = torch.softmax(cfg.temperature * (score[elite] - score[elite].max()), 0)
        return FixedCandidateAnalysis(
            identity, captured["encoded"], (captured["encoded"], *captured["latent"]),
            tuple(captured["reward_logits"]), captured["reward"], captured["terminal_latent"],
            captured["terminal_action"], captured["terminal_logits"], captured["terminal_q"],
            captured["terminal"], score, elite, weight,
            float(1 / weight.square().sum()), captured["head_indices"][0],
        )


def latent_prediction_diagnostics(predicted: Any, observed: Any) -> dict[str, float]:
    """The existing mechanism MSE, squared L2, and relative squared L2."""
    import torch

    if predicted.shape != observed.shape:
        raise ValueError("Latent shape mismatch")
    _finite(predicted, "predicted latent")
    _finite(observed, "observed latent")
    residual = (predicted - observed).square()
    return dict(latent_prediction_mse=float(residual.mean()),
                latent_prediction_squared_l2=float(residual.sum()),
                latent_prediction_relative_error=float(
                    residual.sum() / observed.square().sum().clamp_min(1e-12)))


def terminal_value_decomposition(values: Mapping[str, Any], discount: float) -> dict[str, Any]:
    """Preserve control-Q's F→D→DQ_F→DQ additive attribution and units."""
    if set(values) != {"F", "D", "DQ_F", "DQ"}:
        raise ValueError("Expected F, D, DQ_F, and DQ terminal values")
    if not np.isfinite(discount):
        raise ValueError("Nonfinite discount")
    v = {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}
    if len({item.shape for item in v.values()}) != 1 or not all(np.isfinite(x).all() for x in v.values()):
        raise ValueError("Incompatible or nonfinite terminal values")
    predictor_d = discount * (v["D"] - v["F"])
    predictor_d_to_dq = discount * (v["DQ_F"] - v["D"])
    critic_dq = discount * (v["DQ"] - v["DQ_F"])
    total = discount * (v["DQ"] - v["F"])
    residual = predictor_d + predictor_d_to_dq + critic_dq - total
    if not np.allclose(predictor_d + predictor_d_to_dq + critic_dq,
                       total, rtol=1e-10, atol=1e-8):
        raise ValueError("Terminal additive identity failed")
    return dict(predictor_D=predictor_d, predictor_D_to_DQ=predictor_d_to_dq,
                critic_DQ=critic_dq, total_DQ_minus_F=total,
                reconstruction_error=residual)


def candidate_rank_comparison(a: Any, b: Any, top: int, *,
                              policy: str = "control_q") -> dict[str, Any]:
    """Explicitly preserve the two historical ranking/missingness policies."""
    if policy == "mechanism":
        from common.mechanism_metrics import ranking_comparison

        return ranking_comparison(a, b, k=top)
    if policy != "control_q":
        raise ValueError("Unknown ranking policy")
    from scipy.stats import kendalltau

    left, right = np.asarray(a), np.asarray(b)
    if left.ndim != 1 or right.shape != left.shape or top < 1 or top > len(left):
        raise ValueError("Incompatible candidate rankings")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("Nonfinite candidate scores")

    def rankdata(values: Any) -> Any:
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        end = np.cumsum(counts)
        return ((2 * end - counts + 1) / 2)[inverse]

    ra, rb = rankdata(left), rankdata(right)
    spearman = None if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])
    tau = float(kendalltau(left, right).statistic)
    return dict(spearman=spearman, kendall=tau if np.isfinite(tau) else None,
                top_k=top,
                top_k_overlap=len(set(np.argsort(-left)[:top]) & set(np.argsort(-right)[:top])) / top,
                argmax_disagreement=int(left.argmax() != right.argmax()))


def controlled_q_score_difference(b: FixedCandidateAnalysis, c: FixedCandidateAnalysis,
                                  b_hashes: Mapping[str, str], c_hashes: Mapping[str, str], *,
                                  frozen_online_q_sha256: str) -> Any:
    """Isolate B→C's online-Q contribution on identical dynamics and proposals."""
    import torch

    for component in ("encoder", "dynamics", "reward", "target_q", "actor", "normalization"):
        if b_hashes[component] != c_hashes[component]:
            raise ValueError(f"B/C {component} differs")
    if b_hashes["online_q"] != frozen_online_q_sha256:
        raise ValueError("B must use the verified frozen online critic")
    if b_hashes["online_q"] == c_hashes["online_q"]:
        raise ValueError("B/C online critics are identical")
    if b.candidate_sha256 != c.candidate_sha256:
        raise ValueError("Candidate bank mismatch")
    for name in ("encoded_latent", "terminal_latent", "terminal_action",
                 "reward_sum", "q_head_indices"):
        if not torch.equal(getattr(b, name), getattr(c, name)):
            raise ValueError(f"B/C {name} differs")
    difference = c.total_score - b.total_score
    torch.testing.assert_close(difference, (c.terminal_contribution - b.terminal_contribution).flatten(),
                               rtol=2e-5, atol=2e-4)
    return difference


def native_trace_diagnostics(trace: Any, action: Any, warm_start: Any, cfg: Any) -> tuple[dict[str, Any], ...]:
    """Verify each observer row against native MPPI's elite and Gaussian update.

    The input trace must come from an unchanged `TDMPC2._plan` call. This
    function never generates candidates, picks a final action, or draws RNG.
    """
    import torch

    if (not trace.rows or cfg.horizon < 1 or cfg.num_samples < 1 or
            cfg.num_pi_trajs < 0 or cfg.num_pi_trajs > cfg.num_samples or
            cfg.num_elites < 1 or cfg.num_elites > cfg.num_samples or
            not np.isfinite(cfg.temperature)):
        raise ValueError("Invalid native planner trace/configuration")
    diagnostics = []
    for row in trace.rows:
        candidates, score = row["candidates"], row["score"].reshape(-1)
        if candidates.shape != (cfg.horizon, cfg.num_samples, cfg.action_dim) or score.shape != (cfg.num_samples,):
            raise ValueError("Native candidate shape mismatch")
        _finite(candidates, "native candidates")
        _finite(score, "native scores")
        if not torch.equal(row["elite"], torch.topk(score, cfg.num_elites).indices):
            raise ValueError("Native elite mismatch")
        if not torch.equal(row["reward"] + row["terminal"], row["score"]):
            raise ValueError("Native score decomposition mismatch")
        elite_value = score[row["elite"]].unsqueeze(-1)
        weights = torch.exp(cfg.temperature * (elite_value - elite_value.max(0).values))
        weights = weights / weights.sum(0)
        elite_actions = candidates[:, row["elite"]]
        mean = (weights.unsqueeze(0) * elite_actions).sum(1) / (weights.sum(0) + 1e-9)
        std = ((weights.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(1)
               / (weights.sum(0) + 1e-9)).sqrt().clamp(cfg.min_std, cfg.max_std)
        if not torch.equal(weights, row["weight"]) or not torch.equal(mean, row["mean"]) or not torch.equal(std, row["std"]):
            raise ValueError("Native MPPI Gaussian update mismatch")
        diagnostics.append(dict(elite_indices=row["elite"].clone(), weights=weights,
                                candidate_ranks=torch.argsort(score, descending=True, stable=True),
                                mean=mean, std=std,
                                policy_prior=candidates[:, :cfg.num_pi_trajs].clone(),
                                gaussian_candidates=candidates[:, cfg.num_pi_trajs:].clone()))
    if not torch.equal(trace.action, action) or not torch.equal(trace.mean, warm_start):
        raise ValueError("Native selected action or warm start mismatch")
    return tuple(diagnostics)
