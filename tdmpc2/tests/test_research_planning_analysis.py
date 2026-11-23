"""Canonical read-only analysis against unchanged historical and native paths."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
sys.path.insert(0, str(PROJECT / "experiments/nsgym/scripts"))
sys.path.insert(0, str(PROJECT / "experiments/nsgym/tests"))

import q_planning_core as legacy
from test_q_planning import model
from experiments.nsgym.analysis.control_q.diagnostics.planning import native_plan
from experiments.nsgym.analysis.control_q.metrics import decomposition as legacy_decomposition
from common.mechanism_analysis import prediction_error as legacy_prediction_error
from common.mechanism_metrics import ranking_comparison as legacy_mechanism_ranks
from common import math
from research.dynamics_shift.checkpoints import model_hashes
from research.dynamics_shift.planning_analysis import (
    candidate_rank_comparison, candidate_sha256, controlled_q_score_difference,
    latent_prediction_diagnostics, native_trace_diagnostics,
    score_fixed_candidates, terminal_value_decomposition,
)


@pytest.fixture
def inputs():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    network = model()
    obs = torch.randn(17)
    candidates = torch.as_tensor(legacy.candidate_bank(3, 8, 6, 19, 1))
    return network, obs, candidates


def test_fixed_candidate_native_score_and_historical_parity(inputs):
    network, obs, candidates = inputs
    before = {k: v.clone() for k, v in network.state_dict().items() if torch.is_tensor(v)}
    rng = torch.get_rng_state().clone()
    old = legacy.score(network, obs, candidates, network.cfg, .995, 42)
    new = score_fixed_candidates(
        network, obs, candidates, network.cfg, .995, 42,
        expected_candidate_sha256=candidate_sha256(candidates),
    )
    for old_key, new_key in (
        ("score", "total_score"), ("reward", "reward_sum"),
        ("terminal", "terminal_contribution"), ("native_terminal_q", "terminal_q"),
        ("terminal_latent", "terminal_latent"), ("terminal_action", "terminal_action"),
        ("terminal_logits", "terminal_logits"), ("elite", "elite_indices"),
        ("weight", "elite_weights"),
    ):
        torch.testing.assert_close(old[old_key], getattr(new, new_key), rtol=0, atol=0)
    assert new.effective_elites == old["ess"]
    assert new.encoded_latent.shape == (8, network.cfg.latent_dim)
    assert len(new.latent_trajectory) == 4
    assert len(new.reward_logits) == 3
    decoded = math.two_hot_inv(new.terminal_logits, network.cfg)
    torch.testing.assert_close(decoded[new.q_head_indices].sum(0) / 2, new.terminal_q, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in network.state_dict().items() if k in before)


def test_fixed_candidate_negative_contracts(inputs):
    network, obs, candidates = inputs
    kwargs = dict(model=network, observation=obs, actions=candidates, cfg=network.cfg,
                  discount=.995, seed=42)
    with pytest.raises(ValueError, match="identity"):
        score_fixed_candidates(**kwargs, expected_candidate_sha256="0" * 64)
    with pytest.raises(ValueError, match="bounds"):
        score_fixed_candidates(**(kwargs | {"actions": candidates + 3}))
    with pytest.raises(ValueError, match="Nonfinite"):
        score_fixed_candidates(**(kwargs | {"observation": obs * float("nan")}))
    wrong = deepcopy(network.cfg)
    wrong.num_elites = 9
    with pytest.raises(ValueError, match="elite"):
        score_fixed_candidates(**(kwargs | {"cfg": wrong}))
    wrong = deepcopy(network.cfg)
    wrong.episodic = True
    with pytest.raises(ValueError, match="non-episodic"):
        score_fixed_candidates(**(kwargs | {"cfg": wrong}))
    wrong = deepcopy(network.cfg)
    wrong.num_bins = 7
    with pytest.raises(ValueError, match="Model/planner num_bins"):
        score_fixed_candidates(**(kwargs | {"cfg": wrong}))


def test_q_swap_isolates_online_critic_on_same_candidates(inputs):
    frozen, obs, candidates = inputs
    b_model = deepcopy(frozen)
    c_model = deepcopy(frozen)
    with torch.no_grad():
        for parameter in c_model._Qs.parameters():
            parameter.add_(torch.randn_like(parameter) * .2)
    b = score_fixed_candidates(b_model, obs, candidates, b_model.cfg, .995, 42)
    c = score_fixed_candidates(c_model, obs, candidates, c_model.cfg, .995, 42)
    b_hashes, c_hashes = model_hashes(b_model), model_hashes(c_model)
    frozen_q = b_hashes["online_q"]
    difference = controlled_q_score_difference(
        b, c, b_hashes, c_hashes, frozen_online_q_sha256=frozen_q)
    torch.testing.assert_close(difference, c.total_score - b.total_score, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Candidate bank"):
        controlled_q_score_difference(b, replace(c, candidate_sha256="other"), b_hashes, c_hashes,
                                      frozen_online_q_sha256=frozen_q)
    with pytest.raises(ValueError, match="online critics"):
        controlled_q_score_difference(b, b, b_hashes, b_hashes, frozen_online_q_sha256=frozen_q)
    altered = c_hashes | {"dynamics": "other"}
    with pytest.raises(ValueError, match="dynamics"):
        controlled_q_score_difference(b, c, b_hashes, altered, frozen_online_q_sha256=frozen_q)
    with pytest.raises(ValueError, match="terminal_action"):
        controlled_q_score_difference(b, replace(c, terminal_action=c.terminal_action + 1),
                                      b_hashes, c_hashes, frozen_online_q_sha256=frozen_q)
    with pytest.raises(ValueError, match="verified frozen online critic"):
        controlled_q_score_difference(b, c, b_hashes, c_hashes,
                                      frozen_online_q_sha256="wrong")


def test_terminal_decomposition_rank_and_latent_parity():
    values = {"F": np.array([1., 2., 4.]), "D": np.array([2., 2., 5.]),
              "DQ_F": np.array([3., 2., 6.]), "DQ": np.array([4., 3., 8.])}
    new, old = terminal_value_decomposition(values, .995), legacy_decomposition(values, .995)
    for key in old:
        np.testing.assert_array_equal(new[key], old[key])
    a, b = np.array([1., 4., 2., 3.]), np.array([4., 1., 2., 3.])
    assert candidate_rank_comparison(a, b, 2) == legacy.ranks(a, b, 2)
    assert candidate_rank_comparison(a, b, 2, policy="mechanism") == legacy_mechanism_ranks(a, b, 2)
    assert candidate_rank_comparison([1., float("nan")], [2., 3.], 1, policy="mechanism")["status"] == "UNVERIFIED_NONFINITE_OR_EMPTY"
    with pytest.raises(ValueError, match="Nonfinite"):
        candidate_rank_comparison([1., float("nan")], [2., 3.], 1)
    with pytest.raises(ValueError, match="terminal"):
        terminal_value_decomposition(values | {"other": values["F"]}, .995)
    z, zn = torch.tensor([[1., 2.]]), torch.tensor([[2., 2.]])
    assert latent_prediction_diagnostics(z, zn) == legacy_prediction_error(z, zn)


def test_native_planner_trace_matches_observer_and_rng(inputs):
    network, obs, _ = inputs
    cfg = deepcopy(network.cfg)
    cfg.num_pi_trajs, cfg.iterations = 2, 2
    cfg.min_std, cfg.max_std = .05, 1.
    warm = torch.linspace(-.5, .5, cfg.horizon * cfg.action_dim).reshape(cfg.horizon, cfg.action_dim)
    rng = torch.get_rng_state().clone()
    action, new_warm, trace, events = native_plan(
        network, cfg, .995, obs, warm, False, 18, instrument=True)
    rows = native_trace_diagnostics(trace, action, new_warm, cfg)
    assert len(rows) == cfg.iterations
    assert len(events) > 0
    assert rows[0]["policy_prior"].shape == (cfg.horizon, cfg.num_pi_trajs, cfg.action_dim)
    assert rows[0]["gaussian_candidates"].shape == (
        cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim)
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="selected action"):
        native_trace_diagnostics(trace, action + 1, new_warm, cfg)
