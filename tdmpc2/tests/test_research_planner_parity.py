"""Native stochastic planner trace against the pre-change numerical golden."""

import base64
import hashlib
import json
from pathlib import Path
import platform
import random
import sys

import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
sys.path.insert(0, str(PROJECT / "tests"))
from common.planner_trace import PlannerTrace
from test_planner_instrumentation import agent, load_planner


def fixture_path():
    if platform.system() == "Darwin" and torch.__version__.split("+")[0] == "2.14.0":
        return PROJECT / "tests/fixtures/research_numerical_baselines.json"
    runtime = f"{platform.system().lower()}_{platform.machine().lower()}_torch_{torch.__version__}"
    runtime = runtime.replace(".", "_").replace("+", "_")
    return PROJECT / "tests/fixtures" / f"research_numerical_baselines_{runtime}.json"


def digest_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tensor.dtype, tuple(tensor.shape))).encode() +
                          tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def test_candidate_components_elite_weights_refinement_and_action():
    fixture = fixture_path()
    if not fixture.is_file():
        pytest.skip(f"UNVERIFIED: platform numerical fixture unavailable: {fixture.name}")
    golden = json.loads(fixture.read_text())["planning"]
    threads = torch.get_num_threads()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261119)
            random.seed(20261119)
            np.random.seed(20261119)
            native = agent(load_planner())
            trace = PlannerTrace()
            native._mechanism_trace = trace
            action = native.act(torch.tensor([0.1, -0.2]), t0=True, eval_mode=True)
            assert len(trace.iterations) == len(golden["iterations"])
            for observed, expected in zip(trace.iterations, golden["iterations"]):
                candidates = torch.tensor(observed["actions"])
                reward = torch.tensor(observed["predicted_reward_sum"])
                terminal = torch.tensor(observed["terminal_q_contribution"])
                scores = torch.tensor(observed["actual_planner_score"])
                elite = torch.tensor(observed["elite_ids"], dtype=torch.long)
                assert digest_tensor(candidates[:, :native.cfg.num_pi_trajs]) == expected["policy_prior_hash"]
                assert digest_tensor(candidates[:, native.cfg.num_pi_trajs:]) == expected["gaussian_candidates_hash"]
                assert digest_tensor(reward) == expected["predicted_reward_hash"]
                assert digest_tensor(terminal) == expected["terminal_q_hash"]
                torch.testing.assert_close(scores, reward + terminal, rtol=0, atol=1e-6)
                torch.testing.assert_close(scores, torch.tensor(expected["total_scores"]), rtol=2e-6, atol=2e-7)
                assert elite.tolist() == expected["elite_indices"]
                selected_scores = scores[elite]
                weights = torch.exp(native.cfg.temperature * (selected_scores - selected_scores.max()))
                weights = weights / weights.sum()
                reference_scores = torch.tensor(expected["total_scores"])[elite]
                reference_weights = torch.softmax(native.cfg.temperature *
                                                  (reference_scores - reference_scores.max()), dim=0)
                torch.testing.assert_close(weights, reference_weights, rtol=2e-6, atol=2e-7)
                selected = candidates[:, elite]
                mean = (weights[None, :, None] * selected).sum(1) / (weights.sum() + 1e-9)
                std = ((weights[None, :, None] * (selected - mean[:, None]).square()).sum(1) /
                       (weights.sum() + 1e-9)).sqrt().clamp(native.cfg.min_std, native.cfg.max_std)
                torch.testing.assert_close(mean, torch.tensor(expected["gaussian_mean"]), rtol=2e-6, atol=2e-7)
                torch.testing.assert_close(std, torch.tensor(expected["gaussian_std"]), rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(native._prev_mean, torch.tensor(golden["warm_start_mean"]), rtol=0, atol=0)
            assert trace.selection["selected_candidate_id"] == golden["selected_candidate_id"]
            torch.testing.assert_close(action, torch.tensor(golden["selected_action"]), rtol=2e-6, atol=2e-7)
            assert base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode() == golden["rng_after"]["torch_cpu_base64"]
    finally:
        torch.set_num_threads(threads)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
