"""Read-only historical checkpoint and controlled online-Q composition contracts."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
from tdmpc2 import TDMPC2  # Preserve the upstream module resolution in combined suites.
from research.dynamics_shift.checkpoints import (
    _model, file_sha256, load_checkpoint, model_hashes,
)
from research.dynamics_shift.composer import CheckpointComposer

ROOT = next((candidate / "nsgym" for candidate in (PROJECT.parent / "results", PROJECT / "results")
             if (candidate / "nsgym/nominal-server4-20261108-nohup-01/"
                 "checkpoint-000750000/model.pt").is_file()), PROJECT / "results/nsgym")
NOMINAL = ROOT / "nominal-server4-20261108-nohup-01/checkpoint-000750000"
SNAPSHOTS = {
    "dynamics_only": ROOT / "dynamics-only-750k-v1/runs/fixed-decision/"
    "bthigh_damping-k0.5-decision-seed42000001/snapshot-0003000.pt",
    "dynamics_q": ROOT / "dynamics-q-750k-v1/runs/fixed-decision/"
    "bthigh_damping-k0.5-dynamics-q-seed42000001/snapshot-0003000.pt",
    "joint": ROOT / "full-joint-750k-v1/runs-fixed/"
    "bthigh_damping-k0.5-full-joint-seed42000001/snapshot-0003000.pt",
}
REACHER_NOMINAL = ROOT / "reacher-full-01/checkpoint-000670000"
REACHER_SNAPSHOTS = {
    "dynamics_only": ROOT / "reacher/preview-v1/adapt-home-01/"
    "D0-k2-decision-seed53000001/snapshot-0010000.pt",
    "dynamics_q": ROOT / "reacher-dynamics-q-v1/runs/"
    "D0-k2-dynamics-q-seed53000001/snapshot-0010000.pt",
    "joint": ROOT / "reacher/preview-v1/full-joint-home-01/runs-fixed/"
    "joint0_damping-k2-full-joint-seed53000001/snapshot-0010000.pt",
}


@pytest.fixture(scope="module")
def sources():
    if not all(p.is_file() for p in (NOMINAL / "model.pt", NOMINAL / "optimizers.pt",
                                     NOMINAL / "metadata.json", *SNAPSHOTS.values())):
        if torch.cuda.is_available():
            pytest.fail("BLOCKED: historical NS-Gym source files unavailable on CUDA host")
        pytest.skip("Historical NS-Gym source files unavailable")
    original = {p: file_sha256(p) for p in (NOMINAL / "model.pt", NOMINAL / "optimizers.pt",
                                                *SNAPSHOTS.values())}
    loaded = {"nominal": load_checkpoint(NOMINAL)}
    loaded.update({kind: load_checkpoint(NOMINAL, path) for kind, path in SNAPSHOTS.items()})
    yield loaded
    assert {p: file_sha256(p) for p in original} == original


def test_historical_formats_restore_exact_model_and_metadata(sources):
    nominal = sources["nominal"]
    assert nominal.descriptor.task == "nsgym-halfcheetah"
    assert nominal.descriptor.action_dim == 6
    assert nominal.descriptor.observation_shape == {"state": [17]}
    assert nominal.descriptor.adaptation_identity is None
    assert nominal.descriptor.transitions is None
    assert nominal.descriptor.exact_resume_supported is False
    for kind in SNAPSHOTS:
        item = sources[kind]
        assert item.descriptor.format == kind
        assert item.descriptor.nominal_sha256 == nominal.descriptor.nominal_sha256
        assert item.descriptor.transitions == 3000
        assert item.descriptor.adaptation_identity["adaptation_seed"] == 42000001
        assert item.descriptor.component_hashes == model_hashes(item.model)
        assert item.descriptor.optimizer_keys


def test_reacher_historical_selection_formats():
    if not all(p.is_file() for p in (REACHER_NOMINAL / "model.pt",
                                     REACHER_NOMINAL / "optimizers.pt",
                                     REACHER_NOMINAL / "metadata.json", *REACHER_SNAPSHOTS.values())):
        if torch.cuda.is_available():
            pytest.fail("BLOCKED: historical Reacher source files unavailable on CUDA host")
        pytest.skip("Historical Reacher source files unavailable")
    nominal = load_checkpoint(REACHER_NOMINAL, expected_task="nsgym-reacher", expected_action_dim=2)
    composer = CheckpointComposer(nominal)
    for kind, path in REACHER_SNAPSHOTS.items():
        item = load_checkpoint(REACHER_NOMINAL, path)
        assert item.descriptor.transitions == 10000
        assert composer.compose(kind, item).component_hashes == model_hashes(item.model)
        if kind == "dynamics_q":
            swapped = composer.compose("controlled_q_swap", item)
            assert swapped.component_hashes["online_q"] == nominal.descriptor.component_hashes["online_q"]


def test_controller_composition_and_online_q_isolation(sources):
    composer = CheckpointComposer(sources["nominal"])
    controllers = {"frozen": composer.compose("frozen")}
    for kind in SNAPSHOTS:
        controllers[kind] = composer.compose(kind, sources[kind])
    controllers["controlled_q_swap"] = composer.compose("controlled_q_swap", sources["dynamics_q"])
    before = controllers["dynamics_q"].component_hashes
    after = controllers["controlled_q_swap"].component_hashes
    reference = controllers["frozen"].component_hashes
    for key in ("encoder", "dynamics", "reward", "actor", "target_q", "normalization"):
        assert after[key] == before[key]
    assert after["online_q"] == reference["online_q"]
    assert before["online_q"] != after["online_q"]
    assert sources["dynamics_q"].descriptor.component_hashes == model_hashes(sources["dynamics_q"].model)
    assert sources["nominal"].descriptor.component_hashes == model_hashes(sources["nominal"].model)

    obs = torch.linspace(-0.2, 0.3, 17).reshape(1, 17)
    action = torch.linspace(-0.5, 0.5, 6).reshape(1, 6)
    with torch.no_grad():
        b, c = controllers["dynamics_q"].model, controllers["controlled_q_swap"].model
        z_b, z_c = b.encode(obs, None), c.encode(obs, None)
        torch.testing.assert_close(z_b, z_c, rtol=0, atol=0)
        for _ in range(3):
            torch.testing.assert_close(b.reward(z_b, action, None), c.reward(z_c, action, None), rtol=0, atol=0)
            z_b, z_c = b.next(z_b, action, None), c.next(z_c, action, None)
            torch.testing.assert_close(z_b, z_c, rtol=0, atol=0)
        assert not torch.equal(b.Q(z_b, action, None, return_type="all"),
                               c.Q(z_c, action, None, return_type="all"))


def test_incompatible_task_and_latent_contract_rejected(sources):
    composer = CheckpointComposer(sources["nominal"])
    adapted = sources["dynamics_q"]
    for field, value in (("task", "nsgym-reacher"), ("action_dim", 2),
                         ("latent_dim", 128), ("config_sha256", "wrong")):
        wrong = replace(adapted, descriptor=replace(adapted.descriptor, **{field: value}))
        with pytest.raises(ValueError):
            composer.compose("controlled_q_swap", wrong)
    with pytest.raises(ValueError, match="task"):
        load_checkpoint(NOMINAL, expected_task="nsgym-reacher")
    with pytest.raises(ValueError, match="action"):
        load_checkpoint(NOMINAL, expected_action_dim=2)
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(NOMINAL, SNAPSHOTS["dynamics_q"], expected_sha256="wrong")


def test_missing_unexpected_keys_and_shape_rejected(sources):
    config = json.loads((NOMINAL / "metadata.json").read_text())["config"]
    state = torch.load(NOMINAL / "model.pt", map_location="cpu", weights_only=False)["model"]
    key = next(k for k, v in state.items() if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] > 1)
    for mutation in (lambda x: x.pop(key),
                     lambda x: x.update({"unexpected.weight": torch.zeros(1)}),
                     lambda x: x.update({key: x[key][:-1]})):
        changed = state.copy()
        mutation(changed)
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            _model(config, changed)


def test_temporary_roundtrip_and_legacy_control_q_loader(sources, tmp_path):
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(PROJECT / "experiments/nsgym/scripts"))
    from analyze_q_planning import make_model

    config = json.loads((NOMINAL / "metadata.json").read_text())["config"]
    source = sources["dynamics_q"].model
    destination = tmp_path / "evaluation_model.pt"
    torch.save({"model": source.state_dict()}, destination)
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=[]):
        legacy = make_model(config, payload["model"], "cpu")
    assert model_hashes(legacy) == sources["dynamics_q"].descriptor.component_hashes


def test_controlled_q_swap_native_fixed_candidate_scores(sources):
    sys.path.insert(0, str(PROJECT / "experiments/nsgym/scripts"))
    from mechanism_forward import Config
    from q_planning_core import score

    config = json.loads((NOMINAL / "metadata.json").read_text())["config"]
    composer = CheckpointComposer(sources["nominal"])
    dq = composer.compose("dynamics_q", sources["dynamics_q"])
    swapped = composer.compose("controlled_q_swap", sources["dynamics_q"])
    observation = torch.linspace(-0.2, 0.3, 17)
    candidates = torch.linspace(-0.4, 0.4, 3 * 8 * 6).reshape(3, 8, 6)
    cfg = Config(**config)
    before = score(dq.model, observation, candidates, cfg, 0.995, 7)
    after = score(swapped.model, observation, candidates, cfg, 0.995, 7)
    for key in ("terminal_latent", "reward"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert not torch.equal(before["native_terminal_q"], after["native_terminal_q"])
    assert not torch.equal(before["score"], after["score"])
