"""Historical golden inputs must be identified by content, not a similar filename."""

import hashlib
import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
OUTER = PROJECT.parent
GOLDEN = json.loads((PROJECT / "tests/fixtures/research_golden_references.json").read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source(record):
    relative = Path(record["path"])
    assert relative.parts[:2] == ("results", "nsgym")
    candidates = (OUTER / relative, PROJECT / relative)
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        pytest.skip(f"UNVERIFIED: historical source unavailable: {relative}")
    assert digest(path) == record["sha256"], f"Golden source changed: {relative}"
    return path


@pytest.mark.parametrize("name", sorted(GOLDEN["references"]))
def test_golden_reference_content_and_parent(name):
    reference = GOLDEN["references"][name]
    files = {role: source(item) for role, item in reference["files"].items()}
    if "completion" in files:
        assert json.loads(files["completion"].read_text())["status"] == "PASS"
    if name == "nominal":
        metadata = json.loads(files["metadata"].read_text())
        assert metadata["model_sha256"] == GOLDEN["nominal_checkpoint_sha256"]
        assert digest(files["checkpoint"]) == metadata["model_sha256"]
    elif name == "frozen":
        manifest = json.loads(files["manifest"].read_text())
        assert manifest["expected_checkpoint_sha256"] == GOLDEN["nominal_checkpoint_sha256"]
        assert manifest["optimizer_updates"] == 0
    elif name in {"dynamics_only", "dynamics_q"}:
        manifest = json.loads(files["manifest"].read_text())
        assert manifest["checkpoint_sha256"] == GOLDEN["nominal_checkpoint_sha256"]
        assert manifest["identity"]["adaptation_seed"] == 42000001
        assert manifest["source_hashes"] and manifest["source_revision"]
        if name == "dynamics_q":
            assert manifest["configuration"]["q_target"] == "original_td_target_real_next_encoding_one_step"
            assert manifest["configuration"]["q_gradient_input"] == "detached_actual_encoding"
            completion = json.loads(files["completion"].read_text())
            assert completion["snapshot_sha256"] == digest(files["checkpoint"])
    elif name == "planner":
        completion = json.loads(files["completion"].read_text())
        assert completion["optimizer_updates"] == 0
        assert completion["actual_candidate_ranking"] in ("NOT_MEASURED", False, None)
        assert json.loads(files["input_manifest"].read_text())


def test_reference_manifest_matches_existing_checkpoint_fixture():
    fixture = json.loads((PROJECT / GOLDEN["checkpoint_fixture"]).read_text())
    assert fixture["sources"]["nominal_model"]["source_sha256"] == GOLDEN["nominal_checkpoint_sha256"]
    assert fixture["sources"]["dynamics_q_snapshot"]["source_sha256"] == GOLDEN["references"]["dynamics_q"]["files"]["checkpoint"]["sha256"]
    assert fixture["sources"]["decision_snapshot"]["source_sha256"] == GOLDEN["references"]["dynamics_only"]["files"]["checkpoint"]["sha256"]
