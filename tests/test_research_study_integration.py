"""Study identity, historical provenance, and existing read-only report integration."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))

from experiments.nsgym.analysis.__main__ import main
from experiments.nsgym.analysis.schema import sha256
from experiments.nsgym.analysis.test_unified import fixture
from research.dynamics_shift.interfaces import ExperimentIdentity, ExperimentProgress
from research.dynamics_shift.reporting import ReportBuilder, ReportingConfig


STUDY = PROJECT / "experiments/nsgym/configs/study.json"


def test_identity_keeps_training_and_evaluation_status_separate():
    identity = ExperimentIdentity("online", "one_step_dynamics_consistency",
                                  "replay_encoded_critic", {"parameter": "bthigh_damping", "severity": .5},
                                  "c" * 64, 1, 101, 3000,
                                  {"dynamics": 118, "critic": 118}, "d" * 64)
    assert ExperimentProgress(identity, "PASS", "NOT_RUN").evaluation_status == "NOT_RUN"
    assert ExperimentProgress(identity, "PASS", "PASS").evaluation_status == "PASS"
    with pytest.raises(ValueError, match="Successful evaluation"):
        ExperimentProgress(replace(identity, evaluation_checkpoint_identity=None), "PASS", "PASS")
    with pytest.raises(ValueError, match="Successful evaluation"):
        ExperimentProgress(identity, "NOT_RUN", "PASS")
    with pytest.raises(ValueError, match="overlapping"):
        replace(identity, evaluation_seed=1)


def _fixture_config(root: Path) -> dict:
    fixture(root)
    checkpoint = sha256(root / "arbitrary-run/training/snapshot-0003000.pt")
    config = json.loads(STUDY.read_text())
    config["nominal_checkpoint_sha256"] = "c" * 64
    config["historical_references"] = {
        "frozen": {
            "evaluation_manifest": "shared-baseline/manifest.json",
            "evaluation_completion": "shared-baseline/completion.json",
        },
        "dynamics_only": {
            "checkpoint": "arbitrary-run/training/snapshot-0003000.pt",
            "checkpoint_sha256": checkpoint,
            "training_manifest": "arbitrary-run/training/manifest.json",
            "training_completion": "arbitrary-run/training/completion.json",
            "evaluation_manifest": "arbitrary-run/shifted/manifest.json",
            "evaluation_completion": "arbitrary-run/shifted/completion.json",
            "evaluation_leaf_manifest": "arbitrary-run/shifted/condition/manifest.json",
            "evaluation_leaf_completion": "arbitrary-run/shifted/condition/completion.json",
        },
        "dynamics_q": {
            "training_manifest": "unrun/manifest.json",
            "training_completion": "unrun/completion.json",
            "evaluation_manifest": "unrun/evaluation/manifest.json",
            "evaluation_completion": "unrun/evaluation/completion.json",
        },
    }
    return config


def test_study_cli_regenerates_existing_evidence_and_marks_missing_variants(tmp_path, capsys):
    root = tmp_path / "historical"
    config = _fixture_config(root)
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(config))
    source_hash = sha256(root / "arbitrary-run/shifted/condition/shifted.csv")
    output = tmp_path / "study-report"
    main(["study", "--source-root", str(root), "--study-config", str(config_path),
          "--output", str(output), "--no-plots"])
    capsys.readouterr()
    status = json.loads((output / "REPORT_STATUS.json").read_text())
    assert status["completed_cells"] == 0
    assert status["planned_cells"] == 8
    assert status["unified_evidence"]["canonical_records"] > 0
    assert (output / "evidence/canonical_records.csv").is_file()
    rows = json.loads((output / "SOURCE_PROVENANCE.json").read_text())["historical_references"]
    assert [(row["method"], row["training_status"], row["evaluation_status"]) for row in rows] == [
        ("frozen", "NOT_RUN", "PASS"), ("dynamics_only", "PASS", "PASS"),
        ("dynamics_q", "UNVERIFIED", "UNVERIFIED")]
    assert rows[1]["evaluation_checkpoint_identity"] == config["historical_references"]["dynamics_only"]["checkpoint_sha256"]
    matrix = json.loads((output / "MATRIX_STATUS.json").read_text())
    assert all(row["training_status"] == row["evaluation_status"] == "NOT_RUN" for row in matrix)
    assert all(row["evaluation_checkpoint_identity"] is None for row in matrix)
    assert "Matched-replay versus online" in (output / "STUDY_REPORT.md").read_text()
    assert sha256(root / "arbitrary-run/shifted/condition/shifted.csv") == source_hash
    with pytest.raises(FileExistsError):
        main(["study", "--source-root", str(root), "--study-config", str(config_path),
              "--output", str(output), "--no-plots"])


def test_study_config_validation_precedes_output_creation(tmp_path):
    root = tmp_path / "historical"
    root.mkdir()
    invalid = json.loads(STUDY.read_text())
    invalid["dynamics_objectives"][0] = "stage_a"
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(invalid))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="objective matrix"):
        main(["study", "--source-root", str(root), "--study-config", str(config_path),
              "--output", str(output), "--no-plots"])
    assert not output.exists()


def test_study_excludes_source_conflict_without_relaxing_unified_builder(tmp_path, monkeypatch):
    from experiments.nsgym.analysis import sources

    root = tmp_path / "historical"
    config = _fixture_config(root)
    original_collect = sources.collect

    def conflict(catalog):
        result = original_collect(catalog)
        result.rows[0]["source_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(sources, "collect", conflict)
    with pytest.raises(ValueError, match="source hash mismatch"):
        ReportBuilder(root, ReportingConfig()).build(tmp_path / "strict")
    report = ReportBuilder(root, ReportingConfig(family="study", study_config=config)).build(
        tmp_path / "study", render=False)
    assert report["analysis_status"] == "PARTIAL"
    assert report["unified_evidence"]["excluded_unverified_records"] == 1
    excluded = json.loads((tmp_path / "study/evidence/EXCLUDED_UNVERIFIED_SOURCES.json").read_text())
    assert len(excluded) == 1 and excluded[0]["recorded_sha256"] == "0" * 64


def test_historical_training_and_evaluation_evidence_are_independent(tmp_path):
    root = tmp_path / "historical"
    config = _fixture_config(root)
    definition = config["historical_references"]["dynamics_only"]
    builder = ReportBuilder(root, ReportingConfig(family="study", study_config=config))
    completion = root / definition["training_completion"]
    payload = json.loads(completion.read_text())
    payload["status"] = "FAILED"
    completion.write_text(json.dumps(payload))
    result = builder._study_reference("dynamics_only", definition, "c" * 64)
    assert (result["training_status"], result["evaluation_status"]) == ("UNVERIFIED", "PASS")
    (root / definition["checkpoint"]).write_bytes(b"changed")
    result = builder._study_reference("dynamics_only", definition, "c" * 64)
    assert (result["training_status"], result["evaluation_status"]) == ("UNVERIFIED", "UNVERIFIED")


def test_study_reuses_existing_recovery_figure_renderer(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    root = tmp_path / "historical"
    config = _fixture_config(root)
    output = tmp_path / "rendered"
    result = ReportBuilder(root, ReportingConfig(family="study", study_config=config)).build(
        output, render=True)
    assert result["figures"] >= 1
    assert any((output / "evidence").glob("*adaptation*.png"))
