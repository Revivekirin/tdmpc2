"""Read-only reporting contracts and parity with existing statistical policies."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))

from experiments.nsgym.analysis.control_q.metrics import ci as historical_control_q_ci
from experiments.nsgym.analysis.control_q.reporting.aggregate import hierarchical_interval
from experiments.nsgym.analysis.metrics import summarize_records, paired_improvements
from experiments.nsgym.analysis.schema import sha256
from experiments.nsgym.analysis.test_unified import fixture
from research.dynamics_shift.reporting import (
    ReportBuilder, ReportingConfig, control_q_records, validate_record_sources,
)
from research.dynamics_shift.reporting.metrics import unified_statistics


def test_explicit_source_root_rejects_tampering_escape_and_missing(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "episodes.csv"
    source.write_text("seed,return\n1,0\n")
    digest = sha256(source)
    record = dict(source_path="episodes.csv", source_sha256=digest)
    assert validate_record_sources(root, [record]) == {"episodes.csv": digest}
    source.write_text("seed,return\n1,1\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_record_sources(root, [record])
    with pytest.raises(ValueError, match="outside"):
        validate_record_sources(root, [dict(record, source_path="../elsewhere")])
    with pytest.raises(FileNotFoundError):
        validate_record_sources(root, [dict(record, source_path="missing.csv")])


def test_unified_builder_preserves_legacy_records_statistics_and_source_bytes(tmp_path):
    root = tmp_path / "historical"
    fixture(root)
    from experiments.nsgym.analysis.discovery import Catalog
    from experiments.nsgym.analysis.sources import collect
    from experiments.nsgym.analysis.__main__ import sources

    config = json.loads((PROJECT / "experiments/nsgym/analysis/config.json").read_text())
    sources.SOURCES.clear()
    sources.SOURCES.update(config["tasks"])
    catalog = Catalog(root, config["tasks"])
    original = collect(catalog).rows
    before = {r["source_path"]: sha256(root / r["source_path"]) for r in original}
    output = tmp_path / "new-report"
    result = ReportBuilder(root, ReportingConfig()).build(output)
    assert result["canonical_records"] == len(original)
    assert result["paired_conditions"] == len(paired_improvements(original)[1])
    assert len(pd.read_csv(output / "canonical_records.csv")) == len(original)
    expected = unified_statistics(original)
    assert expected["statistical_summary"] == summarize_records(original)
    for name, records in expected.items():
        assert (output / f"{name}.csv").read_text() == pd.DataFrame(records).to_csv(index=False)
    assert {p: sha256(root / p) for p in before} == before
    with pytest.raises(FileExistsError):
        ReportBuilder(root, ReportingConfig()).build(output)


def test_control_q_unmeasured_status_is_distinct_from_zero():
    tables = {name: pd.DataFrame([dict(task="halfcheetah", family="damping",
                                      severity=.5, metric="return", mean=0.)])
              for name in ("control", "evidence", "ranking", "native", "effects", "returns")}
    rows = control_q_records(tables)
    assert rows[-1].status == "NOT_MEASURED"
    assert rows[-1].values == {}
    assert rows[0].values["mean"] == 0.
    assert rows[0].status == "MEASURED"
    assert control_q_records({**tables, "control": pd.DataFrame([
        dict(task="halfcheetah", family="damping", severity=.5,
             metric="fixed_encoder_latent_mse", mean=0.)])})[0].status == "PREDICTED"
    measured = control_q_records(tables, swap_episodes=pd.DataFrame([
        dict(task="halfcheetah", family="damping", severity=.5,
             adaptation_seed=1, evaluation_seed=2, method="DQ", episode_return=0.)]))
    assert measured[-1].status == "MEASURED"
    assert measured[-1].values["episode_return"] == 0.
    assert not any(row.status == "NOT_MEASURED" for row in measured)
    with pytest.raises(ValueError, match="version 2"):
        ReportingConfig(family="control_q", swap_source=Path("existing"))


def test_statistical_policies_remain_separate_and_deterministic():
    matrix = np.array([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    old = historical_control_q_ci(matrix, ("condition", "return"))
    assert old == historical_control_q_ci(matrix, ("condition", "return"))
    rows = pd.DataFrame([
        dict(adaptation_seed=i, evaluation_seed=j, episode_return=float(value))
        for i, values in enumerate(matrix) for j, value in enumerate(values)])
    hierarchical = hierarchical_interval(rows, "episode_return")
    assert hierarchical == hierarchical_interval(rows, "episode_return")
    assert hierarchical["adaptation_streams"] == 3
    assert hierarchical["episodes"] == 9
    assert hierarchical["mean"] == old[0]
    # These are distinct bootstrap policies; neither interval replaces the other.
    assert (hierarchical["ci_low"], hierarchical["ci_high"]) != old[1:]


def test_reporting_config_rejects_unsupported_family_and_version():
    with pytest.raises(ValueError):
        ReportingConfig(version=3)
    with pytest.raises(ValueError):
        ReportingConfig(family="unknown")
    with pytest.raises(ValueError):
        ReportingConfig(family="unified", version=2, swap_source=Path("swap"))
