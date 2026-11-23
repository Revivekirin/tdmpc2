"""Auditable report assembly from already completed historical experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import pandas as pd

from .figure_data import control_q_figure_data
from .metrics import unified_statistics
from .records import CanonicalMeasurement, control_q_records, unified_records
from .sources import validate_control_q_sources, validate_record_sources


@dataclass(frozen=True)
class ReportingConfig:
    version: int = 1
    family: Literal["unified", "control_q"] = "unified"
    swap_source: Path | None = None

    def __post_init__(self):
        if self.version not in (1, 2) or self.family not in ("unified", "control_q"):
            raise ValueError("Unsupported reporting configuration")
        if self.swap_source is not None and self.family != "control_q":
            raise ValueError("Closed-loop Q-swap source applies only to control_q")
        if self.swap_source is not None and self.version != 2:
            raise ValueError("Optional Q-swap results require reporting configuration version 2")


def _write_rows(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _canonical_rows(records: list[CanonicalMeasurement]):
    for record in records:
        yield dict(report_family=record.family, quantity=record.quantity,
                   measurement_status=record.status, source_table=record.source_table,
                   **record.identity, **{key: value for key, value in record.values.items()
                                       if key not in record.identity})


class ReportBuilder:
    """Own an explicit source root and a new output directory; never run a model."""

    def __init__(self, source_root: Path, config: ReportingConfig):
        self.source_root = Path(source_root).resolve(strict=True)
        if not self.source_root.is_dir():
            raise ValueError("Reporting source root must be a directory")
        self.config = config

    def build(self, output: Path, *, render: bool = False) -> dict:
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(f"Report output already exists: {output}")
        if self.config.family == "control_q":
            return self._control_q(output, render=render)
        return self._unified(output, render=render)

    def _control_q(self, output: Path, *, render: bool) -> dict:
        from experiments.nsgym.analysis.control_q.data import load_swap
        from experiments.nsgym.analysis.control_q.metrics import sha256, require

        paths, tables, arrays, historical = validate_control_q_sources(self.source_root)
        swap = (load_swap(self.source_root, self.config.swap_source)
                if self.config.swap_source is not None else None)
        records = control_q_records(tables, swap_episodes=swap[1] if swap else None)
        terminal_rows, terminal_summary, checks, examples, candidate_rows = control_q_figure_data(
            arrays, paths["terminal"], tables["effects"])
        source_hashes = {str(path): sha256(path) for path in arrays.values()}
        for item in historical.values():
            source_hashes.update(item["sources"])
        for name, path in paths.items():
            for relative, digest in historical[name]["outputs"].items():
                source_hashes[str(path / relative)] = digest
        if swap is not None:
            swap_sources = json.loads((swap[0] / "provenance.json").read_text())["sources"]
            source_hashes.update({path: record["sha256"]
                                  for path, record in swap_sources.items()})
        require(len(arrays) == 108 and len(checks) == 108,
                "Control-Q historical stream coverage changed")

        output.mkdir(parents=True)
        _write_rows(output / "canonical_records.csv", _canonical_rows(records))
        candidate_seed_rows = candidate_rows.groupby([
            "task", "family", "severity", "adaptation_seed", "evaluation_seed"])[[
            "mean_signed_delta_q", "candidate_dependent_sd", "baseline_score_margin",
            "top1_reversal"]].mean().reset_index()
        for name, frame in (("terminal_per_seed", terminal_rows),
                            ("terminal_fixed_head_summary", terminal_summary),
                            ("candidate_identity_checks", checks),
                            ("candidate_q_change_per_state", candidate_rows),
                            ("candidate_q_change_per_seed", candidate_seed_rows)):
            frame.to_csv(output / f"{name}.csv", index=False)
        status = "MEASURED" if swap else "NOT_MEASURED"
        report = dict(family="control_q", conditions=36, adaptation_streams=108,
                      closed_loop_critic_effect=status, canonical_records=len(records),
                      figure_data_rows={"terminal_per_seed": len(terminal_rows),
                                        "terminal_fixed_head_summary": len(terminal_summary),
                                        "candidate_identity_checks": len(checks),
                                        "candidate_q_change_per_state": len(candidate_rows),
                                        "candidate_q_change_per_seed": len(candidate_seed_rows)},
                      figures_rendered=bool(render))
        (output / "SOURCE_PROVENANCE.json").write_text(json.dumps(historical | dict(
            reporting_config_version=self.config.version, source_root=str(self.source_root),
            source_hashes=source_hashes), indent=2, allow_nan=False))
        if render:
            from .rendering import render_control_q
            report["legacy_validation"] = render_control_q(
                self.source_root, output, tables, arrays, paths, swap)
        for name, digest in source_hashes.items():
            require(Path(name).is_file() and sha256(name) == digest,
                    f"Source changed during report assembly: {name}")
        (output / "REPORT_STATUS.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        return report

    def _unified(self, output: Path, *, render: bool) -> dict:
        from experiments.nsgym.analysis.discovery import Catalog
        from experiments.nsgym.analysis.sources import collect
        from experiments.nsgym.analysis.engine import coverage
        from experiments.nsgym.analysis.schema import sha256

        config_path = Path(__file__).resolve().parents[4] / "experiments/nsgym/analysis/config.json"
        config = json.loads(config_path.read_text())
        catalog = Catalog(self.source_root, config["tasks"], output)
        collected = collect(catalog)
        records = unified_records(collected.rows)
        hashes = validate_record_sources(self.source_root, collected.rows)
        statistics = unified_statistics(collected.rows)
        cov = coverage(catalog, collected.rows)
        output.mkdir(parents=True)
        _write_rows(output / "canonical_records.csv", _canonical_rows(records))
        for name, rows in statistics.items():
            _write_rows(output / f"{name}.csv", rows)
        _write_rows(output / "SOURCE_COVERAGE.csv", cov)
        if render:
            from .rendering import render_unified
            render_unified(output, collected.rows, statistics)
        for relative, digest in hashes.items():
            path = self.source_root / relative
            if sha256(path) != digest:
                raise ValueError(f"Source changed during report assembly: {relative}")
        report = dict(family="unified", canonical_records=len(records),
                      paired_conditions=len(statistics["comparisons"]),
                      missing_metric_families=sum(row["evidence"] == "MISSING" for row in cov),
                      figures_rendered=bool(render), invalid_sources=len(catalog.issues),
                      excluded_sources=len(catalog.excluded))
        (output / "REPORT_STATUS.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        (output / "SOURCE_PROVENANCE.json").write_text(json.dumps(dict(
            reporting_config_version=self.config.version, source_root=str(self.source_root),
            source_hashes=hashes, config_sha256=sha256(config_path)), indent=2, allow_nan=False))
        return report
