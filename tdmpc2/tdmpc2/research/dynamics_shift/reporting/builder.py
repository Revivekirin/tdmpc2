"""Auditable report assembly from already completed historical experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd

from .figure_data import control_q_figure_data
from .metrics import unified_statistics
from .records import CanonicalMeasurement, control_q_records, unified_records
from .sources import validate_control_q_sources, validate_record_sources
from research.dynamics_shift.interfaces import validate_study_config


@dataclass(frozen=True)
class ReportingConfig:
    version: int = 1
    family: Literal["unified", "control_q", "study"] = "unified"
    swap_source: Path | None = None
    study_config: Mapping | None = None

    def __post_init__(self):
        if self.version not in (1, 2) or self.family not in ("unified", "control_q", "study"):
            raise ValueError("Unsupported reporting configuration")
        if self.swap_source is not None and self.family != "control_q":
            raise ValueError("Closed-loop Q-swap source applies only to control_q")
        if self.swap_source is not None and self.version != 2:
            raise ValueError("Optional Q-swap results require reporting configuration version 2")
        if self.family == "study":
            if self.version != 1 or self.study_config is None:
                raise ValueError("Study reporting requires version 1 and a study configuration")
            validate_study_config(self.study_config)
        elif self.study_config is not None:
            raise ValueError("Study configuration requires study reporting")


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
        if self.config.family == "study":
            return self._study(output, render=render)
        if self.config.family == "control_q":
            return self._control_q(output, render=render)
        return self._unified(output, render=render)

    def _study_reference(self, name: str, definition: Mapping, parent: str) -> dict:
        from experiments.nsgym.analysis.schema import sha256

        files, payloads, problems = {}, {}, []
        operation_problems = {"training": [], "evaluation": []}
        for role, relative in definition.items():
            if role == "checkpoint_sha256":
                continue
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Study reference outside source root: {relative}")
            source = (self.source_root / path).resolve()
            if not source.is_relative_to(self.source_root):
                raise ValueError(f"Study reference outside source root: {relative}")
            if not source.is_file():
                issue = f"Missing {role}: {relative}"
                problems.append(issue)
                operation_problems["evaluation" if role.startswith("evaluation") else "training"].append(issue)
                continue
            files[role] = dict(path=path.as_posix(), sha256=sha256(source))
            if role != "checkpoint":
                try:
                    payloads[role] = json.loads(source.read_text())
                except (OSError, ValueError) as exc:
                    issue = f"Invalid {role}: {exc}"
                    problems.append(issue)
                    operation_problems["evaluation" if role.startswith("evaluation") else "training"].append(issue)
        checkpoint = definition.get("checkpoint_sha256")
        if checkpoint is not None and files.get("checkpoint", {}).get("sha256") != checkpoint:
            issue = "Adapted checkpoint SHA-256 mismatch"
            problems.append(issue)
            operation_problems["training"].append(issue)
            operation_problems["evaluation"].append(issue)
        trained_snapshot = payloads.get("training_completion", {}).get("snapshot_sha256")
        if checkpoint is not None and trained_snapshot is not None and trained_snapshot != checkpoint:
            issue = "Training completion checkpoint identity mismatch"
            problems.append(issue)
            operation_problems["training"].append(issue)
        training_leaf = payloads.get("training_leaf_completion", {})
        training_leaf_manifest = payloads.get("training_leaf_manifest", {})
        if training_leaf and training_leaf.get("status") != "PASS":
            issue = "Training leaf did not complete successfully"
            problems.append(issue)
            operation_problems["training"].append(issue)
        if checkpoint is not None and training_leaf.get("snapshot_sha256") not in (None, checkpoint):
            issue = "Training leaf checkpoint identity mismatch"
            problems.append(issue)
            operation_problems["training"].append(issue)
        if training_leaf_manifest and training_leaf_manifest.get("checkpoint_sha256") != parent:
            issue = "Training leaf nominal parent mismatch"
            problems.append(issue)
            operation_problems["training"].append(issue)
        for operation in ("training", "evaluation"):
            manifest = payloads.get(f"{operation}_manifest", {})
            recorded_parent = manifest.get("checkpoint_sha256", manifest.get("expected_checkpoint_sha256"))
            if manifest and recorded_parent != parent:
                issue = f"{operation} nominal parent mismatch"
                problems.append(issue)
                operation_problems[operation].append(issue)
            completion = payloads.get(f"{operation}_completion", {})
            if completion.get("expected") is not None and len(completion.get("completed", [])) != completion["expected"]:
                issue = f"{operation} completed count mismatch"
                problems.append(issue)
                operation_problems[operation].append(issue)
        leaf = payloads.get("evaluation_leaf_completion", {})
        leaf_manifest = payloads.get("evaluation_leaf_manifest", {})
        evaluation_checkpoint = (parent if name == "frozen" else
                                 leaf.get("snapshot_sha256") or leaf_manifest.get("snapshot_sha256"))
        if checkpoint is not None and evaluation_checkpoint != checkpoint:
            issue = "Evaluation checkpoint identity mismatch or unverified"
            problems.append(issue)
            operation_problems["evaluation"].append(issue)
        if leaf and leaf.get("status") != "PASS":
            issue = "Evaluation leaf did not complete successfully"
            problems.append(issue)
            operation_problems["evaluation"].append(issue)
        if leaf.get("replay_writes", 0) != 0:
            issue = "Held-out evaluation wrote to replay"
            problems.append(issue)
            operation_problems["evaluation"].append(issue)
        training = ("NOT_RUN" if name == "frozen" else
                    "PASS" if payloads.get("training_completion", {}).get("status") == "PASS"
                    and not operation_problems["training"] else "UNVERIFIED")
        evaluation = ("PASS" if payloads.get("evaluation_completion", {}).get("status") == "PASS"
                      and not operation_problems["evaluation"] else "UNVERIFIED")
        training_record = training_leaf or payloads.get("training_completion", {})
        metrics = training_record.get("metrics", {})
        updates = {key: training_record[key] for key in ("dynamics_steps", "q_steps", "target_steps",
                                                       "total_optimizer_steps", "updates", "optimizer_step_calls")
                   if type(training_record.get(key)) is int}
        if type(metrics.get("gradient_update_steps")) is int:
            updates["gradient_update_steps"] = metrics["gradient_update_steps"]
        identity = (training_leaf_manifest or payloads.get("training_manifest", {})).get("identity", {})
        return dict(method=name, training_status=training, evaluation_status=evaluation,
                    evaluation_checkpoint_identity=evaluation_checkpoint,
                    held_out_replay_writes=leaf.get("replay_writes"),
                    adaptation_seed=identity.get("adaptation_seed"),
                    dynamics_shift={key: identity[key] for key in ("parameter", "severity", "schedule") if key in identity},
                    interaction_budget=training_record.get("transitions", metrics.get("adaptation_transitions")),
                    optimizer_update_counts=updates,
                    sources=files, problems=problems)

    def _study(self, output: Path, *, render: bool) -> dict:
        from experiments.nsgym.analysis.schema import sha256

        config = self.config.study_config
        assert config is not None
        validate_study_config(config)
        output.mkdir(parents=True)
        # Catalog prunes generated report trees even when output is under the source root.
        (output / "manifest.json").write_text(json.dumps(dict(
            pipeline="nsgym_study_report", no_training_or_simulator=True), indent=2))
        evidence = self._unified(output / "evidence", render=render,
                                 exclude_unverified=True)
        references = [self._study_reference(name, item, config["nominal_checkpoint_sha256"])
                      for name, item in config["historical_references"].items()]
        matrix = [dict(adaptation_protocol=protocol, dynamics_objective=dynamics,
                       critic_objective=critic, dynamics_shift=config["dynamics_shift"],
                       nominal_checkpoint_sha256=config["nominal_checkpoint_sha256"],
                       adaptation_seed=None, evaluation_seed=None,
                       interaction_budget=config["interaction_budget"],
                       optimizer_update_counts={}, evaluation_checkpoint_identity=None,
                       training_status="NOT_RUN",
                       evaluation_status="NOT_RUN", analysis_status="NOT_RUN")
                  for protocol in config["adaptation_protocols"]
                  for dynamics in config["dynamics_objectives"]
                  for critic in config["critic_objectives"]]
        (output / "MATRIX_STATUS.json").write_text(json.dumps(matrix, indent=2))
        figures = sorted(path.name for path in (output / "evidence").glob("*adaptation*.png"))
        nominal_relative = config.get("nominal_checkpoint_path")
        nominal_file = self.source_root / nominal_relative if nominal_relative else None
        nominal_verified = bool(nominal_file and nominal_file.is_file() and
                                nominal_file.resolve().is_relative_to(self.source_root) and
                                sha256(nominal_file) == config["nominal_checkpoint_sha256"])
        provenance = dict(study_config=config, source_root=str(self.source_root),
                          nominal_checkpoint_verified=nominal_verified,
                          historical_references=references,
                          unified_provenance_sha256=sha256(output / "evidence/SOURCE_PROVENANCE.json"))
        (output / "SOURCE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
        lines = [f"# {config['title']}", "", "## Verified historical coverage", "",
                 "Historical methods are independent prior experiments; none is a completed cell of the planned 2×2 study.", "",
                 "| Method | Training | Held-out evaluation | Source status |", "|---|---|---|---|"]
        for row in references:
            lines.append(f"| {row['method']} | {row['training_status']} | {row['evaluation_status']} | "
                         f"{'VERIFIED' if not row['problems'] else 'UNVERIFIED'} |")
        lines += ["", "Representative historical update counts are recorded in "
                  "[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json); they do not describe the planned cells."]
        lines += ["", "## Configuration and operation progress", "",
                  f"Nominal checkpoint SHA-256: `{config['nominal_checkpoint_sha256']}`. "
                  f"Interaction budget: {config['interaction_budget']} real transitions. "
                  f"Shift: {config['dynamics_shift']['schedule']} over "
                  f"{', '.join(config['dynamics_shift']['parameters'])} at multipliers "
                  f"{', '.join(map(str, config['dynamics_shift']['severities']))}.", "",
                  f"Existing unified evidence: {evidence['canonical_records']} canonical records, "
                  f"{evidence['paired_conditions']} paired comparisons; "
                  f"{evidence['excluded_unverified_records']} source-conflicted records excluded. "
                  "Training and held-out evaluation statuses are checked separately.", "",
                  "## Matched-replay versus online", "",
                  "NOT_RUN as a paired study comparison. Historical online streams are evidence for their "
                  "own protocols; no matched-replay counterpart is inferred.", "",
                  "## Planned factorial comparison", "",
                  "| Protocol | Dynamics objective | Critic objective | Training | Evaluation |",
                  "|---|---|---|---|---|"]
        for row in matrix:
            lines.append(f"| {row['adaptation_protocol']} | {row['dynamics_objective']} | "
                         f"{row['critic_objective']} | NOT_RUN | NOT_RUN |")
        lines += ["", "## Adaptation recovery figures", ""]
        lines.extend(f"- [Regenerated from verified source rows: {name}](evidence/{name})" for name in figures)
        if not figures:
            lines.append("NOT_RUN: figure rendering disabled or no compatible paired recovery rows.")
        lines += ["", "## Source provenance", "",
                  "See [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) and "
                  "[unified evidence provenance](evidence/SOURCE_PROVENANCE.json). "
                  "[Excluded source-conflicted records](evidence/EXCLUDED_UNVERIFIED_SOURCES.json) "
                  "are omitted from the analysis and make its status PARTIAL. "
                  "A checkpoint alone does not imply evaluation.", ""]
        (output / "STUDY_REPORT.md").write_text("\n".join(lines))
        status = dict(family="study", historical_references=len(references),
                      verified_historical_references=sum(not item["problems"] for item in references),
                      planned_cells=len(matrix), completed_cells=0,
                      matched_replay_vs_online="NOT_RUN", figures=len(figures),
                      nominal_checkpoint_verified=nominal_verified,
                      analysis_status="PARTIAL" if evidence["excluded_unverified_records"] else "PASS",
                      unified_evidence=evidence)
        (output / "REPORT_STATUS.json").write_text(json.dumps(status, indent=2))
        return status

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

    def _unified(self, output: Path, *, render: bool,
                 exclude_unverified: bool = False) -> dict:
        from experiments.nsgym.analysis.discovery import Catalog
        from experiments.nsgym.analysis.sources import collect
        from experiments.nsgym.analysis.engine import coverage
        from experiments.nsgym.analysis.schema import sha256

        config_path = Path(__file__).resolve().parents[4] / "experiments/nsgym/analysis/config.json"
        config = json.loads(config_path.read_text())
        catalog = Catalog(self.source_root, config["tasks"], output)
        collected = collect(catalog)
        rows = collected.rows
        excluded = []
        if exclude_unverified:
            actual_hashes = {}
            verified_rows = []
            for row in rows:
                relative = Path(row["source_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Source outside explicit root: {relative}")
                source = (self.source_root / relative).resolve()
                if not source.is_relative_to(self.source_root):
                    raise ValueError(f"Source outside explicit root: {relative}")
                key = relative.as_posix()
                if key not in actual_hashes:
                    actual_hashes[key] = sha256(source) if source.is_file() else None
                actual = actual_hashes[key]
                if actual != row["source_sha256"]:
                    excluded.append(dict(source_record_id=row["source_record_id"],
                                         source_path=relative.as_posix(),
                                         recorded_sha256=row["source_sha256"], actual_sha256=actual))
                else:
                    verified_rows.append(row)
            rows = verified_rows
        records = unified_records(rows)
        hashes = validate_record_sources(self.source_root, rows)
        statistics = unified_statistics(rows)
        cov = coverage(catalog, rows)
        output.mkdir(parents=True)
        if exclude_unverified:
            (output / "EXCLUDED_UNVERIFIED_SOURCES.json").write_text(json.dumps(excluded, indent=2))
        _write_rows(output / "canonical_records.csv", _canonical_rows(records))
        for name, rows in statistics.items():
            _write_rows(output / f"{name}.csv", rows)
        _write_rows(output / "SOURCE_COVERAGE.csv", cov)
        if render:
            from .rendering import render_unified
            figure_rows = ([row for row in rows if "kind" in row and "metric" in row]
                           if exclude_unverified else rows)
            render_unified(output, figure_rows, statistics)
        for relative, digest in hashes.items():
            path = self.source_root / relative
            if sha256(path) != digest:
                raise ValueError(f"Source changed during report assembly: {relative}")
        report = dict(family="unified", canonical_records=len(records),
                      paired_conditions=len(statistics["comparisons"]),
                      missing_metric_families=sum(row["evidence"] == "MISSING" for row in cov),
                      figures_rendered=bool(render), invalid_sources=len(catalog.issues),
                      excluded_sources=len(catalog.excluded),
                      excluded_unverified_records=len(excluded))
        (output / "REPORT_STATUS.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        (output / "SOURCE_PROVENANCE.json").write_text(json.dumps(dict(
            reporting_config_version=self.config.version, source_root=str(self.source_root),
            source_hashes=hashes, config_sha256=sha256(config_path)), indent=2, allow_nan=False))
        return report
