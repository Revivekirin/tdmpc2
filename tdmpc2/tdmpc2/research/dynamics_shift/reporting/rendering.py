"""Render through unchanged historical visual encodings and validators."""

from __future__ import annotations

from pathlib import Path


def render_control_q(root: Path, output: Path, tables, arrays, paths, swap):
    from experiments.nsgym.analysis.control_q.figures import render
    from experiments.nsgym.analysis.control_q.report import code_audit, guide, validate_figures
    from experiments.nsgym.analysis.control_q.metrics import write

    write(output / "SOURCE_ARTIFACTS.json", {key: str(path) for key, path in paths.items()})
    result = render(root, output, tables, arrays, paths, swap)
    code_audit(output)
    validation = validate_figures(output, result)
    guide(output, validation, {key: str(path) for key, path in paths.items()})
    return validation


def render_unified(output: Path, records: list[dict], statistics: dict):
    from experiments.nsgym.analysis.engine import figures, research_figures

    figures(output, statistics["statistical_summary"], statistics["comparisons"],
            statistics["prediction_control_pairs"])
    research_figures(output, records)
