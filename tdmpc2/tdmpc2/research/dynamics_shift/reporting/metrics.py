"""Numerical adapters to the original, deliberately distinct statistics."""

from __future__ import annotations

def unified_statistics(rows: list[dict]) -> dict[str, list[dict]]:
    from experiments.nsgym.analysis.metrics import (
        summarize_records, paired_improvements, prediction_improvements,
        prediction_control_pairs, method_contrasts, normalized_comparisons,
    )
    summaries = summarize_records(rows)
    streams, comparisons = paired_improvements(rows)
    prediction = prediction_improvements(rows)
    return {
        "statistical_summary": summaries,
        "per_adaptation_stream": streams,
        "comparisons": comparisons,
        "nominal_retention": paired_improvements(rows, kind="nominal")[1],
        "prediction_improvements": prediction,
        "prediction_control_pairs": prediction_control_pairs(streams, prediction),
        "adapted_method_contrasts": method_contrasts(rows),
        "cross_task_normalized": normalized_comparisons(rows, comparisons),
    }
