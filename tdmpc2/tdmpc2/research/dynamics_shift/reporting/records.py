"""Explicit interpretation of historical rows without changing their values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class CanonicalMeasurement:
    family: str
    quantity: str
    status: str
    source_table: str
    identity: Mapping[str, Any]
    values: Mapping[str, Any]


_CONTROL_Q = {
    "ranking": ("predicted_candidate_ranking", "PREDICTED"),
    "native": ("native_mppi_proposed_action", "PREDICTED"),
    "effects": ("predicted_terminal_q_effect", "PREDICTED"),
    "returns": ("actual_heldout_episode_return", "MEASURED"),
}


def unified_records(rows: list[dict]) -> list[CanonicalMeasurement]:
    """Retain the existing unified schema and its evidence/missingness labels."""
    result = []
    for row in rows:
        if not row.get("source_path") or not row.get("source_sha256"):
            raise ValueError("Unified record requires a verified source path and SHA-256")
        status = row.get("evidence", "")
        if status not in ("RAW_REPRODUCED", "DERIVED_ONLY", "MISSING"):
            raise ValueError(f"Unsupported unified evidence: {status}")
        identity = {key: row.get(key, "") for key in (
            "task", "method", "checkpoint_id", "adaptation_seed", "evaluation_seed",
            "shift_parameter", "severity_multiplier", "schedule", "adaptation_budget")}
        result.append(CanonicalMeasurement("unified", row["kind"], status,
                                           row["source_path"], identity, dict(row)))
    return result


def control_q_records(tables: Mapping[str, Any], *, swap_episodes=None) -> list[CanonicalMeasurement]:
    """Keep archived table rows distinct from an unmeasured critic-swap return."""
    result = []
    identity_keys = ("task", "family", "severity", "adaptation_seed", "evaluation_seed",
                     "method", "comparison", "phase", "metric")
    for row in tables["control"].to_dict("records"):
        measured = row.get("metric") == "return"
        result.append(CanonicalMeasurement(
            "control_q", "actual_episode_return" if measured else "one_step_fixed_encoder_latent_mse",
            "MEASURED" if measured else "PREDICTED", "control",
            {key: row.get(key, "") for key in identity_keys}, row))
    for row in tables["evidence"].to_dict("records"):
        identity = {key: row.get(key, "") for key in identity_keys}
        for key, quantity, status in (("return", "actual_episode_return", "MEASURED"),
                                      ("fixed_encoder_latent_mse", "one_step_fixed_encoder_latent_mse", "PREDICTED")):
            if key in row and pd.notna(row[key]):
                result.append(CanonicalMeasurement("control_q", quantity, status, "evidence",
                                                   identity, {"value": row[key], "original_column": key}))
    for name, (quantity, status) in _CONTROL_Q.items():
        frame = tables[name]
        for row in frame.to_dict("records"):
            identity = {key: row.get(key, "") for key in identity_keys}
            result.append(CanonicalMeasurement("control_q", quantity, status,
                                               name, identity, row))
    if swap_episodes is None:
        result.append(CanonicalMeasurement("control_q", "closed_loop_critic_swap_return",
                                           "NOT_MEASURED", "", {}, {}))
    else:
        for row in swap_episodes.to_dict("records"):
            identity = {key: row.get(key, "") for key in (
                "task", "family", "severity", "adaptation_seed", "evaluation_seed", "method")}
            result.append(CanonicalMeasurement("control_q", "closed_loop_critic_swap_return",
                                               "MEASURED", "q_swap/episodes.csv", identity, row))
    return result
