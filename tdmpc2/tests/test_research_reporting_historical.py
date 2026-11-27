"""Compatible-host check against immutable control-Q source families."""

import json
import os
from pathlib import Path
import sys
import hashlib

import pandas as pd
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))

from experiments.nsgym.analysis.control_q.data import load_inputs
from experiments.nsgym.analysis.control_q.figures import _terminal_table
from experiments.nsgym.analysis.control_q.run import main as legacy_report_main
from research.dynamics_shift.reporting import ReportBuilder, ReportingConfig


def test_historical_control_q_report_figure_data_and_rendering(tmp_path):
    configured = os.environ.get("TDMPC2_REPORT_RESULTS_ROOT")
    if not configured:
        pytest.skip("Historical NS-Gym results root not provided")
    root = Path(configured).resolve(strict=True)
    paths, tables, arrays, _ = load_inputs(root)
    output = tmp_path / "canonical-report"
    report = ReportBuilder(root, ReportingConfig(family="control_q")).build(output, render=True)
    assert report["conditions"] == 36
    assert report["adaptation_streams"] == 108
    assert report["closed_loop_critic_effect"] == "NOT_MEASURED"
    assert report["legacy_validation"]["figure_png_count"] == 42
    assert len(list((output / "figures").rglob("*.png"))) == 42
    assert json.loads((output / "REPORT_STATUS.json").read_text())["closed_loop_critic_effect"] == "NOT_MEASURED"
    canonical = pd.read_csv(output / "canonical_records.csv")
    assert canonical.measurement_status.eq("NOT_MEASURED").sum() == 1
    for name in ("control", "ranking", "native", "effects", "returns"):
        original = tables[name].reset_index(drop=True)
        recovered = canonical.loc[canonical.source_table.eq(name), original.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(recovered, original, check_dtype=False,
                                      check_exact=False, rtol=0, atol=1e-12)
    for column in ("return", "fixed_encoder_latent_mse"):
        original = tables["evidence"][column].dropna().reset_index(drop=True)
        recovered = canonical.loc[canonical.source_table.eq("evidence") &
                                  canonical.original_column.eq(column), "value"].reset_index(drop=True)
        pd.testing.assert_series_equal(recovered, original, check_dtype=False,
                                       check_names=False, check_exact=False,
                                       rtol=0, atol=1e-12)

    # Reconstruct the historical figure's input arrays independently from the
    # canonical output path; preserve original order, CI policy and labels.
    discount = pd.read_csv(paths["terminal"] / "tables/terminal_effect_decomposition_per_seed.csv")
    grouped = discount.groupby(["task", "family", "severity", "adaptation_seed"]).discount_power
    powers = {key: float(group.iloc[0]) for key, group in grouped}
    per_seed, terminal, checks, _, candidate_rows = _terminal_table(arrays, powers)
    for name, historical in (("terminal_per_seed", per_seed),
                             ("candidate_identity_checks", checks),
                             ("candidate_q_change_per_state", candidate_rows)):
        pd.testing.assert_frame_equal(pd.read_csv(output / f"{name}.csv"), historical,
                                      check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    # The legacy renderer applies archived critic-only intervals after this
    # reconstruction, so compare the complete post-render table to itself via
    # the validated archive's mean and the expected 36 condition rows.
    final = pd.read_csv(output / "terminal_fixed_head_summary.csv")
    critic = final[(final.method == "Qswap") & (final.metric == "terminal_delta")]
    archived = tables["effects"][tables["effects"].metric == "critic_DQ"]
    merged = critic.merge(archived, on=["task", "family", "severity"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(merged) == 36
    for key in ("mean", "median", "ci_low", "ci_high"):
        np.testing.assert_allclose(merged[f"{key}_new"], merged[f"{key}_old"],
                                   rtol=0, atol=1e-12)

    # The original public reporting CLI is the figure-data and rendering
    # reference. Run it only into a fresh temporary directory, then compare
    # every common CSV and every rendered PNG against the canonical output.
    legacy = tmp_path / "legacy-report"
    legacy_report_main(["--mode", "all", "--results-root", str(root),
                        "--output", str(legacy)])
    common_csv = {path.name for path in legacy.glob("*.csv")} & {
        path.name for path in output.glob("*.csv")}
    assert len(common_csv) == 11
    for name in sorted(common_csv):
        pd.testing.assert_frame_equal(pd.read_csv(output / name),
                                      pd.read_csv(legacy / name),
                                      check_dtype=False, check_exact=False,
                                      rtol=0, atol=1e-12)
    def image_hashes(directory):
        return {path.relative_to(directory).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*.png")}
    canonical_pngs = image_hashes(output / "figures")
    assert len(canonical_pngs) == 42
    assert canonical_pngs == image_hashes(legacy / "figures")
