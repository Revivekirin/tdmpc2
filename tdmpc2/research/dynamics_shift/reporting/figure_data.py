"""Numerical figure inputs, kept separate from rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def control_q_figure_data(arrays: dict, terminal_source: Path, effects: pd.DataFrame):
    """Extract exactly the historical two-head plot data and interval policy."""
    from experiments.nsgym.analysis.control_q.figures import _terminal_table
    from experiments.nsgym.analysis.control_q.metrics import require

    discount = pd.read_csv(Path(terminal_source) / "tables/terminal_effect_decomposition_per_seed.csv")
    grouped = discount.groupby(["task", "family", "severity", "adaptation_seed"]).discount_power
    require(grouped.nunique().eq(1).all(), "Inconsistent saved discount powers")
    powers = {key: float(group.iloc[0]) for key, group in grouped}
    per_seed, terminal, checks, examples, candidate_rows = _terminal_table(arrays, powers)
    # The archived interval used a different deterministic bootstrap key.
    # Verify the recomputed mean, then retain its original median and CI.
    archived = effects[effects.metric == "critic_DQ"]
    for row in archived.itertuples():
        mask = ((terminal.task == row.task) & (terminal.family == row.family) &
                (terminal.severity == row.severity) & (terminal.method == "Qswap") &
                (terminal.metric == "terminal_delta"))
        require(mask.sum() == 1 and np.isclose(float(terminal.loc[mask, "mean"].iloc[0]),
                                               row.mean, atol=1e-9),
                f"Q-swap terminal mean parity: {row.task}/{row.family}/{row.severity}")
        for key in ("mean", "median", "ci_low", "ci_high"):
            terminal.loc[mask, key] = getattr(row, key)
    return per_seed, terminal, checks, examples, candidate_rows
