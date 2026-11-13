"""Shared paired statistics; the adaptation stream is the uncertainty unit."""

from __future__ import annotations

from collections import defaultdict
from .statistics import bootstrap
from statistics import mean, median, stdev

from .schema import numeric


def paired_improvements(evaluations: list[dict], methods: set[str], kind="shifted"):
    grouped = defaultdict(dict)
    for row in evaluations:
        if row["metric"] != "episode_return" or row["kind"] != kind:
            continue
        key = (row["task"], row["shift_family"], row["shift_parameter"],
               row["severity_multiplier"], row["adaptation_seed"], row["evaluation_seed"])
        method = row["method"]
        if method in grouped[key] and grouped[key][method]["value"] != row["value"]:
            raise ValueError(f"Conflicting evaluation records: {key}, {method}")
        grouped[key][method] = row
    per_stream = defaultdict(list)
    for key, observed in grouped.items():
        baseline = observed.get("frozen")
        if baseline is None: continue
        frozen = numeric(baseline["value"])
        if frozen is None: continue
        for method in sorted(methods - {"frozen"}):
            current = observed.get(method)
            if current is None: continue
            adapted = numeric(current["value"])
            if adapted is None: continue
            per_stream[key[:5] + (method,)].append(adapted - frozen)
    streams = []
    for key, values in sorted(per_stream.items()):
        streams.append(dict(task=key[0], shift_family=key[1], shift_parameter=key[2],
                            severity_multiplier=key[3], adaptation_seed=key[4], method=key[5],
                            evaluation_seeds=len(values), mean_return_improvement=mean(values)))
    by_condition = defaultdict(list)
    for row in streams:
        key = (row["task"], row["shift_family"], row["shift_parameter"], row["severity_multiplier"], row["method"])
        by_condition[key].append(row["mean_return_improvement"])
    comparisons = []
    for key, values in sorted(by_condition.items()):
        from .sources import SOURCES
        low, high = bootstrap(values, SOURCES[key[0]].get("adapter", key[0]))
        comparisons.append(dict(task=key[0], shift_family=key[1], shift_parameter=key[2],
                                severity_multiplier=key[3], method=key[4], baseline="frozen",
                                n_adaptation_streams=len(values), evaluation_seeds_per_stream=min(
                                    s["evaluation_seeds"] for s in streams if
                                    (s["task"], s["shift_family"], s["shift_parameter"],
                                     s["severity_multiplier"], s["method"]) == key),
                                mean=mean(values), median=median(values), std=stdev(values) if len(values) > 1 else "",
                                ci_low=low,
                                ci_high=high,
                                improved_streams=sum(v > 0 for v in values),
                                nominal_duplicate=str(float(key[3]) == 1.0),
                                uncertainty_unit="adaptation_seed"))
    return streams, comparisons


def prediction_improvements(predictions: list[dict]):
    grouped = defaultdict(dict)
    for row in predictions:
        key = (row["task"], row["shift_family"], row["shift_parameter"],
               row["severity_multiplier"], row["adaptation_seed"], row["evaluation_seed"])
        grouped[key][row["method"]] = numeric(row["value"])
    result = []
    for key, values in sorted(grouped.items()):
        frozen = values.get("frozen")
        if frozen is None: continue
        for method, current in values.items():
            if method == "frozen" or current is None: continue
            result.append(dict(task=key[0], shift_family=key[1], shift_parameter=key[2],
                               severity_multiplier=key[3], adaptation_seed=key[4],
                               evaluation_seed=key[5], method=method,
                               frozen_mse=frozen, adapted_mse=current,
                               mse_improvement=frozen-current))
    return result


def prediction_control_pairs(streams: list[dict], prediction_pairs: list[dict]):
    grouped = defaultdict(list)
    for row in prediction_pairs:
        key = (row["task"], row["shift_family"], row["shift_parameter"],
               row["severity_multiplier"], row["adaptation_seed"], row["method"])
        grouped[key].append(row["mse_improvement"])
    result = []
    for row in streams:
        key = (row["task"], row["shift_family"], row["shift_parameter"],
               row["severity_multiplier"], row["adaptation_seed"], row["method"])
        errors = grouped.get(key, [])
        if errors:
            result.append({**row, "mse_improvement": mean(errors), "prediction_seeds": len(errors),
                           "quadrant": ("prediction_improved" if mean(errors) > 0 else "prediction_worsened") + "/" +
                           ("control_improved" if row["mean_return_improvement"] > 0 else "control_worsened")})
    return result


def normalized_comparisons(evaluations: list[dict], comparisons: list[dict]):
    baselines = defaultdict(dict)
    for row in evaluations:
        if row["metric"] == "episode_return" and row["kind"] == "shifted" and row["method"] == "frozen":
            key = (row["task"], row["shift_family"], row["shift_parameter"], row["severity_multiplier"])
            baselines[key][row["evaluation_seed"]] = numeric(row["value"])
    result = []
    for row in comparisons:
        key = (row["task"], row["shift_family"], row["shift_parameter"], row["severity_multiplier"])
        values = [v for v in baselines[key].values() if v is not None]
        if not values: continue
        scale = abs(mean(values))
        result.append({**row, "frozen_mean_return": mean(values), "normalized_improvement":
                       row["mean"] / scale if scale > 1e-12 else "",
                       "normalization": "paired_delta_over_abs_condition_frozen_mean"})
    return result
