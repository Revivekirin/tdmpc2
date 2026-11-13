"""Unified read-only analysis CLI. Run from the executable project root."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .metrics import paired_improvements, prediction_improvements, prediction_control_pairs, normalized_comparisons
from .schema import COMMON, IDENTITY, numeric, read_json, sha256, write_csv
from .sources import SOURCES, available, ingest, ingest_completed_reacher_dq

PROJECT = Path(__file__).resolve().parents[3]
WORKSPACE = PROJECT.parent
DEFAULT_RESULTS = WORKSPACE / "results/nsgym"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "unified-analysis/refactor-v1"
COMPARISON_FIELDS = ("task", "shift_family", "shift_parameter", "severity_multiplier", "method",
                     "baseline", "n_adaptation_streams", "evaluation_seeds_per_stream", "mean",
                     "median", "std", "ci_low", "ci_high", "improved_streams", "nominal_duplicate", "uncertainty_unit")


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def classify(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    parent = str(path.parent).lower()
    if name.endswith((".pt", ".pth", ".ckpt")):
        return "B", "checkpoint_or_snapshot"
    if "smoke" in parent or "validation" in parent and "physical-validation" not in parent:
        return "E", "test_or_smoke"
    if "reacher-dynamics-q-v1" in parent:
        return "G", "incomplete_experiment_or_preparation"
    if name in ("episodes.csv", "steps.csv") or name.endswith("_episodes.csv"):
        return "C", "evaluation_or_rollout_record"
    if name.endswith((".npz", ".npy", ".jsonl")):
        return "A", "raw_scientific_record"
    if name.endswith((".png", ".pdf", ".svg")):
        return "D", "derived_figure"
    if name.endswith(".csv"):
        return "D", "derived_or_summary_table"
    if name in ("completion.json", "manifest.json"):
        return "A", "provenance_or_completion"
    return "F", "other_or_superseded_candidate"


def inventory(results: Path):
    output = []
    completion_cache = {}
    for path in sorted(results.rglob("*")):
        if not path.is_file() or path.is_symlink() or "unified-analysis" in path.parts:
            continue
        stage = path.relative_to(results).parts[0]
        stage_path = results / stage
        if stage not in completion_cache:
            c = read_json(stage_path / "completion.json")
            if not c: c = read_json(stage_path / "provenance/completion.json")
            completion_cache[stage] = c.get("status", "UNKNOWN")
        category, role = classify(path)
        task = "reacher" if "reacher" in stage else ("halfcheetah" if stage not in ("preview-v1",) else "unknown")
        output.append(dict(path=rel(path, results), stage=stage, task=task,
                           category=category, role=role, bytes=path.stat().st_size,
                           completion_status=completion_cache[stage],
                           sha256=sha256(path) if path.stat().st_size <= 65536 and
                           path.suffix in (".csv", ".json", ".md") else "",
                           hash_status="sha256" if path.stat().st_size <= 65536 and
                           path.suffix in (".csv", ".json", ".md") else "not_hashed_large_or_binary"))
    return output


def registry(results: Path, config: dict):
    reacher_dq = read_json(results / "reacher-dynamics-q-v1/provenance/completion.json")
    paths = config.get("reacher_dynamics_q", {})
    from .sources import completed_reacher_dq
    dq_ready = completed_reacher_dq(results, paths)
    result = {
        "halfcheetah": {"checkpoint": SOURCES["halfcheetah"]["checkpoint_id"],
                         "methods": {m: "COMPLETE_LOCAL_SOURCE" for m in ("frozen", "dynamics_only", "dynamics_q", "joint")}},
        "reacher": {"checkpoint": "checkpoint-000670000",
                    "methods": {"frozen": "COMPLETE_LOCAL_SOURCE", "dynamics_only": "COMPLETE_LOCAL_SOURCE",
                                "joint": "COMPLETE_LOCAL_SOURCE", "dynamics_q": "COMPLETE" if dq_ready else "IN_PROGRESS — EXCLUDED FROM CURRENT REPORT"},
                    "dynamics_q_gate": {"requires": ["training PASS", "prediction PASS", "shifted control PASS",
                                                     "matched held-out seeds", "snapshot hashes"],
                                        "local_record": reacher_dq.get("status", "MISSING"), "passed": dq_ready}},
    }
    for task, cfg in SOURCES.items():
        if task not in result:
            adapter=cfg['adapter']
            if adapter not in result:raise ValueError(f'Unsupported source schema: {adapter}')
            result[task] = dict(result[adapter], checkpoint=cfg.get('checkpoint_id','unknown'))
    return result



def plot(output: Path, evaluations: list[dict], predictions: list[dict], comparisons: list[dict], physical: list[dict], q: list[dict],
         prediction_control: list[dict], normalized: list[dict], task_metrics: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; figures not generated"]
    fig_dir = output / "figures"
    fig_dir.mkdir(exist_ok=True)
    notes = []
    for task in sorted({r["task"] for r in comparisons}):
        subset = [r for r in comparisons if r["task"] == task]
        if not subset: continue
        fig, ax = plt.subplots(figsize=(11, 5))
        conditions = sorted({(r["shift_family"], r["severity_multiplier"]) for r in subset},
                            key=lambda v: (v[0], float(v[1])))
        labels = [f"{f} ×{s}" for f, s in conditions]
        methods = sorted({r["method"] for r in subset})
        width = .8 / len(methods)
        for i, method in enumerate(methods):
            indexed = {(r["shift_family"], r["severity_multiplier"]): r for r in subset if r["method"] == method}
            heights = [indexed.get(c, {}).get("mean", float("nan")) for c in conditions]
            ax.bar([j-.4+width*(i+.5) for j in range(len(conditions))], heights, width, label=method)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
        ax.set_ylabel("Held-out return difference vs Frozen")
        ax.set_title(f"{task}: adaptation-stream means; nominal ×1 family labels share physics")
        ax.legend()
        fig.tight_layout()
        for suffix in ("png",):
            fig.savefig(fig_dir / f"{task}_adaptation.{suffix}", dpi=160)
        plt.close(fig)
    # Common severity view: physical response, fixed-bank Frozen mismatch, Frozen control.
    for task in sorted({r["task"] for r in physical}):
        for parameter in sorted({r["shift_parameter"] for r in physical if r["task"] == task}):
            metric = task_metrics.get(task, {}).get("physical_response_metric", "")
            probes = [r for r in physical if r["task"] == task and r["shift_parameter"] == parameter and r["metric"] == metric]
            if not probes: continue
            severities = sorted({float(r["severity_multiplier"]) for r in probes})
            family = probes[0]["shift_family"]
            series = []
            for severity in severities:
                matching = lambda r: (r["task"] == task and r["shift_family"] == family and
                                      r["shift_parameter"] == parameter and float(r["severity_multiplier"]) == severity)
                pp = [float(r["value"]) for r in probes if matching(r) and numeric(r["value"]) is not None]
                # Frozen rows are repeated across adaptation streams; deduplicate by evaluation seed.
                for source, measure in ((predictions, "latent_mse"), (evaluations, "episode_return")):
                    unique = {r["evaluation_seed"]: float(r["value"]) for r in source if matching(r) and
                              r["method"] == "frozen" and r["metric"] == measure and
                              r["kind"] in ("fixed_bank", "shifted") and numeric(r["value"]) is not None}
                    if source is predictions:
                        latent = mean(unique.values()) if unique else float("nan")
                    else:
                        control = mean(unique.values()) if unique else float("nan")
                series.append((mean(pp) if pp else float("nan"), latent, control))
            if not series: continue
            fig, axes = plt.subplots(3, 1, figsize=(8, 9))
            for ax, idx, label in zip(axes, range(3), (metric, "Frozen fixed-bank latent MSE", "Frozen held-out return")):
                ax.plot(severities, [r[idx] for r in series], "o-")
                ax.set_ylabel(label)
                ax.set_xticks(severities)
                ax.set_xlabel("Severity multiplier")
            nominal = next((r["applied_parameter_value"] for r in probes if float(r["severity_multiplier"]) == 1), "")
            nominal_label = f"{float(nominal):.5g}" if nominal else "unknown"
            fig.suptitle(f"{task} {parameter}: physical → latent → control; nominal parameter {nominal_label}")
            fig.tight_layout()
            for suffix in ("png",):
                fig.savefig(fig_dir / f"{task}_physical_latent_control_{parameter}.{suffix}", dpi=160)
            plt.close(fig)
    if q:
        rr = [r for r in q if r["metric"] == "entropy" and r["kind"] == "immediate" and numeric(r["value"]) is not None]
        grouped = defaultdict(list)
        for r in rr: grouped[r["method"]].append(float(r["value"]))
        if grouped:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(list(grouped), [mean(v) for v in grouped.values()])
            ax.set_ylabel("Categorical Q entropy (saved per-stream means)")
            ax.set_title("HalfCheetah immediate Q; descriptive, not calibration")
            fig.tight_layout()
            for suffix in ("png",):
                fig.savefig(fig_dir / f"halfcheetah_q_entropy.{suffix}", dpi=160)
            plt.close(fig)
        plans = [r for r in q if r["kind"] == "planning" and r["metric"] in ("reward_sum", "terminal_contribution")
                 and numeric(r["value"]) is not None]
        if plans:
            grouped = defaultdict(list)
            for r in plans: grouped[(r["method"], r["metric"])].append(float(r["value"]))
            fig, ax = plt.subplots(figsize=(7, 4))
            methods = sorted({k[0] for k in grouped})
            for i, metric in enumerate(("reward_sum", "terminal_contribution")):
                ax.bar([j + (i-.5)*.36 for j in range(len(methods))],
                       [mean(grouped.get((m, metric), [float("nan")])) for m in methods], .36,
                       label=metric)
            ax.set_xticks(range(len(methods)), methods)
            ax.set_ylabel("Native planning component (saved per-stream means)")
            ax.legend()
            fig.tight_layout()
            for suffix in ("png",):
                fig.savefig(fig_dir / f"halfcheetah_planning_components.{suffix}", dpi=160)
            plt.close(fig)
    if prediction_control:
        tasks = sorted({r["task"] for r in prediction_control})
        fig, axes = plt.subplots(1, len(tasks), figsize=(5.5*len(tasks), 4), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            for method in sorted({r["method"] for r in prediction_control if r["task"] == task}):
                rr = [r for r in prediction_control if r["task"] == task and r["method"] == method]
                ax.scatter([r["mse_improvement"] for r in rr],
                           [r["mean_return_improvement"] for r in rr], label=method, alpha=.7)
            ax.axhline(0, color="black", linewidth=.8)
            ax.axvline(0, color="black", linewidth=.8)
            ax.set_title(task)
            ax.set_xlabel("Frozen MSE − adapted MSE")
            ax.set_ylabel("Adapted return − Frozen return")
            ax.legend()
        fig.tight_layout()
        for suffix in ("png",):
            fig.savefig(fig_dir / f"prediction_vs_control.{suffix}", dpi=160)
        plt.close(fig)
    if normalized:
        rr = [r for r in normalized if float(r["severity_multiplier"]) != 1 and
              numeric(r["normalized_improvement"]) is not None]
        tasks = sorted({r["task"] for r in rr})
        fig, axes = plt.subplots(1, len(tasks), figsize=(5.5*len(tasks), 4), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            methods = sorted({r["method"] for r in rr if r["task"] == task})
            values = [[r["normalized_improvement"] for r in rr if r["task"] == task and r["method"] == m]
                      for m in methods]
            ax.boxplot(values, tick_labels=methods)
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_title(task)
            ax.set_ylabel("Δ return / |condition Frozen mean|")
        fig.tight_layout()
        for suffix in ("png",):
            fig.savefig(fig_dir / f"cross_task_normalized.{suffix}", dpi=160)
        plt.close(fig)
    for task in sorted({r["task"] for r in evaluations}):
        action = [r for r in evaluations if r["task"] == task and r["kind"] == "shifted" and
                  r["metric"] == "action_saturation" and numeric(r["value"]) is not None]
        if not action: continue
        grouped = defaultdict(dict)
        for r in action:
            key = (r["shift_parameter"], r["severity_multiplier"], r["method"])
            grouped[key][(r["adaptation_seed"], r["evaluation_seed"])] = float(r["value"])
        fig, ax = plt.subplots(figsize=(9, 4))
        for method in sorted({k[2] for k in grouped}):
            rr = sorted(((p, float(s), mean(values.values())) for (p, s, m), values in grouped.items() if m == method),
                        key=lambda x: (x[0], x[1]))
            by_parameter = defaultdict(list)
            for p, s, v in rr: by_parameter[p].append((s, v))
            for parameter, points in by_parameter.items():
                ax.plot([x[0] for x in points], [x[1] for x in points], "o-", label=f"{parameter}: {method}")
        force = any(r["task"] == task and r["metric"] in ("force_saturation", "actuator_force_saturation") and
                    numeric(r["value"]) is not None for r in evaluations)
        ax.set_title(f"{task} commanded action saturation; actuator force saturation " +
                     ("measured separately in source CSV" if force else "NOT_MEASURED"))
        ax.set_xlabel("Severity multiplier")
        ax.set_ylabel("Fraction of action components saturated")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        for suffix in ("png",):
            fig.savefig(fig_dir / f"{task}_action_effort.{suffix}", dpi=160)
        plt.close(fig)
    return notes


def report(output: Path, results: Path, selected: list[str], evaluations: list[dict], predictions: list[dict],
           physical: list[dict], q: list[dict], comparisons: list[dict], normalized: list[dict], reg: dict, notes: list[str]):
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    by_task = Counter(r["task"] for r in evaluations if r["metric"] == "episode_return")
    lines = ["# NS-Gym unified research analysis", "", "Saved records only; no simulator calls or training.", "",
             f"Canonical episode-return records: {dict(by_task)}. Fixed-bank MSE records: {len(predictions)}. "
             f"Physical summary records: {len(physical)}. Q diagnostic records: {len(q)}.", "",
             "Paired return differences use matched evaluation seeds. Confidence intervals resample adaptation-stream means; "
             "the same Frozen observations are shared across streams. Nominal ×1 family aliases are kept for condition coverage "
             "but not interpreted as independent physics tasks.", "",
             ("Reacher Dynamics+Q: completed-gate source included." if reg["reacher"]["dynamics_q_gate"]["passed"] else
              "Reacher Dynamics+Q: **IN_PROGRESS — EXCLUDED FROM CURRENT REPORT**. Local records do not include final PASS training, prediction and shifted-control evaluations."), ""]
    for task in selected:
        lines += [f"## {task}", ""]
        rr = [r for r in comparisons if r["task"] == task and float(r["severity_multiplier"]) != 1.0]
        for method in sorted({r["method"] for r in rr}):
            mm = [r for r in rr if r["method"] == method]
            lines.append(f"{method}: {len(mm)} physical conditions; mean of condition-level paired effects "
                         f"{mean(r['mean'] for r in mm):.6g}; improved conditions {sum(r['mean'] > 0 for r in mm)}/{len(mm)}. "
                         "This average is descriptive across different physical severities.")
        lines += [""]
        if SOURCES[task].get("adapter", task) == "halfcheetah":
            lines += ["The fixed primary checkpoint is 750k (SHA256 " + SOURCES[task]["checkpoint_hash"] + "). "
                      "A separate nominal-convergence audit provisionally selected 950k during training; "
                      "that selection is not substituted into this fixed-750k adaptation comparison.", ""]
        else:
            record = read_json(results / SOURCES[task]["checkpoint_record"]).get("record", {})
            nominal = record.get("nominal_performance", {})
            lines += [f"Selected nominal checkpoint: 670k, SHA256 {record.get('model_sha256', 'unknown')}; "
                      f"20 validation-seed mean return {nominal.get('mean', 'unavailable')}. "
                      "These checkpoint-selection seeds are distinct from the held-out adaptation evaluation seeds.", ""]
        pred = [r for r in predictions if r["task"] == task and r["method"] == "dynamics_only"]
        if pred:
            lines += [f"Dynamics-only fixed-bank prediction records: {len(pred)}; "
                      "prediction error and return are analyzed as separate outcomes.", ""]
    nominal_path = results / "nominal-convergence-audit-01/nominal_checkpoint_summary.json"
    if "halfcheetah" in selected and nominal_path.is_file():
        lines += ["## HalfCheetah nominal validation", ""]
        for row in read_json(nominal_path):
            lines.append(f"- Checkpoint {row['transitions']}: return {row['mean_return']:.6g}, sample SD {row['std_return']:.6g}; {row['episodes']} validation seeds.")
        lines += ["", "## Nominal retention", "",
                  "Reacher nominal-retention comparisons are in nominal_retention.csv, paired by evaluation seed and summarized by adaptation stream. "
                  "HalfCheetah shifted-adapted snapshots on nominal physics: NOT_EVALUATED. The k=1 adaptation condition does not replace that evaluation.", ""]
    lines += ["Q calibration and candidate regret: **NOT_MEASURED** without documented continuation returns.",
              "Joint encoder-changing latent MSE is excluded from fixed-encoder comparisons.",
              "Cross-task raw returns and latent MSE are not pooled; task horizons and reward scales differ.", ""]
    if notes: lines += ["Generation notes: " + "; ".join(notes), ""]
    (reports / "NSGYM_UNIFIED_RESEARCH_REPORT.md").write_text("\n".join(lines))
    for task in selected:
        src = [r for r in comparisons if r["task"] == task]
        text = [f"# {task} analysis", "", f"Completed paired comparison conditions: {len(src)}.", ""]
        for r in src:
            text.append(f"- {r['shift_parameter']} ×{r['severity_multiplier']} {r['method']}: "
                        f"Δ return {r['mean']:.5g}, {r['improved_streams']}/{r['n_adaptation_streams']} streams improved.")
        (reports / f"{task.upper()}_ANALYSIS.md").write_text("\n".join(text) + "\n")
    qlines = ["# Q distribution and planning", "", "Saved categorical Q entropy, JS, scalar Q, Bellman residual, "
              "candidate overlap and native planning components are in `canonical_q_diagnostics.csv`. "
              "The original distributional decoder in `scripts/q_planning_core.py` remains authoritative; "
              "these saved measurements are not re-decoded or treated as Q calibration.", ""]
    for parameter in ("bthigh_damping", "torso_mass"):
        for pair in ("F→D", "F→DQ", "F→J"):
            vals = [numeric(r["value"]) for r in q if r["shift_parameter"] == parameter and
                    float(r["severity_multiplier"]) == 2 and r["method"] == pair and r["metric"] == "js"]
            vals = [v for v in vals if v is not None]
            if vals:
                qlines.append(f"- {parameter} ×2, {pair} immediate JS: mean across {len(vals)} adaptation streams {mean(vals):.6g}.")
    qlines += ["", "Candidate overlap is a comparison of model score rankings on a shared proposal bank, "
               "not ranking accuracy. True continuation returns and planning regret: NOT_MEASURED.", ""]
    (reports / "Q_DISTRIBUTION_AND_PLANNING.md").write_text("\n".join(qlines))
    cross = ["# Cross-task comparison", "", "Each condition's paired return effect is divided by the magnitude "
             "of that condition's Frozen mean return. This is a within-task scale normalization, not a claim that "
             "equal physical multipliers have equal difficulty. Nominal ×1 aliases are excluded below.", ""]
    for task in selected:
        for method in ("dynamics_only", "dynamics_q", "joint"):
            vals = [r["normalized_improvement"] for r in normalized if r["task"] == task and
                    r["method"] == method and float(r["severity_multiplier"]) != 1 and
                    numeric(r["normalized_improvement"]) is not None]
            if vals:
                cross.append(f"- {task} {method}: {len(vals)} physical condition labels; "
                             f"mean normalized paired effect {mean(vals):.5g}, positive {sum(v > 0 for v in vals)}/{len(vals)}.")
    cross += ["", "Reacher Dynamics+Q: IN_PROGRESS — EXCLUDED FROM CURRENT REPORT. "
              "Raw returns, latent MSE coordinates and mass interventions are not pooled across tasks.", ""]
    (reports / "CROSS_TASK_ADAPTATION_COMPARISON.md").write_text("\n".join(cross))
    (reports / "MISSING_EVIDENCE_AND_LIMITATIONS.md").write_text(
        "# Missing evidence\n\n- Reacher Dynamics+Q final PASS training, held-out prediction and control: unavailable locally.\n"
        "- Q Monte Carlo continuation returns, calibration and true candidate regret: NOT_MEASURED.\n"
        "- Joint fixed-encoder prediction: incompatible because Joint changes the encoder.\n"
        "- Local HalfCheetah Joint evaluation raw stages are absent; the validated four-method derived comparison is the local source.\n"
        "- Server process state cannot be inspected from this local macOS workspace.\n")



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", choices=("analyze", "inventory", "validate", "cleanup"), default="analyze")
    parser.add_argument("--task", choices=(*SOURCES, "all"), default="all")
    parser.add_argument("--methods", default="frozen,dynamics_only,dynamics_q,joint")
    parser.add_argument("--analysis", default="full", choices=("full", "control", "physical", "latent", "q_distribution", "cross_task", "nominal", "frozen", "adaptation", "planning"))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--task-metrics", type=Path, default=Path(__file__).with_name("task_metrics.json"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    results = args.results_root.resolve()
    output = args.output.resolve()
    if not results.is_dir(): parser.error(f"Results root absent: {results}")
    config = read_json(args.config)
    task_metrics = read_json(args.task_metrics)
    task_metrics.update(config.get("task_metrics", {}))
    if config.get("version") != 1: parser.error("Unified analysis config version 1 required")
    selected = list(SOURCES) if args.task == "all" else [args.task]
    methods = set(args.methods.split(","))
    if not methods <= {"frozen", "dynamics_only", "dynamics_q", "joint"}:
        parser.error("Unknown method in --methods")
    if args.command == "cleanup":
        print("DRY RUN ONLY: no files deleted. See generated LOCAL_OUTPUT_CLEANUP.md after analyze.")
        return
    if output.exists(): parser.error(f"Output exists; choose a new versioned directory: {output}")
    if args.command != "inventory":
        for task in selected:
            try:
                available(results, task)
            except ValueError as exc:
                parser.error(str(exc))
            cfg = SOURCES[task]
            required = ["comparison", "physical"]
            if cfg.get("adapter", task) == "reacher":
                required += ["predictions", "episodes", "checkpoint_record"]
            if cfg.get("q_completion") and read_json(results / cfg["q_completion"]).get("status") == "PASS":
                required += ["q", "q_divergence", "planning", "ranking"]
            missing = [cfg[key] for key in required if not (results / cfg[key]).is_file()]
            if missing:
                parser.error(f"{task} source files missing: {', '.join(missing)}")
    if args.command == "validate":
        print("PASS: completed local source gates; Reacher Dynamics+Q remains excluded")
        return
    output.mkdir(parents=True)
    inv = inventory(results)
    write_csv(output / "ARTIFACT_INVENTORY.csv", inv,
              ("path", "stage", "task", "category", "role", "bytes", "completion_status", "sha256", "hash_status"))
    write_csv(output / "source_inventory.csv", inv,
              ("path", "stage", "task", "category", "role", "bytes", "completion_status", "sha256", "hash_status"))
    reg = registry(results, config)
    (output / "EXPERIMENT_REGISTRY.json").write_text(json.dumps(reg, indent=2, ensure_ascii=False))
    (output / "ARTIFACT_DEPENDENCIES.json").write_text(json.dumps({
        task: {"source_files": {k: v for k, v in SOURCES[task].items() if isinstance(v, str) and "/" in v},
               "derived_from": "historical completed saved records"} for task in selected}, indent=2))
    if args.command == "inventory":
        print(f"PASS inventory: {len(inv)} files; {output}")
        return
    evaluations, predictions, physical, q = [], [], [], []
    for task in selected:
        allowed = (methods | {"frozen"}) - ({"dynamics_q"} if task == "reacher" and not reg[task]["dynamics_q_gate"]["passed"] else set())
        e, p, ph, qq = ingest(results, task, allowed)
        evaluations.extend(e); predictions.extend(p); physical.extend(ph); q.extend(qq)
        if task == "reacher" and "dynamics_q" in methods and reg[task]["dynamics_q_gate"]["passed"]:
            dq_csv = results / config["reacher_dynamics_q"]["aggregate_per_seed_csv"]
            de, dp = ingest_completed_reacher_dq(results, dq_csv)
            evaluations.extend(de); predictions.extend(dp)
    if args.analysis == "physical":
        evaluations, predictions, q = [], [], []
    elif args.analysis == "control":
        predictions, physical, q = [], [], []
    elif args.analysis == "q_distribution":
        evaluations, predictions, physical = [], [], []
    elif args.analysis == "latent":
        physical, q = [], []
    elif args.analysis == "cross_task":
        physical, q = [], []
    if args.analysis == "nominal":
        evaluations = [r for r in evaluations if r['kind'] == 'nominal' or float(r['severity_multiplier']) == 1]
        predictions, physical, q = [], [], []
    elif args.analysis == "frozen":
        evaluations = [r for r in evaluations if r['method'] == 'frozen']
        predictions = [r for r in predictions if r['method'] == 'frozen']
        q = []
    elif args.analysis == "planning":
        evaluations, predictions, physical = [], [], []
        q = [r for r in q if r['kind'] in ('planning', 'ranking')]
    write_csv(output / "canonical_evaluations.csv", evaluations, COMMON)
    write_csv(output / "canonical_predictions.csv", predictions, COMMON)
    write_csv(output / "canonical_physical_probes.csv", physical, COMMON)
    write_csv(output / "canonical_q_diagnostics.csv", q, COMMON)
    stream, comparisons = paired_improvements(evaluations, methods)
    _, retention = paired_improvements(evaluations, methods, kind="nominal")
    write_csv(output / "nominal_retention.csv", retention, COMPARISON_FIELDS)
    prediction_pairs = prediction_improvements(predictions)
    write_csv(output / "per_adaptation_stream.csv", stream,
              ("task", "shift_family", "shift_parameter", "severity_multiplier", "adaptation_seed", "method",
               "evaluation_seeds", "mean_return_improvement"))
    write_csv(output / "comparisons.csv", comparisons, COMPARISON_FIELDS)
    write_csv(output / "prediction_improvements.csv", prediction_pairs,
              ("task", "shift_family", "shift_parameter", "severity_multiplier", "adaptation_seed",
               "evaluation_seed", "method", "frozen_mse", "adapted_mse", "mse_improvement"))
    conditions = sorted({(r["task"], r["shift_family"], r["shift_parameter"], r["severity_multiplier"], r["kind"])
                         for r in evaluations})
    write_csv(output / "canonical_conditions.csv",
              [dict(zip(("task", "shift_family", "shift_parameter", "severity_multiplier", "kind"), x)) for x in conditions],
              ("task", "shift_family", "shift_parameter", "severity_multiplier", "kind"))
    prediction_control = prediction_control_pairs(stream, prediction_pairs)
    normalized = normalized_comparisons(evaluations, comparisons)
    write_csv(output / "prediction_control_pairs.csv", prediction_control,
              ("task", "shift_family", "shift_parameter", "severity_multiplier", "adaptation_seed",
               "method", "evaluation_seeds", "mean_return_improvement", "mse_improvement", "prediction_seeds", "quadrant"))
    write_csv(output / "cross_task_normalized.csv", normalized,
              COMPARISON_FIELDS + ("frozen_mean_return", "normalized_improvement", "normalization"))
    notes = [] if args.no_plots else plot(output, evaluations, predictions, comparisons, physical, q, prediction_control, normalized, task_metrics)
    report(output, results, selected, evaluations, predictions, physical, q, comparisons, normalized, reg, notes)
    (output / "validation").mkdir()
    (output / "validation/completeness.json").write_text(json.dumps({
        "status": "PASS", "evaluation_records": len(evaluations), "prediction_records": len(predictions),
        "comparison_conditions": len(comparisons),
        "reacher_dynamics_q_included": bool(reg["reacher"]["dynamics_q_gate"]["passed"] and "reacher" in selected and "dynamics_q" in methods)}, indent=2))
    (output / "reports/EXPERIMENT_PROVENANCE.md").write_text(
        "# Provenance\n\nCanonical rows contain relative source file paths, SHA256 hashes and source row numbers. "
        "See `manifest.json`, `ARTIFACT_INVENTORY.csv` and `EXPERIMENT_REGISTRY.json`.\n")
    from .maintenance import cleanup_preview, migration_inventory
    cleanup_preview(results, output)
    migration_inventory(output)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORKSPACE,
                              capture_output=True, text=True, check=False).stdout.strip()
    (output / "manifest.json").write_text(json.dumps({
        "status": "PASS", "analysis": args.analysis, "tasks": selected, "methods": sorted(methods),
        "results_root": str(results), "project_root": str(PROJECT), "git_revision": revision,
        "analysis_config_sha256": sha256(args.config), "task_metrics_sha256": sha256(args.task_metrics),
        "python": platform.python_version(), "source_records": len(evaluations), "source_inventory_files": len(inv),
        "no_training_or_simulator": True}, indent=2))
    print(f"PASS: {len(evaluations)} evaluations, {len(predictions)} predictions, {len(comparisons)} paired conditions; {output}")

