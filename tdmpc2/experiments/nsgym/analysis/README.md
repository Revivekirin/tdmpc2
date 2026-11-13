# NS-Gym saved-artifact analysis

The authoritative postprocessing entry point is `python -m experiments.nsgym.analysis` from the executable project root. The local checkout's executable root is `/Users/jihyekim/Desktop/tdmpc2/tdmpc2`; server root is `/home/jhkim/repos/tdmpc2`.

**Migration status: PARTIAL.** Five independent legacy implementations have moved here and their old scripts are removed. Two identical outer-checkout copies are also removed. Q/planning's measurement runner delegates its summary statistics here. Runtime-imported analysis modules remain protected while Reacher Dynamics+Q is active; this package does not claim complete replacement of those paths.

## Commands

Use new output directories. These commands read saved artifacts; they do not run an environment, model inference or training. `validate` performs numerical regression, not a GPU smoke experiment.

```bash
cd /Users/jihyekim/Desktop/tdmpc2/tdmpc2
PY=/private/tmp/tdmpc2-nsgym-venv/bin/python
RESULTS=/Users/jihyekim/Desktop/tdmpc2/results/nsgym
OUT="$RESULTS/unified-analysis/refactor-v2"
export MPLCONFIGDIR=/private/tmp/mpl-nsgym

"$PY" -m experiments.nsgym.analysis full \
  --source-root "$RESULTS" --output "$OUT"

"$PY" -m experiments.nsgym.analysis validate \
  --source-root "$RESULTS" --unified-output "$OUT" \
  --output "$OUT/regression"
```

`refactor-v1` already exists and is the verified result from this migration. The version above is deliberately new. On server use `PY=python` inside `tdmpc2-nsgym`, update `RESULTS`, and configure transferred-source paths in `config.json` before execution. The current default source paths describe the local transferred artifacts; they are not guaranteed to exist on server.

Operations: `inventory`, `validate`, `nominal`, `physical`, `frozen`, `adaptation`, `q_distribution`, `planning`, `cross_task`, `full`, `cleanup_preview`. All accept explicit source/output roots. `--no-plots` generates tables/reports without figures. `validate` currently covers the registered HalfCheetah and Reacher regression corpus together; it is not a generic validator for arbitrary new task schemas.

```bash
"$PY" -m experiments.nsgym.analysis adaptation task=halfcheetah \
  method=frozen,dynamics_only,dynamics_q,joint \
  --source-root "$RESULTS" --output "$OUT-halfcheetah"

"$PY" -m experiments.nsgym.analysis cleanup_preview \
  --source-root "$RESULTS" --output "$OUT-cleanup"
```

A Frozen reference is loaded for pairing even when only an adapted method is selected. No unrelated adapted method is included. A requested missing method is shown as unavailable in the registry. Reacher Dynamics+Q requires complete seed grids, matched snapshot hashes, child CSV hashes and PASS training/prediction/control/aggregate records; four PASS markers or an empty CSV alone cannot admit it.

## Historical input adapters

The former scripts are not shims left in `scripts/`. Their original calculations and provenance checks live in these modules and are dispatched through the same CLI:

| Removed entry | Adapter | Original arguments forwarded |
|---|---|---|
| analyze_ttt_distribution.py | `adaptation --format ttt` | `--runs --final --reference --mode` |
| analyze_dynamics_only_ttt.py | `adaptation --format budgets` | `--evaluations --runs` |
| refine_severity_analysis.py | `physical --format severity` | `--run` |
| analyze_reacher_research.py | `adaptation --format reacher` | `--source --raw-source` |
| inspect_dynamics_q_results.py | `adaptation --format diagnostics` | `--comparison --runs` |

For example, help for an adapter (no files created):

```bash
"$PY" -m experiments.nsgym.analysis physical --format severity \
  --output /private/tmp/example-unused -- --help
```

Arguments after `--` go to the historical parser; unrecognized pre-separator adapter flags also forward for compatibility with updated run documents. Historical adapters require their original inputs: a transferred summary cannot substitute for missing raw-stage manifests or training logs.

## Configuration and statistics

The existing analysis JSON config is retained. The runtime training Hydra configuration is not imported into this read-only process. `config.json` registers source schemas, paths, methods and historical physical contracts; `task_metrics.json` defines task-specific measured quantities. A new task using an existing source schema can register an entry with `adapter=halfcheetah` or `adapter=reacher`, its paths, checkpoint/environment metadata and task metrics. A different source schema requires a small ingestion adapter, not copied aggregation/plot functions. A synthetic alias registration was exercised; no new benchmark result is claimed.

`statistics.py` owns percentile bootstrap with explicit historical policies: HalfCheetah 20261111/10,000; Reacher research 20261112/5,000; budget/nominal 90210/10,000; Q 817/2,000. `schema.py` owns CSV/JSON/hash I/O. `q_statistics.py` owns observation/head → evaluation-seed → adaptation-stream aggregation. `metrics.py` owns paired comparisons. Original per-format provenance checks remain in their adapters.

Frozen evaluations are shared references. Adaptation streams, not time steps or all 60 paired episodes, are the uncertainty unit for adapted comparisons. Nominal family aliases stay labeled and are excluded from cross-condition pooled claims. Changed-encoder Joint MSE is not combined with fixed-encoder MSE. Candidate overlap is not candidate-return accuracy; calibration/regret remain unavailable without ground truth.

## Verification and limits

`regression.py` checks all conditions of the registered completed comparison, Q, planning and physical-refinement tables, including raw-to-summary paths and optional unified-output agreement. `REGRESSION_COMPARISONS.csv` distinguishes exact/tolerance/failure. `UNIFIED_REGRESSION_REPORT.md` lists uncovered evidence. Regenerated physical figures are not evidence of new simulations; this refactor only reuses saved records.

Retained runtime dependencies: `scripts/analyze_dynamics_q.py`, `scripts/analyze_reacher_pipeline.py`, `scripts/analyze_existing_mechanisms.py`; also `severity_analysis.py` and runtime plotting/evaluation code. Their remaining aggregation paths are a migration gap. Training JSONL convergence, all adaptation budgets and Reacher raw physical probes are not fully covered by the current regression corpus. Full raw Reacher regeneration needs the original bundle, which is no longer in the registered local paths.

Run analysis/unit/Q tests from the workspace with the project's core module directory on `PYTHONPATH`; the local test environment also requires `/private/tmp/q-planning-deps`. Physical tests additionally require the tested NS-Gym revision with `ns_gym.schedulers`; the locally found upstream checkout lacks that API. No dependency installation or remote process change was performed here.

No result archive is deleted. Cleanup is a read-only exact-path manifest; only byte-identical complete directory copies can receive a proposed deletion command. Prior scientific results, checkpoints, and active Reacher D+Q artifacts remain intact.
