# Run Reacher R0–R3 separately

Execute in the actual server project directory containing both `experiments/` and `tdmpc2/`. In this installation the Git root is `/home/jhkim/repos`, but the project root is `/home/jhkim/repos/tdmpc2`. New implementation is in Git-relative `tdmpc2/experiments/...`, not the obsolete outer `experiments/...` tree. Preserve existing results. No command below retrains a nominal model. Each output directory must be new. Do not restore/merge entire directories just to run this pipeline.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RUNNER=experiments/nsgym/scripts/run_reacher_pipeline.py
CFG=experiments/nsgym/configs/reacher_pipeline_v1.json
OUT=/nfs4/jhkim/repos/tdmpc2/results/nsgym/reacher/preview-v1
SEARCH=/nfs4/jhkim/repos/tdmpc2
python "$RUNNER" --help
```

## R0: discover, then nominal validation

```bash
python "$RUNNER" discover --config "$CFG" --search-root "$SEARCH" --output "$OUT/discovery"
cat "$OUT/discovery/discovery.json"
```

If discovery is BLOCKED, inspect its reason and do not run nominal. The project-root search includes `results/`, `outputs/` and other possible training locations. For multiple runs, repeat into a new discovery directory with `--run /actual/discovered/run`, and use that new discovery JSON below. If no candidates exist, inspect actual training artifacts; the nominal loader requires `checkpoint-*/metadata.json`, `model.pt`, `optimizers.pt` and the run's `training.jsonl`. No checkpoint path is fabricated. The remote claim of 9.5M transitions must be checked against this manifest and full training audit.

```bash
python "$RUNNER" nominal --config "$CFG" --discovery "$OUT/discovery/discovery.json" --output "$OUT/nominal"
SELECTION="$OUT/nominal/selected_reacher_checkpoint.json"
cat "$SELECTION"
```

Review nominal reaching quality and training stability before proceeding. The selection rule uses only nominal validation, never shifted results. Candidate count is at least three; 20 shared validation episodes each.

## R1: runtime inventory and common-state physical/latent probes

```bash
python "$RUNNER" inventory --config "$CFG" --output "$OUT/inventory"
python "$RUNNER" probe --config "$CFG" --selection "$SELECTION" --inventory "$OUT/inventory" --output "$OUT/probe"
```

Inspect physical/latent figures, parameter readbacks, mass-only interpretation and per-seed/regime counts. All 24 valid conditions stay in the experiment.

## R2: full Frozen grid

```bash
python "$RUNNER" frozen --config "$CFG" --selection "$SELECTION" --probe "$OUT/probe" --output "$OUT/frozen"
```

This saves 420 unique 50-step episodes (21 unique conditions ×20 seeds); the four k=1 labels share one canonical baseline. Review complete Frozen results before TTT.

## R3: bounded real-checkpoint smoke and runtime budget

```bash
python "$RUNNER" smoke --config "$CFG" --selection "$SELECTION" --output "$OUT/smoke"
cat "$OUT/smoke/budget_protocol.json"
```

Smoke verifies two families ×two methods ×100 online transitions, real checkpoint reload/invariance and held-out replay rejection, plus all physical conditions. Full TTT requires real smoke PASS and a nonempty feasible budget. CPU `--synthetic` smoke is a software fixture only and cannot open this gate. Budget selection uses runtime, not return.

## Full fixed adaptation (explicit user launch)

```bash
python "$RUNNER" adapt --config "$CFG" --selection "$SELECTION" --frozen "$OUT/frozen" --smoke "$OUT/smoke" --method all --output "$OUT/adapt"
```

The full matrix is 24 conditions ×2 methods ×3 independent adaptation streams. At maximum budget this is 1,440,000 transitions. Snapshot budgets and final budget come from the saved runtime protocol. For staged individual launches, supported filters are `--family D0`, `--parameter joint0_damping`, `--severity 1.5`, `--method decision`; always use different output directories. No automatic exhaustive run or job submission occurs during setup.

There is no exact mid-stream resume in this Reacher runner. Preserve interrupted directories and start a new versioned output, or launch selected missing condition/method subsets. Do not mix completed duplicates in aggregation.

## Held-out prediction and control, all saved budgets

```bash
python "$RUNNER" predict --config "$CFG" --selection "$SELECTION" --frozen "$OUT/frozen" --runs "$OUT/adapt" --output "$OUT/prediction"
python "$RUNNER" control --config "$CFG" --selection "$SELECTION" --frozen "$OUT/frozen" --runs "$OUT/adapt" --output "$OUT/control"
```

Both also support `--snapshot /actual/path/snapshot-NNNNNNN.pt` instead of `--runs` for one snapshot. Prediction reuses hashed Frozen held-out data; control runs independent paired episodes plus nominal retention. Full six-budget control can require 34,560 episodes /1,728,000 environment transitions, so launch it separately. No updates occur during evaluation.

## Aggregation and plots

```bash
python "$RUNNER" aggregate --config "$CFG" --inputs "$OUT/probe" "$OUT/frozen" "$OUT/adapt" "$OUT/prediction" "$OUT/control" --output "$OUT/analysis"
```

Reports retain all 24 conditions and flag missing/incomplete ones. Do not include overlapping copies of evaluation directories. Aggregation uses evaluation-seed means within adaptation replicas and then uncertainty over independent adaptation replicas. AUC and best intermediate do not replace the final predeclared-budget primary outcome.

Every stage supports `--dry-run`, which prints arguments, full config, seeds, budgets and source/version manifest without loading checkpoint/CUDA or stepping physics. Seed sets and algorithm controls are specified by `--config`; no implicit seed substitution occurs.

```bash
python "$RUNNER" adapt --config "$CFG" --selection "$SELECTION" --frozen "$OUT/frozen" --smoke "$OUT/smoke" --output "$OUT/adapt" --dry-run
```

For nohup use a fresh log and output and add `python -u`; check exit/completion JSON rather than assuming `nohup` means success. A failure leaves its directory and failure.json for diagnosis; nothing is overwritten or silently deleted.
