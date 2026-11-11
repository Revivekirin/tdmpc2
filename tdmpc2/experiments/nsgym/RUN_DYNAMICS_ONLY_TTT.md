# Server execution commands

For the current **final 3,000-transition Decision versus existing Frozen** benchmark, use `TTT_BENCHMARK_STANDARD.md` and its single `ttt_benchmark.py` entry point. The all-snapshot evaluation loop below is the original broader protocol and is not required for that comparison.

Run manually in tdmpc2-nsgym from the inner project root `/home/jhkim/repos/tdmpc2`. The enclosing Git checkout root is `/home/jhkim/repos`; both paths exist on server4. Existing source results and checkpoints are read-only. Every output below must be new; choose a new suffix for reruns. The full evaluation budget is large (21.6million held-out transitions per schedule); commands do not run until you execute them.

## Location and CLI

```bash
cd /home/jhkim/repos/tdmpc2
RUNNER=experiments/nsgym/scripts/run_dynamics_only_ttt.py
CHECKPOINT=/home/jhkim/repos/tdmpc2/results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000
REFERENCE=/home/jhkim/repos/tdmpc2/results/nsgym/physical-severity-750k-v1
OUT=/home/jhkim/repos/tdmpc2/results/nsgym/dynamics-only-750k-v1
RUNS="$OUT/runs"
test -f "$RUNNER" && test -f "$CHECKPOINT/model.pt"
sha256sum "$CHECKPOINT/model.pt"
python "$RUNNER" --help
python experiments/nsgym/scripts/analyze_dynamics_only_ttt.py --help
python "$RUNNER" run --checkpoint "$CHECKPOINT" --schedule fixed --output "$OUT/fixed-preview" --dry-run
```

Expected SHA256: f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b. Do not cd into the inner tdmpc2 directory.

## Bounded smoke

Actual750k CUDA inference, full original planner, shortened environment only. Each invocation:2 parameters ×3 methods ×1 stream ×48=288 online transitions. Onset8 and horizon24 are **smoke-only**; full runs preserve250 and1000.

```bash
python "$RUNNER" smoke --checkpoint "$CHECKPOINT" --schedule fixed --output "$OUT/smoke-fixed"
python "$RUNNER" smoke --checkpoint "$CHECKPOINT" --schedule abrupt --output "$OUT/smoke-abrupt"
python "$RUNNER" evaluate --checkpoint "$CHECKPOINT" \
  --snapshot "$OUT/smoke-abrupt/bthigh_damping-k1.5-decision-seed42000001/snapshot-0000048.pt" \
  --output "$OUT/smoke-heldout"
```

For CPU implementation validation only, append `--synthetic` to smoke and evaluation; these results do not establish750k performance. Independent bounded integration tests:

```bash
python -m unittest discover -s experiments/nsgym/tests -p test_dynamics_only_ttt.py -v
```

## Frozen baseline

```bash
python "$RUNNER" run --checkpoint "$CHECKPOINT" --method frozen --schedule fixed --output "$RUNS/fixed-frozen"
python "$RUNNER" run --checkpoint "$CHECKPOINT" --method frozen --schedule abrupt --output "$RUNS/abrupt-frozen"
```

## Resume an interrupted stage

If the stage output exists but has no top-level `completion.json`, rerun the same command with `--resume`. Completed condition directories with `PASS`, matching configuration, checkpoint and unchanged model/adapter sources are skipped. An incomplete condition is moved intact to `OUT/interrupted/` and that condition restarts from the original 750k checkpoint; a partial episode is never continued. This keeps the analysis input free of incomplete duplicates.

```bash
python "$RUNNER" run --checkpoint "$CHECKPOINT" --method frozen --schedule abrupt \
  --output "$RUNS/abrupt-frozen" --resume
```

Do not use `--resume` after the stage's top-level completion is `PASS`.

## Fixed severity online adaptation

```bash
python "$RUNNER" run --checkpoint "$CHECKPOINT" --method decision --schedule fixed --output "$RUNS/fixed-decision"
python "$RUNNER" run --checkpoint "$CHECKPOINT" --method episode --schedule fixed --output "$RUNS/fixed-episode"
```

## Abrupt online adaptation

```bash
python "$RUNNER" run --checkpoint "$CHECKPOINT" --method decision --schedule abrupt --output "$RUNS/abrupt-decision"
python "$RUNNER" run --checkpoint "$CHECKPOINT" --method episode --schedule abrupt --output "$RUNS/abrupt-episode"
```

## Held-out evaluation, all predefined snapshots

This explicitly launches the large held-out stage. Subshell stops on any failure. Results are written outside each adaptation run. A partially completed evaluation stage needs a new destination or manual selection of not-yet-evaluated snapshots; no silent overwrite/resume.

```bash
(
set -eu
for schedule in fixed abrupt; do
  if [ "$schedule" = fixed ]; then DATA="$REFERENCE/frozen"; else DATA="$REFERENCE/nonstationary-abrupt"; fi
  for method in frozen decision episode; do
    for RUN in "$RUNS/$schedule-$method"/*; do
      [ -d "$RUN" ] || continue
      NAME="${RUN##*/}"
      for SNAP in "$RUN"/snapshot-*.pt; do
        TAG="${SNAP##*/}"
        python "$RUNNER" evaluate --checkpoint "$CHECKPOINT" \
          --snapshot "$SNAP" --fixed-data "$DATA" \
          --output "$OUT/evaluations/$schedule/$NAME/${TAG%.pt}"
      done
    done
  done
done
)
```

For a single snapshot, use the same evaluate command with its explicit path. Fixed-data is optional only for smoke; without it MSE is labelled NOT_MEASURED. It is not substituted with closed-loop prediction error.

## Aggregate and plot

```bash
python experiments/nsgym/scripts/analyze_dynamics_only_ttt.py \
  --evaluations "$OUT/evaluations" --runs "$RUNS" --output "$OUT/analysis"
```

Outputs include per_seed_evaluation.csv, adaptation_seed_means.csv, seed_summary.csv, paired improvement and nominal retention curves, severity_return_improvement.png, interaction_gradient_cost.png, held-out shift_aligned plots, and separate online_episode_* plots. Empty panels indicate missing conditions; do not claim a complete sweep from partial output. Synthetic and actual-checkpoint evaluations cannot be mixed.
