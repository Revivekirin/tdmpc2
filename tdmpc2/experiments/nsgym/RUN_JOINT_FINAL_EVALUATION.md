# HalfCheetah Full Joint RL final-snapshot evaluation

Run from `/home/jhkim/repos/tdmpc2` in `tdmpc2-nsgym`. This evaluator never starts training, never alters a source snapshot or reference dataset, and does not automatically start the full 720-episode evaluation. The source Joint, Frozen and TTT stages must all have `PASS` completion records. The local desktop copy does not contain the Joint snapshots or final TTT evaluation; execute the artifact audit and CUDA smoke on server4.

```bash
cd /home/jhkim/repos/tdmpc2
conda activate tdmpc2-nsgym
EVAL=experiments/nsgym/scripts/evaluate_joint_final.py
ROOT=results/nsgym/joint-final-evaluation-v1
CHECKPOINT=results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000
RUNS=results/nsgym/full-joint-750k-v1/runs-fixed
REFERENCE=results/nsgym/physical-severity-750k-v1
TTT=results/nsgym/dynamics-only-750k-v1/final-snapshot-v2/shifted
mkdir -p "$ROOT"
python "$EVAL" audit --checkpoint "$CHECKPOINT" --runs "$RUNS" \
  --reference "$REFERENCE" --ttt-shifted "$TTT" --output "$ROOT/audit"
cat "$ROOT/audit/completion.json"
AUDIT="$ROOT/audit/audit.json"
```

The audit hashes all 36 final Joint snapshots and checks source model/environment files, original checkpoint and optimizer hashes, MPPI configuration, physical conditions, training/evaluation seed split, 3,000 interactions, 2,001 update calls and 4,002 optimizer step calls. It also validates the existing Frozen bank and the completed TTT shifted stage. It does not run CUDA or simulation.

```bash
python "$EVAL" smoke --checkpoint "$CHECKPOINT" --runs "$RUNS" \
  --reference "$REFERENCE" --ttt-shifted "$TTT" --audit "$AUDIT" \
  --output "$ROOT/smoke-damping2"
cat "$ROOT/smoke-damping2/completion.json"
```

The bounded smoke loads the bthigh damping ×2 final Joint snapshot twice and executes 24 shifted steps on the first predeclared held-out evaluation seed. Its partial return is **not** compared with a full Frozen episode. It checks actual parameter readback, shape, finiteness, snapshot reload, zero optimizer updates and unchanged model/optimizer/scale state. The exact Frozen-style full 1,000-step episode behavior is checked by the final evaluation.

Only after the audit and smoke pass, inspect current GPU use and the TTT evaluation progress. Then run the final stages explicitly. Each stage uses a fresh output directory. Re-run an interrupted stage with `--resume` using the same output directory; valid completed stream directories are skipped, and incomplete directories are retained under `interrupted-*` before retry.

```bash
nvidia-smi
PRED_LOG="$ROOT/prediction-$(date +%Y%m%d-%H%M%S).log"
nohup python -u "$EVAL" evaluate --checkpoint "$CHECKPOINT" --runs "$RUNS" \
  --reference "$REFERENCE" --ttt-shifted "$TTT" --audit "$AUDIT" \
  --mode prediction --output "$ROOT/prediction" \
  > "$PRED_LOG" 2>&1 < /dev/null &
echo $! > "$ROOT/prediction.pid"
tail -n 50 -f "$PRED_LOG"
```

The fixed-bank prediction stage has 36 × 20 × 1,000 one-step samples. It reports Joint latent self-consistency only. Joint changes its encoder; this MSE is **not** a representation-invariant improvement over Frozen or TTT. Check `"status": "PASS"` in `$ROOT/prediction/completion.json` before proceeding.

```bash
SHIFT_LOG="$ROOT/shifted-$(date +%Y%m%d-%H%M%S).log"
nohup python -u "$EVAL" evaluate --checkpoint "$CHECKPOINT" --runs "$RUNS" \
  --reference "$REFERENCE" --ttt-shifted "$TTT" --audit "$AUDIT" \
  --mode shifted --output "$ROOT/shifted" \
  > "$SHIFT_LOG" 2>&1 < /dev/null &
echo $! > "$ROOT/shifted.pid"
tail -n 50 -f "$SHIFT_LOG"
```

The shifted stage runs 36 × 20 full 1,000-step episodes (720,000 environment transitions maximum), seeds policy RNG per held-out evaluation seed, and pairs Joint returns with existing Frozen and TTT returns. No Frozen stage or intermediate snapshot is rerun. Check `"status": "PASS"` in `$ROOT/shifted/completion.json` before aggregation.

```bash
python "$EVAL" aggregate --checkpoint "$CHECKPOINT" --runs "$RUNS" \
  --reference "$REFERENCE" --ttt-shifted "$TTT" --audit "$AUDIT" \
  --prediction "$ROOT/prediction" --shifted "$ROOT/shifted" \
  --output "$ROOT/comparison"
cat "$ROOT/comparison/completeness.json"
```

The comparison writes all 720 per-seed paired rows, 36 independent adaptation-stream summaries, 12 condition summaries, the three reports and figures. It averages evaluation seeds *within* each adaptation stream before reporting variation among the three adaptation seeds. The `k=1` rows are also written to `nominal_k1_control.csv`; they describe control under nominal physics after adaptation, not full retention across all shifted training conditions. Actual Joint optimization cost is 2,001 update calls / 4,002 optimizer step calls per stream; TTT uses 118 update calls, while all three methods share 3,000 environment interactions.

If an evaluation stage was interrupted, use the same command with `--resume` and the existing stage path, redirecting to a new log name. Do not use `--resume` on the audit, smoke or aggregation stage.
