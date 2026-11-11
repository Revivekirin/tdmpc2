# Final-snapshot Dynamics-only TTT evaluation

This is the single HalfCheetah evaluation entry point: `scripts/evaluate_ttt_final.py`. It reads the existing 750k, fixed-Frozen and fixed-Decision runs. It does not start adaptation, run a smoke stage, or evaluate intermediate snapshots. Run from the inner project root on server4:

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
git pull --ff-only origin main

EVAL=experiments/nsgym/scripts/evaluate_ttt_final.py
CHECKPOINT=/home/jhkim/repos/tdmpc2/results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000
REFERENCE=/home/jhkim/repos/tdmpc2/results/nsgym/physical-severity-750k-v1
RUNS=/home/jhkim/repos/tdmpc2/results/nsgym/dynamics-only-750k-v1/runs
OUT=/home/jhkim/repos/tdmpc2/results/nsgym/dynamics-only-750k-v1/final-snapshot-v2

test -f "$EVAL" && test -f "$CHECKPOINT/model.pt" && \
test -f "$REFERENCE/frozen/episodes.csv" && \
test -f "$REFERENCE/heldout-latent/latent_errors.csv" && \
test -f "$RUNS/fixed-frozen/completion.json" && \
test -f "$RUNS/fixed-decision/completion.json"
sha256sum "$CHECKPOINT/model.pt"
python -m unittest experiments/nsgym/tests/test_dynamics_only_ttt.py experiments/nsgym/tests/test_ttt_final.py -q
```

Expected checkpoint SHA256: `f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b`. `OUT` must be new. Version `v2` avoids mixing outputs from the earlier smoke-dependent CLI. If a stage is already `PASS`, do not rerun it. Only a `PARTIAL` or interrupted `evaluate` stage may use `--resume` with its original output path.

If the audit is already `PASS` but an evaluation stage is `PARTIAL`, keep the audit and resume the same stage after updating the code. First verify that no earlier evaluator process is still running:

```bash
AUDIT="$OUT/audit/audit.json"
python "$EVAL" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode prediction --output "$OUT/prediction" --resume
nohup python -u "$EVAL" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode shifted --output "$OUT/shifted" --resume \
  > "$OUT/shifted-resume.log" 2>&1 < /dev/null &
echo $! > "$OUT/shifted-resume.pid"
tail -n 50 -f "$OUT/shifted-resume.log"
```

```bash
# 1. Audit all 36 Frozen and 36 Decision streams and existing Frozen reference.
python "$EVAL" audit \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --output "$OUT/audit"
cat "$OUT/audit/completion.json"

AUDIT="$OUT/audit/audit.json"

# 2. Full predefined grid: 12 conditions × 3 streams × 20 seeds × 1,000 saved transitions.
python "$EVAL" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode prediction --output "$OUT/prediction"
cat "$OUT/prediction/completion.json"

# 3. Full shifted control: 720,000 new environment transitions.
mkdir -p "$OUT"
nohup python -u "$EVAL" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode shifted --output "$OUT/shifted" \
  > "$OUT/shifted.log" 2>&1 < /dev/null &
echo $! > "$OUT/shifted.pid"
tail -n 50 -f "$OUT/shifted.log"
```

After the shifted job finishes:

```bash
cat "$OUT/shifted/completion.json"
python "$EVAL" aggregate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --prediction "$OUT/prediction" \
  --shifted "$OUT/shifted" --output "$OUT/summary"
cat "$OUT/summary/completion.json"
```

The summary contains the per-seed prediction/control CSVs, 12-condition coverage, paired Frozen comparisons and figures. It averages held-out seeds within each adapted stream and summarizes across the three streams. A lower latent MSE alone is not evidence of improved control. Optional nominal retention is a separate `evaluate --mode nominal --output "$OUT/nominal"` run, followed by `aggregate` into a new summary directory with `--nominal "$OUT/nominal"` added. It costs another 720,000 environment transitions and is not required for the primary comparison.

For a new task or benchmark, retain this stage order and output contract (`audit.json`, per-identity and stage `completion.json`, per-seed prediction/control CSVs, complete condition grid). Its task-specific evaluator must validate the new checkpoint, task dimensions, simulator, condition grid, seed split, horizon, baseline compatibility and adaptation scope. HalfCheetah constants in this evaluator must not be reused unchanged for Reacher.
