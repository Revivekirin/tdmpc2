# Dynamics+Q: explicit server commands

Run on the current server project `/home/jhkim/repos/tdmpc2`. Change PROJECT only
if using a different mount; do not mix the old /nfs4 outputs with /home copies.
These commands never stop other processes and never automatically start the
next stage. Existing model/results remain read-only. Run each stage after the
previous stage succeeds. No server scientific run was executed by the local agent.

## 1. Environment and paths

```bash
conda activate tdmpc2-nsgym
PROJECT=/home/jhkim/repos/tdmpc2
cd "$PROJECT"
DQ=experiments/nsgym/scripts/run_dynamics_q.py
CHECKPOINT="$PROJECT/results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000"
REFERENCE="$PROJECT/results/nsgym/physical-severity-750k-v1"
TTT_RUNS="$PROJECT/results/nsgym/dynamics-only-750k-v1/runs"
TTT_FINAL="$PROJECT/results/nsgym/dynamics-only-750k-v1/final-snapshot-v2"
JOINT_FINAL="$PROJECT/results/nsgym/joint-final-evaluation-v1"
DQ_ROOT="$PROJECT/results/nsgym/dynamics-q-750k-v1"
DQ_RUNS="$DQ_ROOT/runs/fixed-decision"
GPU=0
COMMON=(--checkpoint "$CHECKPOINT" --reference "$REFERENCE" --ttt-runs "$TTT_RUNS" --gpu "$GPU")
test -f "$DQ" && test -f "$CHECKPOINT/model.pt"
python "$DQ" --help
mkdir -p "$DQ_ROOT/logs"
```

If deliberately sharing a GPU, append `COMMON+=(--allow-shared-gpu)` after
checking `nvidia-smi`. Otherwise the existing occupancy/free-memory guard applies.

## 2. Unit tests and one real bounded smoke

```bash
python -m unittest discover -s experiments/nsgym/tests -p test_dynamics_q.py -v
python -u "$DQ" smoke "${COMMON[@]}" --output "$DQ_ROOT/smoke-damping2"
cat "$DQ_ROOT/smoke-damping2/completion.json"
cat "$DQ_ROOT/smoke-damping2/bthigh_damping-k2-dynamics-q-seed42000001/parameter_invariance.csv"
```

Expected: real 100-transition damping ×2 smoke, 2 dynamics + 2 Q steps; only
dynamics/Q/target Q change. CPU development-only smoke has an explicit flag:
`python "$DQ" smoke --cpu-fixture --output /tmp/dq-cpu-fixture-01`.
The CPU fixture is not physical evidence and cannot satisfy the full-run gate.

## 3. Optional standalone full-budget single condition

The unit test already executes 3,000 synthetic transitions and verifies 118+118
steps. For a real 3,000-transition single-condition check, use this separate output:

```bash
python -u "$DQ" run "${COMMON[@]}" --smoke-result "$DQ_ROOT/smoke-damping2" \
  --parameter bthigh_damping --severity 2 --seed 42000001 \
  --output "$DQ_ROOT/single-damping2"
```

This standalone result is not silently merged into the independently launched
36-stream matrix. Do not treat it as an additional independent seed.

## 4. Explicit full adaptation launch

```bash
LOG="$DQ_ROOT/logs/adapt-$(date +%Y%m%d-%H%M%S).log"
nohup python -u "$DQ" run "${COMMON[@]}" \
  --smoke-result "$DQ_ROOT/smoke-damping2" --output "$DQ_RUNS" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$DQ_ROOT/logs/adapt.pid"
tail -n 50 -f "$LOG"
```

After disconnect, `tail` can be restarted without restarting training. To resume
a failed/incomplete matrix, use the identical command with `--resume` and a new
log name. Completed streams are hash-checked and skipped; partial streams restart
from nominal with the old files archived. PASS is idempotently skipped too.

```bash
python "$DQ" status --output "$DQ_RUNS"
```

## 5. Final snapshot audit

```bash
python -u "$DQ" audit "${COMMON[@]}" --runs "$DQ_RUNS" --output "$DQ_ROOT/audit"
cat "$DQ_ROOT/audit/completion.json"
AUDIT="$DQ_ROOT/audit/audit.json"
```

Audit requires 36/36 PASS, 3,000 transitions, 118 dynamics/118 Q/118 EMA updates,
frozen-component invariance, 3,000 aligned replay records, correct seeds, source
and checkpoint hashes. It emits the run inventory and parameter invariance table.

## 6. Held-out fixed-bank prediction

```bash
LOG="$DQ_ROOT/logs/prediction-$(date +%Y%m%d-%H%M%S).log"
nohup python -u "$DQ" evaluate "${COMMON[@]}" --audit "$AUDIT" \
  --mode prediction --output "$DQ_ROOT/prediction" > "$LOG" 2>&1 < /dev/null &
echo $! > "$DQ_ROOT/logs/prediction.pid"
tail -n 50 -f "$LOG"
```

## 7. Held-out shifted control (separate explicit launch)

After prediction has PASS status, launch shifted control explicitly:

```bash
LOG="$DQ_ROOT/logs/shifted-$(date +%Y%m%d-%H%M%S).log"
nohup python -u "$DQ" evaluate "${COMMON[@]}" --audit "$AUDIT" \
  --mode shifted --output "$DQ_ROOT/shifted" > "$LOG" 2>&1 < /dev/null &
echo $! > "$DQ_ROOT/logs/shifted.pid"
tail -n 50 -f "$LOG"
```

Prediction/shifted/nominal commands support `--resume` with the same output and
an unchanged audit. No frozen baseline or intermediate snapshots are re-evaluated.

## 8. Optional separate nominal retention

```bash
LOG="$DQ_ROOT/logs/nominal-$(date +%Y%m%d-%H%M%S).log"
nohup python -u "$DQ" evaluate "${COMMON[@]}" --audit "$AUDIT" \
  --mode nominal --output "$DQ_ROOT/nominal-retention" > "$LOG" 2>&1 < /dev/null &
echo $! > "$DQ_ROOT/logs/nominal.pid"
tail -n 50 -f "$LOG"
```

## 9. Aggregate and generate six figures

```bash
python "$DQ" aggregate --audit "$AUDIT" \
  --prediction "$DQ_ROOT/prediction" --shifted "$DQ_ROOT/shifted" \
  --ttt-final "$TTT_FINAL" --joint-final "$JOINT_FINAL" \
  --output "$DQ_ROOT/comparison"
cat "$DQ_ROOT/comparison/completion.json"
```

To include completed nominal retention, add `--nominal "$DQ_ROOT/nominal-retention"`
and use a new comparison directory. The first comparison does not require it.
Tables are in comparison/, six figures in comparison/figures/, and
DYNAMICS_Q_HELDOUT_RESULTS.md / DYNAMICS_Q_VS_TTT_JOINT.md in comparison/reports/.
Missing training logs are listed exactly rather than imputed. No command launches
another method, selects an intermediate best snapshot, or automatically evaluates
a just-completed adaptation matrix.
