# Reacher Dynamics+Q: explicit stages, no automatic full run

Run from `/home/jhkim/repos/tdmpc2` on server4 in `tdmpc2-nsgym` after copying the new code/config/tests.
Existing `run_dynamics_q.py`, `common/dynamics_q.py` and Reacher environment/model code are unchanged.
`run_reacher_dynamics_q.py` binds the shared adapter to Reacher and reuses `reacher_pipeline` for load,
physics, MPPI actions and held-out episodes. This avoids invalidating existing HalfCheetah source-hash gates.

## Setup and metadata inventory

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RUNNER=experiments/nsgym/scripts/run_reacher_dynamics_q.py
REF=/home/jhkim/repos/tdmpc2/results/nsgym/reacher/preview-v1
SELECTION="$REF/nominal-home-01/selected_reacher_checkpoint.json"
CHECKPOINT=/home/jhkim/repos/tdmpc2/results/nsgym/reacher-full-01/checkpoint-000670000
OUT=/home/jhkim/repos/tdmpc2/results/nsgym/reacher-dynamics-q-v1
python "$RUNNER" --help
python "$RUNNER" inventory --reference "$REF" --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --output "$OUT/inventory-server"
```

Inventory verifies original completed baseline identities/budgets and nominal origins. On the local evidence
bundle (without model files) inventory is metadata-only; server real stages additionally require model and optimizer
bytes, check the selected hash, and enforce original package versions using the unchanged loader.
No edits to existing Frozen/Dynamics-only/Joint results are required.

## Bounded parameter/target tests

```bash
python -m unittest discover -s experiments/nsgym/tests -p test_dynamics_q.py -v
python -m unittest discover -s experiments/nsgym/tests -p test_reacher_dynamics_q.py -v
python "$RUNNER" smoke --cpu-fixture --output "$OUT/cpu-fixture-server"
```

Synthetic fixture evidence is never sufficient for the full run gate. Tests exercise actual WorldModel,
categorical Q loss, true-terminal/TimeLimit target semantics, EMA, replay/reset boundaries, optimizer ownership,
held-out mutation blocking and serialization. The fixture uses 10 observations, 2 actions, gamma .95 and two
50-transition episodes. It is not selected-checkpoint/CUDA evidence.

## One-condition real checkpoint smoke (100 transitions)

```bash
python -u "$RUNNER" smoke --reference "$REF" --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --family D0 --severity 1.5 --seed 53000001 \
  --gpu 0 --output "$OUT/smoke-real"
cat "$OUT/smoke-real/completion.json"
```

Expected 2 dynamics +2 Q optimizer steps (75,100); two episode boundaries. Fixed physics is checked after resets
and steps, all losses finite, only dynamics/Q/target-Q changed, and serialization is verified. Real loading,
MPPI actions and inference after the first update are exercised. Use `--allow-shared-gpu` only deliberately;
it does not change model or planning settings.

## Preview budgets, then optionally one full stream

```bash
python "$RUNNER" run --dry-run --output "$OUT/runs"
python -u "$RUNNER" run --reference "$REF" --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --smoke-result "$OUT/smoke-real" \
  --family D0 --severity 1.5 --seed 53000001 --gpu 0 \
  --output "$OUT/pilot"
python -u "$RUNNER" audit --reference "$REF" --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --runs "$OUT/pilot" \
  --family D0 --severity 1.5 --seed 53000001 --gpu 0 --output "$OUT/pilot-audit"
```

The pilot is optional and adds 10,000 interactions beyond the matrix budget. It is not silently counted as a
second replicate or copied into the matrix. Full matrix: 24 conditions ×3 adaptation seeds =72 streams,
720,000 interactions, 28,656 dynamics steps +28,656 Q steps and 28,656 target EMAs. Counts derive from baseline
config and scheduler: floor(T/25)−floor((max(warmup,batch)−1)/25). Baseline metadata is checked at execution.
Each control evaluation mode has 1,440 episodes /72,000 environment transitions. Prediction only reads
72,000 existing bank transitions across 72 models. No wall-clock estimate is claimed before real smoke timing.

## Full sweep — launch explicitly only after real smoke PASS

```bash
(
  set -eu
  test ! -e "$OUT/runs"
  test ! -e "$OUT/runs.log"
  nohup python -u "$RUNNER" run --reference "$REF" --selection "$SELECTION" \
    --checkpoint "$CHECKPOINT" --smoke-result "$OUT/smoke-real" \
    --gpu 0 --output "$OUT/runs" > "$OUT/runs.log" 2>&1 < /dev/null &
  echo $! > "$OUT/runs.pid"
) && tail -n 50 -f "$OUT/runs.log"
```

Use `--resume` with the identical command to skip verified PASS children and archive/restart failed children.
This is not exact mid-stream resume: environment/planner state is not serialized. Keep failed evidence.
Do not run a second copy against the same output directory; use `ps` to check before any restart.

## Final snapshot audit

```bash
python -u "$RUNNER" audit --reference "$REF" --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --runs "$OUT/runs" --gpu 0 --output "$OUT/audit"
```

Reloads each model and optimizer/replay snapshot, verifies fixed-component invariance against the nominal
model, snapshot hashes, scheduled counters, replay shapes and boundaries. Each evaluation also reloads/checks
its snapshot; the explicit audit is useful before paying the full held-out cost.

## Held-out prediction, then shifted control, then nominal retention

```bash
python -u "$RUNNER" evaluate --mode prediction --reference "$REF" --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --runs "$OUT/runs" --gpu 0 --output "$OUT/prediction"
python -u "$RUNNER" evaluate --mode shifted --reference "$REF" --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --runs "$OUT/runs" --gpu 0 --output "$OUT/control"
python -u "$RUNNER" evaluate --mode nominal --reference "$REF" --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --runs "$OUT/runs" --gpu 0 --output "$OUT/nominal-retention"
```

Prediction requires original Frozen NPZ files and verifies each SHA256. All evaluations have no optimizer
updates and no replay writes. Nominal retention evaluates the shifted-adapted model at k=1; it is not adaptation
at k=1. For one-model evaluation, supply the same `--family --severity --seed` triple and a distinct output.

## Four-way aggregation and figures

```bash
python "$RUNNER" aggregate --reference "$REF" --runs "$OUT/runs" \
  --prediction "$OUT/prediction" --control "$OUT/control" \
  --nominal "$OUT/nominal-retention" --output "$OUT/comparison"
```

Omit `--nominal` if not evaluated. The aggregate requires all 72 streams and all 20 held-out seeds; it fails on
incomplete/mismatched records rather than interpolating. It reads the completed original Dynamics-only/Joint
CSV stages directly. Prediction bank hashes match across methods. Individual evaluation-seed differences are
separate from three adaptation-stream means; CI is descriptive stream bootstrap conditional on the evaluation
bank. Nominal family aliases are not pooled as independent environments. Q entropy is not calibrated uncertainty.

## Local-review ZIP

```bash
python "$RUNNER" package --runs "$OUT" --output "$OUT.zip"
```

Uses a new ZIP path; excludes .pt, .npz, training JSONL and large per-step CSVs, retains summaries/reports,
figures, manifests and per-seed results. Prints absolute path, SHA256, byte size and file count; checks ZIP CRC.
No upload or transfer is automatic. Existing output paths cause an error rather than overwrite.
