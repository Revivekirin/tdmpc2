# Server commands: final 3,000-transition Decision TTT

Run in `/home/jhkim/repos/tdmpc2` (the inner TD-MPC2 project). The enclosing Git root on this server is `/home/jhkim/repos`. The new script must first be synced to `experiments/nsgym/scripts/evaluate_ttt_final.py` in the **inner project**. Use a fresh output suffix for reruns. These commands never start adaptation or Joint RL.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
FINAL_RUNNER=experiments/nsgym/scripts/evaluate_ttt_final.py
CHECKPOINT=/nfs4/jhkim/repos/tdmpc2/results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000
REFERENCE=/nfs4/jhkim/repos/tdmpc2/results/nsgym/physical-severity-750k-v1
RUNS=/nfs4/jhkim/repos/tdmpc2/results/nsgym/dynamics-only-750k-v1/runs
FINAL=/nfs4/jhkim/repos/tdmpc2/results/nsgym/dynamics-only-750k-v1/final-snapshot-v1
test -f "$FINAL_RUNNER" && test -f "$CHECKPOINT/model.pt"
python "$FINAL_RUNNER" --help
sha256sum "$CHECKPOINT/model.pt"
```

Expected checkpoint SHA256 is `f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b`.

1. **Read-only run/baseline audit.** No physics is run when manifests establish compatibility. A single 20-seed Frozen damping 2× reference check runs only if old/new source hashes differ.

```bash
python "$FINAL_RUNNER" audit \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --output "$FINAL/audit"
cat "$FINAL/audit/completion.json"
```

2. **Predefined 2× damping and 2× mass smoke.** Prediction only reads old hashed NPZ files; shifted control evaluates 6 final snapshots ×20 seeds ×1,000 steps =120,000 new environment transitions. Both stages include all three adaptation streams.

```bash
AUDIT="$FINAL/audit/audit.json"
python "$FINAL_RUNNER" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode prediction --subset smoke \
  --output "$FINAL/smoke-prediction"
python "$FINAL_RUNNER" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode shifted --subset smoke \
  --output "$FINAL/smoke-shifted"
cat "$FINAL/smoke-prediction/completion.json"
cat "$FINAL/smoke-shifted/completion.json"
```

A negative smoke performance result is still a valid PASS; it does not prevent full-grid evaluation. A missing or invalid result is PARTIAL and must be diagnosed.

3. **All 12 predefined conditions, final snapshot only.** Prediction covers 720,000 saved transitions and shifted control covers 720,000 environment transitions in total. The six smoke identities are reused, so this stage adds 600,000 prediction samples and 600,000 new shifted transitions. Each command uses a fresh output directory. Run separately; the second is the long GPU/MuJoCo job.

```bash
python "$FINAL_RUNNER" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode prediction --subset full \
  --smoke-prediction "$FINAL/smoke-prediction" \
  --smoke-shifted "$FINAL/smoke-shifted" \
  --output "$FINAL/full-prediction"

python "$FINAL_RUNNER" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode shifted --subset full \
  --smoke-prediction "$FINAL/smoke-prediction" \
  --smoke-shifted "$FINAL/smoke-shifted" \
  --output "$FINAL/full-shifted"
```

For a long shifted batch, use the same command after `nohup python -u`, redirect output to a fresh log, store `$!` as a PID and monitor with `tail -f`. If a batch ends `PARTIAL`, rerun its **same** command with `--resume` and the same `--output`: PASS identities are skipped, incomplete condition directories are preserved under a unique interrupted name, and failed conditions restart from their unchanged final snapshot. A completed PASS batch rejects `--resume`.

4. **Analyze primary prediction/control comparison.** This reads only completed per-seed CSVs and includes MISSING/PARTIAL conditions explicitly.

```bash
python "$FINAL_RUNNER" aggregate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --prediction "$FINAL/full-prediction" \
  --shifted "$FINAL/full-shifted" \
  --output "$FINAL/summary"
```

The summary contains `TTT_FINAL_SNAPSHOT_COMPARISON.md`, `ttt_final_prediction.csv`, `ttt_final_control.csv`, `ttt_final_vs_frozen.png`, `ttt_prediction_vs_control.png`, replica/condition summaries and failure records.

5. **Optional, separately budgeted nominal retention.** Run after the primary comparison if degradation under original dynamics matters for the decision. This is another 720,000 environment transitions and does not change the selected snapshot.

```bash
python "$FINAL_RUNNER" evaluate \
  --checkpoint "$CHECKPOINT" --runs "$RUNS" --reference "$REFERENCE" \
  --audit "$AUDIT" --mode nominal --subset full \
  --smoke-prediction "$FINAL/smoke-prediction" \
  --smoke-shifted "$FINAL/smoke-shifted" \
  --output "$FINAL/full-nominal"
```

To include retention, run `aggregate` into a **new** summary directory with `--nominal "$FINAL/full-nominal"` added. Every stage supports `--dry-run`; it prints the condition/seed budget without loading CUDA or running physics.
