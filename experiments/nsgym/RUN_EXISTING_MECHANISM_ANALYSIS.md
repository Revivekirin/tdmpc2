# Task B: existing HalfCheetah mechanism analysis

One CLI reads completed Frozen/TTT/Joint results and creates reports, tables, figures and a review ZIP. No environment is constructed or stepped. No optimizer is constructed or stepped. It does not depend on Dynamics+Q or Reacher.

Run on server4 from the project root containing `experiments/` and `tdmpc2/`. All paths below use `/home/jhkim/repos/tdmpc2`.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
git pull --ff-only origin main

BASE=/home/jhkim/repos/tdmpc2/results/nsgym
OUT="$BASE/mechanism-analysis-750k-v1"
LOG="$BASE/mechanism-analysis-750k-$(date +%Y%m%d-%H%M%S).log"

nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u \
  experiments/nsgym/scripts/analyze_existing_mechanisms.py \
  --results-root "$BASE" --output "$OUT" \
  --inference --device cpu \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$BASE/mechanism-analysis-750k.pid"
tail -n 50 -F "$LOG"
```

CPU is the default to avoid consuming GPU memory used by other experiments. `--device cuda` is available when GPU resources are free. Both modes only execute model forward passes. No separate smoke or evaluation is required.

The bounded inference uses the first two **already declared** held-out seeds per condition, with128 evenly spaced saved transitions per bank. All36 final adapted snapshots are included. This is3072 labelled source transitions reused across methods/streams, up to27648 forward transition measurements across Frozen/TTT/Joint. Each model also scores32 identical3-step action tapes per stream, extracted from existing Frozen actions and transplanted to the same initial observation. These are predicted ranking comparisons, not simulator-return rankings. Nothing selects new severities or a best snapshot.

The main control comparison still uses all720 completed held-out episode rows and all12 condition IDs. Its uncertainty uses three adaptation stream means, not60 independent episodes. It preserves k=1 under both labels but excludes both from shifted-only association calculations.

After completion:

```bash
cat "$OUT/provenance/completion.json"
cat "$OUT/reports/PRIMARY_CONTROL_FINDINGS.md"
cat "$OUT/reports/REWARD_Q_DIAGNOSTICS.md"
cat "$OUT/reports/PLANNING_MISMATCH.md"
sha256sum "$OUT.zip"
unzip -t "$OUT.zip"
```

The final log JSON prints the absolute ZIP path, file count, bytes and SHA256. The ZIP includes derived tables and candidate scores, reports, figures, source fingerprints and the two analysis scripts. It excludes checkpoints, optimizer states, raw training logs and trajectories. No transfer is performed.

For a CSV/log-only analysis, omit `--inference`. Reward/Q/representation inference is then explicitly NOT_MEASURED and corresponding empty figures are not created. Use a fresh versioned output for every rerun; existing output and ZIP files are never overwritten.

Required sources are the completed raw `final-snapshot-v2/{prediction,shifted}`, `joint-final-evaluation-v1/{prediction,shifted,audit}`, Frozen baseline/bank, `full-joint-750k-v1/runs-fixed` logs and manifests. Forward inference additionally requires nominal `model.pt`, final Decision snapshots and final Joint snapshots. It constructs `WorldModel` directly, strictly loads saved target Q state, uses the actual two-hot decoder, and never calls the original agent constructor, which would create optimizers.

Q calibration against empirical return remains NOT_MEASURED: the saved Frozen continuation is not a valid on-policy action-matched target for all adapted Q functions. Actual candidate ranking/regret also remains NOT_MEASURED. No supported existing CLI computes true candidate returns, so no speculative simulator command is supplied. Joint self-consistency uses its own encoder and is kept separate from Frozen/TTT fixed-encoder MSE.

Local validation: CSV-only contract tests include the actual distinct Joint/TTT child layouts, all36-stream/720-episode fixture aggregation and ZIP CRC checks. Real Frozen planning traces were checked against `score=reward_sum+discounted_terminal_q`. Native snapshot inference needs the server's installed Torch/TensorDict environment and saved models; it was not executed on the desktop.
