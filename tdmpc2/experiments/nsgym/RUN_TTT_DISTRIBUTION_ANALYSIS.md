# HalfCheetah Frozen vs Dynamics-only TTT distribution analysis

Run from `/home/jhkim/repos/tdmpc2` in `tdmpc2-nsgym`. This CLI reads existing JSON/CSV only. It does not load a checkpoint, use CUDA, run MuJoCo, train, or evaluate an agent. Each output directory must be new.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
BASE=results/nsgym/dynamics-only-750k-v1
REF=results/nsgym/physical-severity-750k-v1
python -m experiments.nsgym.analysis --help
```

Analyze all existing complete prediction streams now. An incomplete shifted stage is recorded in `condition_completeness.csv` and excluded from control statistics.

```bash
python -m experiments.nsgym.analysis adaptation --format ttt \
  --runs "$BASE/runs" --final "$BASE/final-snapshot-v2" --reference "$REF" \
  --mode prediction --output "$BASE/analysis-distribution-v1"
cat "$BASE/analysis-distribution-v1/analysis_manifest.json"
```

Once `final-snapshot-v2/shifted/completion.json` says `PASS`, run control and the combined prediction–control analysis in one read-only pass. This command requires all 36 compatible prediction and control streams.

```bash
python -m experiments.nsgym.analysis adaptation --format ttt \
  --runs "$BASE/runs" --final "$BASE/final-snapshot-v2" --reference "$REF" \
  --mode full --output "$BASE/analysis-distribution-v2"
cat "$BASE/analysis-distribution-v2/analysis_manifest.json"
```

To regenerate reports when additional existing evaluation files have arrived, choose a fresh versioned output and use `--mode auto`. It includes control only when the shifted stage is complete; it never resumes or starts evaluation.

```bash
python -m experiments.nsgym.analysis adaptation --format ttt \
  --runs "$BASE/runs" --final "$BASE/final-snapshot-v2" --reference "$REF" \
  --mode auto --output "$BASE/analysis-distribution-v3"
```

`prediction_per_seed.csv` contains paired **evaluation-seed mean** MSEs from the same hashed 1,000-transition Frozen bank. Individual transition-level adapted errors were not saved; no transition-level improvement fraction or distribution is inferred. `prediction_distribution.csv` and `control_distribution.csv` summarize the three adaptation-stream means per condition, with exploratory stream-level bootstrap intervals. The 20 held-out episodes per stream describe conditional evaluation variability. `k=1` is nominal continual update, not an OOD shift. The saved TTT control results have no root/joint trajectory series, and this analysis does not infer actuator-force saturation from action-component saturation.

The primary prediction–control association excludes both families' `k=1` labels, leaving 30 shifted streams. Both nominal labels are retained in the condition tables and figures, but do not count as six independent nominal replicates. Spearman correlations use average ranks for tied values. Regenerate into a fresh output directory to apply these reporting corrections; no evaluation needs to be repeated. A nominal `k=1` adaptation experiment does not measure nominal retention of a model adapted at `k!=1`.
