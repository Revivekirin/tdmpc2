# NS-Gym raw-to-report analysis

Run from the executable project root (the directory containing `experiments/` and `tdmpc2/`). On this Mac that is `/Users/jihyekim/Desktop/tdmpc2/tdmpc2`; the user-provided server result copy is `/Users/jihyekim/Desktop/tdmpc2/results/nsgym`. Server commands below use `/home/jhkim/repos/tdmpc2`.

`config.json` registers task/environment/schema semantics and methods, not historical run directories. The analyzer recursively discovers metadata and completion records under `--source-root`. CSV column schemas select adapters. Existing analysis output directories are excluded by provenance markers. A new run of a supported schema requires no config edit and no Mac-to-server archive.

## Execution

Use a new output directory for each invocation. These commands only read saved experiment data; they do not load model checkpoints, simulate, train, or alter source files.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RESULTS="$PWD/results/nsgym"
OUT="$RESULTS/unified-analysis/raw-to-report-v2"
export MPLCONFIGDIR=/tmp/mpl-nsgym

python -m experiments.nsgym.analysis discover --source-root "$RESULTS" --output "${OUT}-discovery"
python -m experiments.nsgym.analysis validate --source-root "$RESULTS" --output "${OUT}-validation" --no-plots
python -m experiments.nsgym.analysis full --source-root "$RESULTS" --output "$OUT"
python -m experiments.nsgym.analysis full --task halfcheetah --source-root "$RESULTS" --output "${OUT}-halfcheetah"
python -m experiments.nsgym.analysis full --task reacher --source-root "$RESULTS" --output "${OUT}-reacher"
python -m experiments.nsgym.analysis q_distribution --source-root "$RESULTS" --output "${OUT}-q"
python -m experiments.nsgym.analysis planning --source-root "$RESULTS" --output "${OUT}-planning"
```

After a newly completed supported method/run is present, run the same command with a new output name. Reacher Dynamics+Q requires matching completed training/snapshot, prediction and shifted control records; training alone does not enable it.

```bash
python -m experiments.nsgym.analysis full --source-root "$RESULTS" --output "${OUT}-next"
```

Selections are independent and comma-separated:

```bash
python -m experiments.nsgym.analysis full --source-root "$RESULTS" \
  --task halfcheetah --method frozen,dynamics_only,dynamics_q,joint \
  --shift-family damping --severity 0.5,1,2 --schedule fixed \
  --analysis adaptation --output "${OUT}-selection"
```

Validate an existing generated report's input hashes as well:

```bash
python -m experiments.nsgym.analysis validate --source-root "$RESULTS" \
  --unified-output "$OUT" --output "${OUT}-regression" --no-plots
```

## Evidence and statistics

- `RAW_SOURCE_INVENTORY.csv`: file-level raw/derived/provenance roles and schema recognition.
- `DISCOVERED_EXPERIMENTS.csv`: metadata-backed stages, including incomplete/excluded stages.
- `SOURCE_COVERAGE.csv`: raw-reproduced, derived-only and missing metric families/methods.
- `canonical_records.csv`: measurements with checkpoint, method, schedule, seed, budget, source rows, dependencies and hashes.
- `statistical_summary.csv`, `comparisons.csv`: seed-level means, medians, sample standard deviations, bootstrap CIs and descriptive Tukey outlier counts. Small n is explicit.
- `SOURCE_PROVENANCE.json`: hashes of all consumed sources including snapshots/arrays that were verified.
- `DUPLICATE_RECORDS.csv`, `INVALID_SOURCES.csv`, `EXCLUDED_SOURCES.csv`: aliases, conflicts, incomplete and smoke records.
- `RAW_TO_REPORT_CHECKS.csv` / `RAW_TO_REPORT_REGRESSION.md`: raw arithmetic, historical references, seed reductions and plotted data checks.
- `UNIFIED_RESEARCH_REPORT.md`, `UNRESOLVED_DATA_GAPS.md`: scientific comparisons and actual limitations.

Final snapshots use the declared final budget, never the best-performing intermediate snapshot. Intermediate Frozen replicas and S0/S1 aliases are deduplicated. Frozen uncertainty uses shared evaluation seeds once; adapted methods use adaptation seeds. Different checkpoint hashes, schedules and adaptation budgets are not pooled. Joint latent self-consistency is kept separate from fixed-encoder MSE. Nominal k=1 aliases are retained as labelled conditions but never averaged as independent tasks.

Q logits are decoded with the existing TD-MPC2 distribution function. Stored terminal probabilities use the same symexp(expected symlog support) semantics. Targets not recorded in the archive cannot be reconstructed: Bellman residual/CE remain `DERIVED_ONLY`. Candidate ranks refer to model scores, not realized candidate returns. No missing value is imputed.

After CPU verification, original per-observation GPU float32 Q measurements remain the statistical inputs, preserving historical summary numerics. Cancellation-sensitive CPU differences outside the direct tolerance are explicitly counted as `NUMERICAL_DIFFERENCE` when within the recorded propagated float32 bound; larger discrepancies fail. See the measured maximum errors in `RAW_TO_REPORT_CHECKS.csv`.

`PASS_WITH_COVERAGE_GAPS` means supported analyses passed while explicit evidence is absent/derived-only. `PARTIAL` means source inconsistencies remain. Numeric regressions that fail return a nonzero exit code. Previously generated comparison CSVs are optional regression references, not required analysis inputs.

## Tests

```bash
python -m unittest experiments.nsgym.analysis.test_unified experiments.nsgym.analysis.test_migration
```

Temporary fixtures test new-run discovery, relocation, incomplete training, snapshot tampering, duplicate/resume exclusion, output recursion, schedule isolation and automatic Reacher DQ gating. They are removed by `TemporaryDirectory`; scientific datasets are never deleted.

## Local server-copy verification

The full scientific-data run uses the server originals copied by the user, not a remote server session. From the outer local repository:

```bash
cd /Users/jihyekim/Desktop/tdmpc2
PYTHONPATH=tdmpc2 MPLCONFIGDIR=/tmp/mpl-nsgym \
  /private/tmp/tdmpc2-nsgym-venv/bin/python -m experiments.nsgym.analysis full \
  --source-root /Users/jihyekim/Desktop/tdmpc2/results/nsgym \
  --output /Users/jihyekim/Desktop/tdmpc2/results/nsgym/unified-analysis/raw-to-report-v2
```

`full` includes raw-to-report regression and source immutability checks. Running `validate` separately repeats ingestion; it is optional after a successful full run. The task-specific and Q/planning commands above use the same tested CLI; their dispatch and new-method inclusion are also checked with temporary fixtures. A genuinely new schema or adaptation procedure still needs explicit schema/registry support; a new run ID of a supported procedure does not.
