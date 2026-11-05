# Stage 3B: generalization and adaptation horizon

Implementation reuses `adapt_shift.py`, `EpisodeAdapter`, Stage 3A held-out
shifted/nominal evaluation, fixed-data identity/collection/scoring, and Stage 2
PlannerTrace. The actual Stage 3A implementation has one `EpisodeAdapter` with
`frozen`/`dynamics_only` modes. Classes literally named FrozenAdapter or
DynamicsOnlyAdapter are absent from this revision; no replacement adapter was invented. No WorldModel/checkpoint keys, `_plan`, or TD-MPC2 update
were changed. Existing Stage 2/3A artifacts and pre-existing local edits are untouched.

Configuration: `tdmpc2/stage3b.yaml` inherits `stage3a_matched`. Defaults remain
Adam lr=3e-5, one-step consistency coefficient from TD-MPC2 config, shifted-only
FIFO replay, batch/warmup=16, 50 optimizer steps per naturally completed episode.
Budgets are 0, 800, 1600, 3200, 6400, 12800 native steps. Three seed sets use the
original stride=10000; adaptation base=1000, evaluation=4000–4009, fixed=5000–5009.
The 2000-episode safety cap is unchanged. Increase it with care to maintain seed
disjointness (the validator rejects overlap with evaluation/fixed-data seeds).

`benchmarks/stage3b_suite.json` declares methods, seed sets, budgets and severities.
Magnitude rationale is in `docs/stage3b_shift_selection.md` (also copied into the result directory).
Changing suite seed sets expands independent replicates. To expand evaluation
counts, use matching `stage3a.evaluation_seeds` overrides for all methods, keeping
all three seed pools disjoint. Changing collection seeds requires new fixed data.

## Native boundaries and resume

Stage 3B adds an opt-in repeat path to ManiSkillWrapper. The ordinary Stage 3A
path remains unchanged. Each native call finishes and advances the existing shift
wrapper counters. Then a milestone is evaluated BEFORE decision replay commit
and BEFORE a terminal episode's optimizer update. Evaluation restores planner and
RNG and uses the existing model/optimizer/replay invariance checks.

At any milestone, including native termination and the last native call in a
repeat, the pending action, observation, reward sum, done/info and native records
are saved. No transition is discarded. Final stopping suspends the pending
repeat; it neither truncates the episode nor creates an optimizer update. A
continued run finishes that saved decision, commits its transition exactly once,
and performs the normal episode update only when its original done condition is
processed. Evaluation milestones therefore do not change the learning schedule.

`resume-budget-B.pt` stores predictor, Adam state, FIFO replay, replay RNG,
controller/simulator/shift/delay-queue state, environment RNG and wrapper counters,
pending repeat, episode context, planner mean, Python/NumPy/Torch/CUDA RNG, and
prior evaluation/episode records. The existing environment snapshot API is reused
with an opt-in pending-decision flag. Resume requires the same environment stack,
checkpoint hash, dataset hash and experimental controls. Controller snapshot
capability is checked before the first adaptation interaction. CPU fixture tests
verify continuation; real SAPIEN solver/cache continuation remains to be checked
by the representative server smoke. No unsupported fallback reset is used.

Resume always writes to a NEW output path. Prior evaluation and episode rows are
copied from the checkpoint; diagnostics in the new directory contain only new
native steps. The prior run is linked by `stage3b.resume` in metadata. Saved
milestones must remain a prefix of the new grid. Future budgets may be extended.
Use the same pretrained checkpoint (not `adapted.pt`); it supplies the unchanged
encoder/Q/reward/actor and initial-predictor baseline. If combining original and
resumed results, place only the selected continuation in the analysis root;
duplicate method/shift/seed runs are rejected.

## Server commands

Run from repository root in the established Linux CUDA + ManiSkill2 environment.
Verify its revision and local changes first (`git rev-parse HEAD`, `git status
--short`). No server address was available during local implementation.

```bash
# Previously recorded server checkpoint location; confirm it still exists.
export TD_MPC2_CHECKPOINT="$PWD/checkpoints/maniskill2/pick-cube-1.pt"
test -f "$TD_MPC2_CHECKPOINT" || { echo "Checkpoint missing: $TD_MPC2_CHECKPOINT"; exit 1; }
# Print commands only (default); this does NOT start the full suite.
python scripts/run_stage3b.py --checkpoint "$TD_MPC2_CHECKPOINT" --phase collect
python scripts/run_stage3b.py --checkpoint "$TD_MPC2_CHECKPOINT" --phase adapt

# Representative real smoke: delay=1, set=0, both methods, budgets 0/5/80/240.
python scripts/run_stage3b.py --checkpoint "$TD_MPC2_CHECKPOINT" \
  --phase collect --smoke --output results/stage3b-server-smoke --execute
python scripts/run_stage3b.py --checkpoint "$TD_MPC2_CHECKPOINT" \
  --phase adapt --smoke --output results/stage3b-server-smoke --execute

# Full run ONLY when explicitly requested; 12 collections + 24 continuous runs.
python scripts/run_stage3b.py --checkpoint "$TD_MPC2_CHECKPOINT" --phase collect --execute
python scripts/run_stage3b.py --checkpoint "$TD_MPC2_CHECKPOINT" --phase adapt --execute

# CSV-first aggregate and all five figures, output must be a new directory.
python scripts/analyze_stage3b.py --root results/stage3b-generalization/runs \
  --output results/stage3b-generalization/report
```

A run's full native budget is 12800, not the sum of its milestones. The full suite
uses 307200 adaptation native steps total; evaluation/collection costs are extra
and depend on completion lengths. There are 2880 online evaluation episodes
(24 runs ×6 milestones ×20 episodes), plus 120 fixed-data collection episodes.
No wall-clock estimate is asserted without CUDA timing.

For smoke aggregation, pass `--suite results/stage3b-server-smoke/configs/adapt-suite.json`.
The launcher records the effective suite and exact command list per phase. The
240-step smoke extends beyond a maximum-length 200-native-step episode, allowing
the episode-wise optimizer to execute. It retains the batch/warmup of 16.
`--allow-partial` permits missing/failed runs in a report, records them in
validation.json and leaves gaps; it does not permit mismatched controls.

Example resume (replace placeholders and retain the original overrides):

```bash
python tdmpc2/adapt_shift.py --config-name stage3b \
  task=pick-cube model_size=5 episodic=false checkpoint="$TD_MPC2_CHECKPOINT" \
  dynamics_shift=action_delay schedule=fixed dynamics_shift.parameter.shifted=1 \
  stage3a.mode=dynamics_only stage3a.seed_set=0 \
  stage3b.resume=/absolute/path/to/resume-budget-1600.pt \
  stage3a.output=results/stage3b-resumed/runs/action_delay
```

The default fixed data path still points to the original collection. For a run
with custom fixed-data path, seeds or optimizer settings, preserve those overrides.
Stage 3A's predictor-only snapshots/adapted.pt cannot be resumed as Stage 3B.

## Artifacts and interpretation

Runs preserve Stage 3A's metadata/evaluations/diagnostics/episode/complete/adapted
filenames and fields. Stage 3B adds complete resume files, parameter drift,
replay samples, cumulative update compute time and nominal return/length deltas.
Fixed-data MSE is the equally weighted mean of per-collection-episode MSEs.
The baseline predictor is scored on the same transitions at every milestone.

The report writes `tables/stage3b_raw_results.csv`, `stage3b_aggregated_results.csv`,
`stage3b_auc_seed_results.csv`, `stage3b_auc_summary.csv`, and
`stage3b_interaction_budget.csv`. Raw rows retain evaluation/collection seeds.
Mean/sample SD first averages within a seed set, then across independent sets.
Missing values are blank. With a single set SD is undefined, not zero uncertainty.
Success values in CSV are fractions; success figures show percent.

Five figures are written as PNG (220 DPI) and PDF. The overview is metric rows ×
method columns; standalone panels extend to a two-column grid for extra methods.
Shift colors and ordering agree across the line figures. AUC uses grouped bars
(method legend, shift x-axis). Plots read the actual aggregate CSV. Full-grid
trapezoidal AUC is computed per seed before aggregation; no partial AUC is silently
compared to a full-horizon AUC. Paired Frozen differences and initial-to-final/
best success are retained. No post-hoc recovery threshold is invented.

Three sets give imprecise uncertainty: bands show sample SD, NOT a 95% CI.
Prediction improvement need not imply control recovery; high Frozen success is
retained. Fixed data reflects Frozen visitation. Reused evaluation seeds are not
independent across budgets. Long-run saturation/instability require real results.
Mass/inertia/friction mechanisms differ from delayed action history. Stage 3C can
add methods to analysis with the same protocol; this implementation exposes only
Frozen and dynamics-only TTT for execution.

## Local validation

```bash
python -m unittest discover -s tests -p 'test_stage3a*.py' -v
python -m unittest discover -s tests -p 'test_stage3b*.py' -v
python tests/test_stage3b_report.py --render-smoke results/stage3b-plot-smoke
```

The last command writes explicitly SYNTHETIC fixture CSVs/figures; it does not run
TD-MPC2 and must never be reported as experimental success or MSE. Real runs need
CUDA and a checkpoint; neither was available on this local host.

New files: `tdmpc2/stage3b.yaml`, `tdmpc2/common/stage3b.py`,
`benchmarks/stage3b_suite.json`, `scripts/run_stage3b.py`,
`scripts/analyze_stage3b.py`, `tests/test_stage3b.py`, `tests/test_stage3b_report.py`,
this document, shift selection and validation/report artifacts.
Modified files: `tdmpc2/adapt_shift.py` (dispatch/return metric),
`tdmpc2/common/stage3a.py` (state serialization/cost/drift/Frozen invariance),
`tdmpc2/envs/maniskill.py` (opt-in resumable repeat),
`tdmpc2/envs/wrappers/dynamics_shift/logging.py` (pending-repeat snapshot support).

## Continue a suite: skip completed runs and restart incomplete runs

Use the updated `scripts/run_stage3b.py` and `scripts/stage3b_results.py` on the
server, together with the previously updated `scripts/analyze_stage3b.py`.
`--continue-suite` now applies this policy:

- Completed original run: validate and skip.
- Completed matching recovery: validate and skip; keep its incomplete original.
- Incomplete run without a completed recovery: archive its whole run directory,
  then restart from the pretrained checkpoint at native budget zero.
- Unstarted condition: execute normally.

The shared selector checks matching config, checkpoint/data hashes and a complete
milestone grid, and rejects conflicting completed results. A `complete.json`
marker with invalid artifacts is an error, not permission to restart a completed
condition. Missing or partially written metadata without a completion marker is
handled as an incomplete directory.

Archival uses a same-filesystem rename to
`<experiment>/archived-incomplete/<attempt-id>/<shift>/set-N/<method>/1/`.
Incomplete recovery directories for the requested conditions are also archived
outside their own `runs/` root. Model snapshots, optimizer/replay state and logs
are retained there, but are NOT loaded into the restarted run. Shared fixed data
and the pretrained checkpoint are unchanged. Archived runs are not included in
analysis. This does not free disk space; disk-full failures still require available
space before retrying.

```bash
# Preview only. For the reported delay set-0/set-1 results: skip 4, run 20.
python scripts/run_stage3b.py --checkpoint "$TD_MPC2_CHECKPOINT" \
  --phase adapt --continue-suite \
  --recovered-root results/stage3b-recovered/runs

# Archive incomplete runs, execute fresh restarts/new runs, then aggregate.
(
set -e
python scripts/run_stage3b.py --checkpoint "$TD_MPC2_CHECKPOINT" \
  --phase adapt --continue-suite \
  --recovered-root results/stage3b-recovered/runs --execute
python scripts/analyze_stage3b.py \
  --root results/stage3b-generalization/runs \
  --recovered-root results/stage3b-recovered/runs \
  --suite results/stage3b-generalization/configs/adapt-suite.json \
  --output results/stage3b-generalization/report-continued
)
```

Collection is not repeated. Completed runs are validated and skipped before
checking dataset files. Only pending or restarted runs require the corresponding
fixed-data file to exist. All missing pending datasets are reported together before
archival or launching. If a paired method is already complete, restore its original
dataset rather than replacing it with newly collected data. Missing data for
completed conditions does not prevent CSV-based aggregation, but prevents later
re-evaluation on those original transitions until restored. Continuation command/selection/suite manifests get
unique names in `configs/`; previous manifests remain untouched. The selection
manifest records archive sources and attempt ID. All conditions are preflighted
before any archival. Omitting `--execute` previews commands and archive candidates
without writing or moving files. Re-running the same continuation command skips
newly completed runs and archives/restarts any newly incomplete runs. Report output must still be a new
directory. Only analysis of the complete selected suite generates full results;
`--allow-partial` may be used separately for an explicitly partial report.

## Prepare missing data after consolidating runs

Once completed recovery runs have been moved into the canonical `runs/` tree,
`recovered/` may contain only empty directories. On the server, from the experiment
root, `rmdir -p recovered/runs/action_delay/set-1/dynamics_only` removes that empty
chain only; it refuses to remove nonempty directories. Do not use recursive
forced deletion for this cleanup.

The full suite needs twelve shared datasets (four shifts × three seed sets).
Run the following from repository root with the current launcher/helper scripts:

```bash
# Read-only inventory and completed-run hash checks.
python scripts/prepare_stage3b_data.py --checkpoint "$TD_MPC2_CHECKPOINT"
# Collect missing datasets only; existing files remain untouched.
python scripts/prepare_stage3b_data.py --checkpoint "$TD_MPC2_CHECKPOINT" --execute
```

This uses the existing Stage 3A collector. Candidates are first written under
`data-candidates/<attempt>/`. When a condition already has completed results, the
candidate must match the original dataset SHA-256 in metadata before it can enter
`data/`. Matching files are installed without overwriting an existing destination.
Mismatches are preserved as candidates and the command ends with an error after
processing the missing files. Restore the original dataset from backup, or make
an explicit decision to archive and rerun every method for that condition with a
new shared dataset. Same seeds alone are not proof of identical transitions.
Completed runs and their metadata are never changed by preparation. The existing
resume checkpoint contains adaptation replay, not the shared held-out dataset.

Preparation does not launch the full adaptation suite. On successful completion,
continue with `--phase adapt --continue-suite --execute` without `--recovered-root`
when all completed runs have been consolidated into `runs/`.
