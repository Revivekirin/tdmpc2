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
export TD_MPC2_CHECKPOINT=/absolute/path/to/pick-cube-1.pt
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
