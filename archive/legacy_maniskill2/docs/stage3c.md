# Stage 3C: policy-decision-wise dynamics-only TTT

Implemented against the local checkout. The user will run on the server; the latest
server source and real CUDA/ManiSkill execution have not been verified here.

## Configuration and update order

Use `--config-name stage3c`. This inherits Stage 3B and selects
`stage3a.mode=decision_ttt`, following the existing Stage 3A/B method namespace.
The legacy `adaptation` Hydra group remains a Stage 2 interface; do not use
`adaptation=decision_ttt`. Existing `frozen` and `dynamics_only` invocations remain
unchanged. `updates_per_decision=1` is the sole cap; `update_every_decisions=1`
sets the cadence. Both live under `stage3a`. Values greater than one for the cap
are rejected. `updates_per_episode` is inherited but unused for decision TTT.

1. MPC selects a commanded action.
2. The existing wrapper executes its two native simulator steps (or terminates early).
3. Store the complete `(observation, commanded action, next observation)` tuple.
4. If ready and the global completed-decision index is divisible by the cadence,
   take one Adam step using the existing one-step latent loss.
5. Log decision diagnostics; perform terminal/truncation bookkeeping if applicable.
6. Evaluate any decision-boundary milestone, then use the updated predictor for MPC.

No episode-end optimizer step is added. The terminal transition is included once.
Replay, optimizer, predictor and cadence persist across episodes. Tuples are
self-contained one-step transitions, so sampling cannot join adjacent episodes.
Decision logs contain a zero-based episode ID and one-based cumulative completed
decision/native-step IDs. Native records retain the existing wrapper identifiers.

The actual Stage 3B defaults inherited from `stage3a_matched.yaml` are **warmup=16
stored transitions, batch_size=16**, capacity=100000, Adam lr=0.00003. The default
first eligible/actual update is completed decision 16 (normally native step 32;
early terminations can change that native count). Readiness is
`len(replay) >= max(warmup, batch_size)`; thresholds have not been reduced.
Metadata includes the effective threshold and explicit timing units. With cadence
K, the first update is the first ready completed decision divisible by K.

Only `_dynamics` is trainable. Encoder, reward, Q, target Q, actor and any other
non-dynamics model state are covered by the existing frozen-state assertion.
Loss scaling, clipping, batch sampling and optimizer code are shared with
episode-wise TTT. `TDMPC2.update()`, WorldModel, checkpoint keys, commanded-action
semantics and MPC are unchanged. No executed-action oracle enters the loss.

## Native milestones and resume

Budgets remain 0, 800, 1600, 3200, 6400, 12800 native steps. If a milestone lands
on a completed decision, decision TTT checkpoints after its update and terminal
bookkeeping. If it lands inside an action repeat, the existing Stage 3B pending
repeat snapshot is used at the exact native count, before that incomplete
transition can enter replay. No budget overshoot or forced termination is used.
An unfinished final-budget decision remains pending; extending the budget on
resume finishes it once. Legacy episode-wise checkpoint timing is unchanged.

`resume-budget-N.pt` contains the existing predictor/Adam/replay state, adapter RNG,
global RNG, planner, rollout/episode/native/decision counters, environment/controller
snapshot and pending repeat, plus the decision scheduler. The original pretrained
checkpoint and fixed dataset hashes remain mandatory. Frozen weights are supplied
by that original checkpoint. Episode-wise checkpoints are rejected as decision
resumes. `adapted.pt` alone is **not** a full training continuation.

The existing simulator/controller snapshot capability check fails before the first
adaptation interaction if unsupported; there is no silent model-only fallback.
CPU fixtures verify continuation, not bitwise real-simulator reproducibility.
All resumed/evaluated milestones preserve the native budget, even mid-episode.
Evaluation uses separate environments and the existing held-out mutation guards.

## Server commands

Run from the repository root in its existing CUDA environment. Set `CKPT` to the
same nominal pretrained checkpoint used by Stage 3B:

```bash
cd /nfs4/jhkim/repos/tdmpc2
export CKPT=/absolute/path/to/nominal-pretrained.pt

# Config-only smoke; does not load the checkpoint or simulator.
python tdmpc2/adapt_shift.py --config-name stage3c --cfg job --resolve \
  task=pick-cube model_size=5 episodic=false checkpoint="$CKPT" \
  dynamics_shift=action_delay schedule=fixed dynamics_shift.parameter.shifted=1

# Eight-condition dry run: prints commands; does not launch or write results.
python scripts/run_stage3b.py --suite benchmarks/stage3c_suite.json \
  --checkpoint "$CKPT" --phase adapt
```

Every printed condition starts a new process loading the nominal checkpoint.
`benchmarks/stage3c_suite.json` specifies this set-0 matrix:

| Shift | Severities | Method | Native budget |
|---|---|---|---|
| action_delay | 1, 2 | decision_ttt | 12800 each |
| action_gain | 0.7, 0.3 | decision_ttt | 12800 each |
| object_mass | 1.5, 3.0 | decision_ttt | 12800 each |
| contact_friction | 0.5, 0.3 | decision_ttt | 12800 each |

Default fixed-data paths are `results/stage3b-generalization/data/<shift>/<severity>/set-0.pt`.
To reuse existing Stage 3B-2 fixed data, add `fixed_data_path` to each shift entry
in a copy of the suite, using the exact matching severity's existing dataset path.
The runner verifies checkpoint/shift/planner/seed provenance; the aggregator also
requires identical dataset hashes across methods. Do not recollect or substitute
fixed data for an existing comparison. Only genuinely new conditions need
`--phase collect` (prints commands), with `--execute` to explicitly collect.

Bounded real smoke: the following uses a **separate** output/data root, one shift,
one evaluation seed, one fixed seed, native budgets 0/5/80/240, and the unchanged
warmup/batch defaults. Collection is additional held-out interaction outside the
240-step adaptation budget. These smoke datasets are not full-suite datasets.

```bash
python scripts/run_stage3b.py --suite benchmarks/stage3c_suite.json \
  --checkpoint "$CKPT" --output results/stage3c-smoke --phase collect --smoke --execute
python scripts/run_stage3b.py --suite benchmarks/stage3c_suite.json \
  --checkpoint "$CKPT" --output results/stage3c-smoke --phase adapt --smoke --execute
```

One full condition (after setting `DATA` to its verified existing fixed dataset):

```bash
export DATA=/absolute/path/to/action-delay-1-fixed-set-0.pt
python tdmpc2/adapt_shift.py --config-name stage3c \
  task=pick-cube model_size=5 episodic=false checkpoint="$CKPT" \
  dynamics_shift=action_delay schedule=fixed dynamics_shift.parameter.shifted=1 \
  stage3a.seed_set=0 stage3a.fixed_data_path="$DATA" \
  stage3b.output_dir=results/stage3b-generalization/runs/action_delay/set-0/decision_ttt/1
```

To resume, repeat the same command and append:

```bash
stage3b.resume=/absolute/path/to/resume-budget-800.pt \
stage3b.output_dir=results/stage3b-generalization/runs/action_delay/set-0/decision_ttt/1/versions/resumed-run
```

The existing budget prefix must remain unchanged. Decision TTT automatically uses
`versions/<unique-id>` when its destination already exists. Existing Frozen and
episode-wise artifacts are never overwritten. The legacy methods still reject
existing destinations. Inspect metadata paths when selecting versioned results.

Full-suite execution is available by adding `--execute` to the eight-condition
dry-run command, **only when you choose to launch it**. No full runs were launched
during this implementation.

## Aggregation and figures

The existing `scripts/analyze_stage3b.py` now separates severities in duplicate,
dataset and control checks. With multiple severities, CSV `shift_type` condition
labels include `=severity`; `shift_magnitude` is retained. Seed/episode data are
retained. Decision TTT is automatically added when encountered in a valid input
suite's shift/seed grid. Each method's schedule must agree across its runs; shared
learning/model/MPC settings and per-condition fixed-data hashes must match.

Select exactly one run version per condition/method/set in a report input tree
(symlinks to selected run directories are sufficient). Do not point analysis at
multiple versions: duplicates are deliberately rejected. Use a copy of the
Stage 3C suite with `methods: ["frozen", "dynamics_only", "decision_ttt"]` for a
three-column comparison, and matching shift magnitudes, seeds and budget grid.
Frozen severity-only evaluations are not native-budget curves and must not be
fabricated into them. Missing compatible runs are explicit gaps with `--allow-partial`.

```bash
python scripts/analyze_stage3b.py --root /absolute/path/to/selected-runs \
  --suite /absolute/path/to/comparison-suite.json \
  --output results/stage3c-report --allow-partial
```

The pipeline writes PNG and PDF versions of `shifted_success_by_adaptation`,
`fixed_data_mse_by_adaptation`, `nominal_retention_by_adaptation`, and
`stage3b_comprehensive_overview`. Method columns share metric scales and condition
colors. It also writes `gradient_updates_vs_native_steps`,
`shifted_success_vs_gradient_updates`, `predictor_mse_vs_gradient_updates`, and
`update_wall_time_vs_native_steps`. Efficiency plots retain paired seed-level
observations and do not interpolate updates or invent missing measurements.
`update_compute_time` is cumulative optimizer wall time in seconds; new
`update_wall_time_ms` fields explicitly use milliseconds. They exclude evaluation
and checkpoint I/O and are not end-to-end rollout latency.

## Validation and limitations

Local bounded CPU tests cover config parsing for all eight commands, readiness,
predictor changes and frozen-state invariance, held-out replay/optimizer isolation,
terminal ordering/no double update, counters, save/load, mid-repeat/terminal resume,
existing frozen/episode-wise execution, and three-method/two-severity aggregation.
Final local result: **15 tests passed** in 6.595 seconds; `git diff --check` passed.
The dry-run planner emitted exactly eight independent adaptation commands.
Synthetic report tests generate all requested PNG/PDF outputs, labelled synthetic;
they are not experimental evidence. Run locally/on the server with:

```bash
PYTHONPATH=tests python -m unittest test_decision_ttt test_stage3b test_stage3a test_stage3b_report
```

Real CUDA MPC, ManiSkill rollout and actual server snapshot fidelity are pending
the user's bounded server smoke. No new real experiment plots or performance
claims are provided. No nominal fixed-data dataset interface exists in the current
runner, so metadata explicitly marks nominal fixed-data MSE unavailable; nominal
held-out success/return/length and retention are still evaluated. Existing Stage 3B
snapshot support is reused without a new simulator pairing investigation.

Modified implementation: `tdmpc2/common/stage3a.py`, `common/decision_ttt.py`,
`common/stage3b.py`, `tdmpc2/adapt_shift.py`, `tdmpc2/stage3c.yaml`,
`scripts/run_stage3b.py`, `scripts/analyze_stage3b.py`,
`benchmarks/stage3c_suite.json`, `tests/test_decision_ttt.py`, `tests/test_stage3b.py`.
