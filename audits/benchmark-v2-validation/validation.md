# Benchmark v2 validation — 2026-10-30

Implementation and CPU fixture tests are complete. Stage 1 research acceptance
is pending simulator/CUDA validation. No real benchmark success rates were
produced in this task.

## Evidence

- `unittest.txt`: 43 tests, 40 passed, 3 skipped. The skipped tests require
  ManiSkill2 0.4.1 and a SAPIEN-capable host. Hydra/NumPy configuration tests executed, not skipped.
- `legacy-dry-run.txt`: all 14 archived E0–E7 commands generated successfully;
  the checkpoint path is deliberately a non-existent placeholder (dry-run only).
- `../audit-20261030T102856-1aea4474/simulator_audit.json`: requested audit command
  executed with seeds 17/18/19, 50 decisions, atol=1e-6 and contact probe;
  **UNVERIFIED: ModuleNotFoundError: No module named 'gym'**.
- `test_legacy_archive_and_conversion`: archive byte equality and all 14 runtime
  shift dictionaries equal the old composed preset dictionaries, including
  parameter, clock, onset, reset behavior and delay options.
- Configuration tests exercise 120 parameter/schedule/evaluation combinations
  (including expected rejection of unsupported physics/delay schedules),
  overrides, conflicts, unsupported modes and deterministic canonical hashes.
- Exposure tests cover pre-onset termination, first shifted step, shifted global
  starts with zero within-episode transitions, repeated transitions and sine
  nominal crossings. Existing tests cover delay queue warm-up/reset and onset
  between the two native steps in one decision.
- Frozen runner fixture verifies actual artifact writing, nominal/shifted
  exposure, initial-state comparison and exposure-constraint failure. All
  synthetic values are confined to unit tests; they are not benchmark results.

## UNVERIFIED

Real disabled/enabled nominal simulator identities; gain=1, delay=0, mass=1,
friction=1 identity; live SAPIEN property readback and controlled contact response;
real pretrained checkpoint evaluation and E0–E7 outcome reproduction; complete
simulator/controller/RNG pairing. Exposed-component comparisons do not certify
full simulator internals. The unchanged TD-MPC2 evaluator requires CUDA.

## Changed files and rationale

- `tdmpc2/config.yaml`: optional schedule and adaptation/evaluation Hydra defaults.
- `tdmpc2/configs/schedule/{fixed,abrupt,repeated,sinusoidal,sequence,drift}.yaml`:
  composable schedule presets, with native clocks and explicit reset policy.
- `tdmpc2/configs/adaptation/{frozen,dynamics_only,joint_rl}.yaml`: executable
  frozen declaration plus explicitly unsupported future modes.
- `tdmpc2/configs/evaluation/{standard,paired,continual}.yaml`: evaluation schema,
  seeds, pairing, exposure constraints; continual is unsupported.
- `tdmpc2/common/benchmark_config.py`: short CLI aliases, canonical normalization,
  legacy conversion, deterministic config hashing and protocol validation.
- `tdmpc2/common/benchmark_validation.py`: schedule/parameter preflight without
  importing simulator or model packages.
- `tdmpc2/common/benchmark_metrics.py`: measured exposure, nominal crossings,
  component-wise initial state capture/comparison.
- `tdmpc2/common/benchmark_logging.py`: mandatory v2 artifacts, reproducibility
  metadata, exposure fields and constraint failure status; old fields retained.
- `tdmpc2/common/shift_protocol.py`: explicit joint_rl unsupported declaration.
- `tdmpc2/envs/maniskill.py`: adapt canonical config to existing wrapper API.
- `tdmpc2/envs/wrappers/dynamics_shift/base.py`: additive native diagnostic fields.
- `tdmpc2/evaluate_shift.py`: config preflight, reusable frozen rollout, paired
  reference/shifted runs, saved initial states and pair differences.
- `benchmarks/archive/dynamics_shift_suite_v1.json`: byte-identical original suite.
- `tests/test_benchmark_v2.py`: configuration, compatibility, exposure and runner
  fixture regressions.
- `EXPERIMENTS.md`: exact CLI, schema, units, artifacts, limitations and rerun guide.
- This validation directory and the new simulator audit directory: new evidence.

The pre-existing modifications to `DYNAMICS_SHIFT_AUDIT.md`, deleted old audit
files and untracked older audit directories were present before this task and
were left untouched. Original suite, prior results and checkpoints were not
modified. No training or model/optimizer/planner changes were made.
