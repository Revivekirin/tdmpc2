# Stage 3B-1: Frozen dynamics severity sweep

This extends `scripts/run_stage3b.py` with `--phase severity`. It evaluates the pretrained Frozen agent; it never enters Stage 3B adaptation, creates an EpisodeAdapter, collects adaptation replay, or performs optimizer steps. The existing Stage 3B `episode()` loop supplies exactly the same seed/reset, `t0`, `eval_mode=True`, two-native-step action repeat, reward sum, timeout and success protocol. Q-input/latent-error diagnostics from the adaptation report are not computed. Each severity gets one shifted and one independently constructed nominal evaluation over the same seeds.

## Code audit and physical units

| Condition | Existing default locations | Actual application |
|---|---|---|
| action_delay=1 | `benchmarks/stage3b_suite.json`, `tdmpc2/configs/dynamics_shift/action_delay.yaml` | `action_shift.py`: FIFO delay on arm commands, units of native `env.step` (control steps), not policy decisions, seconds or simulator physics substeps. One policy command is repeated twice. Gripper is current. Zero initial arm command; queue reset per episode. |
| action_gain=0.7 | suite JSON and `configs/dynamics_shift/action_gain.yaml` | Arm command multiplied once, then clipped to native action bounds. Gripper unchanged. |
| object_mass=1.5 | suite JSON and `configs/dynamics_shift/object_mass.yaml` | `physics_shift.py`: original object mass AND inertia multiplied by factor, using existing SAPIEN actor rebuild; center of mass, pose, velocities and geometry preserved. Applied at reset, held through episode. |
| contact_friction=0.5 | suite JSON and `configs/dynamics_shift/contact_friction.yaml` | Original static AND dynamic friction of object and plane multiplied once, restitution unchanged, independent materials avoid changing robot fingers. Applied at reset, held through episode. |

Mass/friction restore baseline before reset, and the severity evaluator restores it in `finally` before closing each environment. Per-native-step getter readback verifies effective mass/inertia or material coefficients against baseline. Friction evidence describes material coefficients, not a measured contact force or independently measured PhysX combine rule. Action gain validation compares command × factor after clipping to delivered action. Delay validates delivered history and actual queue length each native step.

No wrapper physics was changed. `base.py` gained an optional read-only post-step observer and `adapt_shift.episode()` gained an optional reset callback. Both are absent in legacy runs.

Important historical path detail: old Stage 3B's final `/1` is `cfg.seed`, not severity (`common/stage3b.py`). Old paths, datasets and metadata remain untouched. New severity paths explicitly encode severity and have a versioned child directory; new metadata names are intentionally distinct so old recursive Stage 3B analyzers ignore them.

## Commands (run from repository root on the CUDA server)

Use the existing environment and checkpoint:

```bash
conda activate tdmpc2-ms2
cd /nfs4/jhkim/repos/tdmpc2
```

One condition, default ten evaluation seeds (4000–4009 for set 0):

```bash
python scripts/run_stage3b.py --phase severity \
  --checkpoint checkpoints/maniskill2/pick-cube-1.pt \
  --shift-type action_gain --shift-value 0.5 \
  --set-id 0 --method frozen --log-eef --execute
```

Bounded server smoke: ONE shifted episode plus ONE nominal reference episode; no adaptation:

```bash
python scripts/run_stage3b.py --phase severity \
  --checkpoint checkpoints/maniskill2/pick-cube-1.pt \
  --shift-type action_delay --shift-value 1 --set-id 0 \
  --episodes 1 --log-eef --run-id delay-smoke-v1 --execute
```

Dry-run full pilot matrix (16 conditions, starts no simulators and writes no files):

```bash
python scripts/run_stage3b.py --phase severity \
  --checkpoint checkpoints/maniskill2/pick-cube-1.pt \
  --pilot --set-id 0 --log-eef --dry-run
```

| Shift | Pilot values |
|---|---|
| action_delay | 0, 1, 2, 3 |
| action_gain | 0.9, 0.7, 0.5, 0.3 |
| object_mass | 1.0, 1.5, 2.0, 3.0 |
| contact_friction | 1.0, 0.7, 0.5, 0.3 |

Select shifts with `--shift-type action_delay object_mass --pilot`. Select custom values for ONE shift:

```bash
python scripts/run_stage3b.py --phase severity \
  --checkpoint checkpoints/maniskill2/pick-cube-1.pt \
  --shift-type action_gain --shift-values 0.7 0.5 0.3 \
  --set-id 0 --log-eef --dry-run
```

Only replace `--dry-run` with `--execute` when ready. Default is dry-run even without the flag. No pilot/sweep was launched during implementation. Without `--pilot` or explicit values, use the four legacy magnitudes (or the selected shift's magnitude). Explicit values take precedence. Explicit values across multiple shift types, duplicate values, unsafe run IDs, NaN/infinity, nonpositive factors, negative/fractional delay, adaptation resume flags and dynamics_only are rejected. The existing delay capacity limit of 10000 native steps is retained. `--episodes` is 1–10000 to keep evaluation sets disjoint; set S uses 4000+10000*S onward. No independent sets are started automatically.

Legacy `--phase collect` / `--phase adapt` and their `--suite` inputs are unchanged. New severity flags require `--phase severity`; legacy adaptation can still use its existing suite/Hydra shift overrides. The severity evaluator loads the same shift YAML and wrapper as TTT, so a later explicitly requested TTT experiment can use the identical physical definition.

## Outputs and completeness

Default base: `results/stage3b-generalization` (override with `--output`).

```text
runs/{shift_type}/set-{set_id}/frozen/{severity}/severity-eval/{run_id}/
  severity-metadata.json       # requested/nominal values, units, config, seeds, checkpoint/source hashes, versions
  severity-effective.json      # final metadata with measured physical values for every episode
  severity-episodes.jsonl      # append-only episode records, useful if interrupted
  severity-complete.json       # only published after all episodes, invariance and initial-state checks
  shifted-seed-4000.npz        # arrays for every native step
  shifted-seed-4000.json       # provenance, initial state/pose, effective readback, episode metrics
  nominal-seed-4000.npz
  nominal-seed-4000.json
severity_sweep/{run_id}/
  manifest.json                # complete requested matrix, including missing/failed conditions
  summary.csv
  {shift_type}.png / .pdf
severity_sweep/summary.csv     # latest launch summary; regenerate rather than treating as immutable run data
```

Generated run IDs are unique UTC timestamp + UUID suffix. Supplying an existing `--run-id` target fails before launching any condition. Interrupted runs are preserved; rerun with a new ID. There is no checkpoint resume or incomplete-directory deletion in this evaluation mode. Root summary is a replaceable latest view; versioned run data stays intact.

Summary marks COMPLETE, PARTIAL, MISSING or PROTOCOL_INCOMPATIBLE. Missing is relative to the manifest, not all conceivable severity values. Only complete compatible conditions contribute rates/plots. It checks checkpoint/config/source/environment compatibility, seed counts, zero model drift/updates, trajectory presence and array lengths. Each new evaluation compares shifted and nominal reset observations, object pose/velocity and robot joint position/velocity within 1e-6. It also stores the full simulator state/hash separately for audit: object-mass actor rebuilding can change that representation without changing the measured initial kinematics. Legacy results without kinematic evidence still require exact full-state hash equality. The summary compares nominal initial-state hashes across severities for a seed set. Duplicate evaluations of the same set/condition are rejected instead of treated as independent repeats. Each plot uses the actual parameter value, displays episode/set counts and matched nominal references. Sample SD error bars appear only for multiple independent sets, never inferred from repeated evaluation of the same set. No CI is claimed.

Combine selected manifests explicitly (same evaluation episode counts and source/protocol):

```bash
python scripts/analyze_stage3b_severity.py \
  --manifest results/stage3b-generalization/severity_sweep/RUN_A/manifest.json \
             results/stage3b-generalization/severity_sweep/RUN_B/manifest.json \
  --output results/stage3b-generalization/severity_sweep/combined-v1
```

One-episode smoke is intentionally protocol-incompatible with ten-episode pilots for aggregation. Summary generation also runs after a subprocess failure; if killed abruptly, use the analyzer manually. Independent source revisions are conservatively not averaged together.

## Trajectory schema

Always saved, compressed NPZ (`allow_pickle=False`): `timestep` [T] (zero-based native step), `commanded_action` [T,A], `executed_action` [T,A], `observation` [T,O] (post-step state observation), `reward`, `success` (native signal), `done` (native signal), `episode_success` (final episode outcome repeated), `queue_length` (-1 when inapplicable). Native `done` can remain false at an outer policy timeout; final episode metrics capture the evaluator termination.

With `--log-eef`: `eef_xyz` [T,3] from simulator TCP pose, world-frame meters, measured AFTER each native step; `eef_quaternion_wxyz` [T,4], NaN if orientation unavailable. No action integration. Requested EEF logging fails clearly if TCP measurement is unavailable. JSON includes initial EEF pose, initial simulator state/hash, task, seed, checkpoint ID, shift/value/units, condition and final episode success/return/length. Delay JSON records requested delay, queue capacity and final queue length; the array records length every step.

CPU tests use deterministic NumPy/SAPIEN API-contract fixtures and the real Stage 3B episode loop. They do not establish real simulator correctness/performance. Local macOS has no CUDA, ManiSkill2, SAPIEN or gym, so a real TD-MPC2 one-episode smoke remains to be run with the bounded server command above. No GPU experiments or new TTT runs were launched.

## Modified files and local validation

- `scripts/run_stage3b.py`: existing CLI adds the severity phase; legacy phases retain their original paths.
- `scripts/stage3b_severity.py`: validated matrix planning, dry-run, unique-run preflight and subprocess launch.
- `tdmpc2/severity.yaml`, `tdmpc2/evaluate_severity.py`: Frozen-only Hydra configuration, checkpoint loading, provenance and result commit.
- `tdmpc2/common/severity_protocol.py`: shared value/path/protocol validation and source hashing.
- `tdmpc2/common/severity_evaluation.py`: episode evaluation, simulator measurements, physical readback validation, state invariance and NPZ/JSON persistence.
- `tdmpc2/adapt_shift.py`, `tdmpc2/envs/wrappers/dynamics_shift/base.py`: optional callbacks only; existing default behavior unchanged.
- `scripts/analyze_stage3b_severity.py`: completeness/provenance checks, summary CSV and four per-shift plot types.
- `tests/test_stage3b_severity.py`: CPU tests for CLI dry-run, legacy Hydra compatibility, model-state metadata, a one-episode evaluator fixture, physical bookkeeping differences at reset, and summary/plot generation.
- This document: audit, commands, schemas, limitations.

Validation at implementation: 60 CPU tests passed (Stage 3A/3B + severity + physics, benchmark, state/logging and dynamics-shift audit). After the object-mass initial-state correction, 27 focused CPU tests passed. `git diff --check` and Python compilation passed. Tests use a local CPU PyTorch environment plus the existing temporary Hydra dependency directory; no simulator smoke was possible locally.
