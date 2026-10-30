# Dynamics Shift Benchmark v2

Stage 1 implements frozen evaluation only. Simulator validation and checkpoint
rollouts remain **UNVERIFIED** on the development machine; see
`audits/benchmark-v2-validation/validation.md`. Unit fixtures are not benchmark
success-rate evidence. No TD-MPC2 architecture, loss, Q-learning, planner, or
checkpoint was changed.

## Configuration and compatibility

Run from `tdmpc2/`. The existing Hydra package layout remains authoritative.
The evaluation entrypoint translates short group selectors to their existing
qualified package names before Hydra composition:

| Public selector | Hydra selector |
| --- | --- |
| `dynamics_shift=action_gain` | `configs/dynamics_shift@dynamics_shift=action_gain` |
| `schedule=abrupt` | `configs/schedule@schedule=abrupt` |
| `adaptation=frozen` | `configs/adaptation@adaptation=frozen` |
| `evaluation=paired` | `configs/evaluation@evaluation=paired` |

Python Hydra `compose()` callers should use the qualified names (or call
`common.benchmark_config.cli_aliases`). All new preset YAML files live under
`tdmpc2/configs`; no duplicate alias preset files are maintained.

The canonical resolved execution schema is:

```yaml
dynamics_shift:
  enabled: true
  application: step
  parameter: {type: action_gain, target: arm, nominal: 1.0, shifted: 0.7}
  observability:
    expose_shift_parameter_to_agent: false
    expose_executed_action_to_agent: false
schedule:
  type: abrupt
  clock: episode_native_step
  reset_on_episode: true
  switch_step: 10
  interval: 200
  duration: 200
adaptation: {mode: frozen, update_enabled: false}
evaluation:
  mode: paired
  episodes: 10
  seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  pairing: true
  min_exposure_fraction: 0.0
  require_shift_exposure: false
  pairing_atol: 1.0e-6
```

The old dynamics presets retain their nested schedule for old commands and
wrapper consumers. At the evaluation boundary `normalize()` removes that
nested field: an explicitly selected schedule group replaces it; otherwise the
old schedule is moved to the top level. Combining an explicit group and nested
schedule CLI overrides is rejected. Hydra's raw `--cfg job` output is the
compatibility input, not the canonical execution config. `resolved_config.yaml`
is the canonical config actually used. The wrapper adapter produces a temporary
nested dictionary; it is not a second editable configuration.

`eval_episodes` and `eval_seeds` remain compatibility inputs; evaluation presets
reference them. New runs may override `evaluation.episodes` and
`evaluation.seeds` directly. Seeds default to `seed + episode_id`. The old
`adaptation_mode` is checked, and any non-frozen mode fails explicitly.
`dynamics_only`, `joint_rl`, and `continual` are declarations only and raise
`NotImplementedError`, before loading a checkpoint or constructing a simulator.

Legacy `shift_mode` remains supported by `evaluate.py` and its native legacy
wrapper. `evaluate_shift.py` rejects non-nominal `shift_mode`; it never silently
combines old and new transformations. Training and legacy evaluation entrypoints
are otherwise unchanged.

## Run examples

Use a compatible Linux/CUDA environment with the repository's ManiSkill2/SAPIEN
stack and an existing pretrained checkpoint. Supply its actual absolute path.
The model size must match the checkpoint (released single-task models use 5).

```bash
cd tdmpc2
python evaluate_shift.py task=pick-cube model_size=5 checkpoint=/absolute/model.pt \
  dynamics_shift=action_gain schedule=fixed adaptation=frozen evaluation=standard \
  dynamics_shift.parameter.shifted=0.7 save_video=false

python evaluate_shift.py task=pick-cube model_size=5 checkpoint=/absolute/model.pt \
  dynamics_shift=action_gain schedule=abrupt adaptation=frozen evaluation=paired \
  schedule.switch_step=10 save_video=false

python evaluate_shift.py task=pick-cube model_size=5 checkpoint=/absolute/model.pt \
  dynamics_shift=action_gain schedule=sequence adaptation=frozen evaluation=paired \
  save_video=false
```

Use `schedule.switch_step=20` for the alternate onset. The abrupt preset uses
an episode-native clock. For global onset, override both
`schedule.clock=global_native_step schedule.reset_on_episode=false`.

Other combinations:

- Fixed delay: `dynamics_shift=action_delay schedule=fixed dynamics_shift.parameter.shifted=2`.
- Fixed mass/friction: `dynamics_shift=object_mass` or `contact_friction`,
  `schedule=fixed dynamics_shift.parameter.shifted=0.5`.
- Alternating regimes: `schedule=repeated schedule.interval=20`.
- Sine: `schedule=sinusoidal schedule.interval=40`. Amplitude is
  `schedule.amplitude * (shifted - nominal)`; the trajectory is centered at
  nominal and may exceed the nominal-to-shifted interval in the other direction.
- Drift: `schedule=drift schedule.switch_step=10 schedule.duration=40`;
  canonical schedule type remains the existing `linear_drift`.
- Sequence: `schedule=sequence` supplies A(1.0,20), B(0.7,40), A(1.0,20),
  C(0.5,40), then holds the final regime. Override `schedule.segments` with
  `{duration, value, label}` entries for another sequence. These default values
  are gain-specific; integer delay or vector/matrix parameters require suitable
  segments. Sequence clocks are configured once for the entire sequence.

Native-step units count ManiSkill `env.step` calls, not PhysX substeps. One
TD-MPC2 decision repeats its action up to twice; termination can stop it after
one native step. The schedule is evaluated before each native step, including
between the two repeats. The action wrapper transforms and clips the native
command before calling the environment/controller. The last action component
(gripper) is preserved by default. Delay history is updated every native step
and reset per episode by default; initial actions are zeros. No diagnostic
parameter or executed action is appended to agent observations.

Physics shifts apply and read back at reset, using the existing SAPIEN backend.
V2 allows only nominal/fixed physics schedules. Abrupt/repeated/continuous or
sequence physics combinations are rejected rather than silently sampled and
held at reset. Continuous delay schedules and fractional/negative delays are
also rejected. Existing low-level wrapper behavior remains available to old
consumers; v2's stricter rules are in benchmark preflight.

## Evaluation and artifacts

Every completed condition has a unique timestamp/UUID directory beneath
`logs/<task>/<seed>/<exp_name>/benchmark/<split>/`. Existing runs are never reused.
Mandatory files are `resolved_config.yaml`, `metadata.json`, `summary.json`,
`episodes.csv`, and `timesteps.jsonl`. V2 always saves step/episode evidence,
regardless of legacy logging disable flags. Optional trajectory NPZ, video,
segment snapshots, `shift_events.jsonl`, and `code.diff` remain supported.

Protocol/schema version is 2. Metadata includes checkpoint SHA256, source
revision and dirty status, Python source hashes, canonical configuration SHA256,
package versions, explicit seeds, shift/schedule/clock and adaptation mode.
The configuration hash uses sorted canonical JSON of the saved resolved YAML;
paths are strings. A checkpoint hash is checked again after evaluation.

Old field meanings are retained. New aliases include `global_native_step`,
`episode_native_step`, `policy_decision_step`, and `regime_id`. Native `done`
remains native termination; `decision_done` also includes outer timeout.
`intended_parameter`, `applied_parameter`, commanded/executed actions, native
reward and native success are recorded for every native step.

Exposure metrics are independent:

- `shifted_native_steps` counts applied parameters different from nominal,
  with absolute tolerance 1e-8 and zero relative tolerance. `exposure_fraction`
  divides this count by the episode native length.
- `shift_transition_count` compares adjacent **regime IDs inside the episode**.
  The first sample is never a transition, including a globally shifted start.
  Repeated/sequence IDs identify segments; sine IDs identify completed cycles;
  drift IDs identify before/ramping/after intervals. This is not a count of
  every continuous parameter change.
- `first_shift_native_step` is the zero-based **global** native index, or null.
- Min/max applied values, nominal sample count, and intended/applied agreement
  are saved per episode. For scalar trajectories `nominal_crossing_count`
  counts sign reversals across nominal, ignoring tolerance-level nominal
  samples. It does not count the initial departure from nominal. Sine amplitude
  and period are saved in metadata; sine and drift trajectories are in JSONL.

Summary `actual_shift_observed=false` explicitly identifies runs with no
measured exposure. Short episodes are not assumed to reach the onset. To fail
such episodes, set `evaluation.require_shift_exposure=true`; optionally require
`evaluation.min_exposure_fraction=0.1`. Failure is recorded in summary and
raises an error. Global schedules may produce different exposures under
policies with different episode lengths.

Paired evaluation runs a nominal reference and the requested condition, with
matching episode seeds and policy RNG reseeding. Both produce complete artifacts.
`initial-<episode>.json` stores available observation/simulator/controller/RNG
state. `paired.json` in the shifted run links both runs and reports each
component comparison, success/failure transitions, return/length differences,
and each condition's exposure. Independent global clocks advance according to
each rollout's native length. Full simulator state pairing remains
**UNVERIFIED**, even when all exposed components match: internal contact caches
and every RNG source have not been certified. A component mismatch is `FAIL`;
unavailable components are `UNVERIFIED`. No seed-only full pairing claim is made.

## Legacy suite and verification

The original suite and launcher are retained. The archive is byte-identical:

```bash
# From repository root; remove --dry-run to execute.
python scripts/evaluate_shift_suite.py \
  --suite benchmarks/archive/dynamics_shift_suite_v1.json \
  --checkpoint /absolute/model.pt --dry-run

python -m unittest discover -s tests -v
python scripts/audit_shift_simulator.py \
  --out-root audits --seeds 17 18 19 --decisions 50 --atol 1e-6 --contact-probe
python scripts/audit_shift_logs.py --logs logs --out-root audits
```

The suite still uses its original presets, native clocks, onset=100 where
specified, reset behavior, seeds, and action transforms. The evaluator now
adds v2 artifacts/fields without renaming old fields. Compatibility tests cover
all 14 E0–E7 conditions, but do not establish equality of actual success rates.
Do not overwrite historical results if a simulator rerun differs: compare
checkpoint/config/source hashes, package versions and native traces first.

The test suite requires NumPy, hydra-core and hydra-submitit-launcher. Tests
cover composition/overrides, invalid combinations, unimplemented modes, hash
stability, exposure before/at onset and after global onset, repeated transitions,
sine crossings, delay warm-up/reset, and a schedule change inside action repeat.
An in-memory environment tests evaluation/artifact orchestration; its rewards
are synthetic test fixtures and are not research results.

The remaining Stage 1 acceptance work is to run the simulator audit and real
frozen/paired checkpoint evaluations on the supported stack. No online replay,
TTT, joint RL, repeated adaptation or plasticity implementation is included.
