# TD-MPC2 dynamics shift benchmark

## Status and scope

Six shift types, a common schedule interface, hidden diagnostic traces, frozen
entrypoint, future protocol declarations, environment state serialization and an
E0–E7 suite are implemented. No architecture, loss, optimizer, training or online
adaptation changes were made. `agent.act` arguments and checkpoint loading match
the existing evaluator. The legacy Phase 1 CLI remains supported through
`dynamics_shift/legacy.py`; default training/evaluation config disables the new
benchmark wrapper. Benchmark nominal is an identity action wrapper used only to
retain diagnostic clocks. Gains/physics parameters are never added to observations.

**Local validation is based on NumPy toy environments and SAPIEN API fixtures. It
is not evidence of successful real physics or checkpoint evaluation.** No checkpoint
exists in the workspace and the available Python runtimes lack PyTorch, Gym,
Gymnasium, Hydra, ManiSkill2 and SAPIEN. Actual simulator and Hydra composition tests
are supplied but skipped locally. No performance CSV or frozen success rates have
been produced. See `results/dynamics_shift_benchmark/validation.txt` and
`validation.json` for the executed checks. Current result: **24 passed, 4 skipped (3 simulator, 1 Hydra)**.
The parameter/schedule matrix test exercises 46 supported combinations and verifies
that the two continuous-delay combinations are rejected.

## Support matrix

| Parameter | nominal | fixed | abrupt | repeated | sinusoidal | linear_drift | random_piecewise | sequence |
|---|---|---|---|---|---|---|---|---|
| action_gain | yes | yes | yes | yes | yes | yes | yes | yes |
| action_bias | yes | yes | yes | yes | yes | yes | yes | yes |
| action_coupling | yes | yes | yes | yes | yes | yes | yes | yes |
| action_delay | yes | yes | yes | yes | error | error | integer values | integer values |
| object_mass | reset | reset | reset | reset | reset sampling | reset sampling | reset | reset |
| contact_friction | reset | reset | reset | reset | reset sampling | reset sampling | reset | reset |

`reset` means sample the schedule during `reset()` and hold that value for the entire
episode. **All physics wrappers report `mid_episode_supported=false`; `application=step`
is rejected.** These wrappers do not claim continuously varying in-episode physics.
Use `clock=episode_index` for explicit episode-boundary interventions. With a global
native/decision clock, an event that falls within an episode is deferred until the
next reset; the applied parameter and actual application clock are logged.

Physics support is restricted to PickCube's single free object. Other tasks require
an actor/cache adapter and are rejected. Unknown schedules/parameters, shape
mismatches, non-finite values, non-positive physical multipliers, negative gains,
non-integer delay, conflicting clocks, exposed parameters/actions and non-frozen
protocol execution are errors. One parameter axis is applied per environment;
compositions of multiple shifts are not implemented.

## Config and parameter semantics

Presets live under `tdmpc2/configs/dynamics_shift/`; Hydra group selection is
`configs/dynamics_shift@dynamics_shift=action_gain` (or the other preset names).
`nominal.yaml` contains the shared schema. Root protocol/split/probe fields remain
outside the environment config. Old `shift_mode!=nominal` and enabled structured
`dynamics_shift` may not be mixed.

- Gain: scalar or arm-sized vector; nominal is 1.
- Bias: scalar or arm-sized vector; nominal is 0.
- Coupling: target-sized matrix; nominal must be identity. The supplied preset is
  3x3 for `pd_ee_delta_pos`.
- Delay: integer **native-call** latency, independent of the schedule clock. The
  queue stores commanded actions at each native call, including both repeats.
  Missing history uses `parameter.initial_action` (default zeros). By default reset
  empties history; `reset_queue_on_episode=false` carries history across episodes.
  History is retained up to the largest configured delay, including nominal periods.
- Mass/friction: scalar multipliers relative to captured native properties; nominal
  is 1. No undocumented runtime mass setter is used.

`parameter.target=arm` preserves the last, gripper component from the **current**
command even for delay. `target=all` applies the transform to the entire action.
Actions are copied and clipped to each native bound; inputs are not mutated.
Nominal equivalence is defined for finite in-bounds actions.

Schedule definitions, with zero-based time `t` in the configured clock:

- nominal: nominal parameter.
- fixed: shifted parameter from reset/start.
- abrupt: nominal for `t < switch_step`, shifted afterwards.
- repeated: nominal/shifted alternate in blocks of `interval`.
- sinusoidal: `nominal + amplitude * (shifted - nominal) * sin(2*pi*t/interval)`.
  Thus gain nominal=1, shifted=1.3, amplitude=1 yields 1 ± 0.3. Amplitude is a
  dimensionless interpolation multiplier, unlike the legacy `shift_amplitude`.
- linear_drift: clamp `(t-switch_step)/duration` to [0,1], interpolate endpoints.
- random_piecewise: seeded uniform choice from explicit `schedule.values` every
  `interval` ticks. Duplicate consecutive choices are valid; RNG state is saved.
- sequence: explicit `{duration, value, label}` segments, holding the last value
  indefinitely. `action_gain_sequence.yaml` demonstrates A → B → A; add C for an
  unrestricted A → B → A → C experiment.

## Counters and boundaries

All indices are zero-based and logged **before** the native action:

| Field | Definition |
|---|---|
| native_step / global_step | Cumulative native calls, preserved across resets |
| episode_step | Native calls since reset |
| decision_step | Cumulative outer TD-MPC2 decisions |
| episode_decision_step | Decisions since reset |
| episode_id | Reset index, starting at zero |
| shift_segment_id | Schedule block/cycle/stage index |
| shift_change_count | Cumulative applied parameter changes, excluding initial assignment |

`ManiSkillWrapper` calls `begin_decision/end_decision`; both repeats share a decision
index. Native early termination still breaks the repeat exactly as before. Direct
native wrapper usage counts each call as its own decision. Episode clocks require
`reset_on_episode=true`; global clocks require false. `episode_index` requires false.

For continuous schedules the change count can increase every native call; segment
IDs represent full sinusoidal cycles or pre/ramp/post drift stages. Event callbacks
fire on segment boundaries and discrete parameter changes, including the initial
assignment. Initial assignment is not counted as an episode change event. Actual
native and decision indices identify a boundary inside an action repeat.

## SAPIEN/ManiSkill API evidence and limitations

Official references inspected:

1. [SAPIEN 2.2 API](https://sapien.ucsd.edu/docs/latest/apidoc/sapien.core.html):
   `ActorBase.get_builder/get_collision_shapes`, `ActorBuilder.set_mass_and_inertia`,
   `build`, dynamic mass/inertia/COM getters, velocity/pose setters, collision-shape
   material getter/setter and `Scene.create_physical_material`.
2. [ManiSkill2 0.4.1 BaseEnv](https://github.com/mani-skill/ManiSkill/blob/v0.4.1/mani_skill2/envs/sapien_env.py):
   previously downloaded source confirms `reset(seed=None,reconfigure=False)`,
   four-value step, `_actors` cache, `get_state/set_state`, and scene ground creation.
3. [Panda configuration](https://github.com/mani-skill/ManiSkill/blob/v0.4.1/mani_skill2/agents/configs/panda/defaults.py):
   arm followed by gripper in `pd_ee_delta_pos`.
4. [PhysX 4.1 material API](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/include/PxMaterial.h):
   default friction combine mode is average.

For mass, the retained actor builder reconstructs the object at reset with mass and
principal inertia both multiplied by the same factor. COM position/orientation,
pose, linear/angular velocity, collision groups and builder geometry/visuals are
preserved. The configured task attribute (`actor: obj`) and the BaseEnv actor cache
are replaced; the old actor is removed. Actor identity/segmentation IDs can change.
The original mass/inertia/COM are retained for restoration. Missing builder/cache
support raises an error. This reset-time implementation still requires real SAPIEN
verification; documented API availability is not a runtime pass.

For friction, new materials are assigned to the object's collision shapes and the
selected contact surface shapes. Both static and dynamic coefficients are scaled;
restitution is preserved. Shared original materials, including robot finger
materials, are not mutated. The default surface selector `@plane` requires exactly
one actor containing a `PlaneGeometry`; explicit actor names are also accepted.
Unknown/ambiguous targets fail. Original materials are restored before reset, and
new scene objects are rebound after `reconfigure=True`.

Readback is asserted after application/restoration. SAPIEN 2's documented Python
API does not expose combine-mode readback; logs explicitly label average as the
PhysX default, not a measured effective coefficient. Object/ground pairs have both
sides scaled, while object/finger pairs have only the object changed. Empirical
contact behavior and mass sensitivity remain unverified locally. Extra PickCube
source download was declined; target attributes are checked at runtime rather than
claimed as inspected. No support is asserted for SAPIEN 3 APIs.

## Running the benchmark

On the configured CUDA/ManiSkill2 0.4.1/SAPIEN host, from repository root:

```bash
python scripts/evaluate_shift_suite.py --checkpoint /absolute/path/pick-cube-1.pt
# Select a subset:
python scripts/evaluate_shift_suite.py --checkpoint /absolute/path/pick-cube-1.pt --ids E0 E2-mid E5-d1
# Preview commands without dependencies or checkpoint:
python scripts/evaluate_shift_suite.py --checkpoint /path/to/pick-cube-1.pt --dry-run
```

`benchmarks/dynamics_shift_suite.json` declares E0–E7 intensity, schedule and shared
seed settings. There are 14 runs including variants: nominal, three fixed gains,
both abrupt clocks, repeated, sinusoidal, two delays, two masses and two frictions.
All use the same checkpoint and episode seeds 1–50. The released single-task
checkpoints require model_size=5 (see README.md); the filename suffix is not a model
size declaration. For a custom checkpoint, use --model-size with its actual size. Checkpoint files are read only and SHA-256 verified at end.

From `tdmpc2/`, an individual mid-episode intervention after 50 decisions:

```bash
python evaluate_shift.py configs/dynamics_shift@dynamics_shift=action_gain task=pick-cube model_size=5 checkpoint=/absolute/path/pick-cube-1.pt eval_episodes=50 save_video=false dynamics_shift.schedule.type=abrupt dynamics_shift.schedule.clock=episode_decision_step dynamics_shift.schedule.reset_on_episode=true dynamics_shift.schedule.switch_step=50
```

Friction drift across episodes (sample-and-hold, no in-episode changes):

```bash
python evaluate_shift.py configs/dynamics_shift@dynamics_shift=contact_friction task=pick-cube model_size=5 checkpoint=/absolute/path/pick-cube-1.pt eval_episodes=50 save_video=false dynamics_shift.schedule.type=linear_drift dynamics_shift.schedule.clock=episode_index dynamics_shift.schedule.switch_step=0 dynamics_shift.schedule.duration=49
```

For randomized delay, select `action_delay` and override
`dynamics_shift.schedule.type=random_piecewise`,
`dynamics_shift.schedule.values=[0,1,2]`, `dynamics_shift.schedule.seed=42`.

## Outputs and reproducibility

Each run has a unique directory beneath
`<work_dir>/benchmark/<train|eval|probe>/<benchmark_id>-<type>-<schedule>-<UTC>-<id>/`.
`work_dir` follows the existing parser's logs/task/seed/exp_name convention.

- `metadata.json`: full config, protocol, split, seeds, environment/package/Python
  versions, checkpoint path/hash, git revision/status, Python source hashes.
- `code.diff`: tracked working tree diff against HEAD.
- `timesteps.jsonl`: native calls, all clocks, schedule/parameter, segment/change
  indices, commanded/executed actions, native reward/done and decision timeout done.
- `shift_events.jsonl`: actual segment/change events with native and decision clocks.
- `episodes.csv`: checkpoint, seed, episode/shift ID, return, success, decision/native
  lengths, events, NPZ path and SHA-256.
- `episode-NNNN.npz`: decision observation/next observation, **commanded** action,
  reward/done and seed. Native executed actions/parameters are explicitly prefixed
  `diagnostic_`; they are not substituted into learner transitions.
- `summary.json`: completion/failure status, completed-episode means and population
  standard deviations, checkpoint-integrity result. Failed runs retain partial logs.

Logging flags control step/event/episode logs. `benchmark_save_trajectory` controls
NPZ files separately. Empty trace/event files may remain when disabled. Only one
episode of trajectories is buffered. Video is independently optional.

## Resume, probe and future adaptation interfaces

`ShiftWrapper.state_dict/load_state_dict` stores clocks, counters, current segment,
previous parameter/events, schedule RNG and cached random value, plus action delay
history or held physical multiplier. Config/version mismatch is rejected. These
snapshots are independent of any learning algorithm.

`capture_environment/restore_environment` in `dynamics_shift/logging.py` additionally
save public simulator state, controller state, native RNGs and outer timeout counters.
Create/reset a matching destination with the same seed/assets before restore. Missing
controller state APIs are rejected. Full snapshots require a completed decision;
`benchmark_save_segment_state=true` writes snapshots immediately after the decision
containing a boundary and labels that timing. Exact pre-action segment callbacks are
available as `shift.boundary_callbacks` for future **model** checkpoint saves. No
model checkpoint is saved/modified automatically in this phase.

State uses lossless tagged JSON for arrays/tuples (`encode_state/decode_state`). The
state APIs do not claim to capture PhysX contact solver caches, global torch RNG,
TD-MPC2 MPC warm-start state or optimizer/replay state. Full agent-run resume needs
those caller-owned states; only wrapper/schedule replay is unit-tested bitwise here.

`common/shift_protocol.py` declares frozen, dynamics_only, value_only,
dynamics_value, dynamics_value_policy and full component sets. Only `FrozenProtocol`
executes; others raise `NotImplementedError` instead of silently running frozen.
The future connection is `protocol.observe(Transition(...))` in `evaluate_shift.py`.
`Transition.commanded_action` is the replay action; diagnostics remain separate.
Environment classes never read adaptation mode.

Use `benchmark_split=train|eval|probe` for separate output roots. For an A→B→A sequence
with withheld C=0.5, select `action_gain_sequence` and
`benchmark_probe_values=[0.5]`. Validation rejects overlap of reserved values with
train/eval schedules, including continuous ranges. Evaluate C separately with
`benchmark_split=probe`, fixed gain .5 and an explicit `eval_seeds=[...]` list. NPZ
files, recorded seeds and SHA-256 identify the fixed probe transition dataset;
reusing that saved dataset avoids collecting different probe data on subsequent runs.

## Validation commands

Machine-readable report plus captured test output:

```bash
/Users/jihyekim/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/validate_shift_benchmark.py
```

The full changed/added file list and execution limitations are also recorded in
`results/dynamics_shift_benchmark/implementation_report.json`. YAML syntax was checked
with Ruby's standard YAML parser; this does not substitute for Hydra composition.

```bash
/Users/jihyekim/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests -v
python3 -m compileall -q tdmpc2 tests scripts
git diff --check
python3 scripts/evaluate_shift_suite.py --checkpoint /path/to/pick-cube-1.pt --dry-run
```

On the simulator host:

```bash
TD_MPC2_TEST_MANISKILL=1 python -m unittest discover -s tests -v
```

The real tests compare identical initial state/action sequences, assert nominal
transition/reward/success equivalence and action-shift effects. Physics tests check
readback/restoration/reconfiguration and compare a controlled lateral-force response
under the same action sequence; this is a sensitivity test, not an agent score.
Hydra composition is tested when Hydra is installed. All these integration checks
must pass on the target host before treating physics results as validated.

## File map

- `envs/wrappers/dynamics_shift/{schedule,base,action_shift,physics_shift,factory,logging}.py`:
  schedules, counters/events, parameter transforms, physical API adapter, construction,
  state serialization. `legacy.py` preserves Phase 1 and `__init__.py` exports both.
- `envs/maniskill.py`: insertion, decision brackets and native repeat trace retention.
- `envs/__init__.py`: explicit benchmark dispatch so validation errors are preserved.
- `envs/wrappers/tensor.py`: reset keyword forwarding from Phase 1.
- `config.yaml`, `configs/dynamics_shift/*.yaml`: defaults, protocol fields and presets.
- `evaluate_shift.py`: new benchmark entrypoint. Existing `evaluate.py` retains Phase 1.
- `common/{shift_protocol,shift_metrics,benchmark_logging,shift_evaluation}.py`: learning
  boundary, statistics, benchmark/legacy artifacts and provenance.
- `benchmarks/dynamics_shift_suite.json`, `scripts/evaluate_shift_suite.py`: declarative suite.
- `tests/test_shift_{benchmark,physics,state_and_logging,config}.py`,
  `tests/benchmark_imports.py`: dependency-light unit/API fixture/config tests.
- `tests/test_maniskill_integration.py`: optional real action/physics tests; earlier
  Phase 1 tests remain in `test_dynamics_shift.py` and `test_shift_logging.py`.

## Checkpoint shape mismatch troubleshooting

A released single-task checkpoint loaded with `model_size=1` can fail with
`Tensor weight has shape [5, 512] ... batch-size [2]`. The 1M preset creates two Q
networks, whereas the released 5M single-task architecture uses five. Use
`model_size=5`; do not slice weights, change `num_q` alone, or disable strict loading.
The suite default and legacy script have been corrected. The suite can also take
`--model-size 5` explicitly. A preceding GLFW/DISPLAY warning is not the stopping
error in the supplied trace: execution continued to checkpoint loading.
