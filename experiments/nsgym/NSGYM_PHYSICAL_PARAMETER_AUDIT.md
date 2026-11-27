# NS-Gym HalfCheetah physical parameter audit

Status: **local CPU source inspection and bounded physics smoke PASS; server inventory and 750k CUDA experiments NOT_RUN**. The server is operated by the user. No training, adaptation or large sweep was launched.

## Adapter and source boundary

The existing `tdmpc2/envs/nsgym/halfcheetah.py` wraps Gymnasium HalfCheetah-v5 with official `MujocoWrapper`, `StepWiseUpdate` and `DiscreteScheduler`. It supports torso-mass nominal/fixed/abrupt/reversible schedules. `envs.nsgym.make_env` applies the existing tensor wrapper, rejects enabled legacy dynamics shifts and exposes only 17 state observations and scalar reward. Existing training/model/planner/checkpoint code was not edited.

The new experiment-only `tdmpc2/envs/nsgym/severity.py` inherits that observation/reward boundary and compiles absolute parameter targets into the same official update functions. It does not implement new physics. It exposes a single official scalar parameter per condition; the full inventory is broader than the initial experimental grid.

Inspected installed source: `/private/tmp/tdmpc2-nsgym-modern/lib/python3.11/site-packages/ns_gym/{base.py,schedulers.py,wrappers/mujoco_env.py,update_functions/single_param.py}`. All four files were byte-identical to the local upstream checkout at `127df4b714fe4fcf680721d8e9278a96090acd7e`. Local NS-Gym reports `0.0.post1`; the original server metadata reports `1.0.12.post2`. Therefore version-label equality is **not** asserted. Every server run records actual versions, executable, repository revision and installed-source hashes. The inventory stage also saves the actual wrapper/base/Gymnasium environment source. Local Gymnasium 1.3.0, MuJoCo 3.3.7, NumPy 2.4.6 and Torch 2.14.0 were observed. Server compatibility must be checked with the provided environment/inventory commands; learned-model stages require NS-Gym/Gymnasium/MuJoCo version equality with the checkpoint's nominal provenance.

## Official inventory and local nominal readback

The keys below are enumerated from installed `NSWrapper.TUNABLE_PARAMS` and resolved through `param_look_up('HalfCheetahEnv', key)`. `ffeet_*` is the upstream spelling and targets `ffoot`. The generated CSV includes exact setter source and indices.

| Official parameter | Target / scope | Local nominal | Unit | Wrapper constraint |
|---|---|---:|---|---|
| gravity | `model.opt.gravity`; update changes z only | [0, 0, −9.81] | m/s² | no gravity-specific bound |
| torso_mass | torso `body_mass` | 6.25020920502092 | kg | >1e−6 |
| bthigh_mass | bthigh `body_mass` | 1.5435146443514645 | kg | >1e−6 |
| bshin_mass | bshin `body_mass` | 1.5874476987447697 | kg | >1e−6 |
| bfoot_mass | bfoot `body_mass` | 1.0953974895397491 | kg | >1e−6 |
| fthigh_mass | fthigh `body_mass` | 1.4380753138075317 | kg | >1e−6 |
| fshin_mass | fshin `body_mass` | 1.200836820083682 | kg | >1e−6 |
| ffeet_mass | ffoot `body_mass` | 0.8845188284518829 | kg | >1e−6 |
| floor_friction | floor `geom_friction[geom_id,0]` | 0.4 | dimensionless | no friction-specific check |
| bthigh_damping | bthigh `dof_damping` | 6.0 | N·m·s/rad | ≥0 |
| bshin_damping | bshin `dof_damping` | 4.5 | N·m·s/rad | ≥0 |
| bfoot_damping | bfoot `dof_damping` | 3.0 | N·m·s/rad | ≥0 |
| fthigh_damping | fthigh `dof_damping` | 4.5 | N·m·s/rad | ≥0 |
| fshin_damping | fshin `dof_damping` | 3.0 | N·m·s/rad | ≥0 |
| ffeet_damping | ffoot `dof_damping` | 1.5 | N·m·s/rad | ≥0 |

All mapped parameters can be updated before actions through the official wrapper. The reported bounds are source constraints, not validated engineering operating ranges. No upper bound is imposed in the inspected mass/damping checks. Gravity and friction are inventoried but excluded from the initial mandatory sweep. The positive scalar protocol does not silently treat a vector or negative gravity component as an interchangeable multiplier.

## Update path and physical meaning

`SeverityHalfCheetah.step → MujocoWrapper.step → UpdateFn(current_value, t) → DiscreteScheduler → StepWiseUpdate absolute target → constraint checker → official setter → _dependency_resolver → Gymnasium.step`.

Damping changes one back-thigh hinge's viscous resistance, conceptually the corresponding diagonal term of `D qdot` in `M(q) qddot + C(q,qdot) qdot + g(q) + D qdot = tau + Jᵀ f_contact`. It is not motor torque degradation. The upstream damping index uses joint ID; this is checked against `jnt_dofadr` at runtime. All HalfCheetah joints here are scalar slide/hinge joints. The initial scope is **only bthigh**, not all six actuated joints.

The mass setter changes **only torso body_mass**. It does not rescale `body_inertia`, move the torso local inertial position (`body_ipos`), or change geometry. The system/subtree mass and center of mass can change when weighting the same body positions differently. NS-Gym invokes `mj_setConst` to recompute derived constants and `mj_forward`, preserving/restoring `mjSTATE_INTEGRATION` around the operation. A mass-only edit is therefore not uniform density scaling of a rigid body. Tests verify unchanged inertia and local inertial position; model-array checks distinguish derived constants from independent physical parameters.

The local actuator gear magnitudes are [120, 90, 60, 120, 60, 30]; the action is the original six-dimensional normalized control in [−1,1]. Damping/mass edits leave gear/control ranges unchanged. Gymnasium reward is `1.0 * x_velocity - 0.1 * sum(action²)` in the observed runtime. It has no direct mass/damping term; changed velocity and actions can change reward. HalfCheetah-v5 does not terminate for falling; the 1000-step time limit truncates it. Time-limit completion is not a success rate.

## Leakage, clocks and reset

NS-Gym's observation contains privileged metadata, and its info can include change notifications even when disabled. The inherited adapter strips observation to `state` and allowlists x position/velocity and reward components. The policy sees no schedule, severity, parameter or notification. Diagnostic files are experiment-owned. Frozen policies can react to new states without identifying a hidden physical parameter.

The official scheduler uses zero-based episode-local transitions before action, with five MuJoCo substeps per transition (`dt=0.05 s`). Reset restores nominal values and update-function state. Fixed changes apply on action 0, not on reset. Global step is an independent logged counter. Full-state restoration does not restore a diverged trajectory when physics is reversed.

Nominal equivalence was checked against unwrapped Gymnasium with an inactive legacy-adapter mass scale of 0.7 and an inactive new severity of 0.5. A regression test compares disabled legacy `dynamics_shift.shift_gain=0.7` with 1.0 through `make_env`; inactive config does not reach the NS-Gym physical setter. Unsupported kwargs are rejected rather than interpreted as hidden shifts.

## Measured smoke evidence and limitations

Preserved outputs: `results/nsgym/severity-local-smoke-20261110/` (first bounded run), with a final validation rerun documented in `validation/physical-severity/VERIFICATION.md`. Both families were checked at k=1 and k=1.5 over the six schedules. The initial same-state probe used one explicitly labelled bounded exploration seed, two initial states, and three-step action tapes. Nominal repeats matched; measured mean one-step joint velocity differences were about 0.08747 rad/s for bthigh damping and 0.15765 rad/s for torso mass at k=1.5. These are **local smoke measurements**, not policy-bank calibration or a severity ranking.

The probe records full integration state, warmstart/control/applied-force state, qpos/qvel/qacc, torso quaternion/body velocity, actuator controls/forces and contact snapshots/forces. Exact nominal replay is checked from saved state before making controlled comparisons. This claim is limited to the recorded runtime and tolerance; contact snapshots are at control-step boundaries, not every solver substep. No CUDA checkpoint inference was executed locally. Server stages remain NOT_RUN until their completion records exist.
