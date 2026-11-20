# Adaptation methods and invariants — Phase 1B-1

Decision TTT and Dynamics-only use the **same learning objective and optimizer step**. `DecisionAdapter` inherits `EpisodeAdapter.loss` (one-step latent dynamics MSE on commanded actions) and `EpisodeAdapter._optimizer_steps` (Adam, gradient clipping, uniform sampling without replacement). It overrides `observe`, `update` and `end_episode` to run updates at eligible decision indices inside an episode instead of at episode boundaries. The different schedule, replay timing and checkpoint scheduler fields are scientifically material and remain distinct.

| Method | Original constructor / entry point | Update timing | Mutable learned components | Original state format |
|---|---|---|---|---|
| Frozen | `EpisodeAdapter(model, settings, coef, clip, seed)` | None | None | Episode adapter version 1 state |
| Dynamics-only | Same constructor | Episode end after warmup | Dynamics only | Episode adapter version 1 state |
| Decision TTT | `DecisionAdapter(model, settings, coef, clip, seed)` | Eligible policy decisions | Dynamics only | Episode state plus scheduler and last decision |
| Dynamics+Q | `DynamicsQAdapter(agent, config, identity)` | 64-transition warmup, every 25 transitions; first at 75 | Dynamics, online Q and target-Q EMA | Existing D+Q adapter state |
| Joint | `run_full_joint_rl.run_stream(args, config, identity, target, budget)` | Original episode/replay-dependent agent update | Original joint optimizer-owned components | Existing Joint snapshot and optimizer format |

`compose_adaptation` selects the existing class for Frozen, Dynamics-only, Decision TTT and Dynamics+Q. It wraps the created object without copying its replay, RNG, optimizer or model. The optional `run_stream` callback is bound without modifying its arguments. Joint requires that callback and does not instantiate a substitute training algorithm.

The original TD-MPC2 training replay (`common.buffer.Buffer`) is not replaced. `AdaptationReplayBuffer` names only the shared storage view (`len` and iteration); each implementation keeps its own record schema, sampler and RNG. In particular, D+Q still samples 64 distinct real transitions and uses actual next observations for categorical TD targets; Joint still uses the original slice sampler over completed training episodes.

The immutable CPU and Linux numerical fixtures remain unchanged. Their source hashes refer to the pre-refactor implementations, which this phase leaves at their original paths. A future source move must record path/hash provenance separately from any numerical comparison and must never refresh a fixture to conceal changed outputs.

Phase 1B-1 does not alter evaluation, checkpoint serialization, figure generation or historical artifacts. Exact training continuation remains unsupported by historical snapshots lacking replay, environment and global RNG state. ManiSkill2/SAPIEN checks require their separate runtime and are not inferred from NS-Gym results.
