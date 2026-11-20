# Adaptation methods and behavioral equivalence — Phase 1B-3

## Pre-migration difference matrix

| Contract | Frozen | Dynamics-only | Decision TTT | Dynamics+Q |
|---|---|---|---|---|
| Transition | (observation, commanded action, next observation), detached CPU clones | Same | Same | Dict with those three float32 CPU clones, reward, discount, terminal/truncation, episode/step, physics/identity metadata |
| Replay | Bounded deque of tuples | Same | Same; decision observes and may update inside episode | Bounded deque of dicts; one explicit next observation per real transition |
| Sampler/RNG | No samples; seeded `random.Random` retained | `rng.sample(list(replay), batch_size)`, without replacement | Same inherited sampler | Same call, one 64-record batch shared by D and Q |
| Trigger | None | Episode end, after max(warmup, batch) | Eligible decision index during episode, after max(warmup, batch), no repeated update at one index | max(warmup, batch) then transition index divisible by 25; first at 75 |
| Objective | None | One-step encoded-next-latent MSE on commanded action | **Exact inherited Dynamics-only objective** | Same one-step MSE plus separate categorical soft cross-entropy Q objective |
| Optimizer | None | One Adam for dynamics, configured LR | Same inherited optimizer/step | Independent dynamics and Q Adam optimizers, original betas/eps/LRs, zero weight decay |
| Gradient clipping | None | Dynamics clip after coefficient-scaled backward | Same | Separate dynamics and Q clipping after separate coefficient-scaled backward |
| Mutable model | None | Dynamics | Dynamics | Dynamics, online Q, target-Q by EMA |
| Fixed model | All components | Encoder, reward, Q, policy, target | Same | Encoder, reward, actor; target-Q EMA is permitted |
| Model mode | Eval throughout | Eval throughout | Eval throughout | Eval normally; online Q train/dropout only within update; target Q remains eval |
| State | Episode state v1, replay/RNG/counters and no optimizer | Same with optimizer | Episode state plus scheduler and last decision | Original D+Q optimizers, replay, RNG, transition/D/Q/target counters and elapsed time |

These differences are based on the original `common.stage3a`, `common.decision_ttt` and `common.dynamics_q` implementations at `76563b5`. No new objective, schedule or replay distribution is introduced.

## Canonical ownership and unchanged semantics

`DynamicsAdaptation` owns the original episode predictor behavior. `DecisionAdapter` inherits its exact `loss` and `_optimizer_steps`, adding only decision scheduling, decision records and scheduler serialization. `DynamicsCriticAdaptation` owns D+Q's existing losses, two optimizers and target critic EMA. The pure `one_step_dynamics_mse` helper expresses the identical `F.mse_loss(model.next(z, action, None), next_z)` call made by both. The original agent `_td_target` and `common.math.soft_ce` remain authoritative for D+Q.

`AdaptationReplayBuffer` subclasses `deque`, preserving maxlen, insertion order, list serialization and the exact `rng.sample(list(self), batch_size)` random call. It does not replace TD-MPC2's training replay. `UpdateSchedule` reads the same settings mapping used by each adapter; its `ready` and `due` predicates reproduce the original warmup and modulo checks. It has no RNG and adds no checkpoint field. D+Q still makes **118 dynamics, 118 critic and 118 target updates after 3000 transitions** under the existing 64/25 contract.

`ParameterInvarianceAudit` captures frozen state with each original policy: Dynamics-only moves tensor snapshots to CPU and deep-copies non-tensor metadata; D+Q deep-copies the fixed-state subset. Public `initial_fixed`, `fixed_state`, `assert_fixed` and D+Q `parameter_audit` stay available. D+Q also checks absent gradients outside its two optimizer parameter groups.

The old public `observe`, `update`, `state_dict`, `load_state_dict`, `held_out`, and metric methods retain their signatures, guards, return values and state keys. The only public class metadata that changes is `__module__`, because classes now live in the canonical package. Historical snapshots store state dictionaries, not adapter objects; the golden checkpoint tests and production loaders check their supported schemas. Exact continuation remains unsupported for historical files missing replay, environment and global RNG state.

The original frozen CPU and Linux fixtures preserve historical source hashes. Tests explicitly separate those path hashes from current code ownership and still compare all recorded inputs, RNG states, losses, model outputs, gradients, optimizer summaries, EMA, update counts and planner traces. A numerical difference remains a test failure.

Joint RL now uses the canonical event runner and `JointAdaptation`, which delegates each learning step to the original agent. `ExistingRunnerAdapter` remains a full-stream factory compatibility delegate; old `common.*` modules are import shims. Native planning, evaluation runners, historical results and figures are unchanged. ManiSkill2/SAPIEN parity requires its own compatible runtime.

## Unified NS-Gym runner behavioral matrix

| Contract | Frozen / Dynamics-only / Decision | Dynamics+Q | Joint RL |
|---|---|---|---|
| Environment | HalfCheetah `SeverityHalfCheetah` or Reacher `SeverityReacher`; original task constructor | Same task split, original constructors | Same task split, original constructors |
| Agent/checkpoint | Original task-specific nominal loader; evaluation-frozen agent | Same nominal loader, fresh D/Q optimizers | `training_support.load_checkpoint`; restore both nominal Adam states and scale, then enable optimizer-owned parameters |
| Collection | `agent.act(..., t0=first step, eval_mode=True)` | `eval_mode=True`; Reacher retains `r.action` contract | `eval_mode=False`; one CUDA graph step mark before each action |
| Transition/replay | `(observation, commanded action, next observation)` to bounded adaptation deque | Real observation, action, actual next observation, reward and physics metadata to bounded adaptation deque | TensorDict dummy initial row plus real transitions; original `common.buffer.Buffer` receives **completed episodes only** |
| Boundaries | Episode end calls `adapter.end_episode`; Decision may update during episode | Observe may update each eligible transition; reset on HalfCheetah termination/truncation, Reacher truncation | Only truncation commits replay, horizon must match; reset only after truncation |
| Update trigger | Frozen none; Dynamics at episode end after warmup; Decision every configured decision after warmup | 64-transition warmup and every 25 transitions; 118 D/Q/EMA updates in 3000 | HalfCheetah first at 1000 then every transition: 2001 in 3000; Reacher first at 75 then every 25: 398 in 10000 |
| Learning owner | Canonical Dynamics/Decision strategy; same one-step dynamics loss | Canonical DynamicsCritic; one-step D MSE plus categorical Q CE | `JointAdaptation` calls **unchanged `TDMPC2.update(Buffer)`**, retaining multi-position rollout, reward, Q, actor, target EMA and optimizer order |
| RNG | Global action/planner RNG plus strategy-local replay sampler | Same with D+Q local sampler | Global planner/action RNG and original TorchRL `SliceSampler`; runner adds no random calls |
| Optimizers/mode | Frozen none; one dynamics Adam; model eval | Independent D/Q Adam; Q train/dropout during update, other components frozen | Restored model and actor Adam states, training/dropout set by original `_update` |
| Snapshot/resume | Existing versioned adapter state, replay and local RNG; stream snapshot lacks environment/planner state | Existing D+Q model/adapter state; stream snapshot lacks environment/planner state | Existing `model/optim/pi_optim/scale` payload, evaluation-only resume marker; no exact continuation |
| Evaluation/output | Existing CSV/NPZ, physical diagnostics, optional snapshots and held-out guards | Existing JSONL, CSV, snapshot and held-out scripts | Existing training JSONL, completion, task-specific explicit evaluation and provenance |

The common runner owns only environment and event order. Each old script still assembles its original manifest, task metrics and snapshot bytes. HalfCheetah and Reacher have different observation/action sizes, horizons, discounts, physical readback and seed/checkpoint selection; callbacks retain those contracts. `common.stage3b.run` and `adapt_shift.episode` remain the separate ManiSkill2 native-step path with exact simulator/controller/planner/RNG resume support.

Controlled runner tests compare old and new episode/Decision, D+Q and Joint step loops for actions, real transitions, replay insertion and samples, RNG state, update positions, optimizer counters, checkpoint payloads and logged records. The one-update test executes the original `TDMPC2.update()` on an identical saved CPU batch through direct and `JointAdaptation` paths, comparing losses, all model tensors including target Q, both optimizer states, scale and RNG. CUDA and simulator parity are separate execution gates.
