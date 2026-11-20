# Adaptation methods and behavioral equivalence — Phase 1B-2

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

Joint RL remains delegated to its original runner. `ExistingRunnerAdapter` is the only forwarding wrapper in `research/dynamics_shift`; old `common.*` modules are import shims. No native planner, evaluation runner, historical result or figure is changed. ManiSkill2/SAPIEN parity requires its own compatible runtime and cannot be inferred from NS-Gym.
