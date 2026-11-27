# Dynamics+Q implementation audit

The inspected implementation is the NS-Gym HalfCheetah 750k pipeline. No existing
Frozen, Dynamics-only, Joint RL, checkpoint, or evaluation artifact is modified.
New code: `tdmpc2/common/dynamics_q.py`, `scripts/run_dynamics_q.py`,
`scripts/analyze_dynamics_q.py`, `configs/dynamics_q.json`, and
`tests/test_dynamics_q.py`. Paths here are relative to the server project root
except `scripts/`, `configs/`, and `tests/`, which are relative to experiments/nsgym.

## Original parameter ownership, inspected before implementation

| Component | Original owner | Dynamics+Q owner |
|---|---|---|
| `_encoder` | `agent.optim`, nominal lr × enc_lr_scale | frozen, no optimizer membership |
| `_dynamics` | `agent.optim` | fresh dynamics Adam |
| `_reward` | `agent.optim` | frozen |
| `_Qs.params` | `agent.optim` | fresh Q Adam |
| `_detach_Qs_params` | alias of online Q data | follows online Q through shared storage; not a separate model |
| `_target_Qs_params` | `soft_update_target_Q()` | same EMA, once after each Q step |
| `_pi` | `agent.pi_optim` | frozen |
| termination head | conditional original model optimizer group | absent for episodic=false |
| `agent.scale` | updated by `update_pi` | frozen; not needed to compute the TD target |
| MPPI previous mean | planner warm start | inference state, reset at each episode start; not learned weights |

The loader restores nominal model, both original Adam states and RunningScale,
then freezes all weights. The new method leaves those original optimizer states
untouched and creates two fresh Adams. `agent.update`, original model optimizer
step, and original policy optimizer step are forbidden by the existing loader.
Only dynamics and online Q parameters regain `requires_grad=True`.

## Actual TD-MPC2 targets, distributions and planner

`TDMPC2._td_target` uses **real replay reward** plus
`discount * (1 - terminated) * target_Q(next_z, actor(next_z))`.
The action is sampled by the current actor; there is no separate target actor.
`WorldModel.Q(... target=True, return_type='min')` randomly samples **two of five**
target heads, decodes each head to a scalar, and takes their minimum. There is no
SAC entropy subtraction in this TD target.

`common.math.soft_ce` projects the scalar TD target with `two_hot`: symlog,
clipping to [-10, 10], then interpolation between neighboring bins of the
101-bin support. This is a two-hot encoding of a scalar Bellman target, not a
new C51 projection of the complete next-state categorical distribution.
`two_hot_inv` computes **symexp of the expected symlog support**, not the expected
symexp of each support point. Those exact functions are reused.

Target Q follows `target.lerp_(online_detached, tau)` with verified nominal
`tau=0.01`. Each new Q update calls the original method and checks the resulting
EMA tensors. The target parameters never belong to either adaptation optimizer.

MPPI retains the nominal horizon 3, 512 candidates, 6 iterations and other
checkpoint settings. It rolls through dynamics and the frozen reward model,
then appends a discounted terminal value using the mean of two sampled online
Q heads. Thus updating Q changes the planner's terminal value, even with the
actor and reward head fixed. `eval_mode=True` suppresses the final additive
exploration noise; MPPI candidate sampling and elite selection are still random.

## New gradient and supervision contract

For a uniformly sampled batch of real `(o, action, next_o, reward, terminal)`:

1. Encode `z=h(o)` and `next_z=h(next_o)` under no-grad, with the unchanged nominal
   encoder and unchanged float32 preprocessing.
2. Compute the original `_td_target(next_z, real_reward, terminal, None)` under
   no-grad. The frozen learned reward head is **not** substituted for real reward.
3. Dynamics loss is the same one-step MSE as predictor-only TTT, coefficient 1.
4. Q loss is original soft categorical CE on `Q(z.detach(), real_action)`, averaged
   across all five heads and the batch, coefficient 0.1.
5. Separate backward/optimizer steps update dynamics and Q respectively. Assertions
   reject cross-group gradients. Online Q uses its original training dropout;
   dynamics retains the TTT eval-mode behavior and target Q remains eval-mode.
6. Apply original target-Q EMA once. Freeze checks cover all other parameters and
   buffers; online-Q alias storage is correctly recognized as mutable.

Original Joint RL unrolls dynamics for horizon 3, evaluates Q on actual initial
and subsequently imagined latents, weights time steps by rho, and permits value
gradients into encoder/dynamics. The new ablation intentionally uses a **one-step,
actual-latent Q update with isolated gradients**. It does not claim equivalence
to the entire Joint RL value-training objective.

## Replay and time semantics

Replay is a fresh capacity-10,000 deque of aligned real transitions, sampled
uniformly without replacement with a stream-local Python RNG. Dynamics and Q use
the same 64 records at each event. Every record stores observations, commanded
action, final next observation, real reward, discount, termination, truncation,
episode index, episode-local step, global step, adaptation seed, parameter,
severity, nominal value and actual readback. No nominal replay is imported.

Discount is 0.995. HalfCheetah truncates at 1,000 transitions; true termination
is rejected by the benchmark contract. Truncation does **not** zero bootstrap.
The stored next observation is the final observation before reset, never the next
episode's initial observation. Held-out observation insertion and updates are
forbidden. Snapshot replay preserves the complete 3,000-transition record.

## Optimizers and accounting

| Setting | Dynamics Adam | Q Adam |
|---|---:|---:|
| Learning rate | 1e-4 | 3e-4 |
| Loss coefficient | 1 | 0.1 |
| Gradient clip | 10 | 20 |
| Betas / epsilon / weight decay | (0.9,0.999) / 1e-8 / 0 | same |
| Initialization | fresh | fresh |
| Batch | 64 | same 64 transitions |
| Steps at final 3,000 transitions | 118 | 118 |

Q lr, coefficient and clipping match the actual local Frozen baseline's recorded
750k configuration (`model_integrity.json`), not an inferred generic RL default.
The real runner asserts those values against the loaded checkpoint. Each run
records exact parameter counts and optimizer groups. No RunningScale update or
actor optimizer call is needed for this target.

Warmup 64 and cadence 25 give updates at 75,100,...,3000: 118 events, 236 optimizer
steps, 118 EMA updates. Replay/optimizers persist across three episodes, but a
new condition/seed starts independently from nominal. Original Joint uses 2,001
joint model steps plus 2,001 actor steps; it is not compute matched.

## Validation and limitations

Unit tests use the actual small CPU WorldModel, original `_td_target`, original
distributional math and original MPC terminal-value calculation. They cover
finite targets, support normalization, random target-head reduction, reward and
discount alignment, TimeLimit bootstrap, gradient isolation, exact EMA, replay
copy/boundaries, held-out mutation protection, snapshot round-trip, and actual
3,000-transition/118+118-step execution. Aggregation tests retain negative effects
and require all 36 streams. CPU fixture smoke is explicitly marked synthetic.

Local CUDA/750k physical smoke and all research runs remain NOT_RUN until executed
on the server. `--cpu-fixture` cannot authorize the full matrix. No scientific
performance claim follows from these contract tests.
