# Frozen world model mechanism analysis (Stage 2)

Implementation is available, but **Stage 2 is not certified complete** until the
real official-checkpoint server smoke tests pass. No learning, optimizer step,
parameter adaptation, or architecture change is introduced.

## Running through Benchmark v2

Use the existing CUDA + ManiSkill2 environment described in
`SETUP_SHIFT_BENCHMARK.md`. The three official checkpoints are external files;
none were downloaded or modified by this work. Commands are run at repository root.

```bash
python tdmpc2/evaluate_shift.py task=pick-cube model_size=5 \
  checkpoint=/absolute/path/pick-cube-1.pt episodic=false compile=false \
  analysis=frozen_mechanism dynamics_shift=action_delay schedule=fixed \
  dynamics_shift.parameter.shifted=2 evaluation=paired evaluation.episodes=5 \
  seed=1 save_video=false
```

`analysis=disabled` is the default. It creates no mechanism directory, attaches
no observer, performs no diagnostic forward passes and adds no native record
fields. The config metadata gains the disabled analysis group. Enabled analysis
requires `compile=false`; Python tracing inside CUDA graph capture is unsupported
and rejected instead of silently changing compilation. Existing frozen evaluation,
Hydra groups, checkpoint format, action sampling, and `_prev_mean` are retained.
Unsupported probe rules and invalid config values raise errors. Only single-task
state observations are supported.

Review the complete smoke matrix without executing it:

```bash
python scripts/evaluate_mechanism_suite.py --checkpoints /absolute/checkpoint/directory
```

Run five episodes **per condition** (paired evaluations also run a nominal arm):

```bash
python scripts/evaluate_mechanism_suite.py --checkpoints /absolute/checkpoint/directory \
  --episodes 5 --execute --report mechanism-smoke-status.json
```

After checking smoke artifacts and separate simulator validation, run 50:

```bash
python scripts/evaluate_mechanism_suite.py --checkpoints /absolute/checkpoint/directory \
  --episodes 50 --execute --report mechanism-full-status.json
```

The matrix covers PickCube-1 nominal, delay 2, gain 0.7, abrupt delay at native step
10; PickCube-3 delay 2; StackCube-3 delay 2. Model size is 5 and episodic is false,
as in the existing official checkpoint setup. The evaluator rejects checkpoints
whose termination-head presence disagrees. Change the configuration for differently
trained checkpoints. Run the same suite with `--actor-only` to use `mpc=false`.
Use a separate report path so reports from separate invocations are retained.

## Verified code semantics and tensor shapes

For this single-task state path, observations/actions entering `act` are `[O]`
and `[A]`. `encode` receives `[1,O]` and returns `[1,Z]`; `next` preserves the
latent batch shape. The planner repeats latent states to `[N,Z]` and evaluates
candidate actions `[H,N,A]`. Reward logits are `[N,B]` and all Q-head logits are
`[K,N,B]`, where `B=max(num_bins,1)`. `math.two_hot_inv` decodes each head before
aggregation, yielding `[K,N,1]` for distributional heads. The scalar/no-bin and
single-bin special cases use that same repository decoder.

`ManiSkillWrapper.step` sends one controller command to up to two native steps,
sums native rewards, and stops early on native done. `TensorWrapper` converts
actions without rescaling. Benchmark wrapper commands use the same normalized
Box; execution substitution is allowed only after checking shape, [-1,1] bounds,
matching recorded commands, exactly two native steps, and identical executed
commands in both repeats. These commands are **controller inputs, not actuator
torques**. Gain/delay transformations occur before the native controller.

The evaluator calls `agent.act(..., eval_mode=False)`. MPC samples policy priors,
Gaussian candidates, a Q-head pair and a Gumbel elite, then adds exploration noise.
Actor-only uses the sampled squashed Gaussian action. `eval_mode=True` would use
the actor mean and suppress MPC's final noise; this feature does not change the
existing evaluator to that mode. Actor-only planner metrics are NOT_APPLICABLE.
The two baselines share checkpoint/config/seeds, but closed-loop observations may
diverge; a same-observation action comparison is not implied.

## Actual planner instrumentation

`TDMPC2._estimate_value` captures reward sum and the exact terminal Q contribution
already computed by the original planner. No second Q call is used for this score.
`_plan` captures candidates and top-k elites immediately after their actual
ranking, and the original candidate ID plus the final, noisy/clipped first action
after Gumbel selection. All source tensors are copied without mutation. The
arithmetic and random-operation order of the action path remain unchanged.

Actual score uses the sampled terminal policy action and the randomly selected
Q-head pair from `Q(return_type='avg')`. Reward and terminal contributions include
the exact discount and cumulative predicted termination mask. Reward at a step
is included before that step's predicted termination masks subsequent rewards.
`actual_planner_score` includes the original `nan_to_num(0)` handling; component
sums equal it for finite values. Nonfinite component rankings are UNVERIFIED.

Diagnostic rescoring runs **after the actual environment step**, on saved actual
candidates from the same encoded observation. It uses policy mean and all decoded
Q heads' mean, and repeats the same reward/termination accumulation. It is stored
as `diagnostic_score`, never as actual score. It measures model-internal preference,
not true candidate return. Even obtaining the policy mean invokes stochastic
`pi`, so diagnostic calls preserve and restore Python, NumPy, CPU Torch and all
CUDA RNG states. The planner is never run a second time for diagnostics.

One active decision's full candidate set is held transiently; all rankings use
all N candidates. Disk storage is limited per iteration by
`max_logged_candidates`. IDs are deterministic first IDs, except the final
selected ID replaces the last stored ID if necessary. No sampling is added and
no planner candidates are removed. This subset is not a random representative
sample. Candidate IDs are local to each decision/iteration. `policy_prior` and
`sampled` are saved separately. Sampled slots change trajectories across iterations;
`elite_slot_overlap` is explicitly slot overlap, not identical-trajectory overlap.
The selected action may differ from its candidate's first action due to final noise.

Ranking metrics use descending average ranks, Spearman, Kendall tau-b and top-k
Jaccard overlap with all boundary ties included. All tied top-1 IDs are reported.
Constant ranks yield null correlation. Score margin is best minus second best.
Reward-only and Q-only re-rankings are diagnostics, never alternate executed MPC
policies. Rank changes do not establish control failure.

## Real transition diagnostics

One-step compares `next(encode(o_t), commanded_action)` to `encode(o_next)`.
Both squared L2 sum and coordinate MSE are stored. Relative error is squared L2
residual divided by `max(squared L2 observed latent,1e-12)`. It is scale-sensitive
and a low target norm can inflate it. Early-ended native repeats are marked
PARTIAL_REPEAT; command mismatch is still recorded descriptively.

Execution-based one-step error is stored only for a full, constant two-step
executed controller command in a compatible model action space. Otherwise it is
null with NOT_IDENTIFIABLE. In particular, differing native executions are never
collapsed to their average and never passed through an invented native-step model.
Even valid constant-command substitution does not remove hidden queue history.

Multi-step uses one observed starting latent and the **actual recorded command
sequence**, without replacing intermediate latent predictions with observations.
This is open-loop prediction along teacher-provided actions, not a fresh MPC
imaginary rollout. Endpoints are observed encoded latents at horizons 1,3,5 by
default. A window must fit inside one episode. Partial repeats propagate the
validity flag. Real rewards are the summed decision-interval rewards.

Raw Q diagnostics decode every head and report all values, mean, minimum,
population std and range. Disagreement is not a validated epistemic uncertainty
estimate, nor a correctness measure. Reward absolute error compares decoded
prediction with the real summed reward.

Observed discounted returns follow the actually executed policy continuation.
They are complete MC returns only when an episodic model episode truly terminates;
otherwise they are explicitly TRUNCATED_POLICY_CONTINUATION with no invented tail.
Q-minus-return is still a descriptive difference, not a certified value error.
The Bellman proxy uses online all-head mean Q at the next policy mean action and
mask `1 - (native_terminated AND cfg.episodic)`. It is **not** the training target,
which uses sampled actions, target-network heads and random-pair minimum. Native
legacy Gym `TimeLimit.truncated` is excluded from native termination; outer decision
timeouts are identified separately. The original environment's `info.terminated`
and agent behavior remain unchanged.

Delay history includes each native requested/executed command, action-repeat index,
shift parameter, queue before execution, and the previous five decision commands.
Raw queue state is complete when available. Summary strata include empty/nonempty
command history and command/execution mismatch, along with task/shift/regime/success.
These coarse associations do not identify a causal history effect or model capacity
limits. No history-aware architecture or training is implemented.

## Artifacts and grouping

Additional files live under each existing Benchmark v2 run's `mechanism/`:

- `manifest.json`: configuration, action semantics, score rules, initial model hash.
- `one_step.csv`, `multistep.csv`, `reward_q.csv`: real-transition diagnostics.
- `action_history.csv`: one row per native execution and queue history.
- `planner_candidates.npz`: numeric/string arrays keyed by episode, decision,
  iteration and field; readable using `numpy.load(..., allow_pickle=False)`.
- `planner_summary.csv`: all-candidate ranking comparisons and actual selection.
- `mechanism_summary.json`: separate aggregates and episode-level summaries.
- `validity_report.json`: final full model-state hash, invariance and limitations.

Disabled diagnostic tables are omitted; an actor-only run contains a planner
NOT_APPLICABLE table and no candidate trajectories. Save flags control trajectories,
scores, latent arrays and extra rescoring ranks; structural candidate IDs/source/
elite flags remain. `decision_stride` controls planner tracing and multi-step starts;
one-step, reward/Q and action history cover every real transition. Latents are
stored in the same NPZ only when `save_latents=true`. CSV tables are flushed at
episode boundaries and the archive is finalized even after an evaluation error.

Aggregates distinguish checkpoint, task, shift type, schedule, native regime(s),
exposure, episode success, MPC/actor-only and native steps since shift at decision
end. A transition spanning a regime boundary retains both regimes. At least five
independent episodes are required for the descriptive status; fewer are
INSUFFICIENT_SAMPLES. Unexposed shifted-condition strata are INSUFFICIENT_EXPOSURE
when enough episode samples exist. Five is a reporting threshold, not a power claim.

Existing `paired.json` additionally stores seed-paired episode differences for
return, latent MSE and reward error, with a 2000-replicate private-RNG percentile
bootstrap CI. This does not perturb global RNG. Fewer than five pairs are marked
insufficient; finite-sample bootstrap intervals are descriptive. The two condition
arms have different closed-loop state distributions and actions.

## Counterfactual validity and invariance

Within-state rescoring holds the observed state and candidate actions fixed.
The existing initial-state capture compares observation, exposed simulator,
controller and RNG state. Snapshot/restore helpers exist, but hidden contact caches,
complete simulator determinism and equal continuation have not been certified.
Consequently no simulator candidate branches are executed or claimed here.
Paired rollouts are always `approximate_pairing`, full-state counterfactual remains
UNVERIFIED and arbitrary candidates' true future returns are UNAVAILABLE. This
fallback intentionally retains the existing paired evaluation protocol.

Full `model.state_dict()` hashes cover parameters **and buffers**, including target
Q state, before and after analysis. A mismatch fails the run. Existing BenchmarkLog
also hashes the checkpoint file before/after. There are no optimizer calls and no
checkpoint saves. CPU hashes and RNG preservation have been tested with toy models;
real checkpoint hashes are produced only by actual server runs.

## Validation status (2026-11-02)

Executed locally using the existing `basilisk-rl` conda Python with Torch 2.14.0:

- 7 mechanism unit tests passed: numeric/shape errors, fixed-action multistep,
  episode boundary exclusion, two-hot decoding, tie handling, native delay queue,
  bounded artifacts, RNG and full model-state checks.
- 3 planner tests passed. Production `_plan`, `_estimate_value` and `act` methods
  are executed from the current source, compared against a checked-in independent
  pre-change oracle. Combinations cover analysis hook enabled/disabled, episodic
  termination on/off, eval mode on/off, and four warm-start decisions each.
  Actions, warm-start means and subsequent RNG states match exactly. This uses
  a stochastic CPU toy world, not an official pretrained checkpoint.
- 22 existing dynamics-shift/benchmark regression tests passed.
- Hydra composition tests are supplied but locally skipped: Hydra/OmegaConf missing.
- Real checkpoint integration test is supplied but locally skipped: CUDA and
  the checkpoint directory are unavailable. ManiSkill2 and TensorDict are also
  absent. **Five-episode smoke: NOT RUN / UNVERIFIED. Fifty episodes: NOT RUN.**
- Suite command generation (dry run), Python syntax and `git diff --check` passed.

Reproduce unit tests in a prepared environment:

```bash
python -m unittest discover -s tests -p 'test_mechanism*.py' -v
python -m unittest discover -s tests -p 'test_planner_instrumentation.py' -v
```

Opt in to the real three-checkpoint equivalence test (three decisions per task,
separate from the five-episode matrix):

```bash
TD_MPC2_MECHANISM_CHECKPOINTS=/absolute/checkpoint/directory \
  python -m unittest discover -s tests -p 'test_mechanism_checkpoint.py' -v
```

Simulator validation remains a separate requirement; use the existing
`audit_shift_simulator.py` workflow. A mechanism run succeeding is not proof of
correct physics or exact replay. Neither latent MSE, reward error, Q disagreement,
ranking change, nor success recovery independently demonstrates robustness,
adaptation, optimality, or plasticity loss. Frozen models cannot measure plasticity
loss here. Success/failure and exposed/unexposed strata avoid selecting only failed
episodes, but do not eliminate all selection bias.
