# Pre-shift determinism and paired validity

## What is established from the uploaded 50-pair run

`results/pairing-validation-archive.json` is reproducible with
`python scripts/analyze_pairing_archive.py`. It reads the existing summary and
diagnostic ZIPs without running ManiSkill2. All 50 pairs pass the existing
exposed initial observation, simulator, controller and environment RNG
comparisons; the original `full_state_pairing` is `UNVERIFIED`.

The first selected MPC action matches in all 50 pairs. Before native step 10,
the logged action-delay parameter is zero in both arms. Nevertheless, the
logged requested/executed actions first differ at native step 4 in 7 pairs,
step 6 in 25 pairs, and step 8 in 2 pairs; 16 pairs have no logged action
divergence in the common prefix. The corresponding selected-action differences
start at policy decisions 2, 3 and 4. One-step latent MSE first differs after
decision 1 in 7 pairs, after decision 2 in 25, and after decision 3 in 2.
The archive does not contain per-native-step observations, controller states,
simulator states or policy RNG states, so the **first physical divergence and
its cause are not established**. The archive JSON deliberately keeps
`pre_shift_equivalence`, `first_divergence_step`, and `pairing_validity` as
`UNVERIFIED`/null.

The nominal schedule's delay queue retains one action, whereas the abrupt
schedule retains up to three because its future possible delay is two. Their
queue contents first differ at native step 2 in all 50 pairs. At delay zero,
both wrappers execute the current arm command, so this queue-capacity
difference is an expected structural difference, **not proof** that it caused
the pre-shift action divergence.

## Source review and possible causes

`evaluate_shift.py` constructs a fresh environment and agent per arm, seeds
before construction and again before each reset, then runs separate closed-loop
rollouts. `ManiSkillWrapper.step` repeats one policy command for two native
steps. `ActionShiftWrapper.transform` appends to its queue even while delay is
zero, but selects the current command in that regime. The abrupt schedule
itself does not draw from its RNG. The planner samples policy trajectories,
Gaussian MPPI candidates, a terminal policy action, a Q-head pair, an elite
and final action noise; equal initial actions do not establish equal later
candidate sets. Stage 2 diagnostics preserve Python, NumPy and PyTorch RNG
around their extra model calls, but the previous artifacts did not verify
the RNG state at each planner call. ManiSkill2 reset and step are invoked via
legacy Gym in `envs/maniskill.py`; this repository does not implement their
internal contact solver or certify its hidden state. Simulator nondeterminism,
an untracked RNG source, and analysis instrumentation remain possible causes.
None is confirmed by the archived data. No world-model, Q, policy or planner
optimization logic was changed.

## New validation

`scripts/validate_pairing.py` reuses the benchmark Hydra configuration and
checkpoint. It executes independent prefixes with the same seed:

1. Nominal twice with the same closed-loop MPC RNG; compare actions, RNG
   digests, and first-iteration candidate-array hashes.
2. Replay the first nominal run's action sequence open-loop in a new nominal
   environment and in a native environment without the shift wrapper; compare
   both with the enabled nominal wrapper at every native step.
3. Nominal against an enabled abrupt action-delay wrapper before step 10;
   compare requested/executed actions, delay queue, observation, exposed
   simulator/controller state, environment RNG, and global policy RNG.
4. Analysis on versus off; compare the selected MPC actions.

The paired evaluator now records `pairing-trace-*.json` for abrupt action-delay
paired runs and adds `pre_shift_equivalence`, `first_divergence_step`, and
`pairing_validity` to each pair and to `paired.json` at top level. It also
records the differing fields and the first **behavioral** divergence step.
Existing fields and protocol version remain. Queue-only differences are
reported without calling the behavior divergent. Any observed pre-shift
action, parameter or observation difference makes `pairing_validity` invalid;
missing exposed components remain unverified. **Even all exposed comparisons
passing does not upgrade `simulator_full_state` or exact pairing beyond
`UNVERIFIED`**, because hidden SAPIEN/PhysX solver state is not captured.

Run on the CUDA server from the repository root, using the same checkpoint:

```bash
python scripts/validate_pairing.py --seed 7 --out results/pairing-validation.json \
  task=pick-cube checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt \
  episodic=false compile=false model_size=5 dynamics_shift=action_delay \
  schedule=abrupt dynamics_shift.parameter.shifted=2 evaluation=paired \
  evaluation.episodes=1 seed=1 mpc=true analysis=frozen_mechanism
```

Use `--seed` for another episode seed. A prefix can end before step 10; the
JSON reports `reached_switch` for each run, so a short prefix must not be
interpreted as a full ten-step check. For the actual paired evaluation, rerun
the same suite command on the server after updating the code; prior ZIPs cannot
be retroactively populated with the new native state and RNG traces.

## Server result: fresh seed 7

`results/pairing-validation-seed7.json` uses a fresh environment for seed 7.
All six variants reach native step 10. Nominal A/A open-loop and the
disabled/enabled wrapper comparison pass on the exposed native prefix.
Nominal versus abrupt also matches in commanded/executed actions,
observations, exposed simulator/controller state, environment RNG and policy
RNG through native step 9. The first structural difference is the delay queue
after native step 1; it does not change behavior in this prefix. Two
closed-loop nominal runs match in all five actions, RNG digests and first
iteration candidate hashes. Analysis on/off matches in all five actions.
All initial exposed components pass; hidden simulator state and exact pairing
remain `UNVERIFIED`.

This **does not reproduce** the previous seed-7 discrepancy. In the original
50-episode paired evaluation, seed 7 was episode index 6, after seeds 1–6 had
run on the same environment and agent in each arm. The fresh run rules out a
repeatable first-episode seed-7 discrepancy under these settings, but does
not establish the earlier cause. To compare the same sequential context,
use the updated validator with complete warmup episodes:

```bash
python scripts/validate_pairing.py --seed 7 --warmup-seeds 1,2,3,4,5,6 \
  --out results/pairing-validation-seed7-warmup.json \
  task=pick-cube checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt \
  episodic=false compile=false model_size=5 dynamics_shift=action_delay \
  schedule=abrupt dynamics_shift.parameter.shifted=2 evaluation=paired \
  evaluation.episodes=1 seed=1 mpc=true analysis=frozen_mechanism
```

## Server result: seed 7 after six warmup episodes

`results/pairing-validation-seed7-warmup.json` confirms the listed warmup
seeds `[1, 2, 3, 4, 5, 6]`. All variants reach native step 10. The nominal
open-loop replay and disabled/enabled wrapper comparisons pass on the exposed
prefix. Nominal versus abrupt matches in commanded/executed actions,
observations, exposed simulator/controller state, environment RNG and policy
RNG for all ten pre-shift native steps. Closed-loop nominal A/A matches in all
five selected actions, policy RNG digests and first-iteration candidate
hashes; analysis on/off also matches in all five actions. The first reported
difference is only the expected delay-queue capacity/content after step 1.
Hidden simulator state remains `UNVERIFIED`.

Thus a single prior-episode warmup sequence does **not** reproduce the older
seed-7 difference. Neither the queue nor the warmup order is a confirmed
cause. The old ZIPs came from separate full benchmark runs and lack native
post-step observations/controller/RNG traces. The new validator also omits
benchmark logging and stops at the switch, so this result cannot prove that
the full 50-episode paired run is deterministic. The next direct check is a
new seven-episode paired benchmark with `evaluation.episodes=7` and the added
`pairing-trace-*.json` files; inspect episode 6 in its `paired.json`. Repeating
that run can distinguish a reproducible benchmark-path difference from an
intermittent one without changing model or planner behavior.
The archived source hashes match the current `tdmpc2.py`,
`common/mechanism_analysis.py`, `envs/maniskill.py` and action-delay transform;
`evaluate_shift.py` and shift-wrapper base changed to add pairing capture.
The capture's read-only state queries themselves have not been proven neutral
to any hidden simulator timing, so a new run cannot retrospectively certify
the old run.

## Verification in this local workspace

Seven synthetic contract tests in `tests/test_pairing_validation.py` pass,
including an inert pre-switch delay-queue check and an injected first
observation divergence. The local machine lacks the ManiSkill2, PyTorch,
Hydra and checkpoint runtime used on the server. Both seed-7 checkpoint
runs above were performed on the server; neither reproduces the old
pre-shift action divergence. The cause is still unidentified.
