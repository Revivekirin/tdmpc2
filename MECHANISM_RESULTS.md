# Frozen Mechanism Analysis: 50-episode results (partial mechanism report)

Source: `results/mechanism-full-summaries.zip`, generated from the 2026-11-02
server evaluation. This report uses only its summary and paired JSON files. The
archive omits the per-transition and planner CSVs, so ranking, Q calibration,
multi-step error growth, and command/execution decomposition are **not yet
analyzed**. These results describe the frozen MPC controller; actor-only was not
run.

## Run validity

All six suite commands returned code 0, producing 11 complete environment runs
of 50 episodes each. All 11 `summary.json` files report complete status and an
unchanged checkpoint. All 11 `validity_report.json` files report an unchanged
full model state, with matching before/after hashes. The same source revision
`eafa48890b3ef02df5b257668196e79136a65fd5` is recorded throughout.
`simulator_validation` remains `UNVERIFIED_BY_THIS_ANALYSIS`. Initial exposed
observation, simulator, controller, and RNG comparisons pass for all 250 seed
pairs, while full-state pairing remains explicitly UNVERIFIED. Seed-paired
closed-loop runs are not exact counterfactual branches.
The four PickCube-1 nominal runs have identical 50-episode mechanism-summary
records despite their separate execution and wrapper metadata; this confirms
the repeated reference computation in this run, without making the different
metadata interchangeable.

## Control and model diagnostics

Each figure below is an average across the 50 episodes in that condition.
`latent MSE` and `reward absolute error` are the means of episode-level means;
episodes therefore have equal weight even when they have different lengths.
Success is the episode success fraction. Nominal is the corresponding checkpoint
and task's paired nominal reference.

| Checkpoint / task | Condition | Success nominal → shifted | Mean decision length nominal → shifted | Latent MSE nominal → shifted | Reward absolute error nominal → shifted |
|---|---|---:|---:|---:|---:|
| PickCube-1 | Fixed delay 2 | 1.00 → 0.32 | 8.24 → 82.28 | 0.000392 → 0.007133 | 0.823 → 0.481 |
| PickCube-1 | Fixed gain 0.7 | 1.00 → 1.00 | 8.24 → 10.60 | 0.000392 → 0.001642 | 0.823 → 0.698 |
| PickCube-1 | Abrupt delay 2 at native step 10 | 1.00 → 0.86 | 8.24 → 34.80 | 0.000392 → 0.005294 | 0.823 → 0.974 |
| PickCube-3 | Fixed delay 2 | 1.00 → 0.34 | 8.02 → 80.36 | 0.000436 → 0.006349 | 0.791 → 0.451 |
| StackCube-3 | Fixed delay 2 | 0.98 → 0.08 | 14.32 → 93.66 | 0.000539 → 0.003014 | 2.635 → 2.188 |

For fixed delay, 34/50 PickCube-1 and 33/50 PickCube-3 seed pairs switch from
success to failure. StackCube-3 has 45 such pairs; its nominal arm has one
failure. All 50 fixed-delay episodes in each condition are marked exposed. In
the abrupt condition only 40/50 episodes encounter delay, and all seven failures
are among those 40. The exposed abrupt subset succeeds in 33/40 episodes
(0.825); all ten unexposed episodes succeed. This is a descriptive exposure
split, not a causal estimate.

The paired episode-level latent MSE differences (shifted minus nominal) are:

| Condition | Difference | 95% bootstrap interval |
|---|---:|---:|
| PickCube-1 fixed delay 2 | +0.006741 | [0.006027, 0.007448] |
| PickCube-1 gain 0.7 | +0.001249 | [0.001102, 0.001389] |
| PickCube-1 abrupt delay 2 | +0.004902 | [0.003801, 0.005854] |
| PickCube-3 fixed delay 2 | +0.005913 | [0.005227, 0.006559] |
| StackCube-3 fixed delay 2 | +0.002475 | [0.002157, 0.002767] |

These are 2,000-replicate percentile bootstrap intervals over 50 seed-paired
episodes. Their pairing is approximate because nominal and shifted policies
follow different state/action trajectories.

The pattern rules out a simple one-metric account of control reliability:
PickCube-1 gain 0.7 raises latent MSE while preserving 100% success. Fixed
delay greatly reduces success on both tasks, yet episode-mean reward absolute
error **decreases** for PickCube-1 (paired difference −0.342, interval
[−0.497, −0.212]) and PickCube-3 (−0.341, [−0.492, −0.171]). StackCube-3's
reward-error difference is −0.447 with an interval [−0.965, +0.067]. These
differences are influenced by changed state distributions and episode lengths;
they do not imply that reward predictions become more correct for the same
states, or that lower reward error improves control.

Episode return also increases under fixed delay despite lower success. The
benchmark sums rewards across decisions; delayed episodes last much longer.
Return and success must therefore be interpreted separately. The positive
return change is not evidence that delay improves the task.

## What these files cannot establish

The supplied archive contains `mechanism_summary.json` and paired outcomes but
no `planner_summary.csv`, `reward_q.csv`, `multistep.csv`, `one_step.csv`, or
`action_history.csv`. Consequently this report cannot yet quantify actual
planner score decomposition, reward-only versus Q-only rank disagreement,
candidate or elite selection changes, Q ensemble behavior, multi-step error
growth, or the command-versus-execution error split. In particular, increased
latent MSE plus reduced success does **not** demonstrate that the MPC chose an
incorrectly ranked candidate. Arbitrary candidates' true future returns are
unavailable. The delay queue is hidden from the policy; its causal contribution
cannot be isolated from this summary alone.

The raw summary groups further partition by each exact elapsed native step
since shift. In fixed delay this creates many one-sample groups, each marked
`INSUFFICIENT_SAMPLES`. The episode-level statistics and paired comparisons
above avoid treating those individual groups as reliable aggregates. A useful
raw-data analysis should group elapsed steps into predeclared ranges while
retaining the exact values in the underlying rows.

The next evidence needed for the central research question is an aggregate of
the omitted CSVs, especially the final-iteration `planner_summary.csv` rankings
and selected-candidate fields, separated by checkpoint, shift, exposure,
regime, success and time since shift. Within-state diagnostic rescoring shows
model preference, not correctness; nominal-versus-shifted rank comparisons
remain confounded by different closed-loop states.
