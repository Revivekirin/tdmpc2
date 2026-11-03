# Frozen Mechanism Analysis: 50-episode results

Sources: `results/mechanism-full-summaries.zip` and
`results/mechanism-full-diagnostics.zip`, generated from the 2026-11-02 server
evaluation. The former contains summary/paired JSON; the latter contains all
11 runs' one-step, multi-step, reward/Q, planner-summary and action-history
CSVs. This report describes the frozen MPC controller; actor-only was not run.
`python scripts/analyze_mechanism_artifacts.py` reproduces the compact run
aggregates in `results/mechanism-aggregates.json` from those archives and records
both input ZIP hashes. The paired decision-prefix and bootstrap checks described
below were computed directly from the same archived rows.

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
The planner CSV has 107,100 iteration rows (17,850 decisions × six iterations);
all five ranking comparisons report `VALID` on every row. This is a numeric
validity check, not evidence that candidate values match real returns.
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

## Command versus executed controller input

The table compares model one-step predictions on the **same valid decisions**:
the TD-MPC2 command supplied to the wrapper versus the single controller command
executed in both native repeats. It excludes early-ended partial repeats. The
executed value is still a controller command, not actuator torque. Both forward
passes span a full policy decision; no native-step dynamics model is assumed.

| Condition | Valid / all decisions | Command latent MSE | Executed-command latent MSE | Mean command/execution action difference |
|---|---:|---:|---:|---:|
| PickCube-1 fixed delay 2 | 4,102 / 4,114 | 0.007125 | 0.002312 | 1.049 |
| PickCube-3 fixed delay 2 | 4,003 / 4,018 | 0.006185 | 0.002157 | 1.089 |
| StackCube-3 fixed delay 2 | 4,680 / 4,683 | 0.003016 | 0.000992 | 1.139 |
| PickCube-1 gain 0.7 | 486 / 530 | 0.001802 | 0.000539 | 0.296 |
| PickCube-1 nominal | 368 / 412 | 0.000559 | 0.000559 | 0 |

Fixed-delay native commands differ from requested commands at all recorded
native steps. On the valid full repeats, the executed-command substitution
reduces one-step MSE by about two thirds for each fixed-delay task. The
episode-level mean of the within-decision MSE reduction is +0.00511 for
PickCube-1 (50-episode bootstrap interval [0.00445, 0.00575]), +0.00444 for
PickCube-3 ([0.00383, 0.00502]), and +0.00205 for StackCube-3
([0.00183, 0.00228]). This directly identifies command/execution timing as a
large component of the measured mismatch. Residual executed-command error
remains, and state distribution, queue history, controller memory and model
error are not separated by this substitution. It is not a counterfactual
retraining or a claim that the original model learned delayed dynamics.

At the first fixed-delay decision, command MSE is of similar order in later
successful and failed episodes: PickCube-1 0.00768 versus 0.00908, PickCube-3
0.00857 versus 0.00921, StackCube-3 0.00179 versus 0.00161. StackCube has
only four successes, so these groups do not support a predictive threshold.
Conditioning all subsequent error analysis on success would be strongly
affected by the much longer failed episodes.

## Open-loop rollout and reward/Q diagnostics

For each condition, the following MSEs use **the same starting decisions** that
have a complete five-decision window. The frozen model rolls forward under the
recorded command sequence without intermediate observation correction:

| Condition | Starts | Horizon 1 | Horizon 3 | Horizon 5 |
|---|---:|---:|---:|---:|
| PickCube-1 nominal | 212 | 0.000840 | 0.002202 | 0.002556 |
| PickCube-1 fixed delay 2 | 3,914 | 0.007321 | 0.015753 | 0.016903 |
| PickCube-1 gain 0.7 | 330 | 0.001986 | 0.006779 | 0.013706 |
| PickCube-1 abrupt delay 2 | 1,540 | 0.008468 | 0.019711 | 0.014105 |
| PickCube-3 fixed delay 2 | 3,818 | 0.006348 | 0.013766 | 0.015439 |
| StackCube-3 nominal | 516 | 0.000970 | 0.002611 | 0.004042 |
| StackCube-3 fixed delay 2 | 4,483 | 0.003064 | 0.007308 | 0.008899 |

Fixed delay produces larger error across horizons. Gain 0.7 also accumulates
error while retaining all 50 successes. Abrupt delay's average horizon-5 error
is lower than horizon-3 for these same starts; error growth is not universally
monotone. These are logged-policy open-loop rollouts, not MPC imaginary plans.

The per-transition reward error shows the same qualitative ambiguity as the
episode averages above: PickCube-1 fixed delay 0.422 versus nominal 0.775,
PickCube-3 fixed delay 0.366 versus nominal 0.731, and StackCube-3 fixed delay
2.085 versus nominal 3.019. Q-head population standard deviation rises for
PickCube-1 (1.87 nominal to 9.89 fixed delay) and PickCube-3 (1.89 to 12.10),
using equal-weight episode means. StackCube-3 is already high nominally (40.52)
and rises to 52.85. These are ensemble disagreements on different encountered
states, not validated epistemic uncertainty. All `reward_q.csv` entries mark
the observed return `TRUNCATED_POLICY_CONTINUATION` because these checkpoints
use `episodic=false`; Q-minus-return and the analysis-only Bellman proxy are
not ground-truth Q errors. A larger Q disagreement cannot by itself explain
the loss of success.

## Actual MPC candidate ranking and selection

The observer recorded 512 actual candidates per iteration and six MPPI
iterations per decision. The table uses the final iteration only. For each
decision it compares **the same candidate set** using the actual planner's
reward, terminal-Q and combined score. Top-1 agreement is then averaged within
each episode, and the 50 episode means receive equal weight. The rank-only
columns are diagnostic re-rankings; they were not executed policies.

| Checkpoint / task | Condition | Actual total top-1 = reward-only | Actual total top-1 = terminal-Q-only | Actual total top-1 = deterministic diagnostic top-1 |
|---|---|---:|---:|---:|
| PickCube-1 | Nominal | 5.5% | 37.9% | 20.0% |
| PickCube-1 | Fixed delay 2 | 3.9% | 53.2% | 19.7% |
| PickCube-1 | Gain 0.7 | 4.8% | 35.5% | 23.2% |
| PickCube-1 | Abrupt delay 2 | 6.4% | 29.2% | 25.5% |
| PickCube-3 | Nominal | 3.0% | 33.1% | 26.0% |
| PickCube-3 | Fixed delay 2 | 4.1% | 59.0% | 20.0% |
| StackCube-3 | Nominal | 6.3% | 32.9% | 23.8% |
| StackCube-3 | Fixed delay 2 | 2.7% | 65.9% | 13.8% |

The actual combined score's top candidate aligns far more often with the
terminal-Q-only ranking than with reward-only ranking, particularly in fixed
delay trajectories. Reward-only and Q-only rankings themselves have very low
correlation in these final candidate sets (decision-weighted Spearman roughly
0.03–0.29 across runs). This identifies a strong internal reward/Q preference
conflict. It does not identify which ranking is correct in the real environment.
The top-k Jaccard agreement with the actual total ranking also favors Q-only:
PickCube-1 fixed delay is 0.539 versus 0.059 for reward-only; PickCube-3 fixed
delay is 0.597 versus 0.057; StackCube-3 fixed delay is 0.672 versus 0.042.
The corresponding Kendall rank correlations are 0.750 versus 0.307, 0.793
versus 0.271, and 0.872 versus 0.164. These compare ranking rules on the same
sampled candidates, not their actual task outcomes.
Within the first five fixed-delay decisions, PickCube-1 Q-only top-1 agreement
is 0.50 in eventual failures versus 0.56 in eventual successes; PickCube-3 is
0.46 versus 0.51. These modest, observational differences do not identify a
ranking threshold for success. StackCube-3 has only four successes, making its
corresponding split too small to interpret.
The actual-versus-diagnostic top-1 agreement is only 14–26% even nominally;
the diagnostic uses a policy mean and all Q-heads while the real planner used
a sampled terminal action and random Q-head pair. A disagreement cannot be
attributed entirely to dynamics shift. The planner samples an elite with
Gumbel noise and can add final action noise, so its executed action is not
necessarily the top-scoring candidate's first action.

For all five paired conditions, the **first** selected MPC action and recorded
actual rank summary are identical in all 50 seeds before the environment has
stepped. Fixed shifts show different selected actions for all 50 pairs by the
second decision; the model itself was unchanged. The abrupt condition shows
an important limitation: although both arms record zero delay before native
step 10 and all first actions match, seven pairs have different one-step errors
after decision 1, and seven selected actions differ by decision 2. In an
inspected seed, requested and executed commands are identical in both arms
before this divergence. Thus the exposed initial-state comparison does not
certify deterministic continuation. Later cross-arm planning differences
cannot all be assigned to the shift, even in the abrupt condition.

## Identification and remaining limits

The observations support a bounded mechanism statement: fixed native delay
creates a large requested/executed controller-command mismatch, and replacing
the command by the constant executed command greatly reduces one-step latent
error on valid repeats. Long-horizon prediction error and control failure grow
along the resulting trajectories. The actual planner's candidate preference
is strongly affected by terminal Q relative to predicted reward. The study
does **not** establish that Q misranking caused failures: arbitrary candidate
returns are unavailable, Q diagnostics use observational continuations, and
separate nominal/shifted rollouts occupy different states. Frozen parameters
rule out adaptation or plasticity conclusions. Actor-only was not run.

The raw summary groups further partition by each exact elapsed native step
since shift. In fixed delay this creates many one-sample groups, each marked
`INSUFFICIENT_SAMPLES`. The episode-level statistics and paired comparisons
above avoid treating those individual groups as reliable aggregates. A future
raw-data analysis can group elapsed steps into predeclared ranges while
retaining the exact values in the underlying rows. Candidate-level action/score
arrays remain in `planner_candidates.npz` on the server and were not needed for
the rank summaries here; no arbitrary candidate was rolled out in the simulator.
The available elite-slot overlap tracks indices even though each iteration
resamples candidates, so it cannot establish stability of the same trajectory
across iterations. The bounded candidate archive also cannot reconstruct every
alternate elite membership.
