# Nonstationary frozen evaluation protocol

Status: six schedule implementations and local bounded validators PASS; **large evaluations NOT_RUN**. Frozen primary results must be reviewed first.

All times below are **episode-local zero-based control steps before action**, not MuJoCo substeps and not the separately recorded global counter. Each episode resets physical parameters to nominal and resets the official update sequence. No cross-episode persistence is claimed. The schedule is compiled into absolute targets using official DiscreteScheduler + StepWiseUpdate, avoiding cumulative multiplier drift.

| ID | CLI schedule | Definition at default 1000-step horizon |
|---|---|---|
| S0 | nominal | k(t)=1 |
| S1 | fixed | k(t)=k from action 0 |
| S2 | abrupt | 1 for t<250, k from t=250 |
| S3 | gradual | 1+(k−1)clip((t−250)/250,0,1); endpoint at t=500 |
| S4 | reversible | 1 before 250, k for 250≤t<750, 1 from 750 |
| S5 | repeated | alternate k and 1 every 250 steps starting at 250 |

S0–S2 are the required protocol. S1 may reuse the completed fixed Frozen dataset; an independent identical S1 command is supplied for reproducibility, not as independent evidence. S0 uses k=1 only. S2 includes the entire severity grid and nominal references on the same final evaluation seeds. S3–S5 have independent validators and commands but large execution additionally requires `--allow-optional-large` after separate experiment approval. The command generator prints these only in the labelled optional phase when requested; no command is automatically executed.

Default budgets: S0 40 episodes/40,000 transitions (two family references); each full-grid S1–S5 240 episodes/240,000 transitions. The 12-step smoke changes onset/duration/period to 3 and restore to 9, uses one validation seed and k∈{1,1.5}, two episodes per condition: 96 transitions per schedule. Those integration results are not final scientific episodes.

Every control step records episode/global counters, scheduled onset, actual parameter, NS-Gym time and remaining update targets. The validator records observations immediately before/after and reward, validates scope at every step, compares two repeated reset trajectories, and verifies nominal restoration. Nonstationary data include the full observations/physical trajectories; an `ended_before_onset` flag distinguishes episodes without exposure. Exposure fraction counts transitions whose actual parameter differs from nominal. Reversal changes physics back but does not restore the earlier state trajectory.

`summarize` produces `nonstationary_evaluation.csv`, onset-aligned velocity/reward/action-change curves and raw aligned tables. Return is the discrete reward sum; `reward_time_integral` is sum(reward)×dt, with that convention explicitly retained rather than conflated with a second independent performance metric. Offline `latent` uses the actual pre/post-shift observations and actions, provides error versus aligned step, and retains open-loop windows that cross a shift (the model has no shift input). The physical readback trace is the ground truth for exposure.

Different timings and exposure durations define different control problems even with the same endpoints. Changes in action after onset indicate frozen MPC reacting/replanning from observations; they are not optimizer adaptation or proof of identifying mass/damping. Recovery, model-error deterioration and adaptation necessity require separate evidence. No dynamics-only TTT, joint RL or plasticity experiment is part of this pipeline.
