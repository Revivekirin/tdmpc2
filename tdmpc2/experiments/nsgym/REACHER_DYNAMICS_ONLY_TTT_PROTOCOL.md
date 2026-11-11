# Reacher dynamics-only online adaptation v1

This protocol is preregistered before shifted performance is available. Use only the checkpoint selected by R0 nominal validation, with unchanged training MPPI settings. It is independent of HalfCheetah 750k. No nominal training or Joint RL is launched by this pipeline.

## Splits and conditions

The complete configuration is `configs/reacher_pipeline_v1.json`. Seeds: nominal validation 51000001–51000020; physical calibration 52000001–52000005; independent adaptation streams 53000001–53000003; held-out evaluation 54000001–54000020. The loader excludes training reset seed ranges and training evaluation seeds. A stream is seeded once, then resets continue the environment RNG; its seed is not incremented into another split.

Four independent interventions: D0 joint0_damping, D1 joint1_damping, M0 body0_mass, M1 body1_mass. Every family uses 0.5, 0.75, 1, 1.25, 1.5, 2. No performance-based exclusion or severity selection. Mass changes retain nominal inertia; they are official NS-Gym mass-only interventions, not density scaling. Fixed targets are applied before reset observations and persist between episodes.

## Update semantics

Only `model._dynamics` parameters enter a new Adam optimizer. Encoder, reward, Q, target Q, policy, normalization, original optimizers and RunningScale are frozen and hashed. The existing DecisionAdapter implements mean one-step latent consistency MSE, with stop-gradient encoded targets; coefficient 1, learning rate 1e-4, clip norm 10, batch 64, replay capacity 10000, warmup 64. No reward targets or extra losses. Observe, plan, execute, compute pre-update diagnostic, append the completed transition, then update. Uniform replay sampling without replacement uses only online transitions. Decision updates occur every 25 Gym transitions after warmup; first update is at 75. Frozen performs zero updates. Short 50-step episodes do not reset transition scheduling or replay.

One Gym transition is one action with two MuJoCo integration frames (dt 0.02 s). All budgets count Gym transitions, not simulator frames. No action-repeat conversion from ManiSkill is used. Episode-wise adaptation is deliberately omitted.

## Budget feasibility rule

Proposed snapshots: 0, 500, 1000, 2000, 5000, 10000 transitions. Before any full run, smoke runs both methods on D0/M0 at multiplier 1.5, one adaptation stream, 100 transitions each. It also verifies all 24 physical conditions. Choose the largest proposed budget whose conservative measured maximum wall-time per transition predicts at most 24 GPU-hours for all 144 adaptation streams. Snapshot loading/verification overhead is included. This rule never reads reward or prediction improvement. If even 500 is infeasible, full adaptation is blocked; revise the protocol explicitly instead of silently shortening it. Held-out evaluation cost is not included and is launched separately.

For B≥75, Decision has floor(B/25)-2 updates: 18, 38, 78, 198, 398 at the proposed budgets. Completed episodes are B/50. Both are recorded with actual replay size, update compute time, and total adaptation wall time. A synthetic CPU smoke cannot authorize full runs or choose a budget.

## Snapshots and held-out data

Each budget saves predictor, Adam, replay, adapter RNG and counters plus immutable checkpoint/config hashes. Reload into a fresh agent must preserve predictions and model hashes. These snapshots are for evaluation, not exact interrupted simulator/planner resume.

R2 saves observations/actions for each held-out seed. Prediction evaluation reads those same hashed Frozen trajectories at every budget, never online replay. It reports MSE, counts and action regimes. Control evaluation uses independent episodes on the same paired held-out seeds, both shifted and nominal retention. No updates or replay writes are permitted during either evaluation. Every model starts from the same nominal checkpoint. Global evaluation/planning RNGs are seeded per episode.

## Outcomes and uncertainty

Primary outcome is paired held-out return change at the final preregistered feasible budget. Also report held-out fixed-bank prediction change, reaching distances, distance/control reward decomposition, effort, nominal retention, adaptation AUC and best intermediate as a separate descriptive statistic. Missing conditions remain missing; failures remain recorded. Improved prediction alone does not establish improved control.

Average evaluation seeds within each adapted replica, then summarize across three independent adaptation streams. Evaluation initial states are not additional adaptation replicas. The single nominal training seed limits cross-training inference. No cross-task raw return comparison is valid.

## Gates

R0 selected checkpoint and per-seed nominal validation → R1 physical PASS → R2 full grid Frozen PASS → real checkpoint bounded smoke PASS + feasible budget → fixed TTT. Gate manifests verify configuration, source hashes and checkpoint identity. Outputs require new directories. HalfCheetah results and nominal checkpoints are read-only.
