# Reacher severity calibration

Implementation and bounded real-physics checks are available. Model-based calibration is NOT_RUN until R0 provides the selected checkpoint.

All four families and six predefined multipliers are retained. Five disjoint calibration seeds each generate a nominal random-action trajectory. After five moving-joint warmup steps, fifteen common integration-state/action pairs per seed are probed. Every counterfactual restores mjSTATE_INTEGRATION (including time/control/warmstart), recomputes derived state, and compares identical actions. A second nominal replay must match to 1e-12; k=1 shifted states must match the nominal states. Every nonnominal condition must produce measurable joint-velocity response; otherwise the phase fails rather than silently dropping a condition.

Physical CSV saves raw joint position/velocity/acceleration, fingertip position/velocity, target, control, actuator force, actions, integration state, next observations, reward components and parameter values. Δq, Δqvel, Δtip and finite-difference acceleration each retain their units. No mixed-unit physical norm is used. The NPZ bank additionally preserves observation/action/next-observation alignment.

Frozen selected-model latent MSE, nominal MSE, difference and guarded ratio use the same counterfactual bank. Ratio is missing if nominal MSE≤1e-10. Latent error is not a physical reconstruction error. One-step error is implemented; multi-step prediction is optional and not implemented here.

Statistical unit is calibration seed. Mean/median/std/bootstrap CI and Tukey outlier counts are computed after averaging windows within seed. Per-seed action regime counts use mean absolute action <0.25, [0.25,0.75), ≥0.75. Non-monotonic responses are not grounds for exclusion. Parameter defaults and actual multiplier-scaled values appear in the figures and raw tables.
