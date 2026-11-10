# Physical severity calibration protocol

Status: protocol implemented; **primary-policy server calibration NOT_RUN**. Configuration is in `configs/physical_severity_grid.yaml` (JSON syntax, valid YAML). Save the manifest before shifted returns are observed.

Primary is checkpoint 750k, SHA256 `f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b`. Secondary 950k is sensitivity only. The earlier audit's mean-return selection is superseded by this user-fixed research contract; its historical artifact is preserved.

## Conditions and splits

- Families: single bthigh hinge damping (`bthigh_damping`); single torso body mass (`torso_mass`).
- Multipliers: **0.5, 0.75, 1.0, 1.25, 1.5, 2.0**, each times runtime nominal readback. Twelve conditions including a separate nominal reference for each family.
- All candidates are positive, compatible with the inspected scalar setters, and satisfy source positivity constraints. No initial exclusions. If runtime validation fails, preserve the failed condition and stop the dependent stage; do not silently discard it or replace it after seeing return.
- Calibration seeds: 41000001–41000003. Schedule/software validation: 42000001–42000003. Final control evaluation: 43000001–43000020. Reserved sensitivity: 44000001–44000020. These splits do not overlap training seed 17, training reset range, training evaluation seeds, or the earlier nominal validation seeds.
- The multipliers are engineering candidates, not official NS-Gym severities or equivalent difficulty across physical families. No shifted return is used for checkpoint/grid selection.

## Bank and paired response

`bank` collects three nominal 1000-transition trajectories with the frozen 750k MPPI controller. It stores observations/actions/rewards, full integration states and physical variables. `physical` selects 16 evenly spaced starting indices per seed and replays the same ten actions from each saved full state under every condition. This is 48 state/tape pairs per condition, **576 measured probe rollouts plus 96 nominal repeats = 6720 simulator transitions**. Bank collection adds 3000 transitions. State windows within a seed are correlated and are not 576 independent environment seeds.

`mjSTATE_INTEGRATION` is restored with MuJoCo's API, including solver warmstart/control/applied-force state. `mj_forward` is checked not to alter the saved integration state. The fixed shift applies through official NS-Gym on the next action. Nominal replay must match both the saved bank tape and an independent repeat. No model prediction contributes to physical response metrics. The first transition is the controlled one-step comparison; later steps are prescribed-action open-loop comparisons, with initial state fixed but trajectories diverging.

Outputs retain signed coordinate deltas, separate angular/linear measures, finite-difference acceleration differences, contacts, physical parameters and full raw trajectories. Mixed-coordinate L2 is explicitly descriptive and not a calibrated physical distance. Contact force differences are boundary snapshots. Full model/optimizer files are never modified. Invalid conditions get a failure record and retain partial files; downstream gates require PASS.

## Latent and planning analyses

`latent --input <physical>` measures the frozen model on controlled data. `latent --input <frozen>` uses independent held-out closed-loop data and labels its changed visitation distribution. Observations are unchanged state vectors converted to float32, with no additional normalization. One-step latent MSE and multi-step open-loop latent MSE at horizons 1, 3 and 10 use the actual saved actions. Reward MSE uses the first step of each window. Windows never cross episode boundaries. Results include seed/window, action saturation regime, forward/nonforward velocity regime, and nominal error ratios where the denominator is nonzero. Errors across latent coordinates are not physical state error or proof of control utility. There is no decoder or separately trained physical outcome probe.

After physical and Frozen result review, `planning` uses the existing `_mechanism_trace` hooks. It saves each actual MPPI candidate set, latent rollouts, reward sum, discounted terminal Q, combined score, ranking, selected candidate/action, and CPU/CUDA RNG states. At identical initial observation and RNG, nominal/shift labels must produce bit-identical traces. Shifted next-state observations are compared separately. MPPI policy-prior sampling, random Q-head selection and Gumbel selection are preserved. Candidate simulator returns are **not measured**, so ranking correctness and regret are NOT_MEASURED. The default representative diagnostic is predeclared k=1.5, not chosen for a large return loss.

All stages refuse existing output directories. For regeneration use a new summary directory; raw measured datasets are inputs, never overwritten. Full command and file mapping is in `RUN_PHYSICAL_SEVERITY_SWEEP.md`.
