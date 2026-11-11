# Reacher Frozen generalization

Status: NOT_RUN; requires nominal selection and completed physical calibration manifests.

Evaluate all 24 labels with 20 paired held-out seeds. Equivalent k=1 conditions reuse one nominal evaluation per seed; duplicate labels refer to that raw file and are not pooled into additional independent samples. All other conditions retain their own per-seed observations/actions and hashes. Unchanged model, optimizer and scale hashes are asserted around every episode. Original MPPI settings are loaded from checkpoint configuration.

Return, distance/control reward components, initial/mean/final/min distance, episode termination/truncation, action magnitude/saturation, force RMS/actual force saturation and on-policy latent error are saved per seed. Per-step records include joint velocity and fingertip/target trajectories. Distances over time and return/distance/action figures are generated. Fixed-data calibration error is labeled separately from error on policy-visited trajectories.

Bootstrap samples are seeds, not time steps. Per-severity prediction/paired-return correlations are descriptive, not causal. Complete grid and improved shifted returns remain in the result. No success threshold or success rate is invented. Force saturation is missing when force limits are disabled.
