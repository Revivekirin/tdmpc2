# Reacher / HalfCheetah comparison plan

Status: protocol only. Reacher checkpoint selection and full experiments have not run locally. HalfCheetah fixed/abrupt Frozen completion was reported by the user; completed Dynamics-only results have not been supplied here.

| Dimension | HalfCheetah 750k | Reacher selected nominal |
|---|---|---|
| Control | Locomotion | Goal reaching |
| Shifts | Joint damping / torso mass | One joint damping / one link mass |
| Output | Forward velocity | Fingertip-target distance |
| Reward | Forward motion minus effort | Negative distance minus effort |
| Prediction | Held-out fixed-bank latent MSE | Held-out fixed-bank latent MSE |
| Recovery | Paired return change | Paired return and distance change |

Compare transition budgets, actual optimizer updates, replay sizes and compute. Episodes have different horizons and are not a shared budget unit. Multipliers are not matched dynamic difficulty; display physical response in task-specific units. Mass-only intervention semantics must remain visible.

Show within-task paired return change, prediction error change and nominal retention. Optional descriptive normalization is ΔJ/max(abs(J_nominal),1e-8), with denominator reported. It is not an optimality score. Do not compare raw returns or latent MSE magnitudes as if tasks shared scales. Do not pool evaluations of one adapted model as independent adaptation replicas. No cross-task adaptation result will be inferred without actual held-out records from both tasks.
