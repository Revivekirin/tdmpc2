# NS-Gym HalfCheetah dynamics-only online adaptation

Status: implemented; bounded CPU fixture validation only. Actual 750k CUDA smoke and full experiments require server execution. This protocol supersedes the previous offline/known-onset draft. No severity or checkpoint reselection.

## Identity and data separation

Primary is checkpoint-000750000/model.pt, SHA256 `f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b`. Runner rejects other checkpoints. Existing nominal/Frozen results remain read-only. 950k/500k are excluded from this runner.

All existing severities {0.5,0.75,1,1.25,1.5,2} are retained for bthigh_damping and torso_mass, including both k=1 family-labelled controls. Damping nominal is 6 N m s/rad; torso nominal is approximately 6.250209205 kg. Actual values are read from each environment and logged. No cross-family equal-difficulty interpretation.

The existing seed file has calibration, validation, evaluation and sensitivity, **no adaptation field**. We explicitly assign existing validation seeds 42000001–42000003 to adaptation streams. Calibration 41000001–41000003 remains reserved. Evaluation 43000001–43000020 is held out from updates and hyperparameter selection; it has already been used in the Frozen preview, so this is not a newly untouched final test set. No new seed numbers are invented. Each stream seeds its environment once; two subsequent resets use seed=None to advance that independent environment RNG. Stream seed and episode index identify a rollout. Repeated episodes are not independent statistical replicates.

## Objective and update scope

L = mean((dynamics(stopgrad(encoder(o)), action) - stopgrad(encoder(next_o)))²).

Only `model._dynamics` parameters receive gradients, via a new Adam(lr=1e-4), coefficient 1, batch64, clip norm10, replay capacity10000. Uniform sampling without replacement; warmup64 observed transitions. No reward, termination, value, policy, normalization or target-network update. Encoder/reward/Q/target Q/policy and model buffers are invariant. Original agent optimizer/pi optimizer/RunningScale state is hashed and must be unchanged. Existing general RL update is never invoked.

One Gymnasium transition is one replay tuple `(o_t,a_t,o_{t+1})`; reward and metadata are excluded. MuJoCo frame_skip=5 is internal to that transition. No ManiSkill controller/action-repeat semantics are used. Truncation is a real observed one-step transition and may train consistency; no transition crosses reset and no TD target is constructed.

## Methods and budgets

Each method/condition/schedule/stream starts fresh from identical 750k model and planner configuration. Evaluation uses original MPPI horizon3,512samples,6iterations, eval_mode=True. No reduction of planner settings in the actual-checkpoint smoke.

- Frozen: same collection, zero gradient updates.
- Decision: one gradient update after each globally counted multiple of25 transitions, once replay has64 transitions. First update75, updates38/78/118 at1000/2000/3000.
- Episode:40 gradient updates after each completed1000-step episode; none within that episode. Updates40/80/120.

Budgets are0,250,1000,2000,3000 transitions. Snapshots occur **after** the just-observed transition and permitted updates, including episode-end updates. A250 snapshot may be inside an episode. The optional exactly matched-gradient control is not implemented; 118 vs120 is reported, not described as identical compute.

Predictor, Adam, replay, replay RNG and decision counter persist across three episodes of one run; they reset between methods/conditions/streams. Environment schedule resets each episode and MPC receives t0=True at its first decision. Snapshots serialize adapter predictor/optimizer/replay/RNG/counters and hashes, but omit environment/planner/global RNG continuation. They support frozen evaluation, **not exact online resumption**.

Order: current-model plan → execute → observe next state → measure pre-update error → append replay → scheduled updates → log → snapshot if budget reached. No future observation. Dynamics updater receives no reward, parameter, severity, onset or change notification. Evaluation-only metadata retains these for audit. The global cadence runs both before and after onset and is not triggered by shift detection.

## Physics and evaluation

Fixed applies k on pre-action transition0. Abrupt applies k on pre-action transition250 (zero-based); the transition indexed249 remains nominal. Actual readback is asserted by SeverityHalfCheetah every step. For abrupt, episode0 measures within-episode online response from nominal initialization; episodes1–2 additionally reflect retained learning. Never pool these as independent seeds.

Each snapshot is loaded into a fresh750k agent in a separate evaluation invocation. Both shifted and nominal environments run20 held-out seeds each, with updates disabled. The adapter's model/optimizer/replay/counters/RNG and auxiliary state must be identical before/after. Intentional replay-write attempts are rejected inside the held-out guard. Source snapshot files are read-only.

Fixed-data MSE uses the original Frozen raw observations/actions and next observations for the same schedule/condition and evaluation seeds. Use `frozen/` for fixed, `nonstationary-abrupt/` for abrupt. No fixed data enters replay. Full-episode one-step MSE averages1000 transitions per seed; abrupt MSE includes the250 pre-onset transitions and750 post-onset transitions. Online and frozen evaluation step logs separately support onset-specific analysis.

Record online return separately from held-out return. Retention is paired nominal return difference from budget0. Record velocity, positive control cost, action component saturation at |a|>=.99, actuator-force RMS and saturation only for force-limited actuators, limited actuator count, actual parameter/nominal value, termination/truncation, update count, gradient norm and synchronized update wall time. Zero force-limited actuators means saturation is null/not applicable, not zero measured force saturation. Action vectors are in online NPZ; actuator limits/gears are saved in actuator_contract.json.

## Statistics and interpretation

Preserve all per-evaluation-seed records. For uncertainty, average held-out seeds within each adaptation stream, then bootstrap independent adaptation streams (n=3;10000 resamples). This CI is conditional on the fixed evaluation seed bank, not full population uncertainty across new evaluation environments. Report mean/median/sampleSD/CI and descriptive1.5IQR outlier counts; keep outliers. With n=1 smoke do not report a CI. Across-time snapshots are repeated measurements. A Frozen reference replicated across adaptation seeds is a pairing device, not additional independent policy evidence. Legacy S0/S1 duplicates are not ingested.

Analysis includes held-out MSE/return/retention curves, paired Frozen return gain vs severity, interaction/update cost, frozen held-out onset-aligned curves, and separate online episode-specific onset-aligned curves. No causal inference from error/return correlation. No ranking/regret from the previous six planning examples. No new learning objective or oracle evaluation.

## Execution budget

Per full schedule:12 conditions ×3 methods ×3 streams ×3000 =324000 adaptation transitions. Each method alone108000. All five snapshots ×40 held-out episodes ×108 runs =21600 held-out episodes (21.6million transitions), plus10.8million offline fixed-data prediction pairs. Two schedules double these counts. Run smoke first and launch full stages manually. Missing evaluation outputs remain gaps, never imputed.
