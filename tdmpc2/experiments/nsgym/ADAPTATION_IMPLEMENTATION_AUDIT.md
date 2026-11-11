# Dynamics-only adaptation implementation audit

## Reused code and inspected boundaries

- `common/stage3a.py::EpisodeAdapter`: detached encoder/next-encoder one-step MSE, predictor-only Adam, replay tuple copying, frozen-state equality, optimizer/replay/counter evaluation guard, checkpointable state.
- `common/decision_ttt.py::DecisionAdapter`: observe-before-update cadence and persistent scheduler counters. NS-Gym uses one Gym transition per `observe(..., native_steps=1)`; this inherited counter name does **not** count MuJoCo substeps. Five integration substeps remain internal to HalfCheetah.
- `envs/nsgym/severity.py`: official NS-Gym setter/scheduler and per-step parameter assertion; parameter/change notifications do not enter observation or adapter.
- `training_support.load_checkpoint`: model/optimizer SHA verification and original CUDA agent/config construction. Added primary750k SHA, original planner/task contract and NS-Gym/Gymnasium/MuJoCo version checks. Original RL update, original optimizer.step and pi optimizer.step are explicitly blocked.
- `common/world_model.py`: only `_dynamics` trainable. Encoder, reward, Q aliases/target parameters, prior and buffers remain in `fixed_state`; complete model hash and original optimizer/pi optimizer/scale hash are checked.
- `tdmpc2.py::act/_plan/_estimate_value`: actual-checkpoint path preserves original MPPI; evaluation uses eval_mode=True and resets warm start each episode. No copied ManiSkill controller or action-repeat code.
- `common/buffer.py`: original RL replay uses CUDA sequence slicing and reward-bearing trajectories. It is intentionally not used for TTT; tested adapter replay stores only observed one-step `(observation, action, next_observation)` tuples.
- Existing physical severity config, evaluation split and Frozen collection schema were inspected. Full grid remains unchanged. Missing dedicated adaptation split is resolved explicitly by assigning existing validation seeds to independent adaptation streams, not by silently training on evaluation seeds.

The new runner is `experiments/nsgym/scripts/run_dynamics_only_ttt.py`; analysis is `analyze_dynamics_only_ttt.py`. Imports resolve both the normal server layout and the nested local checkout. Existing core files and original result files were not edited.

## Executable behavior

`smoke`, `run`, `evaluate`, `--help`, and `--dry-run` are implemented. Smoke keeps actual checkpoint planning settings when `--synthetic` is absent; it only shortens environment length and adaptation budgets. CPU fixtures require explicit `--synthetic`, which is rejected for `run`. A real/synthetic snapshot mismatch is rejected.

Each run writes manifest.json, actuator_contract.json, online_steps.csv, online_episodes.csv, per-episode observations/actions NPZ, four smoke or five full snapshots, and completion.json only on success. Each snapshot includes predictor/optimizer/replay/RNG/counters, frozen model hash, full model hash and auxiliary hash. All output directories must be new. Failed work cannot acquire a PASS completion marker.

Each evaluation reconstructs an independent agent from750k, loads only adapted predictor/adapter state, verifies model identity, then performs frozen shifted and nominal episodes. Held-out guard rejects replay insertion and verifies no learned state/replay/optimizer/counter mutation. Fixed-data evaluation reads only original Frozen NPZ and records file SHA256; missing fixed data is explicitly NOT_MEASURED.

Snapshots are evaluation snapshots, not resumable online process checkpoints. Environment/planner/global RNG state is intentionally not claimed to be captured. No hidden-onset detection claim: cadence runs blindly across onset. Metadata readbacks are experiment diagnostics only.

## Validation evidence

Evidence directory: `results/nsgym/dynamics-only-validation-02/`.

- Real NS-Gym HalfCheetah physics, **small synthetic CPU model/controller**, both fixed/abrupt, both parameters at k1.5, all three methods.
- 12 online runs ×48 transitions =576 online transitions.
- 48 snapshots evaluated, each with one shifted and one nominal24-step held-out episode =2304 held-out transitions.
- 48000 offline one-step prediction pairs from original Frozen fixed/abrupt NPZ, read-only, using the synthetic model to validate data plumbing.
- Frozen:0 updates. Decision:12 updates. Episode:12 updates at episode boundaries only. Predictor changes for TTT; all nonpredictor states match hashes. All measured losses/gradient norms finite.
- Physics readback changes at smoke pre-action onset8 and resets for the second episode; full protocol onset remains250. Both damping and mass verified through adapter assertions.
- Snapshot0,8(mid-episode),24,48 serialization/reload and held-out guards pass.
- `test_dynamics_only_ttt.py`:2 integration tests passed, including invalid update/replay attempts and intentional frozen-parameter corruption detection.
- CLI help, dry-run and Python compilation passed. Plot pipeline exercised against completed synthetic evaluations, keeping missing grid cells empty.

Synthetic controller is a frozen encoder/policy network, **not MPPI**, and its action selection does not depend on the adapted predictor. Therefore smoke return differences cannot validate adaptation efficacy. Its predictor is small and has no TD-MPC2 latent architecture claim. Actual750k checkpoint files/CUDA were unavailable locally; actual750k load/inference, GPU optimizer integration, and all full experiments remain **NOT_RUN**. Run server smoke before any large stage.

## Output/statistical limitations

The analysis records all evaluation seeds and adaptation streams. CI resamples three adaptation streams conditional on the fixed20-seed evaluation bank; repeated time steps/snapshots are not iid. Synthetic smoke has one stream, so CI/std are missing. Original evaluation seeds have already been viewed in Frozen analysis; they remain excluded from updates but are not an untouched confirmatory set.

118 decision updates vs120 episode updates at3000 transitions are deliberately labelled, not presented as matched compute. Optional exactly matched-update control is not implemented. All snapshots/conditions are retained; no performance-based selection. Nominal retention and paired Frozen gain are descriptive protocol outcomes, not causal identification of latent error effects.

Actual actuator-force saturation uses saved model force-limit flags/ranges and only limited actuators. No limit enabled gives null/N/A. Action-component saturation is a separate metric. No prediction ranking/regret, oracle MPC, full Gramian, architecture replacement, LoRA or plasticity experiment was introduced.
