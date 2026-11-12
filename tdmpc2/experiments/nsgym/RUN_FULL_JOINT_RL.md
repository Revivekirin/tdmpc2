# Independent Full Joint RL adaptation: HalfCheetah and Reacher

The same `scripts/run_full_joint_rl.py` supports `--task halfcheetah` (default, preserving the original training protocol) and `--task reacher`. Reacher held-out evaluation is an explicit `evaluate` command; training never launches it automatically. An already running process keeps its loaded code, but it hashes the runner source for later stream manifests. While HalfCheetah is active, use a separate Git worktree for the updated runner so its on-disk source remains stable. A new smoke is required before any new full run because the runner source hash changed.

## Reacher selected-checkpoint protocol

Reacher uses only the nominal-validation selection JSON, requiring its checkpoint path, model SHA256 and configuration fingerprint to match. It covers the existing 24 physical conditions (D0/D1/M0/M1 × six severities), three independent adaptation seeds, 50-step episodes and 10,000 transitions per stream. It saves complete episodes to the original TD-MPC2 replay format. The first Joint RL update is at transition 75, then one update every 25 transitions, for 398 updates per stream; the 100-transition real smoke performs two updates. This matches the decision dynamics-only transition cadence, though each Joint RL update invokes both original Adam optimizers and target-Q/scale behavior. Joint RL collects with `eval_mode=False`, while the dynamics-only runner uses `eval_mode=True`; those online trajectories are therefore not action-matched. Both streams start independently from the selected nominal checkpoint. Do not infer held-out performance from online returns.

Run from the updated checkout's `tdmpc2/` directory, with fresh output directories and a GPU memory check. Keep `OUT` and the selected checkpoint at their existing absolute `/home/jhkim/repos/tdmpc2/results/...` paths. `--allow-shared-gpu` permits the existing Reacher and HalfCheetah jobs to remain on GPU 0, provided the memory guard passes. The GPU-throughput estimate from the dynamics-only smoke does not apply to Joint RL.

```bash
# Run in a separate updated worktree while the original checkout has active jobs.
cd /home/jhkim/repos-reacher-joint/tdmpc2
RUNNER=experiments/nsgym/scripts/run_full_joint_rl.py
OUT=/home/jhkim/repos/tdmpc2/results/nsgym/reacher/preview-v1
SELECTION="$OUT/nominal-home-01/selected_reacher_checkpoint.json"
CHECKPOINT=/home/jhkim/repos/tdmpc2/results/nsgym/reacher-full-01/checkpoint-000670000
JOINT="$OUT/full-joint-home-01"
python "$RUNNER" plan --task reacher
python "$RUNNER" gpu-check --task reacher --gpu 0 --allow-shared-gpu
python -u "$RUNNER" smoke --task reacher --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --gpu 0 --allow-shared-gpu \
  --output "$JOINT/smoke-d0-k1.5"
cat "$JOINT/smoke-d0-k1.5/completion.json"
```

After a PASS smoke, launch the full 72-stream training in a new directory. The run rechecks the selection, smoke fingerprint and GPU free memory before starting and before each condition. It does not interfere with the HalfCheetah or Reacher dynamics-only output trees.

```bash
nohup python -u "$RUNNER" run --task reacher --selection "$SELECTION" \
  --checkpoint "$CHECKPOINT" --gpu 0 --allow-shared-gpu \
  --smoke-result "$JOINT/smoke-d0-k1.5" \
  --output "$JOINT/runs-fixed" > "$JOINT/runs-fixed.log" 2>&1 < /dev/null &
echo $! > "$JOINT/runs-fixed.pid"
tail -n 50 -f "$JOINT/runs-fixed.log"
```

The Reacher full run is 72 × 10,000 = 720,000 interactions and 28,656 Joint RL updates. Smoke validates mechanics, not the full runtime or scientific outcome. Do not start a second copy into an existing output directory.

## Reacher held-out evaluation after training PASS

The explicit `evaluate` command reloads each final snapshot into a fresh selected-checkpoint agent, verifies model/optimizer/scale state and snapshot SHA256, then evaluates the same 20 held-out seeds under the fixed shift and nominal physics. It also computes one-step latent error on the existing hashed Frozen trajectories. All evaluation updates are disabled. The root summary averages evaluation seeds within each adapted model, then uses adaptation seed (n=3) for uncertainty. The complete 72-snapshot evaluation is 2,880 control episodes (144,000 transitions) plus 1,440 fixed-data prediction episodes. The training source hashes remain pinned; the runner's source hash may differ because this evaluation command was added after training. Use a new output directory, and exclude a pilot from full aggregation.

```bash
SNAPSHOT="$JOINT/runs-fixed/joint0_damping-k1.5-full-joint-seed53000001/snapshot-0010000.pt"
FROZEN="$OUT/frozen-home-01"
python -u "$RUNNER" evaluate --task reacher --gpu 0 --allow-shared-gpu \
  --checkpoint "$CHECKPOINT" --selection "$SELECTION" --frozen "$FROZEN" \
  --snapshot "$SNAPSHOT" --output "$JOINT/evaluation-pilot"
cat "$JOINT/evaluation-pilot/completion.json"
```

After a PASS pilot, run all final snapshots. The trained `runs-fixed/` and nominal checkpoint remain read-only.

```bash
nohup python -u "$RUNNER" evaluate --task reacher --gpu 0 --allow-shared-gpu \
  --checkpoint "$CHECKPOINT" --selection "$SELECTION" --frozen "$FROZEN" \
  --runs "$JOINT/runs-fixed" --output "$JOINT/evaluation-full" \
  > "$JOINT/evaluation-full.log" 2>&1 < /dev/null &
echo $! > "$JOINT/evaluation-full.pid"
tail -n 50 -f "$JOINT/evaluation-full.log"
```

`reacher_joint_control.csv`, `reacher_joint_prediction.csv` and `reacher_joint_seed_summary.csv` retain per-seed results and adaptation-seed uncertainty. Do not merge pilot rows into the full output.

## Original HalfCheetah protocol

The existing `configs/adaptation/joint_rl.yaml` is a declaration, not a runnable learning adapter. `common/shift_protocol.py` rejects learning modes, and its legacy `joint_rl` label omits encoder/reward updates. This runner explicitly uses **Full Joint RL**: encoder, dynamics, reward, Q, policy prior, target Q Polyak updates and running Q scale. It calls the existing `TDMPC2.update()` and `common.buffer.Buffer` sequence sampler. No original training, evaluation, or Dynamics-only code is modified.

The 750k `model.pt` SHA is `f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b`. Each stream independently reloads that model, both saved Adam states and running scale. Only parameters owned by those original optimizers are unfrozen; target-Q parameters stay outside gradient optimization. Original checkpoint settings are retained: batch 256, latent horizon 3, LR 0.0003 (encoder multiplier 0.3), consistency/reward/value coefficients 20/0.1/0.1, MPPI 512 samples and 6 iterations. Actual checkpoint configuration and optimizer learning rates are saved in each manifest.

The grid is the existing two parameters × six multipliers × three adaptation seeds: 36 streams. Each stream starts a new fresh replay and trains for 3,000 Gym transitions over three 1,000-step episodes. Only the first reset is seeded, with 42000001/42000002/42000003. Evaluation seeds are never reset or loaded into replay. Replay stores complete training episodes with the original leading dummy row, for 3,003 stored rows. Sequence samples use the existing SliceSampler and original observation/action offset; runtime checks reject nonfinite samples and wrong shapes. Truncation is not marked terminal; `episodic=false` retains the original TD bootstrap behavior.

This adaptation schedule is explicit: collect the first completed episode using the pretrained planner, add it to replay at transition 1,000, then call `update()` once at every transition from 1,000 through 3,000 inclusive. This gives **2,001 joint updates and 4,002 Adam step calls per stream**, or 72,036 updates across 108,000 transitions. It is interaction-matched to Dynamics-only, not compute-matched to its 118 updates. This is not a replay of the original 10k random-exploration nominal training startup. Full Joint RL uses standard training actions (`eval_mode=False`); Dynamics-only collected with `eval_mode=True`. Both distinctions must accompany later comparisons.

## Server commands

These paths use `/home/jhkim/repos/tdmpc2`. Do not stop PID 3530262 or any other existing process. The user explicitly authorized concurrent use of GPU 0: the commands below use `--allow-shared-gpu`, which permits other compute PIDs and high utilization. It retains the requirement for at least 8 GiB free and, for full training, 1.5× smoke peak reserved memory + 2 GiB. Sharing is recorded in the manifest and may slow both workloads. Without this option, the default requires an idle GPU. These checks are admission checks, not a GPU reservation; every new condition checks again.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RUNNER=experiments/nsgym/scripts/run_full_joint_rl.py
CHECKPOINT=/home/jhkim/repos/tdmpc2/results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000
JOINT=/home/jhkim/repos/tdmpc2/results/nsgym/full-joint-750k-v1
GPU=0
nvidia-smi
python "$RUNNER" --help
python "$RUNNER" plan
python "$RUNNER" gpu-check --gpu "$GPU" --allow-shared-gpu
```

Real bounded smoke: damping 2×, first adaptation seed, 1,004 transitions and five original joint updates. It keeps production batch size, horizon and MPPI settings. It checks all six component hashes (including target Q), finite metrics/parameters, both Adam step increments, replay samples and checkpoint immutability. Local bookkeeping tests do not count as this smoke.

```bash
python -u "$RUNNER" smoke --gpu "$GPU" --allow-shared-gpu --checkpoint "$CHECKPOINT" \
  --output "$JOINT/smoke-damping2"
cat "$JOINT/smoke-damping2/completion.json"
```

After smoke PASS, choose **one** of the following foreground/background commands. They perform training only, with no automatic held-out evaluation. All output directories are exclusive-create; existing artifacts are never overwritten. Interrupted training cannot resume exactly because replay/environment states are not checkpointed; partial results are preserved.

```bash
# Foreground 36-stream training:
python -u "$RUNNER" run --gpu "$GPU" --allow-shared-gpu --checkpoint "$CHECKPOINT" \
  --smoke-result "$JOINT/smoke-damping2" --output "$JOINT/runs-fixed"
```

```bash
# Alternative: guarded nohup launcher for the same 36 streams:
bash experiments/nsgym/scripts/launch_full_joint_rl.sh \
  "$GPU" "$CHECKPOINT" "$JOINT/smoke-damping2" "$JOINT/runs-fixed" --allow-shared-gpu
tail -n 50 -f "$JOINT/runs-fixed.log"
```

```bash
python "$RUNNER" status --output "$JOINT/runs-fixed"
tail -n 30 "$JOINT/runs-fixed.log"
ps -p "$(cat "$JOINT/runs-fixed.pid")" -o pid,etime,stat,args
```

Outputs per stream: manifest with full configuration, checkpoint/source fingerprints, initial optimizer counters, `training.jsonl`, final model/optimizer/scale snapshot, and completion record with parameter changes, replay contract, update count, timing and peak CUDA reserved memory. Snapshots are evaluation-only; a future Full Joint RL evaluator must load their model/configuration explicitly, not use the Dynamics-only adapter loader.

Validation on the local Mac: CLI/plan, syntax and five bookkeeping tests passed. Actual CUDA smoke and full training are **NOT_RUN** locally; no server GPU mapping is known here. The ongoing TTT evaluation and its aggregation paths remain unchanged.
