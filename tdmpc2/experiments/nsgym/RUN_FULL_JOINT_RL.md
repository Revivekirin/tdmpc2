# Independent HalfCheetah Full Joint RL adaptation

The existing `configs/adaptation/joint_rl.yaml` is a declaration, not a runnable learning adapter. `common/shift_protocol.py` rejects learning modes, and its legacy `joint_rl` label omits encoder/reward updates. This runner explicitly uses **Full Joint RL**: encoder, dynamics, reward, Q, policy prior, target Q Polyak updates and running Q scale. It calls the existing `TDMPC2.update()` and `common.buffer.Buffer` sequence sampler. No original training, evaluation, or Dynamics-only code is modified.

The 750k `model.pt` SHA is `f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b`. Each stream independently reloads that model, both saved Adam states and running scale. Only parameters owned by those original optimizers are unfrozen; target-Q parameters stay outside gradient optimization. Original checkpoint settings are retained: batch 256, latent horizon 3, LR 0.0003 (encoder multiplier 0.3), consistency/reward/value coefficients 20/0.1/0.1, MPPI 512 samples and 6 iterations. Actual checkpoint configuration and optimizer learning rates are saved in each manifest.

The grid is the existing two parameters × six multipliers × three adaptation seeds: 36 streams. Each stream starts a new fresh replay and trains for 3,000 Gym transitions over three 1,000-step episodes. Only the first reset is seeded, with 42000001/42000002/42000003. Evaluation seeds are never reset or loaded into replay. Replay stores complete training episodes with the original leading dummy row, for 3,003 stored rows. Sequence samples use the existing SliceSampler and original observation/action offset; runtime checks reject nonfinite samples and wrong shapes. Truncation is not marked terminal; `episodic=false` retains the original TD bootstrap behavior.

This adaptation schedule is explicit: collect the first completed episode using the pretrained planner, add it to replay at transition 1,000, then call `update()` once at every transition from 1,000 through 3,000 inclusive. This gives **2,001 joint updates and 4,002 Adam step calls per stream**, or 72,036 updates across 108,000 transitions. It is interaction-matched to Dynamics-only, not compute-matched to its 118 updates. This is not a replay of the original 10k random-exploration nominal training startup. Full Joint RL uses standard training actions (`eval_mode=False`); Dynamics-only collected with `eval_mode=True`. Both distinctions must accompany later comparisons.

## Server commands

These paths use `/home/jhkim/repos/tdmpc2`. Do not stop PID 3530262. Choose an idle physical GPU index after inspecting `nvidia-smi`; the runner refuses any GPU with another compute PID even if memory is available. It additionally requires at least 8 GiB free and, for full training, 1.5× smoke peak reserved memory + 2 GiB. These checks are conservative admission checks, not a GPU reservation; every new condition checks again. If no idle GPU exists, leave this launcher unstarted until one is available.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RUNNER=experiments/nsgym/scripts/run_full_joint_rl.py
CHECKPOINT=/home/jhkim/repos/tdmpc2/results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000
JOINT=/home/jhkim/repos/tdmpc2/results/nsgym/full-joint-750k-v1
GPU=0  # Change to an idle physical GPU after inspecting nvidia-smi.
nvidia-smi
python "$RUNNER" --help
python "$RUNNER" plan
python "$RUNNER" gpu-check --gpu "$GPU"
```

Real bounded smoke: damping 2×, first adaptation seed, 1,004 transitions and five original joint updates. It keeps production batch size, horizon and MPPI settings. It checks all six component hashes (including target Q), finite metrics/parameters, both Adam step increments, replay samples and checkpoint immutability. Local bookkeeping tests do not count as this smoke.

```bash
python -u "$RUNNER" smoke --gpu "$GPU" --checkpoint "$CHECKPOINT" \
  --output "$JOINT/smoke-damping2"
cat "$JOINT/smoke-damping2/completion.json"
```

After smoke PASS, choose **one** of the following foreground/background commands. They perform training only, with no automatic held-out evaluation. All output directories are exclusive-create; existing artifacts are never overwritten. Interrupted training cannot resume exactly because replay/environment states are not checkpointed; partial results are preserved.

```bash
# Foreground 36-stream training:
python -u "$RUNNER" run --gpu "$GPU" --checkpoint "$CHECKPOINT" \
  --smoke-result "$JOINT/smoke-damping2" --output "$JOINT/runs-fixed"
```

```bash
# Alternative: guarded nohup launcher for the same 36 streams:
bash experiments/nsgym/scripts/launch_full_joint_rl.sh \
  "$GPU" "$CHECKPOINT" "$JOINT/smoke-damping2" "$JOINT/runs-fixed"
tail -n 50 -f "$JOINT/runs-fixed.log"
```

```bash
python "$RUNNER" status --output "$JOINT/runs-fixed"
tail -n 30 "$JOINT/runs-fixed.log"
ps -p "$(cat "$JOINT/runs-fixed.pid")" -o pid,etime,stat,args
```

Outputs per stream: manifest with full configuration, checkpoint/source fingerprints, initial optimizer counters, `training.jsonl`, final model/optimizer/scale snapshot, and completion record with parameter changes, replay contract, update count, timing and peak CUDA reserved memory. Snapshots are evaluation-only; a future Full Joint RL evaluator must load their model/configuration explicitly, not use the Dynamics-only adapter loader.

Validation on the local Mac: CLI/plan, syntax and five bookkeeping tests passed. Actual CUDA smoke and full training are **NOT_RUN** locally; no server GPU mapping is known here. The ongoing TTT evaluation and its aggregation paths remain unchanged.
