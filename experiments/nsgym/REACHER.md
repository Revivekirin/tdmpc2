# Reacher-v5 nominal training

The Reacher adapter uses official Gymnasium Reacher-v5 under the installed
NS-Gym MuJoCo wrapper. The policy receives only the 10-value physical state,
and the two native actions remain in `[-1, 1]`. This adapter supports nominal
physics only. Reacher-v5 truncates after 50 environment transitions and does
not terminate on reaching the target. The final two observation values are
the fingertip-to-target displacement; their norm is the goal distance.

On server4, first sync the Reacher adapter, modified factory and training
support, new configs, test, inspector, and batch script from the local checkout.
Then, from `/nfs4/jhkim/repos/tdmpc2`:

```bash
conda activate tdmpc2-nsgym
python -m pytest experiments/nsgym/tests/test_reacher.py -q
python experiments/nsgym/scripts/train_nominal.py \
  --config experiments/nsgym/configs/train_reacher_pilot.yaml --check-config
python experiments/nsgym/scripts/train_nominal.py \
  --config experiments/nsgym/configs/train_reacher_pilot.yaml --smoke \
  --output results/nsgym/reacher-smoke-01
cat results/nsgym/reacher-smoke-01/completion.json
```

If the checks pass, submit the 50k pilot with cluster-appropriate resource
flags. The batch script activates `tdmpc2-nsgym` itself:

```bash
sbatch --gres=gpu:1 --time=08:00:00 \
  experiments/nsgym/scripts/run_reacher.sbatch
```

The output is `results/nsgym/reacher-pilot-$SLURM_JOB_ID`. Inspect the final
checkpoint on independent seeds, comparing both return and fingertip-to-target
distance against random and zero actions:

```bash
RUN=results/nsgym/reacher-pilot-JOBID
cat "$RUN/completion.json"
cat "$RUN/quality.json"
python experiments/nsgym/scripts/inspect_reacher.py \
  --checkpoint "$RUN/checkpoint-000050000" --episodes 10 \
  --output results/nsgym/reacher-50k-inspection-JOBID
cat results/nsgym/reacher-50k-inspection-JOBID/summary.json
```

The full 1M configuration starts a **new run from scratch**. Select a time
limit based on pilot runtime and local cluster policy:

```bash
REACHER_CONFIG=experiments/nsgym/configs/train_reacher_nominal.yaml \
  REACHER_OUTPUT=results/nsgym/reacher-full-01 \
  sbatch --gres=gpu:1 --time=48:00:00 \
  experiments/nsgym/scripts/run_reacher.sbatch
```

For an interactive `srun` launch instead of `sbatch`, keep the first shell
open (or run it inside `tmux`). From the repository root with `tdmpc2-nsgym`
activated:

```bash
RUN=results/nsgym/reacher-full-01
srun --ntasks=1 --gres=gpu:1 --time=48:00:00 \
  python -u experiments/nsgym/scripts/train_nominal.py \
  --config experiments/nsgym/configs/train_reacher_nominal.yaml \
  --output "$RUN" > "$RUN.log" 2>&1
```

From a second shell, monitor the run without stopping the first shell:

```bash
cd /nfs4/jhkim/repos/tdmpc2
RUN=results/nsgym/reacher-full-01
tail -n 3 -F "$RUN/training.jsonl"
```

`training.jsonl` is written every transition. Check startup messages or errors
separately with `tail -n 30 "$RUN.log"`; the log does not print each step.

If already inside a Slurm allocation, `srun` creates a step in that allocation;
its existing time limit applies. Remove `--time` and request no more GPUs than
were allocated. Use the site's required partition/account options when `srun`
creates a new allocation.

`train_nominal.py` has no `--resume` argument. Checkpoints contain model,
optimizer and RunningScale state, but not replay buffer, partially collected
episode, environment state, or Python/NumPy/PyTorch random states. They can be
loaded for frozen evaluation. They are not exact resumable training snapshots;
changing a run's transition budget does not continue the original trajectory.
