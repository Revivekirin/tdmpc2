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

`train_nominal.py` has no `--resume` argument. Checkpoints contain model,
optimizer and RunningScale state, but not replay buffer, partially collected
episode, environment state, or Python/NumPy/PyTorch random states. They can be
loaded for frozen evaluation. They are not exact resumable training snapshots;
changing a run's transition budget does not continue the original trajectory.
