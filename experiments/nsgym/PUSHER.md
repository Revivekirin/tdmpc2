# Pusher-v5 nominal TD-MPC2 preparation

The `nsgym-pusher` task uses the installed NS-Gym `MujocoWrapper` and its
`r_shoulder_pan_link_mass` update API. Nominal training changes no physics.
The agent sees the 23-dimensional Pusher state and normalized 7-dimensional
actions in `[-1, 1]`; the adapter scales actions to Pusher's native `[-2, 2]`
torques. Privileged NS-Gym change metadata is removed from observations/info.
The MuJoCo link mass is recorded only through the experiment recorder.

Pusher-v5 episodes last 100 transitions by default. Its return is the sum of
distance, near-object and control terms; larger (less negative) is better.
`goal_distance_final` is the object-to-target distance in metres. The interim
inspector also reports goal progress against random and zero-action baselines;
return alone can improve merely by avoiding control penalties. Pusher-v5
does not expose an official binary success rate. Do not interpret the default
TD-MPC2 `success=0` info value as a measured success rate. Learning evidence
comes from paired nominal policy-vs-random returns and goal distance.

The full 1,000,000-transition configuration is
`configs/train_pusher_nominal.yaml`. A separate 50,000-transition pilot is
`configs/train_pusher_pilot.yaml`; its checkpoint is **not** an exact resume
point for the full run because replay, environment, and RNG state are absent.
Both retain the original TD-MPC2 agent, objectives, optimizer and MPC settings.
The pilot is a diagnostic, not a controller-quality guarantee.

Copy the new Pusher files from the local Mac checkout to the matching server
checkout before running. The following example requires `server4` to resolve
from the Mac; otherwise use the site's login host or existing transfer route.
From `/Users/jihyekim/Desktop/tdmpc2`:

```bash
rsync -avR \
  tdmpc2/envs/nsgym/__init__.py tdmpc2/envs/nsgym/pusher.py \
  experiments/nsgym/scripts/training_support.py \
  experiments/nsgym/scripts/inspect_pusher.py \
  experiments/nsgym/scripts/run_pusher.sbatch \
  experiments/nsgym/configs/train_pusher_nominal.yaml \
  experiments/nsgym/configs/train_pusher_pilot.yaml \
  experiments/nsgym/tests/test_pusher.py \
  experiments/nsgym/PUSHER.md \
  jhkim@server4:/nfs4/jhkim/repos/tdmpc2/
```

On server4, from `/nfs4/jhkim/repos/tdmpc2`:

```bash
conda activate tdmpc2-nsgym
python -m pytest experiments/nsgym/tests/test_pusher.py -q
python experiments/nsgym/scripts/train_nominal.py \
  --config experiments/nsgym/configs/train_pusher_pilot.yaml --check-config
python experiments/nsgym/scripts/train_nominal.py \
  --config experiments/nsgym/configs/train_pusher_pilot.yaml --smoke \
  --output results/nsgym/pusher-smoke-01
cat results/nsgym/pusher-smoke-01/completion.json
```

Use a **Slurm batch job**, not `nohup` inside a time-limited interactive
allocation. The site's partition/account/GPU flags must match its policy.
For example, after checking the available partition and time limit:

```bash
sbatch --gres=gpu:1 --time=08:00:00 \
  experiments/nsgym/scripts/run_pusher.sbatch
squeue -u "$USER"
```

The batch script defaults to the pilot configuration and writes to
`results/nsgym/pusher-pilot-$SLURM_JOB_ID`. The Slurm job ID is shown by
`sbatch`. Check progress and the completed quality gate with:

```bash
RUN=results/nsgym/pusher-pilot-JOBID
tail -n 1 "$RUN/training.jsonl"
tail -n 30 "$RUN.log"
test -f "$RUN/completion.json" && cat "$RUN/completion.json"
test -f "$RUN/quality.json" && cat "$RUN/quality.json"
```

At a saved checkpoint, while the pilot is still running, inspect nominal
performance against seeded random actions. This creates a separate output
directory and does not update the model:

```bash
python experiments/nsgym/scripts/inspect_pusher.py \
  --checkpoint "$RUN/checkpoint-000010000" \
  --episodes 10 --output results/nsgym/pusher-10k-inspection-01
cat results/nsgym/pusher-10k-inspection-01/summary.json
```

If the pilot demonstrates learning and the full run is warranted, launch a
**new run from scratch** with the full configuration after securing a long
enough Slurm allocation:

```bash
PUSHER_CONFIG=experiments/nsgym/configs/train_pusher_nominal.yaml \
  sbatch --gres=gpu:1 --time=48:00:00 \
  experiments/nsgym/scripts/run_pusher.sbatch
```

Fixed, abrupt, and reversible link-mass schedules are available in the
adapter for the later dynamics-shift phase. They are physically tested here;
this nominal run does not claim frozen robustness or online adaptation.
