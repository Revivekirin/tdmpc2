# Dynamics shift Phase 1

## Scope and semantics

Architecture, losses, optimizer, training loop, checkpoint loading and `agent.act`
arguments are unchanged. The existing inference method uses `torch.no_grad()`;
no learning/update/save calls are added. World-model planning still uses commanded
actions. Gain and executed actions appear only in diagnostic info/logs.

All clocks count **native environment calls**, zero-based before execution.
A decision repeats its command twice (unless native termination occurs early).
The horizon remains 100 decisions. `shift_global_step` persists through reset;
`shift_episode_step` restarts at zero. The top-level info describes the final
native call; `shift_native_steps` preserves every native call with its own reward
and native done. `decision_done` in JSONL additionally records the outer timeout.

- nominal: gain 1, including all valid in-bounds actions unchanged.
- fixed: configured gain from the first call.
- abrupt: gain 1 for `t < switch_step`, configured gain thereafter.
- repeated: alternating nominal/shifted blocks of `period` native steps.
- sinusoidal: `1 + amplitude * sin(2*pi*t/period)`; period is a full cycle.

`shift_clock=episode shift_switch_step=100` switches after 50 decisions in every
episode. `shift_clock=global shift_switch_step=100` switches only once in the run.
For a switch at the boundary after a full 100-decision episode, use global 200;
early termination moves the boundary relative to episodes because the clock counts
actual native calls, not episode slots.

Default scaling preserves the last (gripper) component; the first three components
of PickCube's `pd_ee_delta_pos` action are scaled. Set
`shift_preserve_gripper=false` to scale all components. All components are clipped
to their native Box bounds. Nominal equivalence assumes valid in-bounds input.

Official 0.4.1 source inspected:
- [Panda configuration](https://github.com/mani-skill/ManiSkill/blob/v0.4.1/mani_skill2/agents/configs/panda/defaults.py): arm followed by gripper.
- [Base environment](https://github.com/mani-skill/ManiSkill/blob/v0.4.1/mani_skill2/envs/sapien_env.py): four-value step and `reset(seed=None, reconfigure=False)`.

## Evaluation

On a host with the repository dependencies, ManiSkill2 0.4.1 assets, SAPIEN and CUDA:

```bash
bash scripts/evaluate_dynamics_shift.sh /absolute/path/pick-cube-1.pt
```

This runs E0–E3 plus both E4 clock variants, each with 50 episodes and seeds 1–50.
The model size must match the checkpoint (the script assumes the 1M checkpoint).
To run individual conditions from `tdmpc2/`:

```bash
python evaluate.py task=pick-cube model_size=1 checkpoint=/absolute/path/pick-cube-1.pt eval_episodes=50 save_video=false shift_mode=abrupt shift_gain=0.7 shift_clock=episode shift_switch_step=100
python evaluate.py task=pick-cube model_size=1 checkpoint=/absolute/path/pick-cube-1.pt eval_episodes=50 save_video=false shift_mode=repeated shift_gain=0.7 shift_period=200
```

Existing nominal CLI syntax remains accepted; all shift defaults are nominal.
ManiSkill evaluation now explicitly seeds each episode with `seed + index`, or the
matching entry of `eval_seeds=[...]`. This intentional reproducibility change means
historical unseeded episode sequences need not reproduce their old aggregate scores.
Other environment families retain their existing reset behavior; non-nominal shifts
are rejected outside single-task ManiSkill evaluation.

Each run creates a unique directory below
`<work_dir>/evaluations/<mode>-<clock>-g<gain>-<UTC>-<unique-id>/`:

- `episodes.csv`: task/checkpoint/episode seed, schedule, return/success, decision
  length, native length, global start/end (end exclusive), output directory.
- `timesteps.jsonl`: one row per native call including both actions, gain, counters,
  native reward/done, decision index and decision done.
- `metadata.json`: resolved config, package/Python versions, source file hashes,
  git commit/status, checkpoint SHA-256 and clock convention.
- `code.diff`: tracked working-tree changes against HEAD; source hashes also cover new Python files.
- `summary.json`: task means and population standard deviations (`ddof=0`), metadata,
  checkpoint unchanged check. Videos, if enabled, also use this unique directory.

The checkpoint is read only; SHA-256 is compared before/after evaluation. Streaming
CSV/JSONL remain available on an interrupted run; `summary.json` indicates completed
runs. Incomplete runs may lack a summary and final checkpoint hash verification.

## Validation performed on this workspace

```bash
/Users/jihyekim/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests -v
python3 -m compileall -q tdmpc2 tests
git diff --check
bash -n scripts/evaluate_dynamics_shift.sh
```

Unit tests validate nominal transitions/reward/success against a deterministic toy
environment, fixed actions/clipping/gripper preservation, hidden observation shape,
global/episode resets and kwargs, abrupt/repeated/sinusoidal schedules, real
`ManiSkillWrapper` class logic loaded with a minimal dependency shim, both native
logs per decision, an odd native-step boundary within one decision, output creation,
summary statistics, unique run paths and checkpoint hashing.

**These are not simulator performance measurements.** The optional real simulator
test compares the same restored initial simulator state and action for nominal and
fixed gain; execute it on the configured simulator host:

```bash
TD_MPC2_TEST_MANISKILL=1 python -m unittest discover -s tests -v
```

Current result: 7 tests passed, 1 real simulator test skipped; syntax/whitespace
checks passed. See `results/dynamics_shift_phase1/validation.txt` and
`validation.json`. No evaluation CSV or performance summary JSON was generated.
Logging tests use disposable synthetic fixtures, removed after the test.

Missing prerequisites: no `.pt` checkpoint exists in this repository, specifically
`pick-cube-1.pt`; default Python lacks NumPy/PyTorch/Gym/Gymnasium/Hydra/ManiSkill2.
Bundled Python supplies NumPy for unit tests only. CUDA, simulator assets, actual
nominal equivalence, physical transition differences and E0–E4 scores remain
unverified. A configured CUDA/SAPIEN host and an absolute checkpoint path are needed.
