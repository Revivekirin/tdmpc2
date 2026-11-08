# NS-Gym HalfCheetah experiments

This is a separate research entry point. No TD-MPC2 learning algorithm, adaptation implementation, existing environment factory, or training script has changed. Use `envs.nsgym.make_env(cfg)` explicitly for future NS-Gym runners. The existing `envs.make_env` continues to select only the original tasks; passing `nsgym-halfcheetah` to it is intentionally unsupported.

## Installation and version boundary

Use Python 3.11 in a **separate** environment:

```bash
python3.11 -m venv /tmp/tdmpc2-nsgym
/tmp/tdmpc2-nsgym/bin/python -m pip install -r experiments/nsgym/requirements.txt
```

The original repository pins Gymnasium 0.29.1, NumPy 1.24.4 and MuJoCo 3.1.2 in `docker/environment.yaml`. The tested NS-Gym revision requires newer MuJoCo/NumPy and imports `CliffWalking-v1` and MuJoCo-v5 at import time. It cannot import with Gymnasium 0.29.1. Do not upgrade the legacy experiment environment in place. NS-Gym was not installed in the initially inspected base, basilisk-rl or pde-shift environments. Validation used temporary environments without altering those installations.

Official API inspected: [NS-Gym source at the pinned revision](https://github.com/scope-lab-vu/ns_gym/tree/127df4b714fe4fcf680721d8e9278a96090acd7e), especially `wrappers/mujoco_env.py`, `base.py`, `schedulers.py`, and `update_functions/single_param.py`. The exported wrapper is `MujocoWrapper` (not `NSMujocoWrapper`). We use its official `DiscreteScheduler`, `StepWiseUpdate`, scalar reward option, and nonpersistent reset behavior.

## Protocol

- Environment: `HalfCheetah-v5`, state Box `(17,)`, action Box `(6,)`. No rendering or model training in smoke tests.
- Shift: absolute torso mass targets derived from the initial simulator mass times `mass_scale`. The scope is torso mass, not inertia scaling, gravity/friction shifts or action perturbations.
- The scheduler clock counts **Gymnasium transitions**, zero-based before each action. Each transition executes HalfCheetah's five MuJoCo integration steps. No additional TD-MPC2 action repeat is added.
- Nominal: empty tunable-parameter dictionary. Fixed: apply mass target before transition 0. Abrupt: apply before `switch_step`. Reversible: apply before `switch_step`, restore the original mass before `restore_step`.
- Reset restores nominal model parameters and reinitializes the update sequence/clock; fixed mass takes effect on the first action, not in the reset record. This is an episodic protocol, not a continuing/global schedule. Reversal restores physics, not the already-diverged simulator trajectory.
- First reset defaults to the constructor seed. Explicit reset seeds reseed the environment and action space. Subsequent unseeded resets advance the environment RNG while restarting the dynamics schedule. Random smoke actions use an independent seeded NumPy generator.
- Gymnasium reset returns `(observation, info)`; step returns five values, with a Python `float` reward. Stepping before reset or after termination/truncation raises `ResetNeeded`.
- Agent observations contain only `state`. Only `x_position`, `x_velocity`, `reward_run`, and `reward_ctrl` are allowed through in `info`; NS-Gym's ground-truth change dictionaries and scheduler time are excluded. Physical logs are delivered to an experiment-owned callback, never through policy observations/rewards/info. As with normal Gymnasium wrappers, a caller intentionally inspecting `env.unwrapped` can access the simulator; this is an agent-input boundary, not a security sandbox.
- `envs.nsgym.make_env` wraps this with the unchanged TD-MPC2 `TensorWrapper`: float32 observation/reward tensors, four-return step, and separate terminated/truncated flags. It fills cfg observation/action dimensions and episode length. It rejects legacy dynamics-shift configuration and multitask/pixel modes.

## Exact smoke commands (from repository root)

The output must not already exist; the runner refuses to overwrite results.

```bash
/tmp/tdmpc2-nsgym/bin/python -m pytest experiments/nsgym/tests -q
/tmp/tdmpc2-nsgym/bin/python experiments/nsgym/scripts/smoke.py --config experiments/nsgym/configs/nominal.json --output results/nsgym/nominal-smoke.json
/tmp/tdmpc2-nsgym/bin/python experiments/nsgym/scripts/smoke.py --config experiments/nsgym/configs/fixed.json --output results/nsgym/fixed-smoke.json
/tmp/tdmpc2-nsgym/bin/python experiments/nsgym/scripts/smoke.py --config experiments/nsgym/configs/abrupt.json --output results/nsgym/abrupt-smoke.json
/tmp/tdmpc2-nsgym/bin/python experiments/nsgym/scripts/smoke.py --config experiments/nsgym/configs/reversible.json --output results/nsgym/reversible-smoke.json
```

On this machine, the already-prepared interpreter is `/private/tmp/tdmpc2-nsgym-modern/bin/python`; substitute it for `/tmp/tdmpc2-nsgym/bin/python` above. Temporary environments are not durable installs. The four `*-validation.json` files are the preserved actual runs from this task. Each includes seed/config, version metadata, actions, observations, rewards, physical readbacks, adapter hash and source revision/dirty status.

Legacy checks use the separate legacy interpreter:

```bash
/private/tmp/tdmpc2-nsgym-venv/bin/python -m pytest tests -q
MUJOCO_GL=glfw /private/tmp/tdmpc2-nsgym-venv/bin/python experiments/nsgym/scripts/check_legacy.py
python3 archive/legacy_maniskill2/verify_archive.py
```

`MUJOCO_GL=glfw` is for macOS; use your existing renderer setting on the Linux simulator host. Real ManiSkill2 regression tests remain opt-in with `TD_MPC2_TEST_MANISKILL=1`; checkpoint tests additionally require CUDA and `TD_MPC2_MECHANISM_CHECKPOINTS`. See `VALIDATION.md` for the exact limitations and known baseline failure.

## Controlled physics and training pipeline

See [PIPELINE.md](PIPELINE.md) for paired physical validation, the separate nominal training entry point, frozen checkpoint evaluation, all execution commands, and the current CUDA blocker. Physical validation passes; learned-controller robustness has not been evaluated.
