# Controlled physics → nominal TD-MPC2 → frozen severity evaluation

The new runners are `scripts/validate_physics.py`, `scripts/train_nominal.py`, `scripts/evaluate_frozen.py` and `scripts/plot_results.py`. Configurations are `configs/physical_validation.json`, `configs/train_nominal.yaml`, and `configs/frozen_sweep.yaml`. Experiment-only helpers implement bookkeeping, evaluation and checkpoint packaging, not learning algorithms. The original agent, WorldModel, MPC, optimizers, losses, replay Buffer and OnlineTrainer.to_td are reused. `envs.nsgym.make_env` is called explicitly. The original environment factory, adaptation code and archive are unchanged. A subsequently reproduced Ensemble meta-template device-transfer defect required the narrow shared-layer compatibility fix documented at the end of this file.

## Current status

| Stage | Status | Evidence |
|---|---|---|
| Controlled physics | Physically validated, PASS | `results/nsgym/physical-validation-v3/` |
| Pipeline tests | 27 passed, 1 skipped | `validation/pipeline/tests.txt` |
| Real WorldModel | CPU forward inference tested | `test_real_world_model_cpu_forward_compatibility` |
| Replay/optimizer training smoke | BLOCKED, not passed | Original implementation requires CUDA; host reports CUDA unavailable |
| Nominal controller quality | Not established | No nominal-trained checkpoint exists |
| Frozen evaluator | Implemented, statistics unit-tested | End-to-end trained-checkpoint test skipped without CUDA |
| Frozen scientific robustness | Not evaluated | Requires nominal training and controller-quality evidence |
| Legacy regression | 123 passed, 4 skipped, 1 pre-existing failure | `validation/pipeline/legacy-tests.txt` |

The old failure is `test_mechanism_analysis.py::MechanismTests::test_episode_artifacts_boundary_rng_and_frozen`: the fixture lacks `cfg.evaluation`. It was reproduced before the earlier archival task. It remains unchanged.

## Physical results and interpretation

The final controlled run uses the initial grid `[0.7, 1.0, 1.3]`, three initial states (seeds 101/202/303) crossed with three independent action tapes (401/502/603), each 60 transitions long. `dt=0.05 s` per transition, five integration steps, no extra action repeat. Torso masses are approximately **4.37514644, 6.25020921, 8.12527197 kg**.

All comparisons restore `mjSTATE_INTEGRATION` with `mj_getState`/`mj_setState` and verify restoration after `mj_forward`, including warmstart and additional integration inputs. [MuJoCo documents this state as the full set of inputs to forward dynamics](https://mujoco.readthedocs.io/en/3.3.5/computation/). The first action of each tape is the paired one-step experiment; the remaining actions establish open-loop divergence. No policy observations determine the action tapes.

Both nonnominal masses produce nonzero paired one-step position and velocity differences. Across nine pairs, the mean mixed-coordinate qvel-difference norms are approximately 0.11928 (0.7) and 0.08637 (1.3), versus zero at 1.0. These combine linear and angular coordinates and are descriptive only; plots separate translational position (m), angular position (rad), linear velocity (m/s) and angular velocity (rad/s). Joint-wise signed differences and finite-difference acceleration over 0.05 s are retained. Raw qacc is the simulator's last substep acceleration, not a fresh final-state acceleration computation.

Nominal NS-Gym and unwrapped Gymnasium trajectories/rewards agree at atol=1e-10, rtol=0. Nominal replay is invariant; all requested masses are read back after every action and restored on reset. All model arrays except body_mass and explicitly identified derived constants are unchanged. MuJoCo options, including gravity, are invariant. Torso inertia is intentionally unchanged: this is a mass-only intervention, not a calibrated density/geometry change.

The first exploratory validation directory (`physical-validation-v1`) records a FAIL because the initial invariant allowlist omitted camera/light offsets relative to center of mass. The installed MuJoCo header identifies `cam_poscom0`/`light_poscom0` as derived qpos0/sub-COM values. They legitimately change with mass. The corrected run v2 passed; v3 additionally saves the nominal model/XML and checks all exposed numeric MuJoCo options. Failed and intermediate runs were preserved. Repeatability tests regenerate independent directories and compare raw arrays exactly.

Raw NPZ files include full initial integration states, the exact action tapes, qpos/qvel/qacc, reward components, mass readbacks, dt and dimensions. The saved model XML/arrays, config, command, versions, Git status and source hashes support reproduction. Result hashes are in `validation/pipeline/physical-artifact-hashes.json`. Plots show mean ± descriptive SD across crossed pairs, not confidence intervals from nine independent initial states. These exploratory scales are not physically calibrated severity classes. A larger state difference does not establish worse control or policy robustness.

## Training semantics and checkpoint contract

The full configuration declares 1,000,000 environment transitions, original model-size 5, 256-sample batches, replay capacity 1,000,000 entries, evaluation/checkpoint intervals of 50,000 transitions, 10 fixed held-out evaluation seeds and the original MPC settings. Original loss coefficients, optimizer settings, rho=0.5 and discount bounds are inherited. HalfCheetah's 1,000-step episode yields the original discount heuristic's 0.995; the legacy MuJoCo factory's optional rho/discount overrides are not applied.

The runner preserves OnlineTrainer's zero-based update boundary: seed_steps=10,000 means 10,001 exploration transitions (`step <= seed_steps`), 10,000 initial updates at zero-based step 10,000, then one update per transition. Both the threshold and actual exploration count are explicit. One `agent.update` performs the original model optimizer step and actor optimizer step; counts report agent updates and metadata states two optimizer steps per update. Unlike the original inclusive `while step <= steps`, the new runner executes exactly the requested transition budget. Complete episodes enter the unchanged replay with the original initial dummy row. Truncation remains separate from true termination.

Smoke mode uses 35 transitions, 17 exploration transitions, short 8-step episodes, model-size 1, batch 8, 32 MPC samples, two planner iterations and two evaluation seeds. This changes only workload settings; the objectives remain original. The shortened episode naturally yields discount 0.95 under the unchanged heuristic. This is an integration test, never convergence evidence. It should execute 34 original agent updates on CUDA and verify fresh-agent model/optimizer/scale reload. It has **not run on this CPU-only host**.

Evaluation uses a separate environment, no updates, explicit per-episode seeds, and preserves training RNG and MPC planning history. It records return, length, forward velocity/variability, reward components, effort, action magnitude/variability/smoothness and termination/truncation. The random-action comparison uses independent seeded NumPy actions. Stability metrics are proxies, not safety/fall certification.

Each checkpoint directory contains original `agent.save` model.pt, optimizers.pt (both optimizer states plus RunningScale), config/dimensions, seed, transition/update counts, version/Git/source/config hashes, and model/optimizer file hashes. A fresh original agent reloads and compares all learned/optimizer/scale state. Exact training resume is explicitly **not supported**: replay, environment and RNG continuation state are not checkpointed. Do not describe these files as exact resumable training snapshots.

## Frozen evaluation and quality gate

All conditions load the same checkpoint and planner configuration. Fixed mass applies before transition 0; `t0=True` resets MPC warm starts at episode boundaries. Model parameters require no gradients; inference is no-grad. Learned model, target, optimizer and RunningScale state hashes must match before/after evaluation; source model/optimizer files must also retain their hashes. Diagnostics remain outside policy inputs.

The default sweep uses 10 distinct held-out seeds per severity. `--episodes 30` or `--episodes 50` creates an explicit larger seed list, persisted in metadata. `--expanded` uses the full preregistered grid. Episode records are preserved in JSONL/CSV. Aggregate intervals use 10,000 percentile-bootstrap resamples of episode seeds; nominal-relative intervals bootstrap paired differences. Relative ratios are omitted when nominal mean is nonpositive/near zero or its interval crosses zero.

A non-smoke checkpoint becomes eligible for **preliminary** scientific reporting only when the last three nominal evaluations, each with at least 10 seeds, have a positive lower 95% paired bootstrap return-gain bound over random. This predeclared consistency gate is not a published benchmark threshold, proof of convergence, or multiple-testing correction. All curves and episode data still need review; evaluation sets are repeatedly reused. If the criterion fails, `quality.json` is INCONCLUSIVE and scientific sweeps are blocked. `--integration-only` permits software checks but explicitly blocks scientific interpretation. Smoke checkpoints cannot pass the scientific gate.

There is no equivalent external HalfCheetah-v5 trained baseline included here. Do not infer adaptation, robustness or causal model-error effects from physical tests, random references, CPU forward tests or smoke checkpoints.

## Commands

Run from the repository root. Every output directory below must be new. Temporary validation environments already exist on this Mac at `/private/tmp/tdmpc2-nsgym-modern` and `/private/tmp/tdmpc2-nsgym-venv`; they are not durable installations. CUDA jobs must run on a CUDA-capable host with this repository checked out. Parsers were checked with `--help`; full training/frozen commands below are checked CLI forms, **not executed experiments**.

```bash
# 1. Create a separate environment using this host's actual Python 3.11.
# On a CUDA host, use its Python 3.11 executable for this creation step.
/Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m venv /tmp/tdmpc2-nsgym-research
/tmp/tdmpc2-nsgym-research/bin/python -m pip install -r experiments/nsgym/requirements-training.txt
NSPY=/tmp/tdmpc2-nsgym-research/bin/python
# On this Mac, use: NSPY=/private/tmp/tdmpc2-nsgym-modern/bin/python

# 2–3. Existing adapter and new physical tests.
"$NSPY" -m pytest experiments/nsgym/tests/test_halfcheetah.py -q
"$NSPY" -m pytest experiments/nsgym/tests/test_physical_validation.py -q
"$NSPY" -m pytest experiments/nsgym/tests -q

# 4. Declared physical grid (add --expanded for the optional grid).
"$NSPY" experiments/nsgym/scripts/validate_physics.py --config experiments/nsgym/configs/physical_validation.json --output results/nsgym/physics-run

# 5. Real training smoke; requires CUDA, including replay/optimizer/save/reload.
"$NSPY" experiments/nsgym/scripts/train_nominal.py --smoke --output results/nsgym/training-smoke

# 6. Expensive full nominal training: NOT launched automatically.
"$NSPY" experiments/nsgym/scripts/train_nominal.py --config experiments/nsgym/configs/train_nominal.yaml --output results/nsgym/nominal-training

# 7. Nominal-only checkpoint evaluation as a software/quality inspection.
"$NSPY" experiments/nsgym/scripts/evaluate_frozen.py --checkpoint results/nsgym/nominal-training/checkpoint-001000000 --nominal-only --integration-only --output results/nsgym/nominal-checkpoint-eval

# Optional frozen integration check of the smoke checkpoint, not robustness evidence.
"$NSPY" experiments/nsgym/scripts/evaluate_frozen.py --checkpoint results/nsgym/training-smoke/checkpoint-000000035 --integration-only --episodes 2 --output results/nsgym/frozen-smoke

# 8. Scientific preliminary sweep: requires successful checkpoint quality gate.
"$NSPY" experiments/nsgym/scripts/evaluate_frozen.py --config experiments/nsgym/configs/frozen_sweep.yaml --checkpoint results/nsgym/nominal-training/checkpoint-001000000 --quality-report results/nsgym/nominal-training/quality.json --output results/nsgym/frozen-sweep

# 9. Regenerate tables/plots into new directories. Runners already produce them.
"$NSPY" experiments/nsgym/scripts/plot_results.py --kind physical --input results/nsgym/physics-run --output results/nsgym/physics-plots
"$NSPY" experiments/nsgym/scripts/plot_results.py --kind training --input results/nsgym/nominal-training --output results/nsgym/training-plots
"$NSPY" experiments/nsgym/scripts/plot_results.py --kind frozen --input results/nsgym/frozen-sweep --output results/nsgym/frozen-plots

# 10. Legacy tests in the separate original-compatible interpreter on this host.
/private/tmp/tdmpc2-nsgym-venv/bin/python -m pytest tests -q
```

For different training budgets, edit the explicit YAML before running; checkpoint names encode the actual transition count. Do not reuse the one-million-step checkpoint path after changing the budget.

## Actual commands executed in this task

- `/private/tmp/tdmpc2-nsgym-modern/bin/python experiments/nsgym/scripts/validate_physics.py --output results/nsgym/physical-validation-v1` — failed invariant allowlist; retained.
- Same command with `physical-validation-v2` — PASS.
- Same command with `physical-validation-v3` — PASS, final controlled data.
- `/private/tmp/tdmpc2-nsgym-modern/bin/python -m pytest experiments/nsgym/tests/test_physical_validation.py -q` — 2 passed.
- `/private/tmp/tdmpc2-nsgym-modern/bin/python -m pytest experiments/nsgym/tests -q` — final 27 passed, 1 CUDA skip. An intermediate newly added CPU forward test initially omitted the existing API's required `task=None`; corrected in the test, not the model.
- `/private/tmp/tdmpc2-nsgym-modern/bin/python experiments/nsgym/scripts/train_nominal.py --check-config` and `--smoke --check-config` — resolved configs saved in validation/pipeline.
- `/private/tmp/tdmpc2-nsgym-modern/bin/python experiments/nsgym/scripts/train_nominal.py --smoke --output results/nsgym/train-smoke-v1` — BLOCKED before creating output: no CUDA. Captured in validation/pipeline/training-smoke.txt.
- `/private/tmp/tdmpc2-nsgym-modern/bin/python experiments/nsgym/scripts/plot_results.py --kind physical --input results/nsgym/physical-validation-v3 --output results/nsgym/physical-plots-v3` — PASS.
- All four entry points were invoked with `--help`; outputs are saved in validation/pipeline.
- `/private/tmp/tdmpc2-nsgym-venv/bin/python -m pytest tests -q` — 123 passed, 4 skipped, 1 unchanged failure; 148 subtests passed.

The separate NS-Gym environment gained tensordict 0.14.3 and torchrl 0.14.1; existing pinned NS-Gym/Gymnasium/NumPy/MuJoCo were not upgraded. The original agent/replay/trainer imports pass there. Matplotlib reports an unwritable default cache location and uses a temporary cache; plots were generated and visually inspected. Results remain local and git-ignored; preserve the complete result directories in research backups. No full training or full frozen sweep was launched, and no checkpoint was fabricated.

## server4: driver 535.230.02 and PyTorch CUDA build

The reported server has an RTX A5000 but PyTorch 2.14.0+cu130 cannot initialize CUDA with driver 535.230.02. The generic `torch==2.14.0` pin did not constrain the CUDA build; it is insufficient for this server. `nvidia-smi` showing CUDA 12.2 reports the driver's supported CUDA level, not the runtime bundled with PyTorch.

The [official CUDA 12.6 wheel index](https://download.pytorch.org/whl/cu126/torch/) provides PyTorch 2.14.0+cu126 for Python 3.11/Linux x86_64. Driver 535 falls in [NVIDIA's CUDA 12.x minor compatibility range](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). This has feature limitations, particularly for PTX/JIT paths; keep the pipeline's existing `compile=false`. This build is a compatibility candidate, not a claim that server-side optimizer tests have passed.

Run in the separate `tdmpc2-nsgym` environment:

```bash
python -m pip install --upgrade 'torch==2.14.0+cu126' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r experiments/nsgym/requirements-server4-cu126.txt
python -m pip check
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('Bundled CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA initialization still fails'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device='cuda')
y = x @ x.T
assert torch.isfinite(y).all()
torch.cuda.synchronize()
print('CUDA allocation/matmul: PASS')
PY
python experiments/nsgym/scripts/train_nominal.py --smoke --output results/nsgym/server4-cu126-smoke-01
```

Use a new smoke output directory on retries. No server driver was changed, no legacy environment was upgraded, and these server-side commands have not been executed from the local Mac. Subsequent installs should use the server-specific requirements file to retain the CUDA build constraint.

## Reproduced meta-template device-transfer defect

After server4 passed CUDA allocation/matmul with cu126, agent construction failed at `WorldModel.to(cuda)`. This was reproduced locally with `WorldModel.to('cpu')`: `_Qs.module`, `_detach_Qs.module`, and `_target_Qs.module` exposed registered meta placeholders. The previous CPU forward test did not perform device transfer or exercise the Q ensemble, so it missed this defect.

The minimal compatibility fix is in `tdmpc2/common/layers.py`: the Ensemble execution template is retained as an unregistered attribute; real stacked weights remain registered in `params`. An explicit `train()` override propagates dropout mode to the template, including target evaluation mode. The vmap/to_module evaluation and all learning losses, optimizers, MPC logic and Polyak update formulas are unchanged. This follows the separation between stacked ensemble parameters and the execution template described by [TensorDict](https://docs.pytorch.org/tensordict/stable/cookbook.html). Do not apply `to_empty()` to the whole world model: that would allocate uninitialized storage rather than restore learned weights.

Regression tests compare ensemble predictions, parameter gradients and an Adam step against independent reference networks; verify CPU device transfer, dropout modes, detached-weight aliasing, target-weight independence/Polyak updates, and real checkpoint save/load. The world-model CPU test now explicitly calls `.to('cpu')`. This is a demonstrated compatibility defect justifying the narrow shared-layer change under the task constraints; no WorldModel/Agent training logic was refactored.

Checkpoint state dictionaries no longer include stateless `*.module.*` meta placeholders. New checkpoint roundtrips are tested. A historical checkpoint containing those placeholder keys may need a separately reviewed conversion before strict loading; no automatic checkpoint rewrite is performed. No NS-Gym trained checkpoint was produced before this failure.

After syncing the corrected source and tests to server4, use a fresh output directory:

```bash
python -m pytest experiments/nsgym/tests/test_ensemble_device.py -q
python experiments/nsgym/scripts/train_nominal.py --smoke --output results/nsgym/server4-cu126-smoke-02
```

Local focused checks: 7 passed, 1 CUDA skip. This fixes the locally reproduced initialization error; the full CUDA training smoke still needs to be rerun on server4.

Full post-fix NS-Gym suite: **30 passed, 1 CUDA-dependent skip** (`validation/pipeline/meta-fix-nsgym-tests.txt`). Legacy suite: **123 passed, 4 skipped, the same 1 pre-existing failure** (`validation/pipeline/meta-fix-legacy-tests.txt`). The focused patch is `validation/pipeline/ensemble-meta-fix.patch`; apply it only to a server checkout that still contains the original template registration.
