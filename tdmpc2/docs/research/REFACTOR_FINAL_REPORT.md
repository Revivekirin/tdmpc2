# TD-MPC2 dynamics-shift refactor: final Phase 4B report

Audit date: 2026-11-23. Repository root: `/Users/jihyekim/Desktop/tdmpc2`; active project: its inner `tdmpc2/`; starting HEAD `d129783`, branch `main`, one Git worktree. The pre-existing `.DS_Store`, Phase 3B reporting documentation/test edits and untracked historical-recovery document were not staged or changed for this cleanup.

## Final architecture and ownership

```text
Original TD-MPC2 model, planner, actor/critic and training replay (unchanged)
  ↑ original model APIs
research/dynamics_shift/
  adaptation: dynamics.py, decision.py, dynamics_critic.py, replay.py,
              schedule.py, audit.py, objectives.py, joint.py, runner.py
  composition: factory.py, checkpoints.py, composer.py
  evaluation and analysis: evaluation.py, planning_analysis.py
  reporting/: source checks, canonical records, metrics, figure data,
              existing renderer delegation and report assembly
  ↑ historical import and CLI contracts
common/{stage3a,decision_ttt,dynamics_q}.py and existing NS-Gym / ManiSkill2 CLIs
  ↑ original source manifests and saved results
experiments/nsgym/analysis/control_q/ and immutable result/checkpoint trees
```

The canonical adaptation layer owns Dynamics-only, decision-scheduled Dynamics TTT and Dynamics+Q numerical updates. Joint retains its original `TDMPC2.update` delegation. `CheckpointComposer` reads validated nominal/adapted formats; `EvaluationRunner` is used only by the verified Joint final evaluation path. `planning_analysis.py` observes fixed-candidate and native planner outputs without modifying `_plan`. `ReportBuilder` reads immutable completed sources and delegates to the original NS-Gym statistics and control-Q renderers. The original TD-MPC2 learning objectives and MPC behavior were not changed in Phase 4B.

## Removal decision and retained layers

**Files removed: none.** The final six-gate audit found no file with zero active callers, a complete proven replacement, passing interface parity, intact historical provenance, no golden-byte dependency and unchanged public CLI behavior. No permanent forwarding wrapper was created. The detailed old-path/replacement/caller/risk table is [CLEANUP_INVENTORY.md](CLEANUP_INVENTORY.md).

| Class | Retained implementation | Concrete reason |
|---|---|---|
| Documented diagnostics | `experiments/nsgym/scripts/check_legacy.py`, `smoke.py` | NS-Gym README still calls their distinct legacy import/task probe and four-mode random-action physical probe. No equivalent command exists. |
| Pilot configurations | `train_pusher_pilot.yaml`, `train_reacher_pilot.yaml` | Both task tests, task guides and Slurm job defaults still resolve them. |
| ManiSkill2 stage scripts | `scripts/run_stage3b.py`, `run_stage3b2.py`, severity, audit, analysis and result scripts | Tests, calibration commands, other scripts and archived command manifests still use their path-specific CLIs and native-step protocols. |
| Compatibility imports | `common.stage3a`, `common.decision_ttt`, `common.dynamics_q`; original `ExistingRunnerAdapter` | Production runners and golden tests import the old paths. Joint still needs the original runner delegate. |
| Historical numerical/CLI modules | `experiments/nsgym/analysis/control_q/`, `analyze_q_planning.py`, `q_planning_core.py`, mechanism helpers, `training_support.py` | Original source hashes, saved-bank inference, report outputs and public CLI contracts require them. The two byte-equivalent `tree_hash` implementations remain separate because the smoke→full source gate hashes their containing files. |
| Mirrored scripts | 22 equal-byte pairs under outer `scripts/` and inner `tdmpc2/scripts/` | Both relative paths are present in callers and archived records; byte equality does not establish path compatibility. |

An older control-Q function inventory labels individual plotting functions `DELETE`; it does not establish that their containing modules are obsolete. `analyze_q_planning.py`, for example, still provides model loading, saved-bank inference, aggregation and its public command. Historical artifacts, checkpoints, source manifests, fixtures, original TD-MPC2 files and all current CLIs remain intact.

## Active commands

These commands are **entry-point examples**, not a request to run training or a new experiment:

```bash
# Existing NS-Gym diagnostic and inspection interfaces
python experiments/nsgym/scripts/smoke.py --config experiments/nsgym/configs/nominal.json --output /tmp/nsgym-smoke.json
python experiments/nsgym/scripts/check_legacy.py
python experiments/nsgym/scripts/inspect_pusher.py --help
python experiments/nsgym/scripts/inspect_reacher.py --help

# Completed-source, read-only reporting into a new directory
python -m experiments.nsgym.analysis.control_q.run --mode all --results-root /path/to/results/nsgym --output /tmp/control-q-report-new

# Existing task-specific runner and analysis interfaces
python experiments/nsgym/scripts/run_dynamics_only_ttt.py --help
python experiments/nsgym/scripts/run_dynamics_q.py --help
python experiments/nsgym/scripts/run_full_joint_rl.py --help
python experiments/nsgym/scripts/analyze_q_planning.py --help
python scripts/run_stage3b.py --help
python scripts/run_stage3b2.py --help
```

NS-Gym commands require its compatible interpreter. ManiSkill2 stage commands require the separate ManiSkill2/SAPIEN runtime. The `--help` examples identify interfaces; this audit did not execute those commands as simulator parity tests.

## Regression and historical compatibility

At Phase 4B starting HEAD, the local compatible CPU interpreter (`/Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python`, with `PYTHONPATH=/private/tmp/tdmpc2-golden-deps`) executed the 23-file golden/adaptation/runner/checkpoint/evaluation/planning/reporting/Stage 3/control-Q suite: **150 passed, 5 subtests passed in 43.30 s**. Its complete command is the Phase 4A local bundle in [CLEANUP_INVENTORY.md](CLEANUP_INVENTORY.md). The stage-numbered and calibration CLI contract bundle (`test_stage3b2.py`, `test_stage3b_severity.py`, `test_stage3b_continue.py`, `test_stage3b_report.py`, `test_stage3a_report.py`, `test_physical_severity_calibration.py`) then reported **39 passed, 25 subtests passed in 26.28 s**. The authentic-source historical report test, run with `TDMPC2_REPORT_RESULTS_ROOT=/Users/jihyekim/Desktop/tdmpc2/results/nsgym`, reported **1 passed in 101.22 s**. All three final local summaries had zero failures and zero reported skips. No runtime code was changed after these runs.

```bash
cd /Users/jihyekim/Desktop/tdmpc2/tdmpc2
PYTHONPATH=/private/tmp/tdmpc2-golden-deps /Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m pytest tests/test_stage3b2.py tests/test_stage3b_severity.py tests/test_stage3b_continue.py tests/test_stage3b_report.py tests/test_stage3a_report.py tests/test_physical_severity_calibration.py -q -rs
TDMPC2_REPORT_RESULTS_ROOT=/Users/jihyekim/Desktop/tdmpc2/results/nsgym PYTHONPATH=/private/tmp/tdmpc2-golden-deps /Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m pytest tests/test_research_reporting_historical.py -q -rs
```

Immediately before Phase 4B, server4 validated the SHA-256-checked Phase 4A implementation candidate in `/tmp/tdmpc2-phase4a-f2d2ab13` using the isolated `/tmp/tdmpc2-nsgym-golden-conda` interpreter: **159 general tests and 5 subtests passed; 5 CUDA historical tests passed; 26 NS-Gym physical tests passed; 1 historical report test passed**. The final historical test compared all 11 common CSVs numerically and all 42 PNG files by SHA-256. Phase 4B changed documentation only and deleted no code; these are prior exact-code server results, not a new Phase 4B server run. No skip was reported by those four final server summaries.

The original source validator requires the three authentic families `q-terminal-planning-v1`, `q-decision-relevance-v1` and `q-terminal-planning-presentation-v2`, their completion and source hashes, **36 conditions** and **108 adaptation streams**. The local ignored results tree contains all three; the server test used the verified copies in `/tmp/tdmpc2-controlq-recovered/results/nsgym`. Historical checkpoint compatibility and native planning golden tests are in the executed suites. Neither a historical result nor an immutable golden fixture was regenerated or overwritten. Exact historical training continuation remains unsupported where replay, environment and RNG state were not saved.

## Limits and readiness

The original ManiSkill2/SAPIEN runtime was not executed in this phase, so its simulator, native-step and stage CLI parity remain unverified. The Mac CPU environment cannot collect the NS-Gym-specific Pusher/Reacher tests; those passed in the preceding server4 combined suite. Closed-loop Q-swap return is `NOT_MEASURED`; no new Q-swap experiment was run. The 22 mirrored script pairs and old import shims remain technical cleanup debt until a path-specific caller migration and compatible-host parity are proven. The unrelated pre-existing working-tree edits remain outside this report's commit.

The canonical interfaces and regression boundaries are ready for **new, separately specified research experiments** on their compatible runtimes. This report does not assert ManiSkill2 simulator parity, exact checkpoint resume, or a measured closed-loop critic effect.
