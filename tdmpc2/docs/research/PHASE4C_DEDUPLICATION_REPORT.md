# Phase 4C physical repository deduplication

The Git root is `/Users/jihyekim/Desktop/tdmpc2` (starting HEAD `2b97d27`); the executable Python project is `tdmpc2/`. Its `tdmpc2/tdmpc2` import package, `tests/`, `experiments/nsgym/`, `scripts/`, `benchmarks/`, `assets/`, `datasets/` and `docker/` form the canonical active tree. Local imports and the server regression commands run from that inner project. The Git root had one worktree at the start of this phase. The root has no configured CI or Python package build file; the active regression suites are under the inner `tests/` and `experiments/nsgym/tests/`. No upstream TD-MPC2 model, trainer, planner or research numerical implementation changed.

## Physical changes

```text
Before: Git root/{assets,benchmarks,datasets,docker,scripts,experiments/nsgym}
                     ↕ mirrored implementations/configs/assets
        Git root/tdmpc2/{assets,benchmarks,datasets,docker,scripts,experiments/nsgym}

After:  Git root/tdmpc2/{assets,benchmarks,datasets,docker,scripts,experiments/nsgym}
        Git root/docs/ → archived ManiSkill2 documents (8 links)
        Git root/experiments/nsgym/ → 2 unique old guides + tracked Finder metadata
```

The seven audited outer families contained 134 tracked paths before cleanup. The [per-file inventory](PHYSICAL_DEDUPLICATION_INVENTORY.csv) records both SHA-256 digests where comparable, classification, caller/provenance evidence, canonical replacement and disposition. This is also the **exact list of removed paths**: filter the `decision` column for `DELETE`.

| Family at Git root | Before | Removed | After | Removed bytes |
|---|---:|---:|---:|---:|
| `assets/` | 9 | 9 | 0 | 5,998,015 |
| `benchmarks/` | 6 | 6 | 0 | 8,275 |
| `datasets/` | 2 | 2 | 0 | 359 |
| `docker/` | 3 | 3 | 0 | 3,411 |
| `scripts/` | 24 | 24 | 0 | 234,685 |
| `experiments/nsgym/` | 82 | 79 | 3 | 337,790 |
| `docs/` archival symlinks | 8 | 0 | 8 | 0 |
| **Total** | **134** | **123** | **11** | **6,582,535** |

The 118 byte-identical files have the same SHA-256 as their inner replacements. Five outer NS-Gym scripts differ and were compared separately: the inner Pusher/Reacher inspection CLIs have the Phase 4A byte-identical mocked CSV and summary outputs; the inner dynamics-only runner has golden stream equivalence; the inner severity printer retains old phase branches and adds a checkpoint option; the inner plotter retains existing plot modes and adds a Reacher mode. Exact path-specific text and the added options are documented differences. No archived manifest or saved source hash was rewritten. Original removed bytes remain in Git history.

The root README now points at the canonical image files and instructs users to enter `tdmpc2/` before running relative shell examples. Inner Slurm jobs, tests and scripts already use this working directory. The three retained outer NS-Gym paths are divergent `REACHER.md`, divergent `RUN_DYNAMICS_ONLY_TTT.md`, and tracked `.DS_Store`; the two documents contain unique archival commands and `.DS_Store` is unrelated generated metadata. The eight outer `docs/` links point to distinct preserved ManiSkill2 archive files. They are not duplicate source bytes.

Follow-up documentation cleanup: the unique interactive Reacher `srun` instructions and the alternative `/nfs4` result path note were incorporated into the inner guides, then the two outer NS-Gym Markdown files were removed. The Phase 4C table above records the earlier commit's state; after this follow-up only the unrelated tracked `.DS_Store` remains in the outer NS-Gym directory. Research test files remain because their coverage or fixture imports are distinct. The mirrored ManiSkill2 archive Markdown files remain required by `verify_archive.py`, which checks original and archived paths against recorded SHA-256 values.

The canonical inner project retains `common.stage3a`, `common.stage3b`, `common.decision_ttt` and `common.dynamics_q` public imports, stage-numbered ManiSkill2 CLIs, `check_legacy.py`, `smoke.py`, pilot configs, and original control-Q sources. Active imports, CLI contracts, task tests, Slurm defaults, and source-hash checks require them. `training_support.tree_hash` and `control_q.models.tree_hash` remain separate because the smoke-to-full control-Q validator hashes their containing files. These are concrete blockers to additional deletion, not a basis for retaining the removed outer mirrors.

`__pycache__/`, `*.py[cod]` and `.pytest_cache/` were already ignored. `.DS_Store` is now ignored for future files; existing tracked metadata was left alone. No checkpoint, raw result, validated source archive, golden fixture, temporary archive of unknown ownership or historical figure was deleted.

## Active development commands

```bash
cd tdmpc2
python -m pytest tests/test_golden_research_baselines.py tests/test_research_runner_equivalence.py -q -rs
python experiments/nsgym/scripts/check_legacy.py
python experiments/nsgym/scripts/smoke.py --help
python experiments/nsgym/scripts/inspect_pusher.py --help
python experiments/nsgym/scripts/inspect_reacher.py --help
python experiments/nsgym/scripts/run_dynamics_only_ttt.py --help
python -m experiments.nsgym.analysis.control_q.run --help
python scripts/run_stage3b.py --help
```

Use the isolated NS-Gym CUDA interpreter for its simulation commands, the legacy Gymnasium 0.29.1 interpreter for `check_legacy.py`, and the separate ManiSkill2/SAPIEN interpreter for its simulator commands. The `--help` examples do not run an experiment. Historical reporting writes only to a new output directory and requires the authenticated historical results root.

## Validation boundary

The validation record below is populated only from executed test summaries. The unchanged local golden fixtures and historical sources are read-only. Historical reporting checks 36 conditions, 108 adaptation streams, 11 CSVs and 42 PNGs. Exact historical resume remains unsupported where replay, environment and RNG states were not saved. ManiSkill2/SAPIEN parity must be reported separately from NS-Gym.

After the 123 physical deletions, the local CPU interpreter ran the 22-file golden/adaptation/runner/checkpoint/evaluation/planning/reporting/control-Q bundle: **147 passed, 5 subtests passed in 42.25 s**. The cleanup utility and stage/CLI bundle ran separately: **42 passed, 25 subtests passed in 8.25 s**. The authentic historical reporting test then passed in **97.57 s** with the original `TDMPC2_REPORT_RESULTS_ROOT=/Users/jihyekim/Desktop/tdmpc2/results/nsgym`. No failures or skips were reported in those final summaries. The union is 190 passed tests and 30 subtests. An earlier attempted command named a nonexistent `test_research_utility_consolidation.py` and collected no tests; the actual file is `test_research_cleanup_utilities.py`, which was executed in the second bundle.

The canonical inner project was also packed by itself, extracted to `/private/tmp/tdmpc2-phase4c-isolated`, and linked read-only to existing results. Its golden, stage CLI, reporting and cleanup utility subset ran **14 passed, 3 subtests passed in 2.63 s**. This checks that those active imports do not rely on the removed outer source tree.

```bash
cd tdmpc2
PYTHONPATH=/private/tmp/tdmpc2-golden-deps /Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m pytest tests/{test_research_reporting_pipeline,test_research_planning_analysis,test_research_evaluation_runner,test_research_checkpoint_composition,test_golden_research_baselines,test_golden_checkpoint_compatibility,test_research_adaptation_interfaces,test_research_runner_equivalence,test_planner_instrumentation,test_stage3a,test_decision_ttt,test_stage3a_runner,test_stage3a_matched,test_stage3b,test_mechanism_analysis}.py experiments/nsgym/tests/{test_dynamics_q,test_q_planning,test_reacher_dynamics_q,test_full_joint_rl,test_evaluate_joint_final,test_control_q,test_control_q_metrics}.py -q -rs
PYTHONPATH=/private/tmp/tdmpc2-golden-deps /Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m pytest tests/test_research_cleanup_utilities.py tests/test_stage3b2.py tests/test_stage3b_severity.py tests/test_stage3b_continue.py tests/test_stage3b_report.py tests/test_stage3a_report.py tests/test_physical_severity_calibration.py -q -rs
TDMPC2_REPORT_RESULTS_ROOT=/Users/jihyekim/Desktop/tdmpc2/results/nsgym PYTHONPATH=/private/tmp/tdmpc2-golden-deps /Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m pytest tests/test_research_reporting_historical.py -q -rs
```

The SHA-256 checked inner-only candidate (`cdb27785695281d6050987b9f3942cecad88d3e23b81307528b260b262996484`) was extracted on server4 to `/tmp/tdmpc2-phase4c-cdb27785` with the original result tree linked read-only. Its isolated `/tmp/tdmpc2-nsgym-golden-conda` interpreter executed the general golden/adaptation/runner/checkpoint/evaluation/planning/reporting/control-Q/Pusher/Reacher suite: **159 passed, 5 subtests passed in 58.69 s**. The CUDA historical suite ran **5 passed in 9.83 s**; the NS-Gym physical suite ran **26 passed in 5.11 s**. The authentic historical reporting test ran **1 passed in 137.44 s**, checking the recovered source manifests, 36 conditions, 108 adaptation streams, 11 historical CSV comparisons and 42 PNG hashes. The server summaries reported no failure or skip. The local CPU run cannot establish CUDA or NS-Gym simulator parity; those passed on server4. ManiSkill2/SAPIEN remains unexecuted in its separate runtime.

On the same server candidate, both canonical `plot_results.py --help` and `print_severity_commands.py --help` exited successfully. This verifies their CLI imports and argument parsers in the compatible runtime; it does not claim byte-identical console text for their newer options.
