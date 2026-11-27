# Validation report — 2026-11-08

## Outcome

The minimal adapter and all four experiment modes pass real MuJoCo validation. Existing learning/adaptation code is unchanged. All original local artifacts remain accessible and byte-identical. Full certification of every optional legacy simulator remains blocked by missing simulator dependencies/checkpoints on this host.

| Check | Result | Evidence |
|---|---|---|
| Archive integrity | PASS: 1,150 existing files match initial SHA-256 at both paths | `python3 archive/legacy_maniskill2/verify_archive.py` |
| File-level audit | 1,160 entries: 92 archive, 71 shared, 502 retain, 495 uncertain | `archive/legacy_maniskill2/migration_manifest.csv` |
| Legacy regressions | 123 passed, 4 skipped, 148 subtests passed; 1 pre-existing failure | `archive/legacy_maniskill2/regression-tests.txt` |
| Baseline failure reproduction | Same missing `cfg.evaluation` failure in an untouched pre-archive HEAD checkout | `archive/legacy_maniskill2/baseline-known-failure.txt` |
| Adaptation imports | EpisodeAdapter, DecisionAdapter, shift protocols, adapt_shift and dynamics_shift package pass | `validation/legacy-imports.json` and adapter tests |
| Original MuJoCo task factory | HalfCheetah-v4 and Walker2d-v4 reset/step pass under Gymnasium 0.29.1 | `validation/legacy-imports.json` |
| NS-Gym tests | 20 passed, real simulator | `validation/adapter-tests.txt` |
| Nominal transitions | Reset, 30 steps and scalar rewards match unwrapped HalfCheetah-v5 at absolute tolerance 1e-10, relative tolerance 0 | `test_nominal_matches_unwrapped_gymnasium` |
| Physical intervention | Torso mass changes and trajectories diverge; restoring mass restores derived subtree masses; same full simulator state yields matching next transition after restoration | `test_mass_shift_changes_real_transitions_and_restores_model_constants` |
| Scheduling/reset | Zero-based pre-action event boundaries; seeded reset trajectory equality; unseeded reset restarts schedule and restores mass | Parameterized timing tests |
| Agent interface | State shape 17, action shape 6, scalar float/tensor rewards, no ground-truth metadata in obs/info, finite valid actions, reset/truncation guard | Contract tests |
| Experiment smoke | 40 transitions each for nominal/fixed/abrupt/reversible; no model training | `results/nsgym/*-validation.json`, hashes in `validation/smoke-summary.json` |

## Measured physical values

Nominal torso mass is 6.25020920502092 kg; scale 1.5 sets 9.37531380753138 kg. Fixed changes before action 0; abrupt changes before action 10; reversible restores 6.25020920502092 kg before action 20. Actual readbacks are logged after every transition. The smoke action tapes are identical across modes (seed 17). Maximum observation difference from the nominal tape is 6.29295 for fixed and 7.42742 for abrupt/reversible. Those differences validate simulator sensitivity, not learned adaptation performance. A reversed parameter does not erase accumulated state divergence.

## Versions and compatibility

- No NS-Gym distribution was present in the initially inspected environments. The official source was fetched and inspected before implementing the adapter. Tested revision: `127df4b714fe4fcf680721d8e9278a96090acd7e`. Shallow checkout metadata reports `ns_gym==0.0.post1`; the commit and installed API source hashes in `validation/smoke-summary.json` are the reproducibility identifiers.
- NS-Gym validation: Python 3.11, Gymnasium 1.3.0, MuJoCo 3.3.7, NumPy 2.4.6, Torch 2.14.0. Legacy validation: Gymnasium 0.29.1, NumPy 1.26.4, Hydra 1.3.2, hydra-submitit-launcher 1.2.0, MuJoCo 3.3.7, Torch 2.14.0. This verifies compatibility on this host, not a re-creation of the original CUDA training stack (which pins different Torch/MuJoCo versions).
- Temporary environments inherit the existing basilisk-rl site packages, then override dependencies locally. `pip check` also reports inherited, unrelated bsk-rl's pytest==7.3.1 requirement versus pytest 9.1.1; the legacy environment additionally has unrelated inherited Gymnasium/NumPy constraints. These are validation environments, not replacements for the original simulator environment. `requirements.txt` is for a clean standalone NS-Gym environment; a clean full install was not separately tested.
- Current NS-Gym imports CliffWalking-v1/MuJoCo-v5 during import and fails on legacy Gymnasium 0.29.1. The original `envs.make_env` also uses older Gymnasium APIs such as `gym.logger.set_level`. It is intentionally unchanged and is tested only in the legacy environment. The NS-Gym factory is explicit and separate; existing training entry points do not automatically dispatch to it.
- The original evaluator sets `MUJOCO_GL=egl`, unsupported on macOS. The legacy checks pass with `MUJOCO_GL=glfw`; no source change was made.
- ManiSkill2/SAPIEN and MetaWorld imports are unavailable (`gym` absent); DMControl is unavailable (`dm_control` absent). MyoSuite wrapper imports, but its optional environment registrations and Box2D tasks were not runtime-certified. The four existing skips require ManiSkill2 or CUDA/checkpoint assets. These checks must run on the original simulator host before claiming all legacy registrations work.
- One existing regression fixture lacks `evaluation`, causing an AttributeError in `common/mechanism_analysis.py:104`. Reproduced before archival; no adaptation code was changed to mask it.

## Archival scope and remaining work

Only confirmed reports, source evidence, status metadata and result bundles were moved. Original relative symlinks preserve old paths and commands. Executable experiment scripts, benchmark suites and Hydra configuration remain at their original paths because imports/root-relative dependencies require a separate coordinated migration. Shared TD-MPC2, dynamics-shift wrappers and adaptation modules stay in place. See `archive/legacy_maniskill2/uncertain_paths.md` for every deferred path.

The pre-existing modified PHYSICAL_RESPONSE_VALIDATION.md and calibration directory deletions/untracked replacements are untouched and excluded from both commits. Ten tracked files were already absent at the initial snapshot; they are not claimed as preserved local files. No local checkpoint binaries were present. MANIFEST.md preserves recorded commands, seeds, checkpoint paths, metadata and source-revision references; external server checkpoints/ZIPs cited by old reports are not recreated.

Previously ignored legacy results remain local and ignored. Their hashes and locations are tracked in the manifest; a Git clone alone does not contain them. The four small NS-Gym validation runs are explicitly tracked as reproducibility evidence; subsequent results stay ignored. No experiments, checkpoints, logs or reports were deleted, and no expensive training was run.

Exact rerun commands are in `README.md`. Cleanup and integration are separate commits; only task-owned paths were staged.
