# Archival validation

- `python3 archive/legacy_maniskill2/verify_archive.py`: 1,150 existing files verified byte-for-byte at original and destination paths. Ten already-deleted tracked files are recorded, not restored.
- `/private/tmp/tdmpc2-nsgym-venv/bin/python -m pytest tests -q`: 123 passed, 4 skipped, 148 subtests passed, 1 failed. Full output in regression-tests.txt.
- The failure is `test_mechanism_analysis.py::MechanismTests::test_episode_artifacts_boundary_rng_and_frozen`: its SimpleNamespace fixture lacks `evaluation`, required by existing mechanism_analysis.py. Reproduced against an untouched `git archive HEAD` checkout; see baseline-known-failure.txt. Left unchanged to avoid modifying adaptation behavior.
- Skips require CUDA/official checkpoints or ManiSkill2/SAPIEN. Those simulator registrations cannot be certified on this Mac without their dependencies.
- Initial system Python had no ML dependencies; basilisk-rl lacked Hydra. Tests used a temporary venv inheriting Torch 2.14.0, with NumPy 1.26.4, Gymnasium 0.29.1, Hydra 1.3.2 and hydra-submitit-launcher 1.2.0. Original environments unchanged.
- Core implementations, experiment scripts/configs and adaptation imports remain in their original locations. No training was run.
