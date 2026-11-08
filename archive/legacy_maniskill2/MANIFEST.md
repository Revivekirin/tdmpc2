# ManiSkill2 archival manifest

Snapshot HEAD: `b809299d46f1c37b5c20fb1e8cbbea40062771de`.

The CSV inventories every non-cache file (including ignored results and tracked files already missing), SHA-256, tracking status, static imports and text references. Dependency scanning is conservative, not a complete runtime proof. No algorithms or executable experiments were relocated. Uncertain paths remain in place; see uncertain_paths.md. Existing dirty calibration changes were excluded.

## Layout and compatibility

Original relative layout is mirrored beneath this directory. Relative symlinks at the original paths preserve existing commands and report references. Run all historical commands from the repository root. Preserved bytes and symlink targets are checked by verify_archive.py. Previously ignored results remain local and ignored: commit alone is NOT a backup of those data. Copy this entire directory with symlinks dereferenced when transferring the archive. No checkpoint binaries were present in the initial inventory; remote checkpoint paths below are references, not locally recoverable weights.

## Historical commands, seeds and checkpoints

Commands below are recorded evidence, not commands executed during archival. Preserve their recorded source revisions and server environment; see the linked reports, docker/environment.yaml and original Hydra/run metadata. Server-only archives mentioned by reports may be absent locally. No missing artifacts have been fabricated.

### mechanism-full-status.json

```json
{
  "episodes": 50,
  "runs": [
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=nominal",
        "schedule=nominal",
        "evaluation.episodes=50",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=standard"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_delay",
        "schedule=fixed",
        "evaluation.episodes=50",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=2"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_gain",
        "schedule=fixed",
        "evaluation.episodes=50",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=0.7"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_delay",
        "schedule=abrupt",
        "evaluation.episodes=50",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=2",
        "schedule.switch_step=10"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-3.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_delay",
        "schedule=fixed",
        "evaluation.episodes=50",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=2"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=stack-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/stack-cube-3.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_delay",
        "schedule=fixed",
        "evaluation.episodes=50",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=2"
      ],
      "returncode": 0
    }
  ],
  "simulator_validation": "SEPARATE_REQUIRED_CHECK",
  "analysis_status": "PASS"
}
```

### mechanism-smoke-status.json

```json
{
  "episodes": 5,
  "runs": [
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=nominal",
        "schedule=nominal",
        "evaluation.episodes=5",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=standard"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_delay",
        "schedule=fixed",
        "evaluation.episodes=5",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=2"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_gain",
        "schedule=fixed",
        "evaluation.episodes=5",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=0.7"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_delay",
        "schedule=abrupt",
        "evaluation.episodes=5",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=2",
        "schedule.switch_step=10"
      ],
      "returncode": 0
    },
    {
      "command": [
        "/nfs4/jhkim/conda_envs/tdmpc2/bin/python",
        "/nfs4/jhkim/repos/tdmpc2/tdmpc2/evaluate_shift.py",
        "task=pick-cube",
        "checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-3.pt",
        "episodic=false",
        "compile=false",
        "model_size=5",
        "analysis=frozen_mechanism",
        "dynamics_shift=action_delay",
        "schedule=fixed",
        "evaluation.episodes=5",
        "save_video=false",
        "seed=1",
        "mpc=true",
        "evaluation=paired",
        "dynamics_shift.parameter.shifted=2"
      ],
      "returncode": 0
    }
  ],
  "simulator_validation": "SEPARATE_REQUIRED_CHECK",
  "analysis_status": "PASS"
}
```

## Other provenance sources

All benchmark suites and Hydra configs remain at their original locations. Every existing result/log/metadata file is inventoried in migration_manifest.csv. In particular, the active calibration tree retains run_frozen.sh, run_probes.sh, calibration_manifest.json, per-seed probe metadata and checkpoint references. Its pre-existing tracked deletions are not repaired or committed by this task.

## Moved roots

- `DYNAMICS_SHIFT_AUDIT.md` → `archive/legacy_maniskill2/DYNAMICS_SHIFT_AUDIT.md` (compatibility symlink retained)
- `DYNAMICS_SHIFT_PHYSICAL_ANALYSIS.md` → `archive/legacy_maniskill2/DYNAMICS_SHIFT_PHYSICAL_ANALYSIS.md` (compatibility symlink retained)
- `DYNAMICS_SHIFT_REDESIGN_DECISION.md` → `archive/legacy_maniskill2/DYNAMICS_SHIFT_REDESIGN_DECISION.md` (compatibility symlink retained)
- `EXPERIMENTS.md` → `archive/legacy_maniskill2/EXPERIMENTS.md` (compatibility symlink retained)
- `MECHANISM_ANALYSIS.md` → `archive/legacy_maniskill2/MECHANISM_ANALYSIS.md` (compatibility symlink retained)
- `MECHANISM_RESULTS.md` → `archive/legacy_maniskill2/MECHANISM_RESULTS.md` (compatibility symlink retained)
- `PAIRING_ROOT_CAUSE.md` → `archive/legacy_maniskill2/PAIRING_ROOT_CAUSE.md` (compatibility symlink retained)
- `PAIRING_VALIDATION.md` → `archive/legacy_maniskill2/PAIRING_VALIDATION.md` (compatibility symlink retained)
- `SETUP_SHIFT_BENCHMARK.md` → `archive/legacy_maniskill2/SETUP_SHIFT_BENCHMARK.md` (compatibility symlink retained)
- `docs/stage3b_severity.md` → `archive/legacy_maniskill2/docs/stage3b_severity.md` (compatibility symlink retained)
- `docs/stage3a.md` → `archive/legacy_maniskill2/docs/stage3a.md` (compatibility symlink retained)
- `docs/dynamics_shift_phase1.md` → `archive/legacy_maniskill2/docs/dynamics_shift_phase1.md` (compatibility symlink retained)
- `docs/stage3c.md` → `archive/legacy_maniskill2/docs/stage3c.md` (compatibility symlink retained)
- `docs/dynamics_shift_benchmark.md` → `archive/legacy_maniskill2/docs/dynamics_shift_benchmark.md` (compatibility symlink retained)
- `docs/stage3b.md` → `archive/legacy_maniskill2/docs/stage3b.md` (compatibility symlink retained)
- `docs/stage3b_shift_selection.md` → `archive/legacy_maniskill2/docs/stage3b_shift_selection.md` (compatibility symlink retained)
- `docs/stage3b2.md` → `archive/legacy_maniskill2/docs/stage3b2.md` (compatibility symlink retained)
- `mechanism-full-status.json` → `archive/legacy_maniskill2/mechanism-full-status.json` (compatibility symlink retained)
- `mechanism-smoke-status.json` → `archive/legacy_maniskill2/mechanism-smoke-status.json` (compatibility symlink retained)
- `analysis/physical_dynamics` → `archive/legacy_maniskill2/analysis/physical_dynamics` (compatibility symlink retained)
- `results/stage3b-generalization` → `archive/legacy_maniskill2/results/stage3b-generalization` (compatibility symlink retained)
- `results/stage3c-report-20261108-170105` → `archive/legacy_maniskill2/results/stage3c-report-20261108-170105` (compatibility symlink retained)
- `results/mechanism-aggregates.json` → `archive/legacy_maniskill2/results/mechanism-aggregates.json` (compatibility symlink retained)
- `audits/audit-20261102T022148-a2a28886` → `archive/legacy_maniskill2/audits/audit-20261102T022148-a2a28886` (compatibility symlink retained)
