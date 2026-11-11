# Reacher pipeline bounded verification — 2026-11-11

No real Reacher nominal checkpoint was found by recursive discovery across `/Users/jihyekim/Desktop/tdmpc2`. Maximum actual training transitions and the reported 9.5M are unverified. No checkpoint selection or scientific prediction/control recovery is claimed.

## Executed

- `python -m pytest experiments/nsgym/tests/test_reacher_pipeline.py experiments/nsgym/tests/test_reacher.py -q`: **10 passed** (7 new pipeline tests, 3 existing nominal Reacher tests).
- Unified runner `--help`: PASS.
- All ten stages with `--dry-run`: PASS; previews created no output directory, loaded no CUDA model and ran no physics.
- `smoke --synthetic`: PASS, with real Reacher-v5 MuJoCo physics and an explicitly labeled small CPU neural fixture. Four streams: D0/M0 × Frozen/Decision, each 100 online transitions. Frozen: 0 updates; Decision: 2 updates per stream at transitions 75 and 100. Fixed components and auxiliary states invariant. Snapshot serialization/reload predictions and held-out replay guards pass.
- All 24 physical conditions: readback, unchanged inertia, positive valid inertia, reset persistence, finite transitions, moving/nonzero-action probes, exact full integration-state restore and repeat nominal agreement at 1e-12 pass. Nominal k=1 agrees with unmodified Gymnasium across resets.
- Full Frozen and counterfactual pipeline exercised with a mocked checkpoint loader returning the CPU fixture: two evaluation seeds and one calibration seed, all 24 labels. Tests verify canonical nominal NPZ reuse with correct family-specific physical metadata.
- Fixture snapshot held-out prediction/control, complete-grid summaries, primary-final-budget/missing-condition logic, seed-unit statistics and plotting: PASS. Fixture outputs exist only in temporary test directories, never presented as actual TD-MPC2 measurements.
- Real physical-response figure generated and visually inspected. Joint position, velocity, fingertip and acceleration panels have separate units; mean/median/seed bootstrap CI, defaults and multipliers shown.

## Evidence directories (local, ignored experiment outputs)

- `results/nsgym/reacher/local-discovery-v2/discovery.json`: BLOCKED, no checkpoint candidates.
- `results/nsgym/reacher/local-smoke-v2/`: actual CPU fixture smoke and real physical inventory/probes. Its budget protocol is intentionally BLOCKED because synthetic throughput cannot choose a CUDA budget.
- `results/nsgym/reacher/local-physics-preview-v2/`: measured physical plots, seed summaries, action regime counts, non-monotonicity flags. No latent/control figure is fabricated without a nominal model.

Versions: NS-Gym 1.0.12.post2, Gymnasium 1.3.0, MuJoCo 3.3.7, PyTorch 2.14.0 CPU. Runtime `/private/tmp/tdmpc2-nsgym-modern/bin/python`. Installed runtime defaults: damping 1 N m s/rad for each joint; each link mass 0.03560471674068433 kg; obs10/action2; horizon50; frame_skip2; dt0.02 s.

## Stage status

| Stage | Status | Meaning |
|---|---|---|
| R0 | BLOCKED | No local nominal training checkpoint/logs; server discovery required |
| R1 | READY | Real physical checks completed; selected-model latent calibration awaits R0 |
| R2 | NOT_RUN | Complete scientific Frozen matrix awaits R0/R1 |
| R3 | NOT_RUN | CPU bounded fixture passes; real checkpoint/CUDA smoke and full adaptation not run |
| R4 | READY | Comparison protocol only; no invented cross-task adaptation outcome |

All changes are new Reacher-specific files in the canonical nested project. Existing HalfCheetah files/results/checkpoints are unchanged. No remote job, nominal training, exhaustive sweep or full adaptation was launched. The code is not automatically committed/pushed by this task.

Limitations: actual CUDA/MPPI loading is not exercised locally; one-step held-out prediction only; no exact interrupted simulator/planner resume; one nominal training seed; force saturation missing when actuator force limits are disabled. Full held-out control cost is separate from the adaptation feasibility estimate.
