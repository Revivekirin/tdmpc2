# Dynamics+Q local verification — 2026-11-12

- New contract/integration tests: **13 PASS**, 9.605 seconds in the recorded run.
- Existing Joint final-evaluator tests: **3 PASS**.
- Existing TTT final-evaluator suite: **8 PASS, 2 environment errors**. The two
  scoring tests import local NS-Gym, whose registry expects CliffWalking-v1 but
  this local Gymnasium exposes only CliffWalking-v0. No dependency or existing
  evaluator code was changed to bypass this mismatch; exact trace is retained
  in existing-ttt-tests.log. The new runner's CPU integration tests do pass both
  scoring paths with a labeled synthetic environment.
- CLI `--help`: PASS without CUDA/model loading; output saved in cli-help.txt.
- CPU bounded CLI smoke: **PASS**, 100 synthetic transitions, 2 dynamics steps,
  2 Q steps; actual small TD-MPC2 WorldModel, distributional target/loss and EMA.
- Full-budget CPU test: 3,000 transitions, exactly 118 dynamics + 118 Q steps,
  first event 75, last event 3,000; all unchanged groups verified.
- Runner integration test: actual small CPU model + labeled synthetic environment,
  final snapshot save/reload, integrity audit, fixed-bank prediction and 20-seed
  control orchestration pass. This tests interfaces, not real physical performance.
- Aggregation fixture: 720 paired rows, negative effects retained, six figures
  generated, partial stages rejected, missing real training logs explicitly marked.
- Existing local Frozen manifest grid, seed list, checkpoint SHA and original
  model/environment source compatibility: PASS.
- Actual 750k CUDA/NS-Gym smoke: **NOT_RUN**. Local CUDA/checkpoint unavailable.
- 36-stream scientific adaptation / held-out prediction / shifted control:
  **NOT_RUN**. No new scientific interaction or performance result is claimed.

Interpreter: `/private/tmp/tdmpc2-nsgym-venv/bin/python` (PyTorch 2.14.0,
TensorDict 0.14.3, CPU). These are local contract-test versions, not substitutes
for the server checkpoint's recorded environment versions. Server loading still
checks the recorded ns_gym/Gymnasium/MuJoCo versions.

Logs in this directory are unit/integration fixtures. Research results will be
written only by the explicit server commands in RUN_DYNAMICS_Q.md.
