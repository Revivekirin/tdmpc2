# Q/planning analysis verification — 2026-11-13

- `analyze_q_planning.py --help`: PASS.
- Tier B `--dry-run`: PASS; resolved server paths and flags without model loading or output creation.
- `analyze_q_planning.py smoke`: PASS, 9 CPU tests in 2.403 seconds. Uses actual small WorldModel, actual softmax/two_hot_inv, and original TDMPC2._estimate_value.
- Coverage: F/D immediate equality; altered Q divergence; categorical normalization/support/scalar decode; finite rejection; candidate action bounds/reproducibility; native reward/Q score decomposition; unchanged state/RNG; prohibited Adam construction; hybrid encoder guard; snapshot format restoration and tamper guard; seed split; nominal labels kept separate; aggregation/plots/report; no output overwrite.
- Snapshot-format test uses synthetic states and a stub only for the legacy whole-tree checksum function; actual model tensors are restored and compared. This does not certify real server snapshot compatibility.
- Local real Frozen bank: 240 episode NPZ hashes and fixed physical readbacks verified. The production canonical bank samples 16 transitions per episode (3,840 rows). No simulator transitions.
- Plot rendering was inspected on synthetic fixture outputs in `/private/tmp/q-planning-preview`; these are not scientific results and are not included in the research package.
- CPU dependencies: existing Torch 2.14.0 plus tensordict 0.14.3 installed under temporary `/private/tmp/q-planning-deps`. No repository/server environment packages changed.
- Actual nominal 750k checkpoint and final adapted snapshots are absent locally. Four-method compatibility, scientific immediate Q, terminal Q, rankings, and Q-control associations are NOT_RUN locally. Server commands are in RUN_Q_PLANNING_ANALYSIS.md.
- No original training, planner, evaluation, checkpoint or result files modified. Only new analysis code, documentation, tests and versioned outputs added.
