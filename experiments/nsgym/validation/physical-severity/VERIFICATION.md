# Physical severity implementation verification

Verified locally on 2026-11-10. No remote run, additional training, optimizer update, adaptation, or large sweep was executed.

## Runtime

Python: `/private/tmp/tdmpc2-nsgym-modern/bin/python`.
Observed packages: NS-Gym 0.0.post1, Gymnasium 1.3.0, MuJoCo 3.3.7, NumPy 2.4.6, Torch 2.14.0. CUDA checkpoint execution is unavailable on this Mac. The four inspected NS-Gym core files are byte-identical to upstream checkout `127df4b714fe4fcf680721d8e9278a96090acd7e`; installed hashes are in each smoke metadata file. This does not replace fresh server verification.

## Checks and actual evidence

- `run_severity.py --help` and `print_severity_commands.py --help`: PASS; saved help outputs alongside this report.
- Every command produced by the command builder was parsed by the actual entry point and resolved against the protocol, including optional commands: PASS. Primary SHA256 remained fixed. Dependency-free syntax/dry-run checks passed.
- `pytest experiments/nsgym/tests/test_severity_protocol.py experiments/nsgym/tests/test_halfcheetah.py experiments/nsgym/tests/test_physical_validation.py -q`: **41 passed** (final run, 3.00 seconds).
- Explicit schedule sequences are tested against manually specified expectations for all six schedules and both parameters, over two resets each. Inactive legacy shift_gain=0.7 and 1.0 produce identical nominal trajectories through the real tensor adapter.
- Offline latent/window aggregation tests use an explicitly synthetic deterministic CPU model; expected one-step and three-step errors are checked analytically. Episode/summary plumbing tests use an explicitly fixed test policy. These fixtures live only in pytest temporary directories and are not trained-model measurements.
- Final actual CLI smoke outputs: `results/nsgym/severity-local-smoke-20261110-v3/`. Logs: `final-smoke-00.log` through `final-smoke-10.log`. Earlier v1/v2 runs remain preserved.

| Stage | Actual simulator transitions | Status |
|---|---:|---|
| Raw Gym / nominal legacy adapter / nominal severity adapter equivalence | 36 | PASS |
| Runtime inventory (15 official parameters) | 0 | PASS |
| Nominal schedule validator | 96 | PASS |
| Fixed schedule validator | 96 | PASS |
| Abrupt schedule validator | 96 | PASS |
| Gradual schedule validator | 96 | PASS |
| Reversible schedule validator | 96 | PASS |
| Repeated schedule validator | 96 | PASS |
| Explicit bounded-exploration bank | 12 | PASS |
| Full-state physical probes and nominal repeats | 36 | PASS |
| Tables and scientific PNG regeneration | 0 | PASS |

Total final CLI smoke: **660 transitions**. The validators use both k=1 and k=1.5, one validation seed, and two resets per condition. Mass readback is 6.2502092050→9.3753138075 kg; bthigh damping is 6→9 N·m·s/rad. Scope, inertia/local inertial-position invariants, reward semantics, observation shape and reset behavior pass. The final physical smoke uses one calibration exploration seed and two sampled states, with three-step prescribed tapes; its nominal replay differences are zero and shifted physical responses are nonzero. The physical-response PNG was visually inspected for readable axes and legends.

## NOT_RUN

- Actual 750k CUDA checkpoint load/nominal bank collection and GPU latent/planning diagnostics.
- Full six-value, two-family physical calibration on primary-policy states.
- 240-episode held-out Frozen grid.
- Required large S0–S2 and optional large S3–S5 evaluations.
- Secondary 950k sensitivity, 500k comparison, any adaptation/RL/plasticity experiment.

The measurement CLI and commands for these stages are implemented. They must be run on the server in order, with prerequisite completion gates. Full result CSVs/figures have not been fabricated. Source inspection and synthetic software tests do not establish GPU performance or model accuracy.
