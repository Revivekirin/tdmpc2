# Physical dynamics analysis evidence

2026-11-08. No new simulator rollout was executed locally.

- `command_probe.json`: actual production `transform` function extracted by AST and run with prescribed inputs. This checks commands only, not robot physics.
- `source_sha256.json`: local source / official reference / input CSV hashes.
- `reference_ms2_0_4_1_post0/`: selected official wheel sources, NOT the current server installation. Kept for line-accurate review; never installed or added to the runtime import path. Original LICENSE included.
- Wheel source: https://pypi.org/project/mani-skill2/0.4.1.post0/
- Wheel SHA256: `04121326858535186db3fcd9598d005f673c43aaab874e61244b8f6e2c7aa1ec`.
- Local repository HEAD at analysis: `32d8730c13f7aa36adeb5ad76781d669c09093b3`. Existing dirty `scripts/run_stage3b.py` and `.DS_Store` were preserved.
- `action_pipeline`: conceptual execution diagram.
- `gain_command_vs_response`: measured software command transformation plus reference target calculation; no EEF response measurements.
- `delay_native_step_timeline`: software command execution, not a simulator trajectory.
- `mass_physical_response`: historical baseline mass scaled analytically; constant net-force acceleration theory.
- `friction_contact_response`: conditional reference calculation assuming average combination and reference material values; no measured slip or force.
- `shift_severity_physical_comparison`: existing pilot CSV observations, n=10 per condition, one seed set.

Each figure has PNG and PDF versions. Reproduce with a Python environment containing NumPy and Matplotlib:

```bash
python scripts/analyze_physical_dynamics.py
```

The optional server collector `scripts/probe_physical_response.py` has only been syntax checked. Run it in the existing simulator environment, with a native-action tape and a new output path. It does not import or train the world model. See the main report for its measurement limitations.
