# Server commands: physical severity and frozen control

This document is generated from `print_severity_commands.commands()` and validated with the actual `run_severity.py` parser. Commands below are for the user to execute on server4. No large sweep has been run by Codex. Activate the existing environment and use a new result root for each rerun.

```bash
cd /nfs4/jhkim/repos/tdmpc2
conda activate tdmpc2-nsgym
python -c 'import sys; print(sys.executable)'
python experiments/nsgym/scripts/run_severity.py --help
python experiments/nsgym/scripts/print_severity_commands.py --help
```

Every stage saves a full resolved manifest before simulation, prints condition/seed/episode/transition budgets, and refuses existing output directories. The checkpoint directory (or its model.pt path) is accepted; the primary SHA256 is enforced. Use `--dry-run` on any simulation command to inspect its complete plan without importing the runtime or writing files.

Do not pipe all generated commands into a shell. Execute phases separately: environment/readback → physical calibration → review → Frozen generalization → review → planning/nonstationary. The optional S3–S5 large phase needs separate approval.

## Generate commands from the current protocol

```bash
python experiments/nsgym/scripts/print_severity_commands.py --phase physical --dry-run
python experiments/nsgym/scripts/print_severity_commands.py --phase frozen
python experiments/nsgym/scripts/print_severity_commands.py --phase nonstationary
```

Use `--output-root results/nsgym/physical-severity-750k-v2` to generate a fresh rerun plan. Existing results are never overwritten. Offline summaries can be regenerated into any fresh directory with `run_severity.py summarize --input ... --output ...`.

Seed splits: calibration 41000001–41000003; software/schedule validation 42000001–42000003; held-out evaluation 43000001–43000020; reserved paired sensitivity 44000001–44000020. Full protocol files are JSON-compatible YAML under configs/.

## A–B. Environment, preregistration and physical inventory

Budget: 12 resolved conditions, 3 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py manifest --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/preregistered
```

Budget: 4 resolved conditions, 1 seeds, 3 episodes/probe rollouts, at most 36 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py environment --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/environment --smoke
```

Budget: 4 resolved conditions, 1 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py inventory --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/inventory --smoke
```

## Bounded physical/schedule validators (before full runs)

Budget: 4 resolved conditions, 1 seeds, 8 episodes/probe rollouts, at most 96 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py validate --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/validate-nominal --smoke --schedule nominal
```

Budget: 4 resolved conditions, 1 seeds, 8 episodes/probe rollouts, at most 96 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py validate --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/validate-fixed --smoke --schedule fixed
```

Budget: 4 resolved conditions, 1 seeds, 8 episodes/probe rollouts, at most 96 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py validate --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/validate-abrupt --smoke --schedule abrupt
```

Budget: 4 resolved conditions, 1 seeds, 8 episodes/probe rollouts, at most 96 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py validate --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/validate-gradual --smoke --schedule gradual
```

Budget: 4 resolved conditions, 1 seeds, 8 episodes/probe rollouts, at most 96 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py validate --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/validate-reversible --smoke --schedule reversible
```

Budget: 4 resolved conditions, 1 seeds, 8 episodes/probe rollouts, at most 96 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py validate --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/validate-repeated --smoke --schedule repeated
```

## C. Nominal 750k state/action bank and physical response grid

Budget: 1 resolved conditions, 3 seeds, 3 episodes/probe rollouts, at most 3,000 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py bank --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/bank
```

Budget: 12 resolved conditions, 3 seeds, 672 episodes/probe rollouts, at most 6,720 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py physical --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/physical --input results/nsgym/physical-severity-750k-v1/bank
```

Budget: 12 resolved conditions, 3 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/physical-summary --input results/nsgym/physical-severity-750k-v1/physical
```

## D. Controlled latent diagnostics and plots

Budget: 12 resolved conditions, 3 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py latent --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/physical-latent --input results/nsgym/physical-severity-750k-v1/physical
```

Budget: 12 resolved conditions, 3 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/physical-latent-summary --input results/nsgym/physical-severity-750k-v1/physical-latent
```

## E. Held-out Frozen generalization, action and latent diagnostics

Run only after reviewing the full physical result. `--gate` requires its PASS completion and matching protocol/conditions.

Budget: 12 resolved conditions, 20 seeds, 240 episodes/probe rollouts, at most 240,000 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py frozen --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/frozen --gate results/nsgym/physical-severity-750k-v1/physical
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/frozen-summary --input results/nsgym/physical-severity-750k-v1/frozen --seed-split evaluation --physical-reference results/nsgym/physical-severity-750k-v1/physical
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py latent --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/heldout-latent --input results/nsgym/physical-severity-750k-v1/frozen --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/heldout-latent-summary --input results/nsgym/physical-severity-750k-v1/heldout-latent --seed-split evaluation
```

## D2. Bounded planning analysis after Frozen result review

The default k=1.5 diagnostic is predeclared and bounded to six calibration examples. It does not measure candidate simulator returns.

Budget: 2 resolved conditions, 3 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py planning --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/planning --input results/nsgym/physical-severity-750k-v1/physical --severity 1.5 --gate results/nsgym/physical-severity-750k-v1/frozen
```

## F. Required S0–S2 independent evaluations

The S1 fixed command duplicates the original Frozen protocol and can instead reuse that dataset. It is not independent evidence. S0 k=1 is a reference; S2 evaluates the full grid.

Budget: 2 resolved conditions, 20 seeds, 40 episodes/probe rollouts, at most 40,000 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py nonstationary --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-nominal --schedule nominal --gate results/nsgym/physical-severity-750k-v1/frozen --severity 1
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-nominal-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-nominal --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py latent --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-nominal-latent --input results/nsgym/physical-severity-750k-v1/nonstationary-nominal --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-nominal-latent-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-nominal-latent --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 240 episodes/probe rollouts, at most 240,000 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py nonstationary --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-fixed --schedule fixed --gate results/nsgym/physical-severity-750k-v1/frozen
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-fixed-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-fixed --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py latent --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-fixed-latent --input results/nsgym/physical-severity-750k-v1/nonstationary-fixed --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-fixed-latent-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-fixed-latent --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 240 episodes/probe rollouts, at most 240,000 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py nonstationary --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-abrupt --schedule abrupt --gate results/nsgym/physical-severity-750k-v1/frozen
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-abrupt-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-abrupt --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py latent --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-abrupt-latent --input results/nsgym/physical-severity-750k-v1/nonstationary-abrupt --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-abrupt-latent-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-abrupt-latent --seed-split evaluation
```

## F2. Optional S3–S5 large evaluations — separate approval required

These commands are implemented for a later approved phase. The mandatory command generator excludes them unless `--phase optional` is requested.

Budget: 12 resolved conditions, 20 seeds, 240 episodes/probe rollouts, at most 240,000 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py nonstationary --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-gradual --schedule gradual --gate results/nsgym/physical-severity-750k-v1/frozen --allow-optional-large
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-gradual-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-gradual --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py latent --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-gradual-latent --input results/nsgym/physical-severity-750k-v1/nonstationary-gradual --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-gradual-latent-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-gradual-latent --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 240 episodes/probe rollouts, at most 240,000 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py nonstationary --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-reversible --schedule reversible --gate results/nsgym/physical-severity-750k-v1/frozen --allow-optional-large
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-reversible-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-reversible --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py latent --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-reversible-latent --input results/nsgym/physical-severity-750k-v1/nonstationary-reversible --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-reversible-latent-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-reversible-latent --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 240 episodes/probe rollouts, at most 240,000 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py nonstationary --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-repeated --schedule repeated --gate results/nsgym/physical-severity-750k-v1/frozen --allow-optional-large
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-repeated-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-repeated --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py latent --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-repeated-latent --input results/nsgym/physical-severity-750k-v1/nonstationary-repeated --seed-split evaluation
```

Budget: 12 resolved conditions, 20 seeds, 0 episodes/probe rollouts, at most 0 new simulator transitions.

```bash
python experiments/nsgym/scripts/run_severity.py summarize --checkpoint results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000 --output results/nsgym/physical-severity-750k-v1/nonstationary-repeated-latent-summary --input results/nsgym/physical-severity-750k-v1/nonstationary-repeated-latent --seed-split evaluation
```

## G. Outputs and analysis regeneration

| Stage | Measured outputs |
|---|---|
| inventory | physical_parameter_inventory.csv, environment_contract.json, installed source snapshots |
| validate | schedule_readbacks.csv, checks.json, completion.json |
| bank / frozen / nonstationary | per-episode NPZ, step CSV, contact JSON, episodes.csv, dataset_index.json, model_integrity.json |
| physical | controlled NPZ/contact JSON, physical_response_pairs.csv, checks.json |
| latent | latent_errors.csv, dataset hashes, model_integrity.json |
| planning | complete MPPI traces/RNG states, planning_comparisons.csv, model_integrity.json |
| summarize physical | physical_response_by_severity.csv, physical_response_vs_severity.png |
| summarize latent | latent_error_by_severity.csv, latent_error_by_regime.csv, latent_error_aligned.csv, latent_error_vs_severity.png |
| summarize Frozen/nonstationary | frozen_generalization.csv or nonstationary_evaluation.csv, action_diagnostics_by_severity.csv, return/velocity/control and action figures, shift-aligned tables/figures |

Plots and tables are generated only from saved measurements. No NOT_RUN condition has placeholder numeric results. The primary server bank, latent/MPPI diagnostics, full physical grid, Frozen grid and nonstationary grid remain NOT_RUN in the local delivery. The preserved local exploration smoke is explicitly labelled and cannot satisfy the non-smoke Frozen gate.

If a stage fails, retain its directory and failure.json. Inspect the error and use a new output path for a corrected rerun. If primary weight identity, original MPPI configuration, state replay or numerical checks fail, dependent stages stop. No training command is part of this plan.
