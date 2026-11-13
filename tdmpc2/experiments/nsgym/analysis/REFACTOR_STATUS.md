# NS-Gym raw-to-report implementation status

The current work replaces manual historical-source registration in the existing analysis package. It does not claim that every training/runtime-dependent legacy helper has been retired.

## Implemented

- Discover source stages from metadata, completion records, identities, CSV schemas and snapshot provenance. `config.json` holds task/method semantics, not historical CSV paths.
- Read the user-provided full server results copy directly. No transfer archive or previously generated comparison table is a required input.
- Validate snapshot/CSV hashes, training completion, evaluation seed sets and source pairing. Exclude smoke, incomplete stages, conflicting duplicates and resumed overlap. Detect the analyzer's own outputs by provenance markers.
- Reconstruct nominal, physical, Frozen, Dynamics-only, Dynamics+Q, Joint, retention, action-effort, Q and planning measurements wherever saved raw evidence supports them. Keep changed-encoder Joint MSE separate.
- Aggregate by adaptation seed, retain negative results, and separate checkpoints, schedules and budgets. Preserve raw-row lineage through summaries and figure data.
- Auto-include newly completed supported runs without config edits. Reacher Dynamics+Q requires completed training plus matching held-out prediction/control and snapshot evidence.

## Verification and limits

The actual execution used `/Users/jihyekim/Desktop/tdmpc2/results/nsgym`, the server originals copied by the user. No remote server execution is claimed. Automatic discovery inspected 15,782 files and 2,773 experiment entries; these are not independent scientific replicates.

The final full original-data run (`unified-analysis/raw-to-report-v2`) produced:

- 145,780 canonical scalar measurements; 140,980 raw-reproduced and 4,800 explicitly derived-only.
- 84 method/condition comparisons: HalfCheetah 36 and Reacher 48, plus separate adapted-method contrasts.
- 23 figures with saved numeric inputs and source lineage.
- Regression: 93,213 EXACT, 10,306 TOLERANCE, 6 documented NUMERICAL_DIFFERENCE, zero FAIL and zero invalid sources.
- 61,742 duplicate scalar records excluded. This is not a count of independent episodes.
- Source immutability passed; 6,562 consumed source/provenance files fingerprinted.

Eighteen unit/integration tests passed, including a newly introduced run ID contributing to the report without a config edit, automatic Reacher DQ completion gating, incomplete training and nominal retention parent batches, a running partial held-out batch and live log, relocated paths, snapshot tampering, seed mismatch and duplicate exclusion. All requested CLI operations were exercised on temporary fixtures; the full original-data command exercises both tasks and Q/planning end to end. Only temporary fixtures were removed.

Q arrays were independently decoded with the existing TD-MPC2 categorical math. Original GPU float32 per-observation values remain the statistical inputs after verification. Six cancellation-sensitive differences outside direct tolerance but within a recorded propagated float32 bound are documented separately, not called exact matches. Bellman targets were not saved; residual/CE remain DERIVED_ONLY.

Current scientific coverage is `PASS_WITH_COVERAGE_GAPS`: HalfCheetah shifted-adapted nominal retention and Reacher DQ held-out prediction/control are absent. These missing evaluations were not fabricated or launched. A new task/schema/procedure still needs explicit semantic adapter support.

The presentation revision uses matching CI colors, source-backed nominal parameter labels and explicit DERIVED_ONLY figure labels. The local v2 run passed with the counts above; canonical tables and comparisons were byte-identical to v1. A later server run showed Reacher Dynamics+Q training, prediction and shifted control parent batches PASS at 72/72 each, while nominal retention was still running. Source gating now identifies DQ through its training snapshot, requires the parent training batch to be PASS and to list that child, and requires completed held-out parents with matching seeds. Partial nominal-retention children are excluded as INCOMPLETE without invalidating the complete shifted comparison. Unconsumed live `.log` files do not invalidate the analysis. The server must rerun in a new output directory after nominal retention completes or with the updated code if an interim report is needed. See each output's completion and regression records for its actual status.

## Preserved scope

Training algorithms, experiment runners, checkpoints, snapshots, raw data and existing scientific results were not changed. No simulator rollout, training, remote process interruption, commit or push was performed. Runtime-dependent legacy aggregators remain outside this raw-ingestion task; previous migration inventories are historical records, not claims of complete runtime consolidation.

See README.md for discovery, validation, full, per-task, Q/planning and new-run commands. `full` includes regression; separate validation is optional unless checking a later source change.
