# NS-Gym analysis refactor status

**PARTIAL — real migration and retirement completed; full runtime-dependent consolidation remains deferred.**

## Actual changes

- Inspected 52 source files / 415 function records across outer and executable projects. The function-level CSV records actual paths, source fingerprints, static callers, classification and target.
- Migrated 5 independent postprocessors (89 original functions, including their entry functions) into the package. Migrated/delegated 3 Q aggregation functions, for 92 migration records total.
- Retired 7 files: 5 original active-project entry scripts and 2 byte-identical outer-project copies. No hundreds-of-files legacy archive was created; originals are recoverable from Git revision 97c6c67addaa728cb479c09dd2de1ffa105ea5d3.
- Centralized strict JSON/CSV reads, streaming SHA256, CSV writes and policy-specific bootstrap. The Q runner delegates its summary computation to q_statistics.py; native measurement/inference equations remain unchanged.
- Replaced the prototype's incompatible 20261113/2000 CI with original task policies. Preserved planning hybrid labels and method filters. This fixes prototype behavior; it does not revise the historical scientific measurements.
- One package CLI dispatches 11 operations and 5 historical input formats. Source/task schema registration is in the existing JSON analysis config. No second Hydra configuration was introduced.

## Size, without hiding added modules

For the explicitly defined scope in CODE_SIZE_COMPARISON.json, Python lines changed from **5056 to 4659 (-397)**, including new regression/tests. Python file count changed from **14 to 18** because common IO/statistics/validation are separate modules. Seven old script paths are actually gone; public postprocessing interfaces changed from eight to one. This is not a claim that every remaining runtime analysis path has been consolidated.

## Verification

- 37 analysis, migration, duplicate/seed pairing, real CPU WorldModel Q decoding, Bellman/planning, state/RNG and existing-mechanism tests passed in 19.3 seconds.
- 10 operation executions with saved inputs, 5 adapter help checks, and one config-only synthetic task registration passed. The eleventh operation, validate, was run separately on the complete registered corpus.
- 57,160 numerical field checks: **57,072 EXACT; 88 TOLERANCE; 0 FAIL**. Includes all 84 unified method/condition comparisons against original summaries. Tolerance is rtol=1e-9, atol=1e-12. These counts are not independent experiments.
- Original and migrated physical pipelines generated seven CSVs with byte-identical contents, including abrupt alignment and action-regime counts.
- All 815 recorded regression input files still match their SHA256 hashes.
- Protected runtime function bodies were checked unchanged. Q's extracted aggregation is the deliberate exception, exercised by CPU tests and saved-data regression.
- Physical simulator regression was **BLOCKED**: the available local NS-Gym checkout lacks ns_gym.schedulers. No dependency installation, training, adaptation or environment evaluation was launched.
- A combined-test import-path conflict exposed during verification was fixed by appending the analysis import path, preserving TD-MPC2 core-module resolution. The final 37-test run passed.

## Scientific artifacts

The new output contains both-task reports, canonical source-linked tables, nominal-retention comparisons, Q/planning tables and 14 PNGs. PNG-only integrated physical panels replace redundant separate physical figures and duplicate SVG copies. Prior local-v10 and all original experiment results remain unchanged. Read NSGYM_UNIFIED_RESEARCH_REPORT.md under reports/ and the regression coverage gaps before interpreting results.

HalfCheetah validation remains fixed at the experiment's 750k primary; the older nominal audit's provisional 950k selection is not substituted. Reacher nominal retention is available; HalfCheetah shifted-adapted nominal retention remains NOT_EVALUATED. Reacher D+Q is excluded until training, prediction, control, seed grids and snapshot/CSV hashes pass. Joint changed-encoder MSE is not pooled with fixed-encoder MSE. No Q calibration or planning regret was invented.

## Remaining migration gaps

1. `analyze_dynamics_q.py`, `analyze_reacher_pipeline.py`, and `analyze_existing_mechanisms.py` remain byte-identical because they are runtime/active Reacher D+Q dependencies. Their aggregation code is not fully migrated. Server process imports cannot be inspected locally.
2. `severity_analysis.py`, `plot_results.py`, nominal audit/evaluation runners retain runtime responsibilities. Their remaining statistical helpers are not all consolidated.
3. Reacher raw physical probes, all adaptation-budget stages and nominal training JSONL are not covered end-to-end by this registered local regression corpus. Saved Reacher summary evidence remains available, with its original provenance limitations.
4. New tasks with compatible export schemas can register through config; a genuinely new export schema still needs an ingestion adapter. Full numerical validation for a new task must be registered explicitly.
5. The physical test dependency must be available before claiming simulator-suite verification. No full refactor completion claim is made while these gaps remain.

## Cleanup

LOCAL_CLEANUP_MANIFEST.csv lists exact paths and classifications. No complete byte-identical directory backup was certified, so **no result deletion is proposed or performed**. UNKNOWN and IN_PROGRESS directories remain retained. Duplicate script removal is separate from scientific result cleanup.

## Reproduction

See the package README for copy-paste commands. From the executable project use:

```bash
python -m experiments.nsgym.analysis full --source-root /path/to/results/nsgym --output /new/output
python -m experiments.nsgym.analysis validate --source-root /path/to/results/nsgym --unified-output /new/output --output /new/output/regression
```

No commit, push, remote branch change, server job interruption, checkpoint modification or local result-archive deletion was performed.
