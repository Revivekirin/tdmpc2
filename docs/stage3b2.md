# Stage 3B-2: Dynamics-only TTT severity matrix

Run these commands in the updated server repository, in `tdmpc2-ms2`. Remote execution remains manual. The local machine cannot certify server artifacts or run the CUDA simulator.

```bash
cd /nfs4/jhkim/repos/tdmpc2
python scripts/run_stage3b2.py \
  --checkpoint checkpoints/maniskill2/pick-cube-1.pt
```

The default is an audit: it loads the actual checkpoint/datasets/snapshots on CPU, prints all eight classifications and exact subprocess commands, and writes a new manifest/report. It does not launch simulation. It may take time to load large replay checkpoints.

| Shift | Severities | Seed set |
|---|---|---|
| action_delay | 1, 2 native steps | 0 |
| action_gain | 0.7, 0.3 | 0 |
| object_mass | 1.5, 3.0 | 0 |
| contact_friction | 0.5, 0.3 | 0 |

Execute with the same environment and the GPU allocation already assigned to you:

```bash
python scripts/run_stage3b2.py \
  --checkpoint checkpoints/maniskill2/pick-cube-1.pt \
  --execute
```

Exactly one allocated CUDA GPU must be visible. If your shell exposes multiple GPUs, set `CUDA_VISIBLE_DEVICES` to your assigned device before executing; the launcher never selects a free device automatically. It refuses active compute processes on that GPU and existing Stage 3 adaptation processes in the repository. A root-level advisory lock prevents concurrent Stage 3B-2 launchers. Legacy launchers do not honor this lock; do not start another job during execution. Nothing is killed or reassigned.

## Selection and preservation

- `REUSE_COMPLETE`: same resolved experiment controls, nominal checkpoint hash, held-out/fixed seeds, dataset identity/hash, optimizer, complete milestone snapshots and final artifacts. No collection or adaptation command is issued.
- `RESUME_PARTIAL`: resume the latest intact snapshot with a contiguous chain of retained milestone artifacts. Predictor, optimizer, replay, simulator/controller state, planner, RNG and pending action repeat use the existing runner's restoration. The continuation is written in a **new version directory**. Do not delete the original directory: earlier milestones remain there and the new metadata records the ancestor path.
- `RUN_MISSING`: collect a severity-specific fixed dataset if none is compatible, then start from the nominal checkpoint.
- `INCOMPATIBLE`: retain the existing candidate and record the reason; create a fresh nominal-initialized run in an unused/versioned destination. Incompatible checkpoints are never loaded for adaptation.

Severity is read from resolved metadata, not the last directory component. Old runs ending in `.../dynamics_only/1` may represent gain 0.7, mass 1.5 or friction 0.5. Compatible old results are reused at their existing paths; they are not moved.

New runs use `runs/{shift}/set-0/dynamics_only/{severity}`. If occupied, or for a continuation, the destination is `.../{severity}/versions/{unique-run-id}`. All creation is exclusive. Default invocation IDs are unique; a manual `--run-id` must be new on each invocation. Re-running the command after interruption audits again, skips completed jobs and resumes compatible partials. A final 12800 snapshot without a completion marker can be finalized without additional adaptation, provided earlier milestones are retained. If earlier snapshots are missing, the launcher falls back to the latest complete prefix; with no prefix it starts a new version from the nominal checkpoint.

An exact resume preserves the source metadata's checkpoint and fixed-data paths, as required by the existing runner's identity checks. Relocating those files is outside this workflow. Missing or changed fixed data makes a candidate incompatible; old files are never regenerated in place. New fixed datasets are Frozen nominal-checkpoint rollouts **in the requested shifted environment**, used only for prediction evaluation, not replay. New collection artifacts live in `data/stage3b2/{shift}/{severity}/{run-id}.pt`.

## Protocol and outputs

The adaptation method is unchanged: episode-wise DynamicsOnlyAdapter, one-step latent consistency MSE, Adam at 3e-5, batch/warmup 16, 50 updates per eligible completed episode, persistent shifted-only replay/predictor/optimizer. All other model modules stay frozen. Evaluation uses held-out seeds 4000–4009 and fixed-data seeds 5000–5009. Milestones are native steps 0, 800, 1600, 3200, 6400, 12800. The existing pending-repeat boundary semantics are retained; native steps are not optimizer updates.

Reports and logs are saved to:

```text
results/stage3b-generalization/ttt_severity/{invocation-id}/
  manifest.json
  {shift}-{severity}-{command-index}.log
  report/
    STAGE3B2_RESULTS.md
    completion.json / completion.csv
    baseline_comparison.csv
    frozen_references.json
    figures.json
    stage3b_*.csv
    {shift}-{severity}/tables/...
    {shift}-{severity}/figures/...
    shifted_episode_length.png / .pdf
    shifted_return.png / .pdf
```

The manifest records exact CLI arguments, requested controls, candidate audit reasons, source revision/hash, resume provenance and execution outcome. Adaptation stdout/stderr go into per-command logs. Use `tail -f` on the log path printed by the launcher to follow milestones. On failure, keep the log and rerun after correcting the cause; no checkpoints need deleting.

The report invokes `analyze_stage3b.py` separately per severity, then combines its CSVs. AUC is the trapezoidal success integral divided by 12800, using actual native-step intervals; incomplete curves have no AUC. All eight conditions remain in the completion/comparison tables, including missing conditions and Frozen success 100% conditions. Fixed MSE in the comparison table is the equal-episode mean; transition-weighted values remain in raw evaluations.

Completed Frozen severity sweeps are inspected in place, including trajectory artifact integrity. Matching seed/checkpoint/environment/model/MPC controls permit a final-success difference. Incompatible or incomplete references remain separate in `frozen_references.json`; the corresponding difference is blank. There is no new 12800-step Frozen rollout and no invented Frozen adaptation AUC. Existing Frozen Stage 3B artifacts remain untouched.

To regenerate a report from an existing manifest, choose a new output directory:

```bash
python scripts/analyze_stage3b2.py \
  --manifest results/stage3b-generalization/ttt_severity/INVOCATION_ID/manifest.json \
  --output results/stage3b-generalization/ttt_severity/INVOCATION_ID/report-v2
```

This re-renders the recorded selection, not a fresh audit. Run `run_stage3b2.py` without `--execute` for a fresh audit after more jobs complete.

Set 0 alone does not quantify variation across independent seed sets. Legacy metadata lacks some historical source/dependency hashes; compatibility checks cannot prove binary-environment equivalence. Simulator resume feasibility on the server still depends on the existing state-restoration implementation. CPU tests cover artifact selection, corruption, continuation routing, preservation, report handling and the existing runner's synthetic boundary semantics; they are not experimental performance measurements.
