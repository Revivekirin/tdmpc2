# Minimal final-snapshot Dynamics-only TTT comparison

**Status: READY locally; server evaluation NOT_RUN.** The server's `fixed-frozen` and `fixed-decision` directories were reported as 36/36 PASS, 3,000 transitions each, with 0 and 118 optimizer updates respectively. Their actual manifests, snapshots and online CSVs have not been copied to this computer, so improvement cannot yet be concluded. The final report at `final-snapshot-v1/summary/TTT_FINAL_SNAPSHOT_COMPARISON.md` will contain measured results after server execution.

## Baseline already available

The existing `physical-severity-750k-v1/frozen` has 240 PASS fixed episodes (12 conditions ×20 held-out seeds). The existing `heldout-latent/latent_errors.csv` has 240,000 one-step fixed-data latent errors: exactly 1,000 per condition/seed, plus separate 3/10-step diagnostics. Its dataset provenance hashes the same Frozen NPZ bank. Both source manifests use checkpoint SHA256 `f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b` and evaluation seeds 43000001–43000020. Locally, the frozen model/environment/core source hashes still match. The audit rechecks all of this on the server against both TTT run manifests before baseline reuse.

Decision-wise updates are scheduled every 25 completed environment transitions after warmup of 64. The first eligible update is at transition 75. The exact sequence is 75, 100, …, 3000, totaling `(3000−75)/25+1 = 118` Adam steps. `audit` checks each of the 3,000 online step rows in all 36 Decision streams, as well as all Frozen streams; it fails if an update occurs outside this sequence or a recorded loss/gradient is nonfinite. It verifies predictor-only change, frozen auxiliary hashes, 750k checkpoint/config/MPPI, source hashes, fixed schedule, 1,000-step horizon, seed split and final snapshot fingerprint. If manifest source hashes alone cannot establish compatibility with the old Frozen collector, the audit runs one 20-seed damping 2× Frozen reference check and records it. Other contract mismatches stop the audit.

## Evaluation scope

Only `snapshot-0003000.pt` of each Decision stream is selected. Evaluation directly covers all 12 fixed conditions ×3 streams, including negative and failed outcomes. It does not run a preliminary smoke stage, intermediate budgets, or any Frozen snapshot.

Prediction reads the hashed Frozen observations/actions/next observations, computes adapted one-step latent MSE and pairs it with the *existing* Frozen one-step MSE for each of 20 held-out seeds. No simulated environment steps are required for prediction. Shifted control runs fresh adapted episodes on exactly those seeds and pairs return, forward velocity, control cost and action saturation with the existing Frozen episode CSV. The old baseline is conditional on the same seed bank. Prediction on a fixed bank and online-policy prediction error answer different questions; only fixed-bank MSE is used in the paired prediction result.

Nominal retention is a separate optional `--mode nominal` batch. It does not run during shifted control or prediction. Per-seed results include the original 750k hash, final snapshot hash, 3,000 interactions, 118 updates, measured online transition seconds, GPU update seconds and evaluation wall seconds. Online transition seconds omit model loading and snapshot IO, so they are not a full job wall clock.

At most **720,000 new shifted environment transitions** are required for the primary comparison (12×3×20×1,000), plus 720,000 fixed-bank prediction samples. Optional nominal retention adds up to 720,000 environment transitions. No Frozen rollout is rerun when source compatibility passes.

The report averages the 20 paired evaluation seeds within each independently adapted model, then summarizes over three adaptation streams. A 95% bootstrap interval is across these three streams, conditional on the fixed evaluation seed bank. The single nominal training seed remains a limitation. A decrease in latent MSE is not itself proof of control improvement; both deltas and all grid cells are reported. Failed/incomplete conditions remain visible and can be resumed without overwriting passed outputs.

## Local validation

The new evaluation CLI and dry-run require no CUDA. Local Frozen and held-out latent manifests/data were verified; 240 episode groups and 240 one-step MSE groups matched with 1,000 samples each. Bounded tests score the same hashed NPZ with a CPU fixture and execute one real MuJoCo shifted episode; fixture scores are not TD-MPC2 results. Server CUDA snapshot loading and the full 36-stream audit remain to run.

The scripts never alter existing TTT runs, checkpoint files, Frozen episodes or latent bank.
