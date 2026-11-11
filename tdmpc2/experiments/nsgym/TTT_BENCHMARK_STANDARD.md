# Dynamics-only TTT final-snapshot benchmark standard

Use `scripts/ttt_benchmark.py` as the single operator entry point. Its profile fixes the benchmark identity, source runs, baseline, checkpoint hash, destination, and tests. The runner resolves every path relative to `--project-root`, checks the checkpoint and completed adaptation runs, executes tests, then gates each evaluation stage on a `PASS` completion record. Without `--execute`, it prints the full plan without loading CUDA or starting physics. This standard does **not** start nominal training, adaptation, episode-wise TTT, abrupt TTT, or Joint RL.

## HalfCheetah 750k, server4

The committed `configs/halfcheetah_ttt_final_benchmark.json` is the current profile. It refers only to paths below `/home/jhkim/repos/tdmpc2`. The server's enclosing Git root may be `/home/jhkim/repos`; run from the inner project directory.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
git pull --ff-only origin main
BENCH=experiments/nsgym/scripts/ttt_benchmark.py
PROFILE=experiments/nsgym/configs/halfcheetah_ttt_final_benchmark.json
python "$BENCH" --profile "$PROFILE" --stage primary
```

The last command only prints commands and stage statuses. Before execution, check that the profile's output directory is unused or belongs to this exact benchmark. Run each stage separately to review smoke results before the long full-grid rollout:

```bash
python "$BENCH" --profile "$PROFILE" --stage test --execute
python "$BENCH" --profile "$PROFILE" --stage audit --execute
python "$BENCH" --profile "$PROFILE" --stage smoke-prediction --execute
python "$BENCH" --profile "$PROFILE" --stage smoke-shifted --execute
python "$BENCH" --profile "$PROFILE" --stage full-prediction --execute
```

After checking both smoke completion records, run the long stage with `nohup`:

```bash
OUT=/home/jhkim/repos/tdmpc2/results/nsgym/dynamics-only-750k-v1/final-snapshot-v1
nohup python -u "$BENCH" --profile "$PROFILE" --stage full-shifted --execute \
  > "$OUT/full-shifted.log" 2>&1 < /dev/null &
echo $! > "$OUT/full-shifted.pid"
tail -n 50 -f "$OUT/full-shifted.log"
```

Finish with:

```bash
python "$BENCH" --profile "$PROFILE" --stage aggregate --execute
cat "$OUT/summary/completion.json"
```

`--stage primary --execute` runs all seven stages in sequence and stops on the first non-PASS result. An existing PASS stage is skipped only after its checkpoint and audit provenance match the profile. A PARTIAL/interrupted prediction or shifted batch invokes the evaluator's `--resume`: completed identities remain unchanged, incomplete identities are archived by that evaluator and retried. An incomplete audit/aggregate requires a new profile output suffix; the runner does not erase or overwrite it. Each evaluation stage re-audits the source runs and inputs. The profile's `output` must be unique for a new benchmark.

## Measured scope and interpretation

The HalfCheetah smoke is the preregistered 2× damping and 2× torso-mass pair, three independent adaptation streams each. It evaluates 120,000 fixed-bank prediction samples and 120,000 new shifted environment transitions. The full grid reuses those six smoke identities and adds 600,000 prediction samples and 600,000 shifted transitions. No Frozen or intermediate adaptation snapshot is reevaluated when the reference audit passes. Optional nominal retention is **not** part of `primary`; run it separately only if needed:

```bash
python "$BENCH" --profile "$PROFILE" --stage nominal --execute
python "$BENCH" --profile "$PROFILE" --stage aggregate-retention --execute
```

The evaluator records per-seed Frozen/Decision prediction error and shifted return, velocity, control cost, action and actuator-force metrics; final model and checkpoint hashes; online interactions, update count, and wall times. A condition's 20 paired evaluation seeds are averaged within each adapted stream, and uncertainty is across three independent adaptation streams. Missing/failed conditions remain visible. Lower fixed-bank MSE alone does not establish better closed-loop return. The 20 evaluation seeds were already used in the Frozen preview and are held out from adaptation, but are not a newly untouched test set.

## Contract for the next task or benchmark

The **operator runner** is task-independent. Create a new JSON profile with the same fields and an unused output directory. Paths, `reference_files`, `run_completion_stages`, and test paths must be relative to its project root. Its task-specific evaluator must expose the same CLI contract: `audit`, `evaluate --mode prediction|shifted|nominal --subset smoke|full`, and `aggregate`; accept `--checkpoint`, `--runs`, `--reference`, `--output`, `--audit`; accept `--smoke-prediction` and `--smoke-shifted` for the full stage; and implement `--resume` for incomplete evaluate batches. Each successful stage must write `completion.json` with `status: PASS`, and the audit must write `audit.json`. The runner will not treat a PARTIAL stage as a successful prerequisite.

Before registering a new task, its evaluator must specify and test: task observation/action dimensions and simulator, nominal checkpoint fingerprint, immutable base weights, adaptation-only parameter scope, update schedule, condition grid and smoke subset, seed splits, horizon, replay/data provenance, paired baseline compatibility, raw per-seed outputs, and the unit of statistical replication. These are currently HalfCheetah-specific in `evaluate_ttt_final.py` and `run_dynamics_only_ttt.py`; simply changing the profile to Reacher would be invalid. Implement a Reacher evaluator against this CLI contract and run its bounded tests before adding its profile. The existing `REACHER_DYNAMICS_ONLY_TTT_PROTOCOL.md` specifies Reacher's distinct 50-step horizon, condition grid, seeds, and budget feasibility gate.

Keep source adaptation runs, Frozen data, and checkpoint read-only. A profile is a preregistration of inputs and outputs, not a performance-based selector.
