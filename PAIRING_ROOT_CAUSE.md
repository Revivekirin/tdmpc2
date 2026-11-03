# Pre-shift divergence: current evidence and reproduction

## Status

The cause of the archived abrupt delay-2 pre-shift action divergence is **not established**. The original 50-pair Stage 2 run logged a first pre-shift action difference in 34/50 PickCube-1 pairs: 7 at native step 4, 25 at step 6, and 2 at step 8. Its initial exposed observation, simulator state, controller state, and environment RNG matched in all 50 pairs, and its first MPC action matched in every pair. It did not record the intervening native observations or per-decision policy RNG and candidate arrays. Therefore the first physical or planner divergence cannot be recovered from that archive. The derived [per-seed evidence](results/pairing-root-cause-existing.json) retains `first_divergence: UNVERIFIED` for every archived seed.

Independent server reruns of seed 7, both fresh and after complete warmup episodes 1–6, matched through all ten pre-shift native steps on the exposed observations, controller/simulator states, commanded/executed actions, and environment/policy RNG. The two nominal MPC runs matched all five selected actions and first-iteration candidate hashes. Those runs do not reproduce the archived seed-7 difference. They also do not prove hidden SAPIEN/PhysX state equality. See [fresh seed 7](results/pairing-validation-seed7.json), [warmup seed 7](results/pairing-validation-seed7-warmup.json), and [archive audit](results/pairing-validation-archive.json).

## Code findings

- `tdmpc2/evaluate_shift.py` previously omitted `eval_mode` in `agent.act`, which means `False` under `TDMPC2.act`. The new optional `evaluation.policy_eval_mode` passes the choice explicitly; its absent-key default remains `False`. The Hydra evaluation YAML files are unchanged, so the original benchmark configuration and old artifacts stay intact. Use `+evaluation.policy_eval_mode=true` only in a diagnostic run.
- `tdmpc2/tdmpc2.py` uses stochastic policy priors, Gaussian MPPI candidates, a sampled terminal policy action, randomly chosen Q heads, Gumbel elite selection, and, at `eval_mode=False`, extra final action noise. `eval_mode=True` removes only that last noise in MPC. It does **not** make MPC deterministic. The planner warm-start `_prev_mean` is bypassed by the first decision's `t0=True` path and updated after each plan; the diagnostic now compares its exposed tensor before and after every target decision.
- `tdmpc2/common/world_model.py` samples with `torch.randn_like` in `pi` and `torch.randperm` for Q-head selection. `tdmpc2/common/seed.py` seeds Python, NumPy, Torch CPU, and all CUDA devices. Stage 2 analysis wraps its extra model calls in RNG preservation, but this does not certify CUDA kernel determinism.
- `tdmpc2/envs/wrappers/dynamics_shift/schedule.py`'s abrupt value has no per-step random draw. `action_shift.py` retains a queue sized for the *possible future* delay. Thus nominal and abrupt queue contents differ before onset while the delay-zero executed command is the current command in both. This is a confirmed structural difference, not an established cause of behavioral divergence.
- `tdmpc2/envs/maniskill.py` repeats each policy action for two native steps. At switch step 10, the first shifted native transition is the first repeat of policy decision 5. The local wrapper calls legacy Gym's ManiSkill2 reset/step; its solver internals are outside this repository. `tdmpc2/common/shift_evaluation.py` is a legacy log helper, not the Stage 2 benchmark's active runner; `evaluate_shift.py` uses `BenchmarkLog`.

## New diagnostic

`scripts/validate_pairing.py` now records separate hashes of Python, NumPy, Torch CPU and each CUDA RNG before and after each decision; planner warm-start state; the encoder output; every MPPI iteration's candidates, predicted reward, terminal Q, total score, and elite IDs; selected candidate and action; and native-step commanded/executed actions, delay queue, observation, exposed simulator/controller state and environment RNG. The output's `first_divergence` compares these in execution order, with exact and `--atol` thresholds. `first_divergence_context` saves neighboring native records and compact per-iteration hashes. Queue-only differences are tagged structural. The original `compare_prefix` and pairing validity fields remain, with full simulator state `UNVERIFIED`.

The validator runs nominal A/A closed-loop, nominal open-loop replay, native wrapper disabled/enabled, nominal versus abrupt, and analysis on/off. `scripts/validate_pairing_matrix.py` starts a new process for every seed, `eval_mode` value, fresh/sequential context and repeat, then writes one JSON per seed/context/mode/repeat plus `index.json`. Sequential context completes seeds 1 through `seed-1` in the same agent/environment before the target. These are diagnostic runs, not substitutes for the 50-episode benchmark. A run that ends before native step 10 is marked by `reached_switch` and cannot certify the full prefix.

On the CUDA server, after synchronizing this code, run from the repository root:

```bash
python scripts/validate_pairing_matrix.py --seeds 2,3,7 --repeats 2 \
  --out-dir results/pairing-root-cause \
  task=pick-cube \
  checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt \
  episodic=false compile=false model_size=5 \
  dynamics_shift=action_delay schedule=abrupt \
  dynamics_shift.parameter.shifted=2 \
  evaluation=paired evaluation.episodes=1 \
  seed=1 mpc=true analysis=frozen_mechanism
```

For an additional CUDA determinism diagnostic, add `--deterministic-algorithms` and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before starting Python**. The diagnostic enables `torch.use_deterministic_algorithms(True)` and cuDNN deterministic mode and deliberately propagates unsupported-operation errors. The ordinary benchmark never enables these options. The matrix's `eval_mode=true` cases remain stochastic inside MPC by design.

Interpret the first substantive mismatch in this order: initial exposed state/RNG; RNG before action; encoder; MPPI candidates; predicted reward/Q/total score; elites; selected action; executed native command; native transition. A candidate mismatch after equal RNG hashes suggests nondeterministic execution or untracked state, but does not alone establish which. A native observation mismatch after equal executed commands suggests simulator/controller or hidden-state divergence; it is not proof of a particular solver defect. The raw first numeric difference, tolerance, and neighboring records are in each JSON. Any first exact difference within tolerance should be kept separate from a material action difference.

## Verification and remaining work

The preexisting 50-episode results were not rerun or overwritten. The [regression record](results/pairing-root-cause-regression.json) confirms that the historical archive contains 11 complete, unchanged-checkpoint runs of 50 episodes each; it labels these as historical rather than post-change validation. Nine local synthetic pairing tests and six shift-wrapper tests pass, including first-boundary classification, tolerance handling and queue-only divergence. Local `test_benchmark_v2.py` cannot import `omegaconf`, and this machine lacks the server's CUDA/ManiSkill2/checkpoint runtime, so the new checkpoint-backed matrix and frozen benchmark regression still require execution on the server. The latest *server* source revision and its package/CUDA versions have not been compared with this local checkout; inspect the new run metadata before attributing a result to the code reviewed here. If the matrix reproduces an action difference, retain the failing per-seed JSON and compare its first substantive boundary before considering a behavioral fix. No model, checkpoint, MPPI update, or existing artifact was altered.
