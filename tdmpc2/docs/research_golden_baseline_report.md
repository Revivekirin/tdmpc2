# TD-MPC2 dynamics-shift Phase 1A golden baseline report

Audit basis: repository HEAD `4659c66`, upstream reference `332f19e`, migration plan `docs/research_refactor_migration_plan.md`. This phase adds tests and immutable baseline records only. Adaptation and original TD-MPC2 source were not changed. Historical experiment outputs were read only; nominal training and expensive evaluation were not run.

## New evidence and provenance

| File | Purpose | Source and capture |
|---|---|---|
| `tests/test_golden_research_baselines.py` | Deterministic CPU observation and golden comparison for dynamics-only, D+Q, and native planning | Current production `EpisodeAdapter`, `DynamicsQAdapter`, `TDMPC2.act`, and the existing pre-instrumentation planner oracle. Seed `20261119`; independent transition generator seeds `9001` and `9002`; D+Q adaptation seed `42000001`. |
| `tests/fixtures/research_numerical_baselines.json` | Recorded numerical outputs, input/source hashes, complete Torch/NumPy/Python RNG states, replay RNG states, losses, gradients, parameter drift, optimizer summaries, target-Q hashes, and planner trace | Captured by executing the test recorder on unchanged source with Python 3.11.16, Torch 2.14.0 CPU, NumPy 2.4.6. The fixture includes SHA-256 of the five imported research/core source files. Compare exact discrete outputs and hashes, floating values at relative `2e-6`, absolute `2e-7`. |
| `tests/test_golden_checkpoint_compatibility.py` | Historical file SHA-256, CPU load, state-tree digest, temporary save/reload, metadata and parent-hash checks | Five existing checkpoint files under outer `results/nsgym/`; source files are reopened and rehashed after the temporary round-trip. |
| `tests/fixtures/research_checkpoint_compatibility.json` | Source path, byte size, original SHA-256, top-level keys, state-tree SHA-256, tensor counts and executed round-trip outcome | Captured from actual local files with Torch 2.14.0 CPU. It is not a claim about full GPU agent restoration. |
| `tests/test_cuda_historical_research_validation.py` | Server-only production loader, historical bank, native planner, D+Q GPU update and fixed-tensor CPU/CUDA parity checks | After correcting a test-only target-critic hash assertion, server4 executed all four CUDA tests successfully. Historical sources were read only and the model round-trip used a temporary path. |

The numerical recorder refuses to replace a pre-existing fixture. Ordinary pytest runs read the fixture and never update it. Because Torch CPU random draws differed between the macOS CPU and Linux CUDA builds despite matching Torch base version and seed, the recorder now selects a descriptive platform/build-specific file. A new platform fixture can be recorded only while the audited source hashes equal those in the original CPU fixture. Historical experiment outputs are not destinations for any new test. Checkpoint round-trips use `TemporaryDirectory`.

## Contract coverage now frozen

| Contract | Executed evidence | Limit |
|---|---|---|
| Dynamics-only | Four fixed observations/actions/next observations, one-step latent MSE, predictor gradient/update norm, 4-transition warmup, one update, output tensor, fixed-parameter audit, optimizer step and replay RNG | Synthetic CPU agent; no simulator rollout. |
| D+Q inputs and losses | Replay observation/action/actual next-observation hashes; encoded input and next latent; replay batch of 64; `_td_target` recomputed from **actual next observation**, independent one-step dynamics MSE and categorical cross-entropy recomputation; logits/target hashes and numerical samples | CPU synthetic fixture plus server4 real 750k nominal agent and saved-bank forward/update checks. |
| D+Q ownership | Existing test verifies separate optimizer parameter groups and gradient isolation. Golden records dynamics/Q gradient norms, optimizer state steps, component drift, frozen encoder/reward/actor, and exact target critic EMA after one update | Server4 real-bank CUDA update verifies disjoint optimizer groups, one step each, target EMA and frozen components. |
| D+Q schedule | Existing test executes 3000 synthetic transitions and observes events 75, 100, …, 3000, **118 dynamics + 118 critic updates**, 118 target updates. New golden also captures the first update after 64-transition warmup and verifies the schedule function. | Full training run deliberately excluded. |
| Native TD-MPC2 planning | Existing independent oracle checks original vs instrumented selected action, RNG, warm start and score decomposition. New golden records encoded latent, policy-prior and Gaussian candidate hashes, predicted reward and terminal Q hashes, total score vector, elite indices, Gaussian mean/std, final candidate/action and RNG states across three iterations. | CPU golden oracle and trained nominal CUDA planner score/elite/action checks executed. |
| Historical checkpoints | Actual nominal model and optimizer files, Decision, D+Q and Joint 3000-transition snapshots: CPU deserialization and temporary serialization tree parity; metadata SHA-256 and parent nominal-model hash. Tensor counts are 72, 152, 9060, 9138 and 224 respectively. | CPU structural and temporary round-trip plus server4 production CUDA loader for nominal, Decision, D+Q and Joint. Exact training continuation unsupported by saved format. |

## Actual test execution

Run from the outer repository root with the inner project as pytest arguments:

```bash
PYTHONPATH=/private/tmp/tdmpc2-golden-deps /Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m pytest tdmpc2/tests/test_golden_research_baselines.py tdmpc2/tests/test_golden_checkpoint_compatibility.py tdmpc2/experiments/nsgym/tests/test_dynamics_q.py tdmpc2/tests/test_planner_instrumentation.py tdmpc2/tests/test_stage3a.py tdmpc2/tests/test_decision_ttt.py -q
```

The first complete targeted run executed **35 passed, 2 subtests passed in 29.72 s**. After the fixture gained complete NumPy/Python/replay RNG states and numeric output samples, the same suite executed **35 passed, 2 subtests passed in 30.02 s**. Earlier setup attempts failed at collection due missing `pyvers` and Hydra/submitit in the temporary environment; those packages were installed only under `/private/tmp/tdmpc2-golden-deps`. They do not constitute test passes. The temporary environment is not the original server CUDA stack and did not combine NS-Gym and ManiSkill2 simulator installations.

The new golden/checkpoint files alone were rerun against the final test sources after normalizing the Torch version check to accept the matching CUDA build suffix: **8 passed in 3.05 s**.

The new CUDA historical test module was collected locally with **4 skipped in 0.80 s**, each explicitly requiring a compatible CUDA host. These skips do not validate its assertions. A local CPU construction of `WorldModel` from the actual 750k metadata verified that the historical nominal model and current model each expose 72 matching state-dict keys; this is a structural precheck, not a production CUDA load.

The local CPU model loaded the immutable 750k nominal `model.pt` and inferred the first one-step latent MSE from `frozen/condition-00-seed-43000001.npz`: `8.275948493974283e-05` versus the historical `heldout-latent/latent_errors.csv` value `8.275924483314157e-05`, absolute deviation `2.4010660126805305e-10`. This directly checks saved-bank tensor preparation and the historical numerical reference on CPU, but it bypasses `training_support.load_checkpoint` because that production loader requires CUDA.

The first compatible server run reported Torch `2.14.0+cu126` on an NVIDIA RTX A5000. Its six-file suite reported **26 passed, 6 skipped, 3 failed, 2 subtests passed**. The failures were: the original macOS numerical fixture differed at its first `torch.randn` input hash; the checkpoint test looked outside the server checkout for absent historical result files; and the test environment lacked `hydra-submitit-launcher`. The three-file NS-Gym simulator suite separately reported **26 passed in 5.72 s**. These are executed results, not a Phase 1A server golden PASS. The test sources were then corrected to keep separate OS/build numerical fixtures and to search both documented result layouts or accept `TD_MPC2_GOLDEN_RESULTS_ROOT`; missing historical files now produce an explicit skip. The revised local golden/checkpoint suite executed **8 passed in 3.63 s**, and the revised six-file suite executed **35 passed, 2 subtests passed in 30.79 s**. On server4, `hydra-submitit-launcher==1.2.0` was installed, the Linux CUDA-build fixture was explicitly recorded at `tests/fixtures/research_numerical_baselines_linux_x86_64_torch_2_14_0_cu126.json`, and the revised six-file suite executed **34 passed, 1 skipped, 2 subtests passed in 44.38 s**. The skip was later identified and resolved; the Linux fixture was subsequently committed to this repository. This paragraph records the earlier execution state.

The first four-case server production-path run executed **3 passed, 1 failed in 16.11 s**. The nominal 750k checkpoint loaded through `training_support.load_checkpoint`, including its optimizer file and exact model key/shape/tree hash. Its immutable saved bank SHA-256 was `d473bc579f50c414b99785626dcfe7d783acab1aff836ffc780fa795685bd1d3`; the first-step latent MSE was `8.275928848888725e-05` versus historical `8.275924483314157e-05` (absolute deviation `4.3655745685100555e-11`). Native MPPI ran on the trained model and verified score decomposition, elite selection and finite selected action; the temporary model save/load preserved component hashes. Production loaders also restored the historical Decision, D+Q and Joint 3000-transition snapshots. D+Q reported 118 dynamics, 118 Q and 118 target updates; Joint reported 2001 updates. Fixed float32 CPU/CUDA comparisons passed at `rtol=5e-4`, `atol=2e-5`: maximum absolute deviations were encoded latent `1.1920928955078125e-07`, next latent `2.086162567138672e-07`, categorical logits `0`, cross-entropy `4.76837158203125e-07`. The failing case stopped at the test's incorrect target-critic hash assertion, before its remaining real-bank D+Q checks. `_target_Qs` is a wrapper with an empty `state_dict`; the actual registered EMA state is `_target_Qs_params`. The assertion was corrected to hash that parameter container. The subsequent server run passed; local CUDA skips are never counted as validation.

The isolated server prefix `/tmp/tdmpc2-nsgym-golden-conda` used Python 3.11, Torch `2.14.0+cu126`, NS-Gym `1.0.12.post2`, Gymnasium `1.3.0`, MuJoCo `3.3.7`, NumPy `2.4.6`, and `hydra-submitit-launcher==1.2.0`, on an NVIDIA RTX A5000. These versions were reported by the server interpreter, which was kept separate from the ManiSkill2 environment.

The corrected four-case production-path module then executed **4 passed, 1 warning in 16.02 s** on server4. The real saved-bank D+Q test confirmed exactly one dynamics update, one Q update and one target EMA after 75 transitions, 64 distinct replay samples, disjoint optimizer parameter groups with one Adam step each, unchanged encoder/reward/actor hashes, and changed dynamics/Q/target-Q hashes. It independently recomputed the actual-next-observation TD target, one-step dynamics MSE, categorical critic cross-entropy and exact EMA tensor update. Its post-update measured dynamics MSE was `0.0005493831122294068`, categorical Q loss `0.46237149834632874`, and mean TD target `2383.01806640625`; update-time metrics had dynamics gradient norm `0.001860533026047051` and Q gradient norm `0.0022982617374509573`. These post-update loss values differ from the update-time losses because they use a separate replay batch and RNG seed. The earlier warning came solely from converting a gradient-bearing tensor to a scalar for printing; the final server run used the corrected print expression and emitted no warning.

The six-file targeted suite reran with skip reasons and executed **34 passed, 1 skipped, 2 subtests passed in 41.95 s**. The skip is `test_existing_frozen_manifest_grid_and_seeds`: its test-only path assumed historical results at `../results`, while server4 stores them at `./results`; the four-case production-path test successfully read the same historical results there. The test now searches both layouts. The updated full suite subsequently passed with **35 passed, 2 subtests passed, 0 skipped in 42.67 s**; the relevant file alone executed locally with **13 passed in 25.89 s**.

## Missing coverage and blockers

1. **PASS for tested CUDA contracts — numerical parity:** server4 executed nominal saved-bank inference, trained-model MPPI, fixed CPU/CUDA forward/loss parity and real-bank D+Q forward/backward/update/EMA checks. CPU golden values remain frozen; no fixture was rewritten to force a pass. Wider transition-bank and simulator numerical sweeps remain outside these tests.
2. **PARTIAL — NS-Gym simulator:** `ns_gym` is absent locally, but the server's NS-Gym target suite executed 26 passes. Its physical tests check requested mass readback, reset restoration, full integration-state restoration, nominal versus raw Gymnasium transitions, and model-array invariants using temporary outputs. The Linux numerical fixture is committed to this repository; all 240 banks have first-step inference parity. Full-episode and all shift-parameter parity remain open.
3. **BLOCKED — ManiSkill2/SAPIEN simulator:** `mani_skill2` and `sapien` are absent locally. Real action/physics shift and renderer checks were not executed. Existing mock SAPIEN contract tests are separate from real physics validation.
4. Nominal, Decision, D+Q and Joint historical model loading through production paths succeeded on server4. Temporary model save/load succeeded. **Exact training continuation remains unsupported** because the historical snapshots omit replay, environment and RNG state; no resumed one-step equality is claimed.
5. Fixed first-step saved-bank inference has run for all 240 historical transition banks. Full-episode bank inference, environment seed pairing, termination/truncation and action-command versus executed-action parity are not separate numerical golden fixtures. Existing simulator tests cover subsets.
6. Full repository regression and reporting/figure byte parity were not run. Those belong to later gates and must be freshly executed; earlier PASS documents are inventory evidence only.

## Final compatible-host execution and provenance

The final six-file Phase 1A targeted suite ran on server4 with **35 passed, 2 subtests passed, 0 failed, 0 skipped in 42.67 s**:

```bash
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python -m pytest tests/test_golden_research_baselines.py tests/test_golden_checkpoint_compatibility.py experiments/nsgym/tests/test_dynamics_q.py tests/test_planner_instrumentation.py tests/test_stage3a.py tests/test_decision_ttt.py -q -rs
```

The separate NS-Gym physical suite ran with **26 passed in 5.72 s**:

```bash
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python -m pytest experiments/nsgym/tests/test_halfcheetah.py experiments/nsgym/tests/test_physical_validation.py experiments/nsgym/tests/test_dynamics_only_ttt.py -q
```

The production-path CUDA module first ran four cases with **4 passed, 1 test-output warning in 16.02 s**; the test-output conversion was then corrected. Its fifth case separately ran with **1 passed in 7.34 s**, checking all 240 historical saved banks against their SHA-256 and first-step historical latent MSE. The final five-case module then executed together on the final test source with **5 passed, 0 failed, 0 skipped, 0 warnings in 19.23 s**:

```bash
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python -m pytest tests/test_cuda_historical_research_validation.py -q -rs -s
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python -m pytest tests/test_cuda_historical_research_validation.py::test_cuda_all_historical_transition_banks_first_step_inference -q -rs -s
```

The final five-case run repeated the 240-bank comparison and produced maximum absolute deviation `3.14321368932724e-09`, mean absolute deviation `2.863998815882951e-10`, and maximum relative deviation `8.714321871805354e-06`, below the declared `rtol=5e-4`, `atol=2e-7`. The committed Linux fixture `tests/fixtures/research_numerical_baselines_linux_x86_64_torch_2_14_0_cu126.json` has SHA-256 `f7d60dfdcb526e29955ec5e428cf7b54377ca3f314aeaca65edf09e7f1da71d9`, baseline head `4659c66`, and the same five source SHA-256 entries as the macOS CPU fixture. It was added without changing either existing fixture or historical results.

Exact training continuation is **UNSUPPORTED** by the historical snapshot format because replay, environment and RNG states are absent. The temporary nominal model save/load proves model serialization and evaluation restoration, not a resumed training step. ManiSkill2/SAPIEN simulator validation remains **BLOCKED** on this host and must use its separate compatible interpreter; no result for that simulator is claimed here.

## Phase 1B readiness

**Phase 1A CUDA and NS-Gym baseline validation is complete for the specified compatible-host checks. Phase 1B may begin** with frozen macOS and Linux fixtures and the executed targeted, physical, production-loader, real-bank D+Q, MPPI, and 240-bank results as gates. The historical format still cannot prove exact training continuation, and ManiSkill2/SAPIEN requires its separate compatible host. Every later migration change must rerun the goldens against unchanged fixtures; a mismatch requires diagnosis, not a fixture update.

Local read-only fixture inspection confirmed the Linux and macOS baseline files have identical five source hashes and baseline head `4659c66`. Final local `git diff --check` reported no whitespace errors.
