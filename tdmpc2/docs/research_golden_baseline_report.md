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
| D+Q inputs and losses | Replay observation/action/actual next-observation hashes; encoded input and next latent; replay batch of 64; `_td_target` recomputed from **actual next observation**, independent one-step dynamics MSE and categorical cross-entropy recomputation; logits/target hashes and numerical samples | Synthetic CPU network; no real 750k nominal agent forward pass. |
| D+Q ownership | Existing test verifies separate optimizer parameter groups and gradient isolation. Golden records dynamics/Q gradient norms, optimizer state steps, component drift, frozen encoder/reward/actor, and exact target critic EMA after one update | No CUDA optimizer/kernel parity yet. |
| D+Q schedule | Existing test executes 3000 synthetic transitions and observes events 75, 100, …, 3000, **118 dynamics + 118 critic updates**, 118 target updates. New golden also captures the first update after 64-transition warmup and verifies the schedule function. | Full training run deliberately excluded. |
| Native TD-MPC2 planning | Existing independent oracle checks original vs instrumented selected action, RNG, warm start and score decomposition. New golden records encoded latent, policy-prior and Gaussian candidate hashes, predicted reward and terminal Q hashes, total score vector, elite indices, Gaussian mean/std, final candidate/action and RNG states across three iterations. | Native planner on synthetic CPU agent, not a real trained CUDA checkpoint. |
| Historical checkpoints | Actual nominal model and optimizer files, Decision, D+Q and Joint 3000-transition snapshots: CPU deserialization and temporary serialization tree parity; metadata SHA-256 and parent nominal-model hash. Tensor counts are 72, 152, 9060, 9138 and 224 respectively. | Structural compatibility only; full production loader, CUDA and simulator continuation remain unverified. |

## Actual test execution

Run from the outer repository root with the inner project as pytest arguments:

```bash
PYTHONPATH=/private/tmp/tdmpc2-golden-deps /Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m pytest tdmpc2/tests/test_golden_research_baselines.py tdmpc2/tests/test_golden_checkpoint_compatibility.py tdmpc2/experiments/nsgym/tests/test_dynamics_q.py tdmpc2/tests/test_planner_instrumentation.py tdmpc2/tests/test_stage3a.py tdmpc2/tests/test_decision_ttt.py -q
```

The first complete targeted run executed **35 passed, 2 subtests passed in 29.72 s**. After the fixture gained complete NumPy/Python/replay RNG states and numeric output samples, the same suite executed **35 passed, 2 subtests passed in 30.02 s**. Earlier setup attempts failed at collection due missing `pyvers` and Hydra/submitit in the temporary environment; those packages were installed only under `/private/tmp/tdmpc2-golden-deps`. They do not constitute test passes. The temporary environment is not the original server CUDA stack and did not combine NS-Gym and ManiSkill2 simulator installations.

The new golden/checkpoint files alone were rerun against the final test sources after normalizing the Torch version check to accept the matching CUDA build suffix: **8 passed in 3.05 s**.

The new CUDA historical test module was collected locally with **4 skipped in 0.80 s**, each explicitly requiring a compatible CUDA host. These skips do not validate its assertions. A local CPU construction of `WorldModel` from the actual 750k metadata verified that the historical nominal model and current model each expose 72 matching state-dict keys; this is a structural precheck, not a production CUDA load.

The local CPU model loaded the immutable 750k nominal `model.pt` and inferred the first one-step latent MSE from `frozen/condition-00-seed-43000001.npz`: `8.275948493974283e-05` versus the historical `heldout-latent/latent_errors.csv` value `8.275924483314157e-05`, absolute deviation `2.4010660126805305e-10`. This directly checks saved-bank tensor preparation and the historical numerical reference on CPU, but it bypasses `training_support.load_checkpoint` because that production loader requires CUDA.

The first compatible server run reported Torch `2.14.0+cu126` on an NVIDIA RTX A5000. Its six-file suite reported **26 passed, 6 skipped, 3 failed, 2 subtests passed**. The failures were: the original macOS numerical fixture differed at its first `torch.randn` input hash; the checkpoint test looked outside the server checkout for absent historical result files; and the test environment lacked `hydra-submitit-launcher`. The three-file NS-Gym simulator suite separately reported **26 passed in 5.72 s**. These are executed results, not a Phase 1A server golden PASS. The test sources were then corrected to keep separate OS/build numerical fixtures and to search both documented result layouts or accept `TD_MPC2_GOLDEN_RESULTS_ROOT`; missing historical files now produce an explicit skip. The revised local golden/checkpoint suite executed **8 passed in 3.63 s**, and the revised six-file suite executed **35 passed, 2 subtests passed in 30.79 s**. On server4, `hydra-submitit-launcher==1.2.0` was installed, the Linux CUDA-build fixture was explicitly recorded at `tests/fixtures/research_numerical_baselines_linux_x86_64_torch_2_14_0_cu126.json`, and the revised six-file suite executed **34 passed, 1 skipped, 2 subtests passed in 44.38 s**. The single skipped case was not identified by the short output; rerun with `-rs` to identify it. The Linux fixture has not yet been returned to this repository.

The first four-case server production-path run executed **3 passed, 1 failed in 16.11 s**. The nominal 750k checkpoint loaded through `training_support.load_checkpoint`, including its optimizer file and exact model key/shape/tree hash. Its immutable saved bank SHA-256 was `d473bc579f50c414b99785626dcfe7d783acab1aff836ffc780fa795685bd1d3`; the first-step latent MSE was `8.275928848888725e-05` versus historical `8.275924483314157e-05` (absolute deviation `4.3655745685100555e-11`). Native MPPI ran on the trained model and verified score decomposition, elite selection and finite selected action; the temporary model save/load preserved component hashes. Production loaders also restored the historical Decision, D+Q and Joint 3000-transition snapshots. D+Q reported 118 dynamics, 118 Q and 118 target updates; Joint reported 2001 updates. Fixed float32 CPU/CUDA comparisons passed at `rtol=5e-4`, `atol=2e-5`: maximum absolute deviations were encoded latent `1.1920928955078125e-07`, next latent `2.086162567138672e-07`, categorical logits `0`, cross-entropy `4.76837158203125e-07`. The failing case stopped at the test's incorrect target-critic hash assertion, before its remaining real-bank D+Q checks. `_target_Qs` is a wrapper with an empty `state_dict`; the actual registered EMA state is `_target_Qs_params`. The assertion now hashes that parameter container, and the revised test still requires a server rerun. Local collection of the revised module executed **4 skipped** because CUDA is unavailable; that is not validation.

The isolated server prefix `/tmp/tdmpc2-nsgym-golden-conda` used Python 3.11, Torch `2.14.0+cu126`, NS-Gym `1.0.12.post2`, Gymnasium `1.3.0`, MuJoCo `3.3.7`, NumPy `2.4.6`, and `hydra-submitit-launcher==1.2.0`, on an NVIDIA RTX A5000. These versions were reported by the server interpreter, which was kept separate from the ManiSkill2 environment.

The corrected four-case production-path module then executed **4 passed, 1 warning in 16.02 s** on server4. The real saved-bank D+Q test confirmed exactly one dynamics update, one Q update and one target EMA after 75 transitions, 64 distinct replay samples, disjoint optimizer parameter groups with one Adam step each, unchanged encoder/reward/actor hashes, and changed dynamics/Q/target-Q hashes. It independently recomputed the actual-next-observation TD target, one-step dynamics MSE, categorical critic cross-entropy and exact EMA tensor update. Its post-update measured dynamics MSE was `0.0005493831122294068`, categorical Q loss `0.46237149834632874`, and mean TD target `2383.01806640625`; update-time metrics had dynamics gradient norm `0.001860533026047051` and Q gradient norm `0.0022982617374509573`. These post-update loss values differ from the update-time losses because they use a separate replay batch and RNG seed. A warning came solely from converting a gradient-bearing tensor to a scalar for printing; the local test source now detaches it before printing and that warning-only edit has not been rerun on server4.

The six-file targeted suite reran with skip reasons and executed **34 passed, 1 skipped, 2 subtests passed in 41.95 s**. The skip is `test_existing_frozen_manifest_grid_and_seeds`: its test-only path assumed historical results at `../results`, while server4 stores them at `./results`; the four-case production-path test successfully read the same historical results there. The test now searches both layouts. Its updated full-suite result is pending; the relevant file alone executed locally with **13 passed in 25.89 s**.

## Missing coverage and blockers

1. **PASS for tested CUDA contracts — numerical parity:** server4 executed nominal saved-bank inference, trained-model MPPI, fixed CPU/CUDA forward/loss parity and real-bank D+Q forward/backward/update/EMA checks. CPU golden values remain frozen; no fixture was rewritten to force a pass. Wider transition-bank and simulator numerical sweeps remain outside these tests.
2. **PARTIAL — NS-Gym simulator:** `ns_gym` is absent locally, but the server's NS-Gym target suite executed 26 passes. Its physical tests check requested mass readback, reset restoration, full integration-state restoration, nominal versus raw Gymnasium transitions, and model-array invariants using temporary outputs. The Linux numerical fixture must still be returned to this repository; broader saved-bank and all shift-parameter parity remain open.
3. **BLOCKED — ManiSkill2/SAPIEN simulator:** `mani_skill2` and `sapien` are absent locally. Real action/physics shift and renderer checks were not executed. Existing mock SAPIEN contract tests are separate from real physics validation.
4. Nominal, Decision, D+Q and Joint historical model loading through production paths succeeded on server4. Temporary model save/load succeeded. **Exact training continuation remains unsupported** because the historical snapshots omit replay, environment and RNG state; no resumed one-step equality is claimed.
5. Fixed saved-bank inference has run for the nominal checkpoint and one transition bank. Broader bank coverage, environment seed pairing, termination/truncation and action-command versus executed-action parity are not yet numerical golden fixtures. Existing tests cover subsets.
6. Full repository regression and reporting/figure byte parity were not run. Those belong to later gates and must be freshly executed; earlier PASS documents are inventory evidence only.

## Exact compatible-host commands

From the **inner** project root (`tdmpc2/`), on the Linux CUDA server documented by `experiments/nsgym/requirements-server4-cu126.txt`, keep the NS-Gym environment separate:

```bash
python3.11 -m venv /tmp/tdmpc2-nsgym-golden
/tmp/tdmpc2-nsgym-golden/bin/python -m pip install -r experiments/nsgym/requirements-server4-cu126.txt
/tmp/tdmpc2-nsgym-golden/bin/python -c 'import torch, ns_gym; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
/tmp/tdmpc2-nsgym-golden/bin/python -m pytest tests/test_golden_research_baselines.py tests/test_golden_checkpoint_compatibility.py experiments/nsgym/tests/test_dynamics_q.py tests/test_planner_instrumentation.py tests/test_stage3a.py tests/test_decision_ttt.py -q
/tmp/tdmpc2-nsgym-golden/bin/python -m pytest experiments/nsgym/tests/test_halfcheetah.py experiments/nsgym/tests/test_physical_validation.py experiments/nsgym/tests/test_dynamics_only_ttt.py -q
```

On server4, the first `python3.11 -m venv` attempt lacked `ensurepip`, so a separate Conda environment at `/tmp/tdmpc2-nsgym-golden-conda` was successfully installed instead. Add `hydra-submitit-launcher==1.2.0` to that environment before the six-file test. The server must receive the updated test sources before recording its platform fixture. From the inner project root, after checking `git rev-parse --short HEAD` is `4659c66`:

```bash
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python -m pip install hydra-submitit-launcher==1.2.0
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python tests/test_golden_research_baselines.py --record
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python -m pytest tests/test_golden_research_baselines.py tests/test_golden_checkpoint_compatibility.py experiments/nsgym/tests/test_dynamics_q.py tests/test_planner_instrumentation.py tests/test_stage3a.py tests/test_decision_ttt.py -q
```

If historical checkpoint sources are outside either `../results` or `./results`, set `TD_MPC2_GOLDEN_RESULTS_ROOT` to the directory that directly contains `nsgym/` for the checkpoint test. A skipped checkpoint case is **BLOCKED**, not validated compatibility. Return the new Linux fixture to the repository before any refactor; do not replace the macOS CPU fixture.

After copying the new `tests/test_cuda_historical_research_validation.py` to server4, run the production-path completion checks from the inner project root using the existing compatible Conda environment, with output visible and skip reasons reported:

```bash
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python -m pytest tests/test_cuda_historical_research_validation.py -q -rs -s
conda run --no-capture-output -p /tmp/tdmpc2-nsgym-golden-conda python -m pytest tests/test_golden_research_baselines.py tests/test_golden_checkpoint_compatibility.py experiments/nsgym/tests/test_dynamics_q.py tests/test_planner_instrumentation.py tests/test_stage3a.py tests/test_decision_ttt.py -q -rs
```

The completion checks compare nominal saved-bank one-step MSE with the immutable historical `heldout-latent/latent_errors.csv` value using relative `5e-4` and absolute `2e-7`; fixed float32 CPU/CUDA encoder, dynamics, categorical-Q logits and cross-entropy use relative `5e-4` and absolute `2e-5`. These tolerances allow device arithmetic drift while keeping source hashes, update counts, component identity, elite membership, optimizer ownership and frozen-state hashes exact. The test prints measured absolute deviations. Exact training continuation is explicitly unsupported by the historical Joint/nominal snapshot formats because replay, environment and RNG state were not saved; a temporary model save/load validates serialization and evaluation restoration only.

Use the existing original-compatible ManiSkill2/SAPIEN interpreter for its separate opt-in checks, without installing NS-Gym dependencies into it:

```bash
TD_MPC2_TEST_MANISKILL=1 python -m pytest tests/test_maniskill_integration.py tests/test_shift_physics.py -q
```

The last command must be run **after activating that interpreter**; its environment and renderer choice are host-specific. Do not infer a PASS from these unexecuted commands.

## Phase 1B readiness

**Ready for CPU-only import/contract bridges** if they preserve the recorded fixture and existing 35-test targeted suite. The tested CUDA production-path contracts now pass. **Broad Phase 1B sign-off remains pending** until the Linux fixture is returned and frozen in this repository, the corrected manifest test is rerun on server4 without a skip, and remaining saved-bank/environment parity scope is explicitly accepted or covered. Exact training continuation cannot be asserted from historical snapshots lacking replay, environment and RNG state. Every later migration change must rerun the goldens against the unchanged fixture; a mismatch requires diagnosis, not a fixture update.
