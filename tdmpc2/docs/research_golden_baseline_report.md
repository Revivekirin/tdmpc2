# TD-MPC2 dynamics-shift Phase 1A golden baseline report

Audit basis: repository HEAD `4659c66`, upstream reference `332f19e`, migration plan `docs/research_refactor_migration_plan.md`. This phase adds tests and immutable baseline records only. Adaptation and original TD-MPC2 source were not changed. Historical experiment outputs were read only; nominal training and expensive evaluation were not run.

## New evidence and provenance

| File | Purpose | Source and capture |
|---|---|---|
| `tests/test_golden_research_baselines.py` | Deterministic CPU observation and golden comparison for dynamics-only, D+Q, and native planning | Current production `EpisodeAdapter`, `DynamicsQAdapter`, `TDMPC2.act`, and the existing pre-instrumentation planner oracle. Seed `20261119`; independent transition generator seeds `9001` and `9002`; D+Q adaptation seed `42000001`. |
| `tests/fixtures/research_numerical_baselines.json` | Recorded numerical outputs, input/source hashes, complete Torch/NumPy/Python RNG states, replay RNG states, losses, gradients, parameter drift, optimizer summaries, target-Q hashes, and planner trace | Captured by executing the test recorder on unchanged source with Python 3.11.16, Torch 2.14.0 CPU, NumPy 2.4.6. The fixture includes SHA-256 of the five imported research/core source files. Compare exact discrete outputs and hashes, floating values at relative `2e-6`, absolute `2e-7`. |
| `tests/test_golden_checkpoint_compatibility.py` | Historical file SHA-256, CPU load, state-tree digest, temporary save/reload, metadata and parent-hash checks | Five existing checkpoint files under outer `results/nsgym/`; source files are reopened and rehashed after the temporary round-trip. |
| `tests/fixtures/research_checkpoint_compatibility.json` | Source path, byte size, original SHA-256, top-level keys, state-tree SHA-256, tensor counts and executed round-trip outcome | Captured from actual local files with Torch 2.14.0 CPU. It is not a claim about full GPU agent restoration. |

The numerical recorder refuses to replace a pre-existing fixture. Ordinary pytest runs read the fixture and never update it. Because Torch CPU random draws differed between the macOS CPU and Linux CUDA builds despite matching Torch base version and seed, the recorder now selects a descriptive platform/build-specific file. A new platform fixture can be recorded only while the audited source hashes equal those in the original CPU fixture. Historical experiment outputs are not destinations for any new test. Checkpoint round-trips use `TemporaryDirectory`.

## Contract coverage now frozen

| Contract | Executed evidence | Limit |
|---|---|---|
| Dynamics-only | Four fixed observations/actions/next tency MSE, predictor gradobservations, 4-transition warmup, one update, consisient/update norm, output tensor, fixed-parameter audit, optimizer step and replay RNG | Synthetic CPU agent; no simulator rollout. |
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

The first compatible server run reported Torch `2.14.0+cu126` on an NVIDIA RTX A5000. Its six-file suite reported **26 passed, 6 skipped, 3 failed, 2 subtests passed**. The failures were: the original macOS numerical fixture differed at its first `torch.randn` input hash; the checkpoint test looked outside the server checkout for absent historical result files; and the test environment lacked `hydra-submitit-launcher`. The three-file NS-Gym simulator suite separately reported **26 passed in 5.72 s**. These are executed results, not a Phase 1A server golden PASS. The test sources were then corrected to keep separate OS/build numerical fixtures and to search both documented result layouts or accept `TD_MPC2_GOLDEN_RESULTS_ROOT`; missing historical files now produce an explicit skip. The revised local golden/checkpoint suite executed **8 passed in 3.63 s**, and the revised six-file suite executed **35 passed, 2 subtests passed in 30.79 s**. The revised tests and a server-specific recorded fixture still require execution on server4.

## Missing coverage and blockers

1. **BLOCKED — CUDA numerical parity:** local Torch reports `torch.cuda.is_available() == False`. Server4 confirms an A5000 and CUDA Torch import, but the current goldens deliberately instantiate CPU agents. No CUDA forward/backward, optimizer state loading, target EMA, native planning or deterministic kernel parity was executed. CPU values are the frozen comparison; CUDA may need a separately declared tolerance, never an unreported fixture rewrite.
2. **PARTIAL — NS-Gym simulator:** `ns_gym` is absent locally, but the server's NS-Gym target suite executed 26 passes. This establishes the existing simulator tests on that host; a platform-specific numerical golden and full saved-bank/production checkpoint parity are still pending.
3. **BLOCKED — ManiSkill2/SAPIEN simulator:** `mani_skill2` and `sapien` are absent locally. Real action/physics shift and renderer checks were not executed. Existing mock SAPIEN contract tests are separate from real physics validation.
4. Full production `TDMPC2.load` of historical nominal+optimizer and adaptation snapshots under the matching server stack, plus resumed one-step equality, is still required. Current checkpoint tests establish CPU file/tree serialization and parent hashes only.
5. Fixed saved transition-bank evaluation, environment seed pairing, termination/truncation and action-command versus executed-action parity are not yet golden fixtures. Existing tests cover subsets but no numerical baseline is recorded here.
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

Use the existing original-compatible ManiSkill2/SAPIEN interpreter for its separate opt-in checks, without installing NS-Gym dependencies into it:

```bash
TD_MPC2_TEST_MANISKILL=1 python -m pytest tests/test_maniskill_integration.py tests/test_shift_physics.py -q
```

The last command must be run **after activating that interpreter**; its environment and renderer choice are host-specific. Do not infer a PASS from these unexecuted commands.

## Phase 1B readiness

**Ready for CPU-only import/contract bridges** if they preserve the recorded fixture and existing 35-test targeted suite. **Not ready for D+Q algorithm migration or broad Phase 1B sign-off** until the compatible server executes its revised platform golden, the historical checkpoint sources are located and checked, CUDA numerical parity is tested, and missing saved-bank/environment parity fixtures are established. Every later migration change must rerun the goldens against the unchanged fixture; a mismatch requires diagnosis, not a fixture update.
