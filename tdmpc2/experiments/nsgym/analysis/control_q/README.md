# Control–critic research pipeline

Single public entry: `python -m experiments.nsgym.analysis.control_q.run`.
Run from the project root containing `experiments/` and `tdmpc2/`.

## Current verification and remaining work

Existing numerical tests: 9/9 before migration; 13/13 including new paired-aggregation
and critic-isolation tests after migration. Four real checkpoints (HalfCheetah
mass/damping 2×; Reacher D0/M0 2×) pass exact old/new native planner action/mean
parity, immediate Frozen=D invariant and component isolation. This is not a
108-stream inference rerun or a closed-loop physics test.

**New physical evaluation is NOT_RUN locally.** Historical CUDA and simulator
versions are required; local runtime lacks CUDA, MuJoCo and NS-Gym. The evaluator
must pass the server smoke before expansion. Report generation currently marks
PARTIAL and explicitly lists uncompleted physical/prediction figure consolidation
and full-source regression. Do not present it as a completed research report.

## Module ownership

- `run.py`: public stage selection.
- `config.py`: unchanged task/seed/checkpoint contracts.
- `models.py`: inventory, reconstruction, hashes.
- `metrics.py`: unchanged Q/rank/decomposition/bootstrap computations.
- `provenance.py`: content validation and explicit result-path relocation.
- `evaluation.py`: historical closed-loop task loops and parity gates.
- `diagnostics/terminal_value.py`: v1 immediate/terminal forward diagnostics.
- `diagnostics/planning.py`: native planner and decision diagnostics.
- `diagnostics/q_swap.py`: independent online-Q substitution; EMA stays unchanged.
- `reporting/figures.py`, `reporting/decision.py`: moved historical rendering.
- `reporting/aggregate.py`: paired hierarchical return bridge.
- `reporting/html.py`: offline evidence report and ZIP; incomplete evidence is labelled.
- `legacy_cli.py`: original CLI semantics for historical commands.

Original filenames are compatibility imports only. External scripts/tests and the
historical command document still reference them; deleting them would break those
users. Supporting scripts remain shared with unrelated experiments. Native TD-MPC2,
physical wrappers and historical results were not modified.

## Server commands

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RESULTS="$PWD/results/nsgym"
RUN="$RESULTS/control-q-run-$(date +%Y%m%d-%H%M%S)"
export MPLCONFIGDIR=/tmp/mpl-nsgym
mkdir -p "$RUN"

python -m experiments.nsgym.analysis.control_q.run audit \
  --results-root "$RESULTS" --output "$RUN/audit"

python -m experiments.nsgym.analysis.control_q.run q-diagnostics \
  --results-root "$RESULTS" --output "$RUN/q-cache" --reuse-validated
python -m experiments.nsgym.analysis.control_q.run planner-diagnostics \
  --results-root "$RESULTS" --output "$RUN/planner-cache" --reuse-validated
```

Audit/caches do not interact with a simulator. Stop on any failure; do not bypass
hash checks. Missing optional historical caches are allowed only if their original
inventory explicitly says `required=False`.

New smoke: four predeclared 2× streams, first historical evaluation seed, both
controllers: 8 episodes, 4,200 environment transitions. Exact historical D+Q return
parity uses a predefined 1e-6 absolute tolerance. Any failure stops expansion; no
post-hoc tolerance is chosen. Both model and auxiliary state must remain unchanged.

```bash
nohup python -u -m experiments.nsgym.analysis.control_q.run q-swap-eval \
  --results-root "$RESULTS" --output "$RUN/smoke" \
  --device cuda:0 --smoke > "$RUN/smoke.log" 2>&1 < /dev/null &
echo $! > "$RUN/smoke.pid"
tail -n 50 -f "$RUN/smoke.log"
```

Only after the smoke completion is PASS:

```bash
nohup python -u -m experiments.nsgym.analysis.control_q.run q-swap-eval \
  --results-root "$RESULTS" --output "$RUN/q-swap" --device cuda:0 \
  --smoke-result "$RUN/smoke" > "$RUN/q-swap.log" 2>&1 < /dev/null &
echo $! > "$RUN/q-swap.pid"
tail -n 50 -f "$RUN/q-swap.log"
```

Full matrix: 108 streams × 20 seeds × 2 controllers = 4,320 episodes;
1,584,000 environment transitions. No adaptation or training. Optional factorial
cross-combination is not launched. Choose a GPU externally with CUDA_VISIBLE_DEVICES;
the native implementation uses logical cuda:0. No running process is stopped.

After evaluation (report remains PARTIAL until remaining report/regression gaps
are resolved):

```bash
python -m experiments.nsgym.analysis.control_q.run integrate-report \
  --results-root "$RESULTS" --swap-source "$RUN/q-swap" \
  --output "$RUN/report"
```

`all` sequences audit → validated existing diagnostics → smoke → full Q-swap →
report. `all --smoke` stops after smoke. `--validate-only` performs only the source
audit. All outputs must be new; existing sources/checkpoints are read-only.
