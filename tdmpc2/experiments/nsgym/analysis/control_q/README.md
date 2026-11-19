# Final NS-Gym research figures

Run from the project directory containing `experiments/` and `tdmpc2/`:

```bash
python -m experiments.nsgym.analysis.control_q.run \
  --mode all --results-root results/nsgym \
  --output results/nsgym/final-figures-v1
```

`--mode figures` generates the same validated PNGs and figure guide; `--mode report` refreshes the guide in an already completed output. `--validate-only` checks the three source artifacts and full condition/seed coverage without drawing. `--smoke` additionally checks the predeclared HalfCheetah damping 2× saved array. Outputs must be new except for `--mode report`.

This CLI reads three completed and hash-validated archives under `results/nsgym/unified-analysis`: `q-terminal-planning-v1`, `q-decision-relevance-v1`, and `q-terminal-planning-presentation-v2`. It uses saved categorical-decoded Q arrays, fixed candidate scores, native planner traces and measured held-out return records. It does not load a model checkpoint, run a simulator, train, or repeat checkpoint inference.

Each of the six shift families receives four main PNGs: actual return/prediction, fixed-head terminal Q, fixed-candidate ranking and native MPPI proposed-action difference. The three optional appendices per family use a predeclared severity-2 example. A fifth closed-loop critic-swap plot is generated only when a completed, validated actual environment evaluation exists; none is registered in the current source set, so that outcome is marked `NOT_MEASURED`.

`FIGURE_READING_GUIDE.md`, `FIGURE_REPORT.html`, `FIGURE_MANIFEST.csv`, `FIGURE_SOURCE_MAP.csv`, `FIGURE_COVERAGE.csv`, `FIGURE_VALIDATION.json`, `FIGURE_CODE_AUDIT.csv` and `DELETED_PLOT_CODE.csv` document interpretation, provenance, coverage and code cleanup. Source arrays and checkpoints remain unchanged. The older HalfCheetah `analyze_q_planning.py` remains a numerical bank/inference tool used by model reconstruction and tests; its duplicate plot function and `plot` command were removed. The historical standalone figure CLIs and their rendering modules were deleted. Their exact previous versions are recoverable with `git show 99429838effbfd882aa6e4b15e57c6b5282a5ba5:<path>`.

The physical Q-swap evaluator remains available separately as `python -m experiments.nsgym.analysis.control_q.evaluation --results-root results/nsgym --output NEW_DIRECTORY --smoke`; a full run additionally requires `--smoke-result COMPLETED_SMOKE_DIRECTORY`. It is a simulator evaluation runner and is never called by the figure CLI. After a complete validated 108-stream evaluation, pass its output with `--swap-source PATH` to add the conditional Figure 05.

Code ownership is narrow: `run.py` is the sole figure CLI; `data.py` verifies archived inputs and normalizes condition coverage; `metrics.py` contains decoder-adjacent statistics and reproducible intervals; `figures.py` is the only plotting implementation; `report.py` validates output and writes the reading guide. `provenance.py` checks source hashes. `models.py` and `diagnostics/{terminal_value,planning,q_swap}.py` retain numerical/model inference needed for scientific provenance and the separate evaluator; `reporting/aggregate.py` holds the paired closed-loop return estimator. `config.py` fixes task, seed and checkpoint contracts.
