# HalfCheetah Q distribution and planning mechanism analysis

Read-only, no training, no simulator transitions. Existing artifacts are never overwritten.
Commands below are for the server project root `/home/jhkim/repos/tdmpc2`.
Local source is nested under `tdmpc2/experiments/`; deploy the repository normally before running.
All four model variants are required for a matched stream; missing/incompatible streams are excluded and recorded.
No pipeline command launches another scientific stage automatically.

## Implementation and interpretation

- `analyze_q_planning.py`: inventory, canonical bank, inference, aggregation, plots, reports, ZIP.
- `q_planning_core.py`: original WorldModel Q extraction and original `TDMPC2._estimate_value` instrumentation.
- Reuses `mechanism_forward.Config/state_hash`, `analyze_existing_mechanisms` hashes/ranks and original `training_support.tree_hash`.
- Existing loaders instantiate original optimizers and may construct environments. This analysis instead follows their exact **model-state restoration**: F `api_model_conversion`, D nominal + adapter predictor, DQ/J full saved model. Full reconstructed hashes (where saved), component hashes, final counters, metadata and snapshot hashes are checked. This is an intentional analysis-only loader, not a replacement training/resume loader. No optimizer or environment is constructed.
- Categorical value: five heads, 101 symlog bins [-10,10], softmax then `symexp(E[bin])`. Not `E[symexp(bin)]`. Boundary mass: first and last two bins. Model-level JS compares ensemble mixtures; no cross-model head-index matching.
- Immediate F/D equality is asserted at each common observation/action. Joint uses its own encoder.
- Native planner terminal actor is stochastic and Q is the average of two randomly selected heads. Score decomposition calls the original function. Forked seeds preserve native sampling and caller RNG. Identical random-number seeds pair model comparisons; resulting actor actions can differ with latent/model changes.
- Controlled terminal effects instead use the Frozen terminal action across models. These outputs are separate from native planner scores. Seven variants include F, D, DQ-dyn-only, DQ-full, TTT+DQ-Q, F+DQ-Q, and native J. Never mix Joint latent spaces with nominal modules.
- Candidates are fixed clipped Gaussian proposals at nominal max_std. No future data, policy prior, warm start or multi-iteration replanning. Elite weights match native exponent weighting on this bank. Argmax disagreement is **not** native stochastic action disagreement or ranking accuracy.
- Native saved horizon is 3; `--horizon` explicitly changes only analysis horizon. It never mutates saved configuration.
- Termination flags are inferred only from verified complete 1000-step Frozen episodes with no true termination and final truncation. Actual rewards and next observations come from hashed NPZ. TimeLimit bootstraps. Bellman residual and distributional CE are off-policy consistency, not calibration.
- Adaptation seeds are 42000001–42000003; held-out seeds 43000001–43000020. Default Tier A samples 16 evenly spaced transitions per episode: 320 inputs per stream, 11,520 stream/input pairs for 36 matched streams, four model outputs each. Frozen banks reused across adaptation streams are not independent baseline models.
- Tier B defaults to two 2× conditions, three adaptation seeds, first observation from each of two held-out episodes, 64 candidates, seven variants: 12 observation/stream cases and 5,376 model-candidate scores. Cases were selected from existing outcomes and are not representative of all severities.
- Statistical summaries first average within evaluation seed, then adaptation stream, then bootstrap the three adaptation seeds. Shared Frozen baseline uncertainty uses unique evaluation seeds. Nominal k=1 family labels are retained in separate facets and never pooled into independent physical tasks.
- Stored categorical/logit arrays and canonical NPZ banks remain on server; review ZIP excludes them and all checkpoints/replay/optimizer files.

## 1. Variables and bounded CPU smoke

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
Q=experiments/nsgym/scripts/analyze_q_planning.py
RESULTS=/home/jhkim/repos/tdmpc2/results/nsgym
OUT="$RESULTS/q-planning-mechanism-750k-v1"
python "$Q" --help
python "$Q" smoke --output "$RESULTS/q-planning-cpu-smoke-v1"
```

Smoke uses a small real WorldModel with synthetic inputs; it is not a scientific checkpoint or CUDA smoke. It covers probability normalization/decoding, invariance, hybrids, sampling/bounds, score decomposition, state/RNG preservation, no Adam construction, and the table/plot/report path. It does not roll out an environment.

## 2. Compatibility inventory and common bank

```bash
python "$Q" inventory --results-root "$RESULTS" --output "$OUT" --device cpu
cat "$OUT/provenance/inventory.json"
python "$Q" bank --results-root "$RESULTS" --output "$OUT" --observations 16
```

Inventory excludes incompatible streams. Read excluded_models.csv. If the local checkpoint is missing, inventory records NOT_RUN; it does not manufacture compatibility evidence. Use a fresh output directory for changes or retries; there is no implicit resume or overwrite.

## 3. Tier A — all matched conditions

```bash
python "$Q" immediate --tier A --results-root "$RESULTS" --output "$OUT" --device cuda:0
```

Optional background form (run **instead** of the foreground command):

```bash
nohup python -u "$Q" immediate --tier A --results-root "$RESULTS" --output "$OUT" --device cuda:0 \
  > "$OUT/immediate.log" 2>&1 < /dev/null &
echo $! > "$OUT/immediate.pid"
tail -n 50 -f "$OUT/immediate.log"
```

## 4. Tier B — terminal Q and planning ranking together

```bash
python "$Q" terminal --tier B --results-root "$RESULTS" --output "$OUT" \
  --device cuda:0 --planning-observations 2 --candidates 64 --top-k 8
```

`planning` is an alias for `terminal`; do not run both into the same output. The single pass emits terminal_q_metrics.csv, planning_score_components.csv, candidate_ranking_comparison.csv, and controlled terminal_counterfactuals.csv. No actual candidate simulator return is computed.

## 5. Aggregate, plot, report and package

```bash
python "$Q" aggregate --results-root "$RESULTS" --output "$OUT"
python "$Q" plot --results-root "$RESULTS" --output "$OUT"
python "$Q" report --results-root "$RESULTS" --output "$OUT"
python "$Q" package --results-root "$RESULTS" --output "$OUT"
cat "$OUT/provenance/completion.json"
```

Package prints absolute ZIP path, bytes, file count and SHA256, and tests ZIP CRC. No automatic transfer. Existing DQ comparison CSV supplies paired return outcomes. Missing control evidence is reported; no calibration/causal conclusion is fabricated.

## Optional full-grid planning

Use a new output directory, preserving all initial analysis:

```bash
EXPANDED="$RESULTS/q-planning-mechanism-750k-fullgrid-v1"
python "$Q" inventory --results-root "$RESULTS" --output "$EXPANDED" --device cpu
python "$Q" bank --results-root "$RESULTS" --output "$EXPANDED" --observations 16
python "$Q" immediate --tier A --results-root "$RESULTS" --output "$EXPANDED" --device cuda:0
python "$Q" planning --tier B --full-grid --results-root "$RESULTS" --output "$EXPANDED" \
  --device cuda:0 --planning-observations 2 --candidates 64 --top-k 8
python "$Q" aggregate --results-root "$RESULTS" --output "$EXPANDED"
python "$Q" plot --results-root "$RESULTS" --output "$EXPANDED"
python "$Q" report --results-root "$RESULTS" --output "$EXPANDED"
python "$Q" package --results-root "$RESULTS" --output "$EXPANDED"
```

Optional selectors `--parameter`, `--severity`, `--seed` restrict inventory and inference. Use identical selectors when extracting a restricted bank. `--dry-run` prints resolved arguments and does not load models or write files. Current matched inference requires `--models F D DQ J`; arbitrary subsets are intentionally rejected to preserve invariance tests.

Additional stratification: action saturation means any action component |a|≥0.99, not force saturation. Root velocity bins are negative, 0–5 and above 5 m/s. Observation regime is early (<100 transitions) versus later. Physical response regime, when qacc is available, is the within-source-episode lower/middle/upper tertile of generalized acceleration norm. Its mixed generalized-coordinate units and episode-specific cutoffs preclude interpreting this as an absolute cross-condition physical severity scale; thresholds are saved in each canonical bank.
