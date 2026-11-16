# Checkpoint-backed Q terminal and fixed-candidate diagnostics

Run from the directory containing `experiments/` and the Python core `tdmpc2/`.
This is a separate forward-only module. Existing 2×2 dashboards and adaptation
runners are unchanged. No environment, agent constructor, optimizer, or replay
collection is run. Source checkpoints and outputs are hashed and read-only.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RESULTS="$PWD/results/nsgym"
STAMP=$(date +%Y%m%d-%H%M%S)
SMOKE="$RESULTS/q-terminal-planning-smoke-$STAMP"
OUT="$RESULTS/q-terminal-planning-$STAMP"
export MPLCONFIGDIR=/tmp/mpl-nsgym

(
set -euo pipefail
python -u -m experiments.nsgym.analysis.q_terminal_planning_figures \
  --results-root "$RESULTS" --output "$SMOKE" --device cpu --smoke
cat "$SMOKE/completion.json"

# Run after smoke status is PASS. Full grid is independent of the outcome signs.
python - "$SMOKE/completion.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
assert x['status']=='PASS' and x['complete_inference_streams']==4, x
PY
nohup python -u -m experiments.nsgym.analysis.q_terminal_planning_figures \
  --results-root "$RESULTS" --output "$OUT" --device cpu \
  > "$OUT.log" 2>&1 < /dev/null &
echo $! > "$OUT.pid"
tail -n 50 -f "$OUT.log"
)
```

The CLI refuses existing output directories and ZIPs. It exits nonzero on
incomplete coverage, including when only verified archive-based immediate Q
plots could be produced. Read `completion.json` and `COVERAGE_AND_LIMITATIONS.md`.
Full inputs: 12 HalfCheetah family/severity labels × 3 streams, plus 24 Reacher
labels × 3 streams. Each stream uses 20 paired held-out seeds, 16 immediate
states per episode, four fixed terminal-tape starts, and 32 candidates per start.
Reacher planning horizon is read from checkpoint configuration (3), distinct
from its 50-step episode/runner horizon. Discounts are computed from the native
saved configuration: HalfCheetah 0.995, Reacher 0.95.

Primary terminal action is the saved Frozen action at `start + H`, common to
all methods and all candidates. Primary scores use native scalar decoding
head-by-head and fixed heads 0,1. Immediate and terminal distribution summaries
use all-head scalar means. Candidates are new seeded, clipped Gaussian proposals
with saved `max_std`; they are **not** historical MPPI candidates. An additional
separate sensitivity protocol uses native `pi` and random two-head Q averaging
at each planning state, for three paired RNG repetitions. It does not rerun the
MPPI candidate-refinement loop or claim to reproduce historical actions.

Tables named `*_per_seed.csv` retain adaptation/evaluation pairing. Statewise
probabilities, per-head scalar values, latents, actions, terminal decompositions,
candidate scores and native sensitivity scores are in `q_arrays/`. The compact
ZIP includes reports, figures, aggregate summaries and six small severity-2
example arrays, excluding checkpoints and full raw banks. The complete local
output keeps all per-seed and candidate-ranking records.

All effect intervals are descriptive hierarchical bootstrap intervals, with
state/head/candidate aggregation before resampling. Three adaptation streams
limit inference. Severity-1 aliases are not pooled, and task magnitudes remain
separate. No actual candidate returns, planning regret, Q accuracy/calibration,
or causal control recovery are measured.

## Reading the figures

- **A, signed delta:** average decoded scalar Q movement on the same saved
  physical observations/actions. Positive and negative statewise changes may
  cancel. **A, absolute delta:** average of absolute statewise movements, before
  averaging states, so cancellation cannot hide a large change. Head standard
  deviation is critic disagreement, not proven uncertainty calibration.
- **A supplement:** entropy and Frozen-relative JS compare categorical outputs.
  JS uses the mean-head probability distributions. Frozen and D overlap because
  their encoder and critic are unchanged; this is an invariant.
- **B:** all-head mean decoded terminal Q after each model's own predictor runs
  the same saved action tape. The terminal action is also identical. D can now
  differ from Frozen through its predictor despite having the same critic.
- **C:** discounted decomposition with the **fixed head pair 0,1**, matching the
  controlled candidate scoring protocol. Its three terms sum exactly to the
  D+Q-minus-Frozen terminal score on each tape. Do not expect it to equal the
  all-head scalar delta in B: the head aggregation is deliberately different.
- **D:** common newly generated candidates, scored by each model. Spearman and
  Jaccard near one mean similar preferences; lower values mean different
  preferences. They do not identify the physically correct ranking. The native
  terminal sensitivity panel uses a separate stochastic protocol. Margins are
  top-1 minus top-2 predicted scores, not measured action regret. Bottom reward
  and terminal panels average over all candidates, so they are not confounded
  by each model selecting a different winner. Winner-specific components also
  remain in the CSV.
- **E:** one predetermined earliest state at severity 2. Faint curves are heads;
  bold curves are mean-head probability mass. These examples do not replace
  the condition-level multi-state/seed statistics in A–D.

The comparison `DQ_F → DQ` holds the D+Q predictor, reward model, initial state,
terminal action and candidate bank fixed and changes only the critic. The
comparison `D → DQ` includes both the actual predictor difference and critic
update. Neither comparison proves why closed-loop return improved or worsened.

## Presentation v2 from completed v1 (no forward inference)

Use the same module with `--presentation-source`. The v1 source and canonical
return CSV hashes must match. This does not load checkpoints or run smoke again.
Output must be a new directory; v1 artifacts remain read-only.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RESULTS="$PWD/results/nsgym"
SOURCE="$RESULTS/q-terminal-planning-v1"
OUT="$RESULTS/q-terminal-planning-presentation-v2"
export MPLCONFIGDIR=/tmp/mpl-nsgym
(
  set -euo pipefail
  test -f "$SOURCE/completion.json"
  test ! -e "$OUT"
  test ! -e "$OUT.zip"
  nohup python -u -m experiments.nsgym.analysis.q_terminal_planning_figures \
    --results-root "$RESULTS" --presentation-source "$SOURCE" --output "$OUT" \
    > "$OUT.log" 2>&1 < /dev/null &
  echo $! > "$OUT.pid"
) && tail -n 50 -f "$OUT.log"
```

Start with `PRESENTATION_Q_RESULTS.md` and the three figures in `figures/`.
Supplementary figures retain all severities, per-state centered candidate score
examples, and zoom/full-support probability plots with per-head decoded Q.
Per-state spread CSV reports standard deviation, IQR and range across candidates;
these spreads are not independent adaptation samples. Components are centered
separately, so their centered versions need not add to centered total score.

## Decision relevance, including actual native planning logic

The same command dispatches the internal native-planner instrumentation helper.
Locate the completed v1 directory first; it may be under unified-analysis/.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
RESULTS="$PWD/results/nsgym"
find "$RESULTS" -type d -name q-terminal-planning-v1
# Set SOURCE to the path printed above.
SOURCE="$RESULTS/unified-analysis/q-terminal-planning-v1"
OUT="$RESULTS/q-decision-relevance-v1"
export MPLCONFIGDIR=/tmp/mpl-nsgym
(
  set -euo pipefail
  test -f "$SOURCE/completion.json"
  test ! -e "$OUT"
  test ! -e "$OUT.zip"
  nohup python -u -m experiments.nsgym.analysis.q_terminal_planning_figures \
    --results-root "$RESULTS" --decision-source "$SOURCE" --output "$OUT" \
    > "$OUT.log" 2>&1 < /dev/null &
  echo $! > "$OUT.pid"
) && tail -n 50 -f "$OUT.log"
```

No simulator or optimizer runs. Matched behavior Q, terminal tapes and the
32-candidate scores are reused. New immediate multi-action Q uses all original
20 evaluation seeds. Native planning uses the first two evaluation seeds per
stream, respectively cold and imposed common warm initialization (not a
historical cache). Actual `_plan` runs with native priors, samples, elites and
iterations; identical recorded base noises are replayed across all models.
Instrumented and uninstrumented final actions and means must agree exactly.
A parity failure produces PARTIAL, never PASS. Full output includes new native
trace arrays; compact ZIP omits them. No automatic held-out rollouts follow.
