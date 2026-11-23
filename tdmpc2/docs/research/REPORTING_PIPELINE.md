# Phase 3B reporting pipeline

## Boundary and ownership

The canonical `research.dynamics_shift.reporting` package reads completed NS-Gym results only. It never constructs an agent, runs a simulator, trains, evaluates a new controller, or writes into a source root. A `ReportBuilder` owns the explicit source root and a new output directory. Versioned `ReportingConfig` selects `unified` or `control_q`; version 2 is required to admit an explicit, already completed closed-loop Q-swap source. Without one, the critic-only **closed-loop return** status is `NOT_MEASURED`, distinct from a measured return of zero. No Q-swap experiment is executed.

| Layer | Canonical owner | Historical calculation retained |
|---|---|---|
| Discovery and byte validation | `reporting/sources.py` | `analysis.discovery.Catalog`, `analysis.sources.collect`, `control_q.data.load_inputs`, `control_q.provenance.verified_artifact` |
| Canonical record and evidence | `reporting/records.py` | `analysis.schema.canonical`; archived control-Q table rows and their measurement/prediction distinction |
| Metric and comparison dispatch | `reporting/metrics.py` | `analysis.metrics` seed and pair functions; `analysis.statistics` policy seeds/draws |
| Numerical figure inputs | `reporting/figure_data.py` | `control_q.figures._terminal_table`, original fixed-head and archived-interval rules |
| Visualization | `reporting/rendering.py` | Unchanged `analysis.engine.figures/research_figures` and `control_q.figures.render` |
| Assembly, coverage, provenance | `reporting/builder.py` | Existing source validations, output table columns and figure validator |

The builder adds its own `canonical_records.csv`, statistical or terminal figure-data CSVs, `SOURCE_COVERAGE.csv` where applicable, `SOURCE_PROVENANCE.json`, and `REPORT_STATUS.json`. It selects a new directory and refuses to overwrite one. A mixed return/prediction evidence row becomes two canonical measurements, so the actual return is `MEASURED` and the fixed-encoder prediction error is `PREDICTED`; an absent Joint fixed-encoder error produces no prediction measurement. The existing `python -m experiments.nsgym.analysis` and `python -m experiments.nsgym.analysis.control_q.run` CLIs and their output schemas remain unchanged. Their functions are reused without moving their source bytes because historical manifests identify those bytes. The new package is a parallel auditable path, not a silent replacement for either CLI.

## Source-to-output dependency matrix

| Raw source family | Validation and row schema | Metric/aggregation | Figure input and renderer | Output/status |
|---|---|---|---|---|
| Completed nominal, physical, Frozen, adaptation and evaluation records | Metadata/completion and snapshot hashes, schema classification and exclusions in `Catalog`/`collect`; canonical row retains task, model, seed, schedule, budget, unit, source path and SHA-256 | `summarize_records`, complete-seed `paired_improvements`, method contrasts and prediction/control pairs | Existing `engine.figures` and `research_figures` consume the same summary and comparison rows | `statistical_summary.csv`, `comparisons.csv`, coverage, optional PNGs; incomplete/smoke excluded |
| Archived terminal Q arrays | Exact source-hash and 108-array gates in `control_q.data.load_inputs`; fixed heads 0,1 and aligned candidate IDs | Original terminal reconstruction and `control_q.metrics.aggregate`; archived critic-only CI retained after mean parity | `_terminal_table` yields terminal rows, candidate changes and identity checks; existing `control_q.figures.render` | 36-condition summary, 108-stream checks, 42 PNGs when rendered without Q-swap |
| Archived decision relevance | Same three-family source gate, original 20 evaluation seeds and three adaptation seeds | Existing fixed-candidate ranking and native MPPI metrics | Existing 03 ranking and 04 native MPPI panels | Predicted rank/action evidence only |
| Archived paired held-out returns | Same three-family source gate | Existing paired return summary | Existing 01 return panel | Actual episode return, kept separate from Q estimates |
| Optional completed Q-swap episodes | Version 2 configuration plus `load_swap` 108-stream/4320-episode and source-hash gates | Existing `return_bridge` and 4000-draw hierarchical interval | Existing optional 05 closed-loop panel | `MEASURED` only when a complete source is supplied; otherwise `NOT_MEASURED` |
| `plot_results.py`, `analyze_q_planning.py`, `physical.py`, `ttt.py`, `reacher.py` | Their own distinct historical sources and source hashes | Their own documented task-specific formulas | Their existing figures and reports | Legacy entry points retained; no numerical parity or migration claimed for untested figures |

The control-Q gate still requires exactly three historical artifact families: `q-terminal-planning-v1`, `q-decision-relevance-v1` and `q-terminal-planning-presentation-v2`. It still requires six families × six severities = 36 conditions and 108 adaptation streams. Optional future families are explicit configuration changes, never discovered as substitutes for required sources.

## Meaning, units and comparability

Physical severity is a dimensionless multiplier; measured response has its own physical units. One-step latent MSE and multi-step rollout error are separate. Latent MSE is comparable across methods only in a shared encoder coordinate system; Joint changes that encoder and stays on its own metric. Reward prediction error is not episode return. Immediate Q uses an encoded real observation; terminal Q uses an imagined post-horizon latent and sampled terminal action. Both are model estimates. Categorical distributions, head disagreement and entropy describe model outputs, not empirical Q accuracy. Predicted reward plus discounted terminal Q is a candidate score. Candidate rank changes and native MPPI proposed-action changes are not observed control improvement. Actual closed-loop episode return comes only from executed environment trajectories.

Frozen, Dynamics-only, Dynamics+Q and Joint retain their historical definitions. Dynamics-only and Dynamics+Q were trained separately; their predictors may differ. B versus C under the Phase 3A composer isolates online Q on fixed D+Q dynamics. A versus C compares independently adapted predictors and cannot be labelled an isolated Q effect.

## Statistical and missingness conventions

The unified analysis groups by task, checkpoint/cohort, shift family/parameter, severity, schedule, kind, metric, method and budget. Adaptation seed is the sampling unit for adapted methods; a Frozen reference uses unique evaluation seeds. Paired return comparisons require the full common evaluation-seed set. Partial pairs are excluded, not shrunk to an intersection. HalfCheetah uses 10,000 bootstrap draws with seed `20261111`, Reacher uses 5,000 with `20261112`, Q summaries use 2,000 with seed `817`; the original `analysis.statistics` owns their exact formulas.

Control-Q uses a distinct, valid hierarchical convention: its archived `ci` draws 3,000 stream and evaluation indices using a hash of the comparison key. Optional measured Q-swap return intervals use 4,000 nested resamples with seed `20261116`. These confidence intervals are not interchanged. Terminal fixed-head Q-swap intervals in the original figure reuse the verified archived interval after asserting the reconstructed mean. The new figure-data adapter follows that rule exactly. Nonfinite values, missing sources, duplicate controller episodes and unpaired interventions remain errors or explicit exclusions under their original policies. `NOT_MEASURED` is never encoded as zero, NaN success, or a predicted score.

## Figure and provenance contract

The pipeline separates numerical figure inputs from rendering. Control-Q `01` uses actual return and fixed-encoder one-step prediction on separate axes; `02` uses fixed heads 0,1 terminal Q and discounted critic-only score; `03` uses fixed 32-candidate ranks; `04` uses native MPPI proposed-action distance; `05` exists only for measured closed-loop Q-swap. The existing renderer retains labels, units, six-severity order, methods, colors, exemplar rule and PNG settings. Generated images go to a new temporary output directory. Numerical plot-data equality is the primary parity gate; PNG byte equality across runtimes is not assumed. The existing 42/48-figure validator checks coverage, source evidence and no simulator/model calls.

`SOURCE_PROVENANCE.json` contains the explicit root, reporting configuration version, verified source hashes and the original three-family source records. Sources are rehashed after assembly. Unified records retain their original source row and SHA-256. Historical results, source manifests, figures and golden fixtures are never rewritten.

## Validation and known boundaries

Local temporary-fixture tests cover source-hash failure, path escape, missing source, exact unified record/statistical CSV parity, complete pairing, policy distinction, missing versus measured-zero Q-swap, and output non-overwrite. The server-only historical test validates the three artifact families, all 108 arrays, numerical terminal/candidate plot data, archived critic-only intervals, 42 rendered PNGs and source immutability. It runs only when `TDMPC2_REPORT_RESULTS_ROOT` names the compatible result tree; a local skip is not a pass.

The original ManiSkill2/SAPIEN runtime remains separate. Physical, TTT, Reacher, `plot_results.py` and `analyze_q_planning.py` retain their own report calculations and rendering until source-by-source numerical and figure-data parity is established. Adding a new result family requires an explicit versioned configuration, a source validator, canonical status/unit mapping, numerical comparison to its historical calculation, and a separate renderer parity test. No closed-loop Q-swap return is inferred from terminal Q, rankings or proposed actions.
