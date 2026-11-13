# Reacher existing-results research analysis

This is read-only analysis: no model loading, simulator calls or optimizer updates.
The current local input is a copied **analysis export**, not the original experiment directories.
The output therefore reports `PARTIAL_EVIDENCE`, with `analysis_execution: PASS` when calculations succeed.
Existing runtime scripts and their provenance hashes are unchanged. Reuse: CSV/provenance, duplicate checking,
bootstrap and correlation helpers from `analyze_existing_mechanisms.py`; one new Reacher entrypoint.

## 1. Locate remote originals (server4)

These commands print paths and markers; they do not rerun any experiment.

```bash
cd /home/jhkim/repos/tdmpc2
for base in /home/jhkim/repos/tdmpc2/results /nfs4/jhkim/repos/tdmpc2/results /home/jhkim/repos-reacher-joint/tdmpc2/results; do
  if [ -d "$base" ]; then
    find "$base" -type d \( -name 'preview-v1' -o -name 'reacher-full-01' -o -name 'nominal-home-01' -o -name 'probe-home-01' -o -name 'frozen-home-01' -o -name 'adapt-home-01' -o -name 'prediction-home-01' -o -name 'control-home-01' -o -name 'control-home-02-resumed' -o -name 'full-joint-home-01' -o -name 'analysis-comparison-home-01' \) -print
  fi
done
```

After locating the preview root, inspect completion markers (directories alone are not completion evidence):

```bash
ROOT=/home/jhkim/repos/tdmpc2/results/nsgym/reacher/preview-v1
python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ('nominal-home-01','probe-home-01','frozen-home-01','adapt-home-01',
             'prediction-home-01','control-home-01','control-home-02-resumed',
             'full-joint-home-01/runs-fixed','full-joint-home-01/evaluation-full-01'):
    p=root/name; marker=p/'completion.json'
    record=json.loads(marker.read_text()) if marker.is_file() else {}
    print(name, 'status=',record.get('status','NO_COMPLETION'),
          'child_markers=',len(list(p.glob('*/completion.json'))),'path=',p.resolve())
for p in root.rglob('selected_reacher_checkpoint.json'):print('SELECTION:',p.resolve())
PY
```

The new analysis CLI also has a marker inventory mode; it requires only Python and NumPy:

```bash
python experiments/nsgym/scripts/analyze_reacher_research.py \
  --source "$ROOT" --output /tmp/reacher-marker-inventory-01 --stage inventory
```

Marker inventory is not a snapshot or trajectory audit. The summary analysis mode currently consumes the
existing `analysis-comparison-home-01` export. Raw originals must be inspected before upgrading evidence status.
Do not add original `control-home-01` and resumed control as independent samples. They require matching
snapshot hash, condition, budget, method and seed, with conflicting duplicate rows rejected.

## 2. Files needed from the server

Copy the directory hierarchy for these small evidence files first (no training or evaluation is needed):

- Nominal: `selected_reacher_checkpoint.json`, nominal validation CSVs, checkpoint `metadata.json`.
- All stages: `completion.json`, `manifest.json`, configuration JSONs.
- Probe: `reacher_physical_response.csv`, inventory JSONs.
- Frozen: `reacher_frozen_severity.csv` and steps CSVs.
- Prediction/control children: `predict.csv`, `control.csv`, `steps.csv`, completion markers with snapshot hashes.
- Joint runs: `training.jsonl`, `online_episodes.csv` if present, completion markers, manifests.
- Joint evaluation children: `predict.csv`, `control.csv`, `steps.csv`, completion markers.

Missing fields and exact source patterns are listed in `reports/MISSING_EVIDENCE.md`.
Do not infer missing Joint reward components or learning losses from final return.
Model/trajectory byte verification additionally requires the referenced `.pt`/`.npz` files, or a server-side
hash audit; those large files are not needed merely to inspect saved CSV metrics.

## 3. Reproduce summary-based analysis

Run from the project directory containing `experiments/` and `tdmpc2/`.
Set `SOURCE` to the discovered exported summary directory and use a fresh output directory.

```bash
conda activate tdmpc2-nsgym
cd /home/jhkim/repos/tdmpc2
SOURCE=/home/jhkim/repos/tdmpc2/results/nsgym/reacher/preview-v1/analysis-comparison-home-01
OUTPUT=/home/jhkim/repos/tdmpc2/results/nsgym/reacher-analysis-v2
python experiments/nsgym/scripts/analyze_reacher_research.py --help
python experiments/nsgym/scripts/analyze_reacher_research.py \
  --source "$SOURCE" --output "$OUTPUT" --dry-run
python experiments/nsgym/scripts/analyze_reacher_research.py \
  --source "$SOURCE" --output "$OUTPUT" \
  --nominal-record /home/jhkim/repos/tdmpc2/results/nsgym/reacher/preview-v1/nominal-home-01/selected_reacher_checkpoint.json
```

Optional `--halfcheetah PATH` uses a complete `mechanism-analysis-750k-v1` export for dimensionless
within-task comparison. Do not supply an incomplete or unrelated HalfCheetah directory.
`--stage summary` omits plots; `--no-summary` omits narrative reports. `--family M1 --severity 0.5`
filters conditions; `--method 'Dynamics-only'` filters canonical views only, preserving paired contrasts.
`--stage plots` generates plots and supporting tables/provenance. Every analysis run creates a ZIP and SHA256 sidecar.
Nominal family aliases remain labeled but are not pooled as independent environments.

The default return band (0.1% of absolute Frozen return) is an exploratory analysis convention, not a
preregistered equivalence threshold. `classification_sensitivity.csv` includes 0%, 0.01%, 0.1%, 1% bands.
Three adaptation-stream means are bootstrapped conditional on the twenty evaluation seeds; intervals are descriptive.
No individual transition is an independent statistical sample. All outliers remain.

Local reproduction (workspace root has a nested project):

```bash
cd /Users/jihyekim/Desktop/tdmpc2
/Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python \
  tdmpc2/experiments/nsgym/scripts/analyze_reacher_research.py \
  --source results/nsgym/preview-v1/analysis-comparison-home-01 \
  --output results/nsgym/reacher-analysis-v2-reproduced \
  --nominal-record results/nsgym/reacher-analysis-v2/provenance/selected_nominal_record.json \
  --halfcheetah results/nsgym/mechanism-analysis-750k-v1
```

Tests: `python -m unittest discover -s experiments/nsgym/tests -p test_reacher_research.py`.
These exercise duplicate/reference safeguards; original resumed snapshot reconstruction remains unverified
until the raw markers and episode files are available. No test claims to validate missing raw evidence.
