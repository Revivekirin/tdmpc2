#!/usr/bin/env bash
# Submit from the repository root with site-appropriate Slurm resource flags.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the tdmpc2 repository root}"
source /home/jhkim/miniconda3/etc/profile.d/conda.sh
conda activate tdmpc2-nsgym

REACHER_CONFIG="${REACHER_CONFIG:-experiments/nsgym/configs/train_reacher_pilot.yaml}"
REACHER_OUTPUT="${REACHER_OUTPUT:-results/nsgym/reacher-pilot-${SLURM_JOB_ID:?}}"
mkdir -p results/nsgym
python -u experiments/nsgym/scripts/train_nominal.py \
  --config "$REACHER_CONFIG" --output "$REACHER_OUTPUT" \
  2>&1 | tee "${REACHER_OUTPUT}.log"
