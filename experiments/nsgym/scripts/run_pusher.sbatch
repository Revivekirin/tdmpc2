#!/usr/bin/env bash
# Submit from the repository root with Slurm resource flags appropriate to the cluster.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the tdmpc2 repository root}"
source /home/jhkim/miniconda3/etc/profile.d/conda.sh
conda activate tdmpc2-nsgym

PUSHER_CONFIG="${PUSHER_CONFIG:-experiments/nsgym/configs/train_pusher_pilot.yaml}"
PUSHER_OUTPUT="${PUSHER_OUTPUT:-results/nsgym/pusher-pilot-${SLURM_JOB_ID:?}}"
mkdir -p results/nsgym
python -u experiments/nsgym/scripts/train_nominal.py \
  --config "$PUSHER_CONFIG" --output "$PUSHER_OUTPUT" \
  2>&1 | tee "${PUSHER_OUTPUT}.log"
