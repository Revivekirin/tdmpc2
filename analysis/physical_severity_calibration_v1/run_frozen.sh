#!/usr/bin/env bash
set -euo pipefail
CALIBRATION_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${REPO:?Set REPO to the server repository}"
cd "$REPO"
: "${CHECKPOINT:?Set CHECKPOINT to the unchanged pretrained checkpoint}"
python scripts/run_stage3b.py --phase severity --method frozen --checkpoint "$CHECKPOINT" --shift-type action_delay --shift-values 0 1 2 3 --set-id 70 --episodes 3 --log-eef --run-id calibration-action_delay --output "$CALIBRATION_DIR/frozen" --execute
python scripts/run_stage3b.py --phase severity --method frozen --checkpoint "$CHECKPOINT" --shift-type action_gain --shift-values 1 0.9 0.7 0.5 0.3 --set-id 70 --episodes 3 --log-eef --run-id calibration-action_gain --output "$CALIBRATION_DIR/frozen" --execute
python scripts/run_stage3b.py --phase severity --method frozen --checkpoint "$CHECKPOINT" --shift-type object_mass --shift-values 1 1.5 2 3 --set-id 70 --episodes 3 --log-eef --run-id calibration-object_mass --output "$CALIBRATION_DIR/frozen" --execute
python scripts/run_stage3b.py --phase severity --method frozen --checkpoint "$CHECKPOINT" --shift-type contact_friction --shift-values 1 0.7 0.5 0.3 --set-id 70 --episodes 3 --log-eef --run-id calibration-contact_friction --output "$CALIBRATION_DIR/frozen" --execute
