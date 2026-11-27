#!/usr/bin/env bash
# Run from any directory on the configured CUDA / ManiSkill2 0.4.1 host.
set -euo pipefail
checkpoint="${1:?Usage: bash scripts/evaluate_dynamics_shift.sh /absolute/path/pick-cube-1.pt}"
if [[ "$checkpoint" != /* || ! -f "$checkpoint" ]]; then
  echo 'Checkpoint must be an existing absolute path.' >&2
  exit 1
fi
cd "$(dirname "$0")/../tdmpc2"
base=(task=pick-cube model_size=5 "checkpoint=$checkpoint" eval_episodes=50 seed=1 save_video=false)
python evaluate.py "${base[@]}" shift_mode=nominal exp_name=E0
python evaluate.py "${base[@]}" shift_mode=fixed shift_gain=0.9 exp_name=E1
python evaluate.py "${base[@]}" shift_mode=fixed shift_gain=0.7 exp_name=E2
python evaluate.py "${base[@]}" shift_mode=fixed shift_gain=0.5 exp_name=E3
python evaluate.py "${base[@]}" shift_mode=abrupt shift_gain=0.7 shift_clock=episode shift_switch_step=100 exp_name=E4-mid
python evaluate.py "${base[@]}" shift_mode=abrupt shift_gain=0.7 shift_clock=global shift_switch_step=100 exp_name=E4-global
