#!/usr/bin/env bash
# Launch only Full Joint RL training; no existing process is stopped or restarted.
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: bash $0 GPU_INDEX CHECKPOINT PASS_SMOKE_DIRECTORY NEW_RUN_DIRECTORY [--allow-shared-gpu]" >&2
  exit 2
fi
GPU="$1"
CHECKPOINT="$2"
SMOKE="$3"
OUTPUT="$4"
GPU_FLAGS=()
if [ "$#" -eq 5 ]; then
  test "$5" = "--allow-shared-gpu"
  GPU_FLAGS+=(--allow-shared-gpu)
fi
cd "$(dirname "$0")/../../.."
RUNNER=experiments/nsgym/scripts/run_full_joint_rl.py
LOG="${OUTPUT}.log"
PIDFILE="${OUTPUT}.pid"
test -f "$CHECKPOINT/model.pt"
test -f "$SMOKE/completion.json"
test ! -e "$OUTPUT"
test ! -e "$PIDFILE"
python "$RUNNER" gpu-check --gpu "$GPU" "${GPU_FLAGS[@]}"
mkdir -p "$(dirname "$OUTPUT")"
(set -o noclobber; : > "$LOG")
nohup python -u "$RUNNER" run --gpu "$GPU" "${GPU_FLAGS[@]}" --checkpoint "$CHECKPOINT" \
  --smoke-result "$SMOKE" --output "$OUTPUT" >> "$LOG" 2>&1 < /dev/null &
JOINT_PID=$!
(set -o noclobber; printf '%s\n' "$JOINT_PID" > "$PIDFILE")
echo "Full Joint RL PID: $JOINT_PID"
echo "Log: $LOG"
echo "The runner rechecks smoke validity and GPU availability before loading CUDA."
