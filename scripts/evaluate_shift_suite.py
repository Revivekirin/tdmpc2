"""Declarative frozen suite launcher; --dry-run does not need RL dependencies."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT/'benchmarks/dynamics_shift_suite.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if not args.dry_run and not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    known = {c['id'] for c in suite['conditions']}
    if args.ids and set(args.ids) - known:
        parser.error(f'Unknown IDs: {set(args.ids) - known}')
    for condition in suite['conditions']:
        if args.ids and condition['id'] not in args.ids:
            continue
        cmd = [args.python, 'evaluate_shift.py', f"configs/dynamics_shift@dynamics_shift={condition['preset']}",
               f"task={suite['task']}", f"model_size={suite['model_size']}",
               f"checkpoint={args.checkpoint.resolve()}", f"eval_episodes={suite['episodes']}",
               f"seed={suite['seed']}", f"adaptation_mode={suite['protocol']}", 'save_video=false',
               f"benchmark_id={condition['id']}", *condition['overrides']]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT/'tdmpc2', check=True)


if __name__ == '__main__':
    main()
