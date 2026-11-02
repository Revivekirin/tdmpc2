"""Run Stage 2 through the existing evaluator; default is a reviewable dry run.

Use --execute for the real CUDA/ManiSkill checkpoint runs. This script neither
installs dependencies nor downloads checkpoints nor claims simulator validation.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def commands(root, checkpoints, episodes, actor_only=False):
    matrix = [
        ('pick-cube', 'pick-cube-1.pt', 'nominal', 'nominal', None),
        ('pick-cube', 'pick-cube-1.pt', 'action_delay', 'fixed', 2),
        ('pick-cube', 'pick-cube-1.pt', 'action_gain', 'fixed', .7),
        ('pick-cube', 'pick-cube-1.pt', 'action_delay', 'abrupt', 2),
        ('pick-cube', 'pick-cube-3.pt', 'action_delay', 'fixed', 2),
        ('stack-cube', 'stack-cube-3.pt', 'action_delay', 'fixed', 2),
    ]
    for task, checkpoint, shift, schedule, value in matrix:
        args = [sys.executable, str(root/'tdmpc2/evaluate_shift.py'), f'task={task}',
            f'checkpoint={checkpoints/checkpoint}', 'episodic=false', 'compile=false', 'model_size=5',
            'analysis=frozen_mechanism', f'dynamics_shift={shift}', f'schedule={schedule}',
            f'evaluation.episodes={episodes}', 'save_video=false', 'seed=1',
            f'mpc={str(not actor_only).lower()}',
            'evaluation=standard' if schedule=='nominal' else 'evaluation=paired']
        if value is not None:
            args.append(f'dynamics_shift.parameter.shifted={value}')
        if schedule=='abrupt': args.append('schedule.switch_step=10')
        yield args


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--episodes', type=int, choices=[5,50], default=5)
    parser.add_argument('--actor-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--report', type=Path, default=Path('mechanism-suite-status.json'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    results=[]
    for command in commands(root,args.checkpoints.resolve(),args.episodes,args.actor_only):
        print(shlex.join(command),flush=True)
        if args.execute:
            result=subprocess.run(command,cwd=root)
            results.append(dict(command=command,returncode=result.returncode))
            args.report.write_text(json.dumps(dict(episodes=args.episodes, runs=results,
                simulator_validation='SEPARATE_REQUIRED_CHECK', analysis_status='PASS' if all(r['returncode']==0 for r in results) else 'FAIL'),indent=2)+'\n')
            if result.returncode: raise SystemExit(result.returncode)


if __name__=='__main__': main()
