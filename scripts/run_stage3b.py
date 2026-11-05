"""Print Stage 3B commands by default; only --execute launches experiments."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def commands(suite, checkpoint, output, phase, smoke=False):
    for shift in suite['shifts']:
        name = shift['type']
        for seed_set in ([0] if smoke else suite['seed_sets']):
            methods = ['frozen'] if phase == 'collect' else suite['methods']
            for method in methods:
                command = [sys.executable, 'tdmpc2/adapt_shift.py', '--config-name', 'stage3b',
                    f"task={suite['task']}", f"model_size={suite['model_size']}",
                    f"episodic={str(suite['episodic']).lower()}", f'checkpoint={checkpoint}',
                    f'dynamics_shift={name}', 'schedule=fixed',
                    f"dynamics_shift.parameter.shifted={shift['magnitude']}",
                    f'stage3a.mode={method}', f'stage3a.seed_set={seed_set}',
                    f'stage3b.run_id={output.name}', f'stage3a.output={output}/runs/{name}',
                    f'stage3a.fixed_data_path={output}/data/{name}/set-{seed_set}.pt',
                    'stage3a.native_budgets='+json.dumps(([5,80,240] if smoke else suite['budgets'][1:]), separators=(',',':'))]
                if phase == 'collect':
                    command += ['stage3a.fixed_data_collect=true']
                if smoke:
                    command += ['stage3a.evaluation_seeds=[4000]', 'stage3a.fixed_data_seeds=[5000]',
                                'stage3a.updates_per_episode=2']
                yield command
                # One representative real smoke, not a whole shift/seed grid.
        if smoke:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=Path('benchmarks/stage3b_suite.json'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, default=Path('results/stage3b-generalization'))
    parser.add_argument('--phase', choices=['collect','adapt'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    output = args.output.resolve()
    if args.smoke and output.name == 'stage3b-generalization':
        parser.error('--smoke requires a separate --output directory')
    if args.execute:
        (output / 'configs').mkdir(parents=True, exist_ok=True)
        effective_suite = dict(suite, run_id=output.name)
        if args.smoke:
            effective_suite.update(shifts=suite['shifts'][:1], seed_sets=[0], budgets=[0,5,80,240])
        with (output / 'configs' / (args.phase + '-suite.json')).open('x') as stream:
            json.dump(effective_suite, stream, indent=2)
        with (output / 'configs' / (args.phase + '-commands.json')).open('x') as stream:
            json.dump(list(commands(suite, args.checkpoint, output, args.phase, args.smoke)), stream, indent=2)
    for command in commands(suite, args.checkpoint, output, args.phase, args.smoke):
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)
