"""Run pre-shift pairing diagnostics in fresh processes and sequential contexts.

Example: python scripts/validate_pairing_matrix.py --seeds 2,3,7 --repeats 2 \
  --out-dir results/pairing-root-cause task=pick-cube checkpoint=/path/pick-cube-1.pt \
  episodic=false compile=false model_size=5 dynamics_shift=action_delay \
  schedule=abrupt dynamics_shift.parameter.shifted=2 evaluation=paired \
  evaluation.episodes=1 seed=1 mpc=true analysis=frozen_mechanism
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='2,3,7')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--out-dir', type=Path, default=Path('results/pairing-root-cause'))
    parser.add_argument('--deterministic-algorithms', action='store_true')
    args, overrides = parser.parse_known_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    seeds = [int(s) for s in args.seeds.split(',') if s]
    if not seeds or any(s < 1 for s in seeds) or len(set(seeds)) != len(seeds):
        parser.error('--seeds must be unique positive integers')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    validator = Path(__file__).resolve().with_name('validate_pairing.py')
    for seed in seeds:
        for mode in (False, True):
            for context in ('fresh', 'sequential'):
                for repeat in range(args.repeats):
                    label = f'seed{seed}-{context}-eval{int(mode)}-repeat{repeat + 1}'
                    path = args.out_dir / f'{label}.json'
                    command = [sys.executable, str(validator), '--seed', str(seed),
                               '--out', str(path)]
                    if context == 'sequential' and seed > 1:
                        command += ['--warmup-seeds', ','.join(str(s) for s in range(1, seed))]
                    if args.deterministic_algorithms:
                        command.append('--deterministic-algorithms')
                    command += overrides + [f'+evaluation.policy_eval_mode={str(mode).lower()}']
                    print(f'START {label}', flush=True)
                    completed = subprocess.run(command, check=False)
                    row = dict(seed=seed, context=context, policy_eval_mode=mode,
                               repeat=repeat + 1, returncode=completed.returncode,
                               file=str(path.resolve()))
                    if completed.returncode == 0:
                        report = json.loads(path.read_text())
                        row['first_divergence'] = report['first_divergence']
                        row['reached_switch'] = report['reached_switch']
                    results.append(row)
                    (args.out_dir / 'index.json').write_text(json.dumps(dict(
                        schema_version=1, seeds=seeds, repeats=args.repeats,
                        deterministic_algorithms=args.deterministic_algorithms,
                        runs=results), indent=2) + '\n')
                    print(f'DONE {label} exit={completed.returncode}', flush=True)
    print((args.out_dir / 'index.json').resolve())
    if any(row['returncode'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
