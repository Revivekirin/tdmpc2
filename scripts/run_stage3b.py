"""Print Stage 3B commands by default; only --execute launches experiments."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import hashlib
from uuid import uuid4
from stage3b_results import selected_results, completed, read


def remaining_commands(planned, suite, output, recovered_root, checkpoint):
    """Plan completed skips and fresh restarts. Planning never changes artifacts."""
    selected = selected_results(output / 'runs', recovered_root, ignore_incomplete=True)
    digest = hashlib.sha256()
    with Path(checkpoint).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    remaining, skipped, archives = [], [], []
    missing_datasets = set()
    for command in planned:
        overrides = dict(arg.split('=', 1) for arg in command if '=' in arg)
        name, method = overrides['dynamics_shift'], overrides['stage3a.mode']
        group = int(overrides['stage3a.seed_set'])
        key = name, method, group
        path = selected.get(key)
        relative = Path(name) / f'set-{group}' / method / '1'
        destination = output / 'runs' / relative
        # A failed recovery is not a completed replacement. Preserve it outside
        # its analysis root too, so a later report cannot count it as a duplicate.
        if recovered_root is not None:
            recovered = recovered_root / relative
            if recovered.exists() and not (recovered / 'complete.json').exists():
                archives.append(recovered)
        if path is not None and completed(path):
            meta = read(path)
            cfg = meta['config']
            if (meta['checkpoint_sha256'] != digest.hexdigest()
                    or cfg['task'] != suite['task']
                    or cfg['model_size'] != suite['model_size']
                    or cfg['episodic'] != suite['episodic']
                    or cfg.get('seed', 1) != 1
                    or cfg['schedule']['type'] != 'fixed'
                    or cfg['stage3b']['run_id'] != output.name
                    or cfg['dynamics_shift']['parameter']['shifted'] != json.loads(overrides['dynamics_shift.parameter.shifted'])
                    or cfg['stage3a']['native_budgets'] != json.loads(overrides['stage3a.native_budgets'])):
                raise ValueError(f'Completed run does not match requested experiment: {path.parent}')
            skipped.append(str(path.parent))
        else:
            dataset = Path(overrides['stage3a.fixed_data_path'])
            if not dataset.is_file():
                missing_datasets.add(str(dataset))
            if destination.exists():
                if (destination / 'complete.json').exists():
                    raise ValueError(f'Completion marker without valid metadata: {destination}')
                archives.append(destination)
            remaining.append(command)
    if missing_datasets:
        raise ValueError('Missing fixed datasets required by pending runs (no runs archived or launched):\n'
                         + '\n'.join(sorted(missing_datasets))
                         + '\nIf a paired method is already complete, restore its original dataset; '
                           'do not substitute newly collected data.')
    return remaining, skipped, archives


def archive_incomplete(paths, archive_id):
    """Same-filesystem rename, outside runs/; no checkpoint or dataset deletion."""
    moves = []
    for source in paths:
        if (source / 'complete.json').exists():
            raise ValueError(f'Run completed after planning; refusing to archive: {source}')
        # Expected layout: <experiment>/runs/<shift>/set-N/<method>/1.
        runs = source.parents[3]
        if runs.name != 'runs' or source.is_symlink():
            raise ValueError(f'Unexpected run directory: {source}')
        target = runs.parent / 'archived-incomplete' / archive_id / source.relative_to(runs)
        if target.exists():
            raise FileExistsError(target)
        moves.append((source, target))
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        print(f'Archived incomplete run: {source} -> {target}', flush=True)


def commands(suite, checkpoint, output, phase, smoke=False):
    for shift in suite['shifts']:
        name = shift['type']
        for seed_set in ([0] if smoke else suite['seed_sets']):
            methods = ['frozen'] if phase == 'collect' else suite['methods']
            for method in methods:
                command = [sys.executable, 'tdmpc2/adapt_shift.py', '--config-name',
                    'stage3c' if method == 'decision_ttt' else 'stage3b',
                    f"task={suite['task']}", f"model_size={suite['model_size']}",
                    f"episodic={str(suite['episodic']).lower()}", f'checkpoint={checkpoint}',
                    f'dynamics_shift={name}', 'schedule=fixed',
                    f"dynamics_shift.parameter.shifted={shift['magnitude']}",
                    f'stage3a.mode={method}', f'stage3a.seed_set={seed_set}',
                    f'stage3b.run_id={output.name}', f'stage3a.output={output}/runs/{name}',
                    f'stage3a.fixed_data_path={output}/data/{name}/set-{seed_set}.pt',
                    'stage3a.native_budgets='+json.dumps(([5,80,240] if smoke else suite['budgets'][1:]), separators=(',',':'))]
                if suite.get('severity_layout'):
                    from stage3b2_audit import value_label
                    label = value_label(shift['magnitude'])
                    dataset = shift.get('fixed_data_path', f'{output}/data/{name}/{label}/set-{seed_set}.pt')
                    command = [arg for arg in command if not arg.startswith('stage3a.fixed_data_path=')]
                    command += [f'stage3a.fixed_data_path={dataset}',
                        f'stage3b.output_dir={output}/runs/{name}/set-{seed_set}/{method}/{label}']
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
    parser.add_argument('--phase', choices=['collect','adapt','severity'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--continue-suite', action='store_true', help='Skip completed runs; archive incomplete runs and restart them from pretrained weights.')
    parser.add_argument('--recovered-root', type=Path, help='Runs directory containing completed checkpoint recoveries.')
    from stage3b_severity import add_arguments, launch
    add_arguments(parser)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if args.phase == 'severity':
        try:
            launch(args, suite)
        except (ValueError, FileExistsError) as error:
            parser.error(str(error))
        sys.exit(0)
    if (args.shift_type or args.shift_value is not None or args.shift_values or args.pilot
            or args.set_id != 0 or args.method != 'frozen' or args.run_id or args.episodes != 10
            or args.log_eef or args.dry_run):
        parser.error('Severity CLI flags require --phase severity; existing collect/adapt uses --suite.')
    output = args.output.resolve()
    if args.smoke and output.name == 'stage3b-generalization':
        parser.error('--smoke requires a separate --output directory')
    if args.continue_suite and (args.phase != 'adapt' or args.smoke):
        parser.error('--continue-suite is for full adaptation runs only')
    if args.recovered_root and not args.continue_suite:
        parser.error('--recovered-root requires --continue-suite')
    if args.execute or args.continue_suite:
        checkpoint = Path(args.checkpoint).expanduser()
        if not args.checkpoint.strip() or not checkpoint.is_file():
            parser.error(f'Pretrained checkpoint not found: {args.checkpoint!r}. '
                         'Set --checkpoint to an existing pretrained .pt file; '
                         'no output files have been created.')
        args.checkpoint = str(checkpoint.resolve())
    planned = list(commands(suite, args.checkpoint, output, args.phase, args.smoke))
    if suite.get('severity_layout') and args.continue_suite:
        parser.error('Severity-layout continuation uses explicit stage3b.resume and a new output_dir.')
    skipped, archives = [], []
    if args.continue_suite:
        planned, skipped, archives = remaining_commands(planned, suite, output, args.recovered_root, args.checkpoint)
        print(f'Completed runs skipped: {len(skipped)}; remaining runs: {len(planned)}', flush=True)
        for path in archives:
            print(f'Will archive incomplete directory: {path}', flush=True)
    if args.execute:
        (output / 'configs').mkdir(parents=True, exist_ok=True)
        effective_suite = dict(suite, run_id=output.name)
        if args.smoke:
            effective_suite.update(shifts=suite['shifts'][:1], seed_sets=[0], budgets=[0,5,80,240])
        prefix = args.phase + ('-continue-' if args.continue_suite else '-') + uuid4().hex[:12]
        with (output / 'configs' / (prefix + '-suite.json')).open('x') as stream:
            json.dump(effective_suite, stream, indent=2)
        with (output / 'configs' / (prefix + '-commands.json')).open('x') as stream:
            json.dump(planned, stream, indent=2)
        if args.continue_suite:
            with (output / 'configs' / (prefix + '-selection.json')).open('x') as stream:
                json.dump(dict(skipped=skipped, recovered_root=str(args.recovered_root),
                               archive_sources=[str(p) for p in archives],
                               archive_id=prefix, incomplete_policy='archive_and_restart_from_pretrained'), stream, indent=2)
            archive_incomplete(archives, prefix)
    for command in planned:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)
