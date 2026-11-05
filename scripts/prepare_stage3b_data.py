"""Audit shared datasets; collect only missing files with --execute.

Completed runs bind their original dataset SHA-256. A recollected candidate is
installed only if it matches that hash; mismatches stay outside data/ for review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4
from run_stage3b import commands
from stage3b_results import completed, read, selected_results


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def plan_data(suite, checkpoint, output):
    checkpoint_hash = sha256(checkpoint)
    expected = {}
    magnitudes = {s['type']: s['magnitude'] for s in suite['shifts']}
    for (shift, method, group), path in selected_results(output/'runs', ignore_incomplete=True).items():
        if not completed(path):
            continue
        meta = read(path)
        cfg = meta['config']
        if (shift not in magnitudes or group not in suite['seed_sets'] or method not in suite['methods']
                or meta['checkpoint_sha256'] != checkpoint_hash
                or cfg['task'] != suite['task'] or cfg['model_size'] != suite['model_size']
                or cfg['episodic'] != suite['episodic'] or cfg['schedule']['type'] != 'fixed'
                or cfg['dynamics_shift']['parameter']['shifted'] != magnitudes[shift]):
            raise ValueError(f'Completed run does not match requested suite/checkpoint: {path}')
        key = shift, group
        value = meta.get('fixed_data_sha256')
        if not value or (key in expected and expected[key] != value):
            raise ValueError(f'Missing/conflicting fixed-data hashes for {key}')
        expected[key] = value
    plan = []
    for command in commands(suite, str(checkpoint), output, 'collect'):
        args = dict(arg.split('=',1) for arg in command if '=' in arg)
        key = args['dynamics_shift'], int(args['stage3a.seed_set'])
        path = Path(args['stage3a.fixed_data_path'])
        reference = expected.get(key)
        actual = sha256(path) if path.is_file() else None
        if path.exists() and not path.is_file():
            raise ValueError(f'Dataset path is not a file: {path}')
        if actual is not None and reference is not None and actual != reference:
            raise ValueError(f'Existing dataset differs from completed results: {path}. Restore original; no files changed.')
        plan.append(dict(shift=key[0], seed_set=key[1], path=str(path), expected_sha256=reference,
                         status='EXISTS' if actual is not None else 'MISSING', command=command))
    return plan


def install_candidate(candidate, target, expected_hash):
    actual = sha256(candidate)
    if expected_hash is not None and actual != expected_hash:
        return dict(status='HASH_MISMATCH', candidate=str(candidate), actual_sha256=actual,
                    expected_sha256=expected_hash)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Both paths are under the same experiment directory. link() fails if the
    # target appeared concurrently and avoids copying bytes on a near-full disk.
    os.link(candidate, target)
    candidate.unlink()
    return dict(status='INSTALLED', path=str(target), sha256=actual)


def execute_plan(plan, output):
    attempt = uuid4().hex[:12]
    manifest = output/'configs'/f'prepare-data-{attempt}.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x') as stream:
        json.dump(plan, stream, indent=2)
    mismatches = []
    with (output/'configs'/f'prepare-data-{attempt}-results.jsonl').open('x') as log:
        for entry in plan:
            if entry['status'] == 'EXISTS':
                continue
            target = Path(entry['path'])
            candidate = output/'data-candidates'/attempt/entry['shift']/target.name
            command = [f'stage3a.fixed_data_path={candidate}' if arg.startswith('stage3a.fixed_data_path=')
                       else arg for arg in entry['command']]
            subprocess.run(command, check=True)
            result = install_candidate(candidate, target, entry['expected_sha256'])
            log.write(json.dumps(result)+'\n'); log.flush()
            print(json.dumps(result), flush=True)
            if result['status'] == 'HASH_MISMATCH':
                mismatches.append(str(target))
    if mismatches:
        raise ValueError('Recollected data did not match completed results. Candidates preserved; '
                         'restore original datasets before paired runs, or explicitly rerun all methods '
                         'for those conditions with a new shared dataset:\n'+'\n'.join(mismatches))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=Path('benchmarks/stage3b_suite.json'))
    parser.add_argument('--output', type=Path, default=Path('results/stage3b-generalization'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    suite = read(args.suite)
    output = args.output.resolve()
    plan = plan_data(suite, checkpoint, output)
    for entry in plan:
        suffix = ' (must match completed-run hash)' if entry['expected_sha256'] else ''
        print(f"{entry['status']}: {entry['path']}{suffix}")
    print(f"Existing: {sum(e['status']=='EXISTS' for e in plan)}; missing: {sum(e['status']=='MISSING' for e in plan)}")
    if args.execute:
        execute_plan(plan, output)
