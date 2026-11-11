#!/usr/bin/env python3
"""One stage-gated entry point for a final-snapshot TTT benchmark.

Benchmark-specific physics, checkpoint and statistics remain in the evaluator.
This runner owns path resolution, ordering, completion gates and safe resume.
Without --execute it only prints the plan and never loads CUDA or runs physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / 'experiments/nsgym/configs/halfcheetah_ttt_final_benchmark.json'
STAGES = ('test', 'audit', 'smoke-prediction', 'smoke-shifted',
          'full-prediction', 'full-shifted', 'aggregate')
OPTIONAL = ('nominal', 'aggregate-retention')
OUTPUTS = {'audit': 'audit', 'smoke-prediction': 'smoke-prediction',
           'smoke-shifted': 'smoke-shifted', 'full-prediction': 'full-prediction',
           'full-shifted': 'full-shifted', 'aggregate': 'summary',
           'nominal': 'full-nominal', 'aggregate-retention': 'summary-with-retention'}
REQUIRES = {'audit': (), 'smoke-prediction': ('audit',),
            'smoke-shifted': ('audit',),
            'full-prediction': ('audit', 'smoke-prediction', 'smoke-shifted'),
            'full-shifted': ('audit', 'smoke-prediction', 'smoke-shifted'),
            'aggregate': ('audit', 'full-prediction', 'full-shifted'),
            'nominal': ('audit', 'smoke-prediction', 'smoke-shifted'),
            'aggregate-retention': ('audit', 'full-prediction', 'full-shifted', 'nominal')}


def load_profile(path, root):
    profile = json.loads(Path(path).read_text())
    required = {'benchmark_id', 'evaluator', 'checkpoint', 'checkpoint_sha256',
                'reference', 'runs', 'output', 'reference_files',
                'run_completion_stages', 'tests'}
    if set(profile) != required:
        raise ValueError(f'Profile fields differ: missing={required-set(profile)}, extra={set(profile)-required}')
    paths = {'root': root}
    for key in ('evaluator', 'checkpoint', 'reference', 'runs', 'output'):
        relative = Path(profile[key])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'{key} must be relative to project root')
        paths[key] = (root / relative).resolve()
    for key in ('tests', 'reference_files', 'run_completion_stages'):
        if not isinstance(profile[key], list) or not profile[key]:
            raise ValueError(f'{key} must be a nonempty list')
        for item in profile[key]:
            relative = Path(item)
            if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
                raise ValueError(f'{key} entries must be relative paths')
    if len(profile['checkpoint_sha256']) != 64 or any(c not in '0123456789abcdef' for c in profile['checkpoint_sha256']):
        raise ValueError('checkpoint_sha256 must be lowercase SHA256 hex')
    return profile, paths


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def preflight(profile, paths):
    needed = [paths['evaluator'], paths['checkpoint'] / 'model.pt']
    needed += [paths['reference'] / name for name in profile['reference_files']]
    needed += [paths['runs'] / name / 'completion.json' for name in profile['run_completion_stages']]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required inputs missing: ' + ', '.join(missing))
    if file_hash(paths['checkpoint'] / 'model.pt') != profile['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 differs from preregistered profile')
    for stage in profile['run_completion_stages']:
        record = json.loads((paths['runs'] / stage / 'completion.json').read_text())
        if record.get('status') != 'PASS':
            raise ValueError(f'{stage} is not PASS')


def completion(path):
    record = path / 'completion.json'
    return json.loads(record.read_text()).get('status') if record.is_file() else None


def command(stage, profile, paths, python=sys.executable):
    if stage == 'test':
        return [python, '-m', 'unittest', *[str(paths['root'] / x) for x in profile['tests']]]
    cmd = [python, str(paths['evaluator']),
           'aggregate' if stage.startswith('aggregate') else 'audit' if stage == 'audit' else 'evaluate',
           '--checkpoint', str(paths['checkpoint']), '--runs', str(paths['runs']),
           '--reference', str(paths['reference']), '--output', str(paths['output'] / OUTPUTS[stage])]
    if stage not in ('audit',):
        cmd += ['--audit', str(paths['output'] / 'audit/audit.json')]
    if stage.startswith('aggregate'):
        cmd += ['--prediction', str(paths['output'] / 'full-prediction'),
                '--shifted', str(paths['output'] / 'full-shifted')]
        if stage == 'aggregate-retention':
            cmd += ['--nominal', str(paths['output'] / 'full-nominal')]
    elif stage != 'audit':
        mode = 'nominal' if stage == 'nominal' else 'prediction' if stage.endswith('prediction') else 'shifted'
        cmd += ['--mode', mode, '--subset', 'smoke' if stage.startswith('smoke') else 'full']
        if stage.startswith('full') or stage == 'nominal':
            cmd += ['--smoke-prediction', str(paths['output'] / 'smoke-prediction'),
                    '--smoke-shifted', str(paths['output'] / 'smoke-shifted')]
    return cmd


def stage_status(stage, paths):
    return 'READY' if stage == 'test' else completion(paths['output'] / OUTPUTS[stage]) or 'NOT_RUN'


def verify_pass(stage, profile, paths):
    root = paths['output'] / OUTPUTS[stage]
    if stage == 'audit':
        audit = json.loads((root / 'audit.json').read_text())
        expected = {'checkpoint': str(paths['checkpoint']), 'runs': str(paths['runs']),
                    'reference': str(paths['reference']), 'checkpoint_sha256': profile['checkpoint_sha256']}
        if audit.get('status') != 'PASS' or any(audit.get(k) != v for k, v in expected.items()):
            raise ValueError('Existing audit does not match benchmark profile')
        return
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('checkpoint_sha256') != profile['checkpoint_sha256']:
        raise ValueError(f'Existing {stage} checkpoint differs')
    audit_path = paths['output'] / 'audit/audit.json'
    if manifest.get('audit_sha256') != file_hash(audit_path):
        raise ValueError(f'Existing {stage} audit differs')
    if stage.startswith('aggregate'):
        if manifest.get('prediction') != str(paths['output'] / 'full-prediction') or \
           manifest.get('shifted') != str(paths['output'] / 'full-shifted'):
            raise ValueError('Existing aggregate inputs differ')
        expected_nominal = str(paths['output'] / 'full-nominal') if stage == 'aggregate-retention' else None
        if manifest.get('nominal') != expected_nominal:
            raise ValueError('Existing aggregate nominal input differs')
    else:
        mode = 'nominal' if stage == 'nominal' else 'prediction' if stage.endswith('prediction') else 'shifted'
        subset = 'smoke' if stage.startswith('smoke') else 'full'
        if manifest.get('mode') != mode or manifest.get('subset') != subset:
            raise ValueError(f'Existing {stage} identity differs')


def run_stage(stage, profile, paths, execute):
    cmd = command(stage, profile, paths)
    status = stage_status(stage, paths)
    print(f'{stage}: {status}', flush=True)
    if not execute:
        print(' '.join(shlex.quote(x) for x in cmd), flush=True)
        return
    if stage == 'test':
        print(' '.join(shlex.quote(x) for x in cmd), flush=True)
        subprocess.run(cmd, cwd=paths['root'], check=True)
        return
    for prerequisite in REQUIRES[stage]:
        if stage_status(prerequisite, paths) != 'PASS':
            raise RuntimeError(f'{stage} requires PASS {prerequisite}')
    if status == 'PASS':
        verify_pass(stage, profile, paths)
        print(f'SKIP PASS {stage}', flush=True)
        return
    if status != 'NOT_RUN' or (paths['output'] / OUTPUTS[stage]).exists():
        if stage.startswith(('smoke-', 'full-')) or stage == 'nominal':
            if not (paths['output'] / OUTPUTS[stage] / 'manifest.json').is_file():
                raise RuntimeError(f'{stage} has no manifest to resume; select a new output profile')
            cmd.append('--resume')
        else:
            raise RuntimeError(f'{stage} output exists but is not PASS; select a new output profile')
    print(' '.join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=paths['root'], check=True)
    if stage_status(stage, paths) != 'PASS':
        raise RuntimeError(f'{stage} finished without PASS; inspect completion/failure records')
    verify_pass(stage, profile, paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=DEFAULT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('primary', *STAGES, *OPTIONAL), default='primary')
    parser.add_argument('--execute', action='store_true', help='Run selected stage(s); omission only prints commands')
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    profile, paths = load_profile(args.profile, root)
    stages = STAGES if args.stage == 'primary' else (args.stage,)
    print(f"benchmark={profile['benchmark_id']} project_root={root}", flush=True)
    if args.execute and args.stage != 'test':
        preflight(profile, paths)
        paths['output'].mkdir(parents=True, exist_ok=True)
    for stage in stages:
        run_stage(stage, profile, paths, args.execute)


if __name__ == '__main__':
    main()
