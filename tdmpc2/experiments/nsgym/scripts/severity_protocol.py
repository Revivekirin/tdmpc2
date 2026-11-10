"""Dependency-free CLI and resolved protocol shared by runners and command generation.

The three .yaml protocol files use JSON syntax, a YAML subset, so --help and
--dry-run work before the scientific Python stack is installed.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'experiments/nsgym/configs'
STAGES = ('environment', 'inventory', 'manifest', 'validate', 'bank', 'physical',
          'latent', 'planning', 'frozen', 'nonstationary', 'summarize')


def load(name):
    return json.loads((CONFIG / name).read_text())


def parser():
    p = argparse.ArgumentParser(description='Frozen 750k NS-Gym physical severity pipeline; no optimizer updates.')
    p.add_argument('stage', choices=STAGES)
    p.add_argument('--checkpoint', type=Path, default=ROOT/load('physical_severity_grid.yaml')['primary_checkpoint'])
    p.add_argument('--checkpoint-role', choices=['primary', 'sensitivity'], default='primary')
    p.add_argument('--shift-family', choices=['all', 'mass', 'damping'], default='all')
    p.add_argument('--parameter', help='One official scalar parameter listed in the preregistered grid')
    p.add_argument('--severity', type=float, nargs='+', help='Subset of preregistered engineering grid')
    p.add_argument('--schedule', choices=load('nonstationary_schedules.yaml')['schedules'], default='fixed')
    p.add_argument('--seeds', type=int, nargs='+', help='Subset of the stage-specific seed split')
    p.add_argument('--seed-split',choices=['calibration','validation','evaluation','sensitivity'],help='Explicit split for offline diagnostics or a paired sensitivity study')
    p.add_argument('--output', type=Path, required=True, help='New directory, never overwritten')
    p.add_argument('--input', type=Path, help='Bank, response data, or raw evaluation directory')
    p.add_argument('--gate', type=Path, help='Completed prerequisite stage (physical for frozen; frozen for nonstationary)')
    p.add_argument('--physical-reference',type=Path,help='Physical response directory for response/control comparison plot')
    p.add_argument('--policy', choices=['primary', 'exploration'], default='primary', help='Bank source only')
    p.add_argument('--smoke', action='store_true', help='Bounded 12-step integration experiment; not research evidence')
    p.add_argument('--allow-optional-large',action='store_true',help='Explicit execution gate for large S3-S5 after separate experiment approval')
    p.add_argument('--dry-run', action='store_true', help='Print resolved manifest and budgets; do not simulate or write')
    return p


def resolve(args):
    grid = load('physical_severity_grid.yaml'); splits = load('evaluation_seed_splits.yaml')
    sched = load('nonstationary_schedules.yaml')
    flat = [s for values in splits.values() for s in values]
    if len(flat) != len(set(flat)):
        raise ValueError('Seed splits overlap')
    split = ('sensitivity' if args.checkpoint_role == 'sensitivity' else
             'validation' if args.stage in ('validate', 'environment', 'inventory') else
             'evaluation' if args.stage in ('frozen', 'nonstationary') else 'calibration')
    if args.seed_split:
        if args.stage in ('bank','physical') and args.seed_split!='calibration':
            raise ValueError('Bank/physical require calibration seeds')
        if args.stage in ('frozen','nonstationary') and args.seed_split not in ('evaluation','sensitivity'):
            raise ValueError('Frozen control requires held-out evaluation/sensitivity seeds')
        split=args.seed_split
    source_manifest=None
    if args.input and args.stage in ('latent','planning','summarize') and (args.input/'manifest.json').exists():
        source_manifest=json.loads((args.input/'manifest.json').read_text())
        if args.seed_split and args.seed_split!=source_manifest['seed_split']:
            raise ValueError('Requested seed split differs from source data')
        split=source_manifest['seed_split']
    seeds = args.seeds or (source_manifest['seeds'] if source_manifest else splits[split][:1] if args.smoke else splits[split])
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= set(splits[split]):
        raise ValueError(f'Seeds must be a unique subset of {split}: {splits[split]}')
    families = grid['families'] if args.shift_family == 'all' else {args.shift_family:grid['families'][args.shift_family]}
    parameters = [(f, p) for f, ps in families.items() for p in ps if args.parameter is None or p == args.parameter]
    if not parameters:
        raise ValueError('Parameter/family is not in the preregistered grid')
    severities = args.severity or ([1., 1.5] if args.smoke else grid['severities'])
    if len(severities) != len(set(severities)) or not set(severities) <= set(grid['severities']):
        raise ValueError('Severity must be a unique subset of the preregistered grid')
    horizon = 12 if args.smoke else grid['horizon']
    timing = dict(onset=3, duration=3, restore=9, period=3) if args.smoke else {k:sched[k] for k in ('onset','duration','restore','period')}
    cp = args.checkpoint.resolve()
    if cp.name == 'model.pt':
        cp = cp.parent
    conditions = [dict(family=f, parameter=p, severity=k, schedule=args.schedule) for f,p in parameters for k in severities]
    if args.stage=='bank':
        conditions=[dict(family='mass',parameter='torso_mass',severity=1.,schedule='nominal')]
    samples = 2 if args.smoke else grid['samples_per_seed']
    probe = 3 if args.smoke else grid['probe_horizon']
    n = len(conditions); episodes = n * len(seeds)
    transitions = episodes * horizon
    if args.stage == 'bank':
        episodes = len(seeds); transitions = episodes * horizon
    elif args.stage == 'physical':
        # Each nominal condition is repeated for restoration verification.
        episodes = len(seeds) * samples * (n + sum(c['severity']==1 for c in conditions))
        transitions = episodes * probe
    elif args.stage == 'validate':
        episodes *= 2; transitions = episodes * horizon
    elif args.stage in ('manifest','latent','planning','summarize','environment','inventory'):
        episodes = transitions = 0
        if args.stage=='environment':episodes,transitions=3,36
    resolved = dict(stage=args.stage, checkpoint=str(cp), checkpoint_role=args.checkpoint_role,
                    expected_checkpoint_sha256=grid['primary_sha256'] if args.checkpoint_role=='primary' else grid['secondary_sha256'],
                    seed_split=split, seeds=seeds, conditions=conditions, horizon=horizon,
                    probe_horizon=probe, samples_per_seed=samples, schedule_timing=timing,
                    clock=sched['clock'], persistent_params=False, policy=args.policy,
                    smoke=args.smoke or bool(source_manifest and source_manifest['smoke']),
                    allow_optional_large=args.allow_optional_large, optimizer_updates=0, output=str(args.output.resolve()),
                    input=str(args.input.resolve()) if args.input else None,
                    gate=str(args.gate.resolve()) if args.gate else None,
                    physical_reference=str(args.physical_reference.resolve()) if args.physical_reference else None,
                    budget=dict(conditions=n, seeds=len(seeds), episodes_or_probe_rollouts=episodes,
                                maximum_transitions=transitions),
                    protocol=dict(grid=grid, seed_splits=splits, schedules=sched))
    resolved['protocol_sha256'] = hashlib.sha256(json.dumps(resolved['protocol'],sort_keys=True).encode()).hexdigest()
    if source_manifest and args.stage in ('latent','summarize'):
        if args.severity or args.parameter or args.shift_family!='all':
            raise ValueError('Latent/summarize process the complete input dataset; filter in the upstream data stage')
        resolved['conditions']=source_manifest['conditions']
        resolved['schedule_timing']=source_manifest['schedule_timing']
        resolved['horizon']=source_manifest['horizon']
        resolved['budget']['conditions']=len(source_manifest['conditions'])
    return resolved
