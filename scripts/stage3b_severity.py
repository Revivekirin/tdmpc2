"""Frozen severity planning, invoked by the existing run_stage3b.py CLI."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tdmpc2'))
from common.severity_protocol import DEFAULTS, PILOT, PROTOCOL, severity, run_path, digest_file


def add_arguments(parser):
    parser.add_argument('--shift-type', choices=list(DEFAULTS), nargs='+')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--shift-value')
    group.add_argument('--shift-values', nargs='+')
    group.add_argument('--pilot', action='store_true', help='Use pilot values for selected shifts (all shifts if omitted).')
    parser.add_argument('--set-id', type=int, default=0)
    parser.add_argument('--method', choices=['frozen','dynamics_only'], default='frozen')
    parser.add_argument('--run-id')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--log-eef', action='store_true')
    parser.add_argument('--dry-run', action='store_true')


def plan(args, suite):
    if args.method != 'frozen':
        raise ValueError('Stage 3B-1 supports --method frozen only; no TTT.')
    if args.continue_suite or args.recovered_root or args.smoke:
        raise ValueError('Severity evaluation cannot use adaptation resume/recovery/smoke flags; use --episodes 1.')
    if args.execute and args.dry_run:
        raise ValueError('--execute and --dry-run are mutually exclusive.')
    if args.set_id < 0 or not 1 <= args.episodes <= 10000:
        raise ValueError('set-id >= 0 and episodes in [1, 10000] are required.')
    kinds = args.shift_type or list(DEFAULTS)
    if len(kinds) != len(set(kinds)):
        raise ValueError('Duplicate shift types.')
    explicit = [args.shift_value] if args.shift_value is not None else args.shift_values
    if explicit is not None and len(kinds) != 1:
        raise ValueError('Explicit severity values require exactly one --shift-type.')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+uuid4().hex[:8]
    seeds = list(range(4000+10000*args.set_id,4000+10000*args.set_id+args.episodes))
    entries = []
    for kind in kinds:
        values = [severity(kind, v) for v in (explicit if explicit is not None else PILOT[kind] if args.pilot else [DEFAULTS[kind]])]
        if len(values) != len(set(values)):
            raise ValueError('Duplicate severity values.')
        for value in values:
            path = run_path(args.output.resolve(), kind, value, args.set_id, run_id)
            command = [sys.executable,'tdmpc2/evaluate_severity.py',f"task={suite['task']}",
                       f"model_size={suite['model_size']}",f"episodic={str(suite['episodic']).lower()}",
                       f'checkpoint={args.checkpoint}',f'dynamics_shift={kind}','schedule=fixed',
                       f'dynamics_shift.parameter.shifted={value}',f'stage3a.seed_set={args.set_id}',
                       'stage3a.evaluation_seeds='+json.dumps(seeds,separators=(',',':')),
                       f'severity.output={json.dumps(str(path))}', f'severity.run_id={run_id}',
                       f'severity.log_eef={str(args.log_eef).lower()}']
            entries.append(dict(shift_type=kind,shift_value=value,set_id=args.set_id,
                                seeds=seeds,path=str(path),command=command))
    return dict(protocol=PROTOCOL,run_id=run_id,task=suite['task'],model_size=suite['model_size'],
                episodic=suite['episodic'],log_eef=args.log_eef,entries=entries)


def launch(args, suite):
    if args.execute:
        args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    manifest = plan(args,suite)
    for entry in manifest['entries']:
        print(shlex.join(entry['command']),flush=True)
    print(f"{len(manifest['entries'])} Frozen conditions; {args.episodes} shifted + {args.episodes} nominal episodes per condition. No adaptation.",flush=True)
    if not args.execute:
        return
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f'Pretrained checkpoint not found: {checkpoint}')
    # Preflight all paths before any simulation. Never resume/replace existing evaluations.
    for entry in manifest['entries']:
        if Path(entry['path']).exists():
            raise FileExistsError(f"Use a new --run-id: {entry['path']}")
    manifest['checkpoint_sha256'] = digest_file(checkpoint)
    destination = args.output.resolve()/'severity_sweep'/manifest['run_id']
    destination.mkdir(parents=True,exist_ok=False)
    manifest_path = destination/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    try:
        for entry in manifest['entries']:
            subprocess.run(entry['command'],check=True)
    finally:
        from analyze_stage3b_severity import summarize
        summarize([manifest_path],destination, args.output.resolve()/'severity_sweep'/'summary.csv')
