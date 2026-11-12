"""Consolidate completed Reacher control snapshots and evaluate only missing ones.

Usage: python -u resume_reacher_control.py /absolute/path/to/reacher/preview-v1
The original control output, adaptation snapshots and checkpoint remain read-only.
"""
import shutil
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[3]
import reacher_pipeline as r

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
out = Path(sys.argv[1])
adapt = out / 'adapt-home-01'
old = out / 'control-home-01'
new = out / 'control-home-02-resumed'
cfg = r.config(root / 'experiments/nsgym/configs/reacher_pipeline_v1.json')
selection = r.selected(out / 'nominal-home-01/selected_reacher_checkpoint.json', cfg)
frozen = out / 'frozen-home-01'
r.gate(frozen, 'frozen', cfg, selection)
if r.read(adapt / 'completion.json')['status'] != 'PASS':
    raise ValueError('Adaptation is incomplete')
manifest = r.read(old / 'manifest.json')
if (manifest['stage'] != 'control' or
        manifest['checkpoint_sha256'] != selection['model_sha256'] or
        Path(manifest['arguments']['runs']).resolve() != adapt.resolve()):
    raise ValueError('Existing control output does not match adaptation')
snapshots = sorted(adapt.glob('*/snapshot-*.pt'))
expected = len(r.conditions(cfg)) * len(cfg['methods']) * len(cfg['adaptation_seeds']) * len(cfg['snapshot_budgets'])
if len(snapshots) != expected:
    raise ValueError(f'Expected {expected} snapshots, found {len(snapshots)}')

identity = dict(original=str(old.resolve()), adaptation=str(adapt.resolve()),
                frozen=str(frozen.resolve()), checkpoint_sha256=selection['model_sha256'],
                snapshots=expected, config_fingerprint=r.fingerprint(cfg),
                source_manifest_sha256=r.sha(old / 'manifest.json'),
                resume_script_sha256=r.sha(__file__))
if new.exists():
    if r.read(new / 'manifest.json') != identity:
        raise ValueError('Resume output belongs to a different input set')
    if (new / 'completion.json').exists():
        record = r.read(new / 'completion.json')
        if record['status'] != 'PASS':
            raise ValueError('Existing resume output has a non-PASS completion marker')
        print(record)
        sys.exit(0)
else:
    new.mkdir()
    r.write(new / 'manifest.json', identity)

hashes = {}
def valid(folder, snapshot):
    marker = folder / 'completion.json'
    if not (marker.is_file() and (folder / 'control.csv').is_file() and
            (folder / 'steps.csv').is_file()):
        return False
    try:
        record = r.read(marker)
    except (ValueError, OSError):
        return False
    if record.get('status') != 'PASS' or record.get('snapshot') != str(snapshot.resolve()):
        return False
    if snapshot not in hashes:
        hashes[snapshot] = r.sha(snapshot)
    return record.get('snapshot_sha256') == hashes[snapshot]

original_valid = sum(
    valid(old / f'{snapshot.parent.name}-{snapshot.stem}', snapshot)
    for snapshot in snapshots
)
print(f'Original valid snapshots: {original_valid}/{expected}; '
      f'need evaluation: {expected - original_valid}', flush=True)

reused = evaluated = skipped = 0
for number, snapshot in enumerate(snapshots, 1):
    name = f'{snapshot.parent.name}-{snapshot.stem}'
    target = new / name
    if valid(target, snapshot):
        skipped += 1
        continue
    if target.exists():
        archive = new.parent / (new.name + '-interrupted')
        archive.mkdir(exist_ok=True)
        target.rename(archive / f'{name}-{time.time_ns()}')
    if valid(old / name, snapshot):
        shutil.copytree(old / name, target)
        reused += 1
    else:
        target.mkdir()
        r.evaluate_snapshot(cfg, selection, snapshot, target, frozen, 'control')
        evaluated += 1
    print(f'PASS {number}/{expected} {name}', flush=True)

if not all(valid(new / f'{snap.parent.name}-{snap.stem}', snap) for snap in snapshots):
    raise ValueError('Consolidated control result is incomplete')
r.done(new, snapshots=expected, reused_this_attempt=reused,
       evaluated_this_attempt=evaluated, skipped_this_attempt=skipped,
       original_output=str(old.resolve()))
print(f'COMPLETE {expected} snapshots', flush=True)
