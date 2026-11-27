"""Select original or explicitly supplied completed recovery results without writes."""
import json


def read(path):
    return json.loads(path.read_text())


def run_key(metadata):
    cfg = metadata['config']
    return (cfg['dynamics_shift']['parameter']['type'], metadata['active_method'],
            cfg['stage3a']['seed_set'])


def completed(path):
    marker = path.parent / 'complete.json'
    evaluations = path.parent / 'evaluations.jsonl'
    if not marker.exists():
        return False
    meta = read(path)
    budgets = [0] + meta['config']['stage3a']['native_budgets']
    rows = [json.loads(line) for line in evaluations.read_text().splitlines()]
    if ([r['native_budget'] for r in rows] != budgets
            or read(marker)['adaptation_native_steps'] != budgets[-1]):
        raise ValueError(f'Invalid completion artifacts: {path.parent}')
    return True


def provenance(metadata):
    cfg = dict(metadata['config'])
    cfg.pop('work_dir', None)
    cfg['stage3a'] = {k:v for k,v in cfg['stage3a'].items() if k != 'output'}
    cfg['stage3b'] = {k:v for k,v in cfg.get('stage3b', {}).items() if k != 'resume'}
    return cfg, metadata['checkpoint_sha256'], metadata['fixed_data_sha256']


def selected_results(root, recovered_root=None, ignore_incomplete=False):
    selected = {}
    for path in sorted(root.rglob('metadata.json')):
        try:
            meta = read(path)
            key = run_key(meta)
        except (ValueError, KeyError, TypeError):
            if ignore_incomplete and not (path.parent / 'complete.json').exists():
                continue
            raise
        if key in selected:
            raise ValueError(f'Duplicate original result: {key}')
        selected[key] = path
    if recovered_root is not None:
        if not recovered_root.is_dir():
            raise ValueError(f'Recovery root does not exist: {recovered_root}')
        recovered = set()
        for path in sorted(recovered_root.rglob('metadata.json')):
            if ignore_incomplete and not (path.parent / 'complete.json').exists():
                continue
            meta = read(path)
            key = run_key(meta)
            if key in recovered:
                raise ValueError(f'Duplicate recovery result: {key}')
            recovered.add(key)
            if not completed(path):
                raise ValueError(f'Recovery is incomplete: {path.parent}')
            if key in selected:
                original = selected[key]
                if original.resolve() == path.resolve():
                    raise ValueError('Original and recovery roots overlap.')
                if provenance(read(original)) != provenance(meta):
                    raise ValueError(f'Recovery provenance mismatch: {key}')
                if completed(original):
                    raise ValueError(f'Both original and recovery are complete: {key}')
            selected[key] = path
    return selected
