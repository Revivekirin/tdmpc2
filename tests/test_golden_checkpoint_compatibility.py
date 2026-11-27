"""Read-only historical checkpoint inventory and temporary serialization round-trips."""
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
GOLDEN = PROJECT / 'tests/fixtures/research_checkpoint_compatibility.json'
NOMINAL = 'nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000/model.pt'
if os.environ.get('TD_MPC2_GOLDEN_RESULTS_ROOT'):
    RESULTS_ROOT = Path(os.environ['TD_MPC2_GOLDEN_RESULTS_ROOT']).expanduser()
else:
    candidates = [PROJECT.parent/'results', PROJECT/'results']
    RESULTS_ROOT = next((root for root in candidates if (root/NOMINAL).is_file()), candidates[0])
SOURCES = {
    'nominal_model': 'results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000/model.pt',
    'nominal_optimizers': 'results/nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000/optimizers.pt',
    'decision_snapshot': 'results/nsgym/dynamics-only-750k-v1/runs/fixed-decision/bthigh_damping-k0.5-decision-seed42000001/snapshot-0003000.pt',
    'dynamics_q_snapshot': 'results/nsgym/dynamics-q-750k-v1/runs/fixed-decision/bthigh_damping-k0.5-dynamics-q-seed42000001/snapshot-0003000.pt',
    'joint_snapshot': 'results/nsgym/full-joint-750k-v1/runs-fixed/bthigh_damping-k0.5-full-joint-seed42000001/snapshot-0003000.pt',
}


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(value):
    h = hashlib.sha256()
    count = 0

    def visit(item):
        nonlocal count
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(f'T{tensor.dtype}:{tuple(tensor.shape)}:'.encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            count += 1
        elif isinstance(item, np.ndarray):
            h.update(f'N{item.dtype}:{item.shape}:'.encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            h.update(b'{')
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
            h.update(b'}')
        elif isinstance(item, (list, tuple)):
            h.update(b'[' if isinstance(item, list) else b'(')
            for subitem in item:
                visit(subitem)
            h.update(b']' if isinstance(item, list) else b')')
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest(), count


def inspect_source(label, relative):
    source = RESULTS_ROOT / Path(relative).relative_to('results')
    if not source.is_file():
        return dict(status='BLOCKED_MISSING_SOURCE', path=relative)
    before_hash = file_hash(source)
    payload = torch.load(source, map_location='cpu', weights_only=False)
    before_tree, tensors = tree_digest(payload)
    with TemporaryDirectory(prefix='tdmpc2-golden-checkpoint-') as temporary:
        copy = Path(temporary) / 'roundtrip.pt'
        torch.save(payload, copy)
        restored = torch.load(copy, map_location='cpu', weights_only=False)
        after_tree, after_tensors = tree_digest(restored)
        assert after_tree == before_tree and after_tensors == tensors
    assert file_hash(source) == before_hash, 'Historical source changed'
    return dict(status='PASS', path=relative, source_sha256=before_hash,
                source_bytes=source.stat().st_size, top_level_type=type(payload).__name__,
                top_level_keys=sorted(map(str, payload)) if isinstance(payload, dict) else None,
                tree_sha256=before_tree, tensor_count=tensors, temporary_roundtrip_equal=True,
                source_unchanged=True)


def inspect_all():
    return dict(schema=1, torch_version=torch.__version__.split('+')[0],
                sources={name:inspect_source(name,path) for name,path in SOURCES.items()})


@pytest.mark.parametrize('label', sorted(SOURCES))
def test_historical_checkpoint_roundtrip(label):
    expected=json.loads(GOLDEN.read_text())
    assert expected['torch_version']==torch.__version__.split('+')[0]
    result=inspect_source(label,SOURCES[label])
    if result['status']=='BLOCKED_MISSING_SOURCE':
        pytest.skip(f"Historical checkpoint unavailable: {result['path']}")
    assert result==expected['sources'][label]


def test_nominal_metadata_and_adaptation_parent_hashes():
    checkpoint=RESULTS_ROOT/'nsgym/nominal-server4-20261108-nohup-01/checkpoint-000750000'
    required=[checkpoint/'metadata.json', checkpoint/'model.pt', checkpoint/'optimizers.pt']
    required.extend(RESULTS_ROOT/Path(SOURCES[label]).relative_to('results') for label in
                    ('decision_snapshot','dynamics_q_snapshot','joint_snapshot'))
    missing=[str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip('Historical checkpoint unavailable: '+', '.join(missing))
    metadata=json.loads((checkpoint/'metadata.json').read_text())
    nominal_model_hash=file_hash(checkpoint/'model.pt')
    assert metadata['model_sha256']==nominal_model_hash
    assert metadata['optimizer_sha256']==file_hash(checkpoint/'optimizers.pt')
    for label in ('decision_snapshot','dynamics_q_snapshot','joint_snapshot'):
        payload=torch.load(RESULTS_ROOT/Path(SOURCES[label]).relative_to('results'),map_location='cpu',weights_only=False)
        assert payload['checkpoint_sha256']==nominal_model_hash, label


if __name__ == '__main__' and '--record' in sys.argv:
    assert not GOLDEN.exists(), 'Refuse to overwrite existing checkpoint golden fixture'
    GOLDEN.write_text(json.dumps(inspect_all(), indent=2, sort_keys=True)+'\n')
    print(GOLDEN)
