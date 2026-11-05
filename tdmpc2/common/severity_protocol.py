"""Stage 3B-1 protocol and validation; no simulator or learning dependencies."""
import hashlib
import json
import math
import re
from pathlib import Path

DEFAULTS = dict(action_delay=1, action_gain=.7, object_mass=1.5, contact_friction=.5)
PILOT = dict(action_delay=[0,1,2,3], action_gain=[.9,.7,.5,.3],
             object_mass=[1,1.5,2,3], contact_friction=[1,.7,.5,.3])
PROTOCOL = 'stage3b-1-frozen-v1'


def severity(kind, value):
    if kind not in DEFAULTS or isinstance(value, bool):
        raise ValueError('Unsupported shift or boolean severity.')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Severity must be finite.')
    if kind == 'action_delay':
        if value < 0 or not value.is_integer() or value > 10000:
            raise ValueError('Delay must be an integer in [0, 10000] native steps (existing queue limit).')
        return int(value)
    if value <= 0:
        raise ValueError('Gain/mass/friction must be strictly positive.')
    return value


def value_label(value):
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def run_path(root, kind, value, seed_set, run_id):
    severity(kind, value)
    if type(seed_set) is not int or seed_set < 0:
        raise ValueError('set-id must be a nonnegative integer.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run-id must be a safe single directory name.')
    return Path(root)/'runs'/kind/f'set-{seed_set}'/'frozen'/value_label(value)/'severity-eval'/run_id


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def protocol_identity(config):
    # Severity, shift type and seed set vary; all planner/evaluator controls must match.
    controls = {k:v for k,v in config.items() if k not in (
        'dynamics_shift','schedule','stage3a','stage3b','severity','work_dir','checkpoint')}
    return hashlib.sha256(json.dumps(controls, sort_keys=True, default=str).encode()).hexdigest()


def source_digest(root):
    h = hashlib.sha256()
    for directory in ('tdmpc2', 'scripts', 'benchmarks'):
        for path in sorted((Path(root)/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.yaml','.json'):
                h.update(str(path.relative_to(root)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()
