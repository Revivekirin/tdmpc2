"""Canonical v2 config and one-way legacy conversion; no simulator imports."""
import hashlib
import json
import math
from copy import deepcopy
from omegaconf import OmegaConf
from common.shift_protocol import make_protocol, evaluation_seeds

GROUPS = ('dynamics_shift', 'schedule', 'adaptation', 'evaluation')


def cli_aliases(arguments):
    """Short public group names alias the repository's existing Hydra packages."""
    result = []
    for argument in arguments:
        key, sep, value = argument.partition('=')
        result.append(f'configs/{key}@{key}={value}' if sep and key in GROUPS else argument)
    return result


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False, default=str).encode()).hexdigest()


def normalize(cfg, overrides=()):
    """Return a resolved canonical config. Nested schedule is input compatibility only."""
    data = OmegaConf.to_container(cfg, resolve=True)
    nested = data['dynamics_shift'].pop('schedule', None)
    if data.get('schedule') is not None:
        if any(s.startswith('dynamics_shift.schedule.') for s in overrides):
            raise ValueError('Do not mix schedule group with legacy nested schedule overrides.')
    else:
        data['schedule'] = nested
    old = data.get('adaptation_mode', 'frozen')
    if old != 'frozen':
        if data['adaptation']['mode'] not in {'frozen', old}:
            raise ValueError('Conflicting adaptation modes.')
        data['adaptation']['mode'] = old
    make_protocol(data['adaptation']['mode'])
    if data['adaptation']['update_enabled']:
        raise ValueError('Frozen evaluation requires update_enabled=false.')
    data['adaptation_mode'] = data['adaptation']['mode']
    evaluation = data['evaluation']
    if evaluation['mode'] == 'continual':
        raise NotImplementedError('continual is a schema only; continual evaluation is not implemented.')
    if evaluation['mode'] not in {'standard', 'paired'}:
        raise ValueError('Unknown evaluation mode.')
    if evaluation['pairing'] != (evaluation['mode'] == 'paired'):
        raise ValueError('pairing must agree with evaluation mode.')
    for field in ('min_exposure_fraction', 'pairing_atol'):
        if isinstance(evaluation[field], bool) or not isinstance(evaluation[field], (int, float)):
            raise ValueError(f'evaluation.{field} must be numeric.')
    if (not 0 <= evaluation['min_exposure_fraction'] <= 1
            or not math.isfinite(evaluation['pairing_atol']) or evaluation['pairing_atol'] < 0):
        raise ValueError('Invalid exposure constraint or pairing tolerance.')
    count = evaluation['episodes']
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('evaluation.episodes must be a positive integer.')
    data['eval_episodes'] = count
    data['eval_seeds'] = evaluation_seeds(data['seed'], evaluation['episodes'], evaluation['seeds'])
    evaluation['seeds'] = data['eval_seeds']
    if data.get('shift_mode', 'nominal') != 'nominal':
        raise ValueError('Use evaluate.py for legacy shift_mode; do not mix it with benchmark groups.')
    shift = data['dynamics_shift']
    visibility = shift['observability']
    if any(visibility.values()):
        raise ValueError('Only hidden shift observations are supported.')
    physics = shift['parameter']['type'] in {'object_mass', 'contact_friction'}
    if shift['application'] != ('reset' if physics else 'step'):
        raise ValueError('Physics requires reset application; actions require step application.')
    if physics and data['schedule']['type'] not in {'fixed', 'nominal'}:
        raise ValueError('Reset-only physics supports fixed/nominal schedules in v2.')
    # Benchmark nominal retains clocks through the identity diagnostic wrapper.
    shift['enabled'] = True
    data['protocol_version'] = 2
    return OmegaConf.create(data)


def runtime_shift(config):
    """Adapt canonical config to the unchanged native wrapper schema."""
    return dict(deepcopy(config['dynamics_shift']), schedule=deepcopy(config['schedule']))
