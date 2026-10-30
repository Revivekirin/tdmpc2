"""Preflight validation without importing gym, torch or simulator packages."""
import importlib.util
from pathlib import Path
import numpy as np
from common.shift_protocol import validate_probe_holdout


def validate(config):
    path = Path(__file__).resolve().parents[1] / 'envs/wrappers/dynamics_shift/schedule.py'
    spec = importlib.util.spec_from_file_location('_benchmark_schedule', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = config['dynamics_shift']['parameter']
    schedule = module.Schedule(config['schedule'], p['nominal'], p['shifted'])
    kind = p['type']
    values = schedule.possible_values()
    if kind == 'action_delay':
        if schedule.kind not in module.DISCRETE or np.shape(p['nominal']) or p['nominal'] != 0:
            raise ValueError('Delay requires discrete schedules and scalar nominal zero.')
        for value in values:
            if np.shape(value) or module.integer(value, 'delay') > 10000:
                raise ValueError('Delay must be a scalar integer <= 10000.')
    elif kind in {'object_mass','contact_friction'}:
        if config['task'] != 'pick-cube':
            raise ValueError('Physics shifts currently support pick-cube only.')
        if p['nominal'] != 1 or any(np.shape(v) or float(v) <= 0 for v in values):
            raise ValueError('Physics requires positive scalar multipliers and nominal=1.')
    elif kind in {'action_gain', 'action_bias'}:
        if not np.all(np.asarray(p['nominal']) == (1 if kind == 'action_gain' else 0)):
            raise ValueError('Invalid identity nominal parameter.')
        if kind == 'action_gain' and any(np.any(np.asarray(v)<0) for v in values):
            raise ValueError('Gain cannot become negative.')
    elif kind != 'action_coupling':
        raise ValueError('Unknown shift parameter.')
    validate_probe_holdout(schedule, config['benchmark_split'], config['benchmark_probe_values'])
    return schedule
