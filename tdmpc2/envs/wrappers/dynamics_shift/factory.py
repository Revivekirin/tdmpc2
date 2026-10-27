"""One parameter per environment; adaptation settings never enter wrappers."""
from copy import deepcopy
from .action_shift import ActionShiftWrapper
from .physics_shift import PhysicsShiftWrapper


def make_shift(env, config, task=None):
    config = deepcopy(dict(config))
    if not config.get('enabled', False):
        return env
    visibility = config.get('observability', {})
    if visibility.get('expose_shift_parameter_to_agent', False) or visibility.get('expose_executed_action_to_agent', False):
        raise ValueError('This benchmark implements hidden shifts only.')
    kind = config['parameter']['type']
    if kind in {'object_mass', 'contact_friction'}:
        if task != 'pick-cube':
            raise ValueError('Physics actor replacement/material targeting is supported only for pick-cube.')
        return PhysicsShiftWrapper(env, config)
    if config.get('application', 'step') != 'step':
        raise ValueError('Action shifts require application=step.')
    return ActionShiftWrapper(env, config)
