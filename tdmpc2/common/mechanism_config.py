"""Strict analysis schema, importable without torch or simulator dependencies."""

def validate_analysis(options, cfg):
    flags = {'enabled', 'one_step', 'multi_step', 'planning', 'counterfactual',
             'deterministic_probes', 'save_latents', 'save_candidate_actions',
             'save_candidate_scores', 'save_counterfactual_details'}
    allowed = flags | {'mode', 'rollout_horizons', 'decision_stride', 'max_logged_candidates',
                       'q_head_mode', 'policy_probe_mode'}
    if set(options) - allowed:
        raise ValueError(f'Unknown analysis options: {set(options) - allowed}')
    for key in flags & options.keys():
        if type(options[key]) is not bool:
            raise ValueError(f'analysis.{key} must be boolean')
    if not options.get('enabled', False):
        return
    if set(options) != allowed:
        raise ValueError(f'Missing analysis options: {allowed - set(options)}')
    for key, value in dict(mode='frozen_mechanism', deterministic_probes=True,
                           q_head_mode='all', policy_probe_mode='mean').items():
        if options[key] != value:
            raise ValueError(f'Only analysis.{key}={value} is supported')
    if cfg.get('compile', False):
        raise ValueError('Frozen mechanism tracing requires compile=false (Python observation hooks)')
    if cfg.get('multitask', False) is True or cfg.get('obs', 'state') != 'state':
        raise ValueError('Mechanism analysis supports single-task state observations only')
    for key in ('decision_stride', 'max_logged_candidates'):
        if type(options[key]) is not int or options[key] <= 0:
            raise ValueError(f'analysis.{key} must be a positive integer')
    horizons = options['rollout_horizons']
    if not horizons or any(type(h) is not int or h <= 0 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError('rollout_horizons must contain unique positive integers')
