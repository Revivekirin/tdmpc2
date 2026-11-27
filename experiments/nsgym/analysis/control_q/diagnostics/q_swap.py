"""Online-critic substitution; target critic is deliberately not replaced."""
from copy import deepcopy
from ..metrics import require
from q_planning_core import component_hashes


def swap_online_critic(adapted, nominal):
    """Return an independent model, changing online Q only (including its detach alias)."""
    before = component_hashes(adapted)
    reference = component_hashes(nominal)
    for part in ('encoder', 'reward', 'actor'):
        require(before[part] == reference[part], f'DQ changed fixed component: {part}')
    require(adapted.cfg.num_bins == nominal.cfg.num_bins and
            adapted.cfg.vmin == nominal.cfg.vmin and adapted.cfg.vmax == nominal.cfg.vmax,
            'Q support mismatch')
    result = deepcopy(adapted)
    result._Qs.load_state_dict(nominal._Qs.state_dict(), strict=True)
    result.requires_grad_(False)
    result.eval()
    after = component_hashes(result)
    for part in before:
        require(after[part] == (reference[part] if part == 'q' else before[part]),
                f'Critic swap isolation failed: {part}')
    require(component_hashes(adapted) == before, 'Original model mutated')
    return result, {'adapted': before, 'swapped': after, 'nominal': reference}
