"""Frozen-state snapshots shared by the two existing adaptation objectives."""

from copy import deepcopy

import torch


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


class ParameterInvarianceAudit:
    """Capture and compare the unchanged components using each legacy policy."""

    def __init__(self, model, mutable_prefixes, *, cpu_tensors=False):
        self.model = model
        self.mutable_prefixes = tuple(mutable_prefixes)
        self.cpu_tensors = cpu_tensors
        self.initial_fixed = self.fixed_state()

    def fixed_state(self):
        state = {k: v for k, v in self.model.state_dict().items()
                 if not k.startswith(self.mutable_prefixes)}
        if self.cpu_tensors:
            return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
                    for k, v in state.items()}
        return deepcopy(state)

    def assert_fixed(self, reference, message):
        if not equal_state(reference, self.fixed_state()):
            raise AssertionError(message)
