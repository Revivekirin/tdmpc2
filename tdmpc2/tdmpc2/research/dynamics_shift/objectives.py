"""Adaptation objective hooks; the default expressions retain legacy numerics."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from common import math


def one_step_dynamics_mse(model, encoded_observation, action, encoded_next_observation):
    return F.mse_loss(model.next(encoded_observation, action, None), encoded_next_observation)


def replay_encoded_critic_loss(model, encoded_observation, action, target, cfg):
    """Original categorical Q loss on detached encodings of real observations."""
    logits = model.Q(encoded_observation.detach(), action, None, return_type='all')
    loss = torch.stack([math.soft_ce(head, target, cfg).mean() for head in logits]).mean()
    return loss, logits


@dataclass(frozen=True)
class ObjectiveHooks:
    dynamics: Callable[[Any, Any, Any, Any], Any] = one_step_dynamics_mse
    critic: Callable[[Any, Any, Any, Any, Any], tuple[Any, Any]] = replay_encoded_critic_loss


DEFAULT_OBJECTIVES = ObjectiveHooks()
