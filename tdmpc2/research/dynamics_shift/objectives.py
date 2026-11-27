"""Adaptation objective hooks; the default expressions retain legacy numerics."""

from dataclasses import dataclass
import math as scalar_math
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from common import math


def one_step_dynamics_mse(model, encoded_observation, action, encoded_next_observation):
    return F.mse_loss(model.next(encoded_observation, action, None), encoded_next_observation)


@dataclass(frozen=True)
class DynamicsObjectiveConfig:
    """Resolved loss settings; no extra fields are required by legacy runs."""

    dynamics_objective: str = "one_step_dynamics_consistency"
    critic_objective: str = "replay_encoded_critic"
    rollout_horizon: int = 1
    temporal_discount: float = 1.0
    normalization_mode: str = "horizon"

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> "DynamicsObjectiveConfig":
        if not isinstance(settings, Mapping):
            raise ValueError("Objective settings must be a mapping")
        mode = settings.get("dynamics_objective", cls.dynamics_objective)
        critic = settings.get("critic_objective", cls.critic_objective)
        fields = ("rollout_horizon", "temporal_discount", "normalization_mode")
        if mode == cls.dynamics_objective:
            if "dynamics_objective" not in settings and any(key in settings for key in fields):
                raise ValueError("Dynamics objective must be explicit when rollout settings are supplied")
            values = [settings.get(key, default) for key, default in zip(fields, (1, 1.0, "horizon"))]
        elif mode == "multi_step_imagined_dynamics_consistency":
            if any(key not in settings for key in fields):
                raise ValueError("Multi-step dynamics requires horizon, discount, and normalization")
            values = [settings[key] for key in fields]
        else:
            raise ValueError(f"Unsupported dynamics objective: {mode}")
        result = cls(mode, critic, *values)
        if result.critic_objective != "replay_encoded_critic":
            raise ValueError(f"Unsupported critic objective: {critic}")
        if type(result.rollout_horizon) is not int or result.rollout_horizon < 1:
            raise ValueError("Rollout horizon must be a positive integer")
        if type(result.temporal_discount) not in (int, float) or not scalar_math.isfinite(result.temporal_discount) or \
                not 0 < result.temporal_discount <= 1:
            raise ValueError("Temporal discount must be finite and in (0, 1]")
        if result.normalization_mode not in ("horizon", "valid_steps"):
            raise ValueError("Unsupported dynamics normalization mode")
        if result.dynamics_objective == cls.dynamics_objective and \
                (result.rollout_horizon != 1 or result.normalization_mode != "horizon"):
            raise ValueError("One-step dynamics requires horizon=1 and horizon normalization")
        return result

    def as_dict(self) -> dict[str, Any]:
        return dict(dynamics_objective=self.dynamics_objective, critic_objective=self.critic_objective,
                    rollout_horizon=self.rollout_horizon, temporal_discount=float(self.temporal_discount),
                    normalization_mode=self.normalization_mode)


@dataclass(frozen=True)
class MultiStepDynamicsConsistency:
    """Autoregressive consistency on [H,B,A] actions and [H,B,Z] real targets.

    Each valid position contributes mean squared error over latent dimensions.
    `horizon` divides by H*B (original TD-MPC2 scale); `valid_steps` divides
    by the count of valid positions. Weights are rho**k, not sum-normalized.
    """

    rollout_horizon: int
    temporal_discount: float
    normalization_mode: str

    def __call__(self, model, initial_latent, actions, target_latents, valid_mask=None):
        if actions.ndim != 3 or target_latents.ndim != 3 or initial_latent.ndim != 2 or \
                actions.shape[0] != self.rollout_horizon or target_latents.shape[:2] != actions.shape[:2] or \
                initial_latent.shape != target_latents.shape[1:]:
            raise ValueError("Expected latent [B,Z], actions [H,B,A], targets [H,B,Z]")
        if valid_mask is None:
            valid_mask = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        if valid_mask.shape != actions.shape[:2] or valid_mask.device != actions.device:
            raise ValueError("Valid mask must have shape [H,B] on the action device")
        valid_mask = valid_mask.to(dtype=torch.bool)
        if not valid_mask.any():
            raise ValueError("Multi-step objective requires a valid prediction")
        initial_latent = initial_latent.detach()
        target_latents = target_latents.detach()
        if self.rollout_horizon == 1 and valid_mask.all():
            return one_step_dynamics_mse(model, initial_latent, actions[0], target_latents[0])
        predicted = initial_latent
        loss = initial_latent.new_zeros(())
        for step in range(self.rollout_horizon):
            predicted = model.next(predicted, actions[step], None)
            per_item = (predicted - target_latents[step]).square().mean(dim=-1)
            loss = loss + self.temporal_discount ** step * (per_item * valid_mask[step]).sum()
        denominator = (actions.shape[0] * actions.shape[1] if self.normalization_mode == "horizon"
                       else valid_mask.sum())
        return loss / denominator


def replay_encoded_critic_loss(model, encoded_observation, action, target, cfg):
    """Original categorical Q loss on detached encodings of real observations."""
    logits = model.Q(encoded_observation.detach(), action, None, return_type='all')
    loss = torch.stack([math.soft_ce(head, target, cfg).mean() for head in logits]).mean()
    return loss, logits


@dataclass(frozen=True)
class ObjectiveHooks:
    dynamics: Callable[..., Any] = one_step_dynamics_mse
    critic: Callable[[Any, Any, Any, Any, Any], tuple[Any, Any]] = replay_encoded_critic_loss


DEFAULT_OBJECTIVES = ObjectiveHooks()
