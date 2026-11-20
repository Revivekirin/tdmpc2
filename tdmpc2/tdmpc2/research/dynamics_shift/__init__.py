"""Canonical dynamics-shift adaptation and compatibility composition."""

from .factory import AdaptationBinding, compose_adaptation
from .interfaces import (
    AdaptationConfig,
    AdaptationRunner,
    AdaptationState,
    AdaptationStrategy,
    Method,
)
from .replay import AdaptationReplayBuffer
from .schedule import UpdateSchedule
from .audit import ParameterInvarianceAudit
from .dynamics import DynamicsAdaptation
from .decision import DecisionAdapter
from .dynamics_critic import DynamicsCriticAdaptation

__all__ = [
    "AdaptationBinding", "AdaptationConfig", "AdaptationReplayBuffer",
    "AdaptationRunner", "AdaptationState", "AdaptationStrategy", "Method",
    "UpdateSchedule", "ParameterInvarianceAudit", "DynamicsAdaptation",
    "DecisionAdapter", "DynamicsCriticAdaptation",
    "compose_adaptation",
]
