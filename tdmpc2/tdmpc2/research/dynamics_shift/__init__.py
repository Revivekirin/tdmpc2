"""Typed composition over existing dynamics-shift implementations."""

from .factory import AdaptationBinding, compose_adaptation
from .interfaces import (
    AdaptationConfig,
    AdaptationReplayBuffer,
    AdaptationRunner,
    AdaptationState,
    AdaptationStrategy,
    Method,
)

__all__ = [
    "AdaptationBinding", "AdaptationConfig", "AdaptationReplayBuffer",
    "AdaptationRunner", "AdaptationState", "AdaptationStrategy", "Method",
    "compose_adaptation",
]
