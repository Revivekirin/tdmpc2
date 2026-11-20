"""Adaptation replay storage; records and RNG remain method owned."""

from collections import deque
from typing import Any


class AdaptationReplayBuffer(deque):
    """The original bounded deque with its exact list-based uniform sampler."""

    def sample(self, rng: Any, batch_size: int) -> list[Any]:
        return rng.sample(list(self), batch_size)
