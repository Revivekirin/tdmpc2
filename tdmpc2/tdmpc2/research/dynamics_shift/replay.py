"""Adaptation replay storage; records and RNG remain method owned."""

from collections import deque
from collections.abc import Mapping
from typing import Any


class AdaptationReplayBuffer(deque):
    """The original bounded deque with its exact list-based uniform sampler."""

    def sample(self, rng: Any, batch_size: int) -> list[Any]:
        return rng.sample(list(self), batch_size)

    def contiguous_sequences(self, length: int) -> list[tuple[Any, ...]]:
        """Return real-transition windows within one episode, without sampling.

        Sequence metadata belongs to Dynamics+Q records. The predictor-only
        tuple replay keeps using `sample()` and is never reinterpreted here.
        """
        if type(length) is not int or length <= 0:
            raise ValueError('Sequence length must be a positive integer')
        records = list(self)
        if any(not isinstance(record, Mapping) or
               not {'episode', 'episode_step', 'global_step', 'terminated', 'truncated'} <= record.keys()
               for record in records):
            raise ValueError('Sequence replay requires Dynamics+Q transition metadata')
        windows = []
        for start in range(len(records) - length + 1):
            window = records[start:start + length]
            if all(previous['episode'] == current['episode'] and
                   previous['episode_step'] + 1 == current['episode_step'] and
                   previous['global_step'] + 1 == current['global_step'] and
                   not previous['terminated'] and not previous['truncated']
                   for previous, current in zip(window, window[1:])):
                windows.append(tuple(window))
        return windows

    def sample_sequences(self, rng: Any, batch_size: int, length: int) -> list[tuple[Any, ...]]:
        """Uniformly sample eligible windows without replacement."""
        return rng.sample(self.contiguous_sequences(length), batch_size)
