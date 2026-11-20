"""Side-effect-free adaptation update eligibility over original counters."""

from dataclasses import dataclass
from typing import Any, Mapping


def expected_updates(transitions: int, warmup: int = 64, every: int = 25) -> int:
    return max(0, transitions // every - (warmup - 1) // every)


@dataclass(frozen=True)
class UpdateSchedule:
    settings: Mapping[str, Any]
    every_key: str | None = None

    @property
    def every(self) -> int:
        return self.settings[self.every_key] if self.every_key is not None else 1

    def ready(self, replay_size: int) -> bool:
        return replay_size >= max(self.settings['warmup'], self.settings['batch_size'])

    def due(self, replay_size: int, transitions: int) -> bool:
        return self.ready(replay_size) and transitions % self.every == 0

    def expected_updates(self, transitions: int) -> int:
        return expected_updates(transitions, max(self.settings['warmup'], self.settings['batch_size']), self.every)
