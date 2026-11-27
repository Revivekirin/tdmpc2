"""Environment orchestration shared by NS-Gym adaptation streams.

Callbacks retain each method's collection, replay, update, logging and
checkpoint policies.  This module does not import a simulator or an agent.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class TransitionEvent:
    index: int
    episode: int
    episode_step: int
    observation: Any
    action: Any
    next_observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: Any


class AdaptationRunner:
    """Run environment steps while leaving scientific policies at call sites.

    The generator yields after the real step and the strategy callback, before
    the next reset.  Callers can therefore log and checkpoint at the same
    point as their legacy loop.  A close in `finally` preserves error cleanup.
    """

    def __init__(self, env: Any, action: Callable[[Any, int, int], Any],
                 observe: Callable[[TransitionEvent], Any], *,
                 horizon: int, initial_seed: int | None,
                 before_step: Callable[[], None] | None = None,
                 on_reset: Callable[[Any, int], None] | None = None,
                 reset_on: Callable[[TransitionEvent], bool] | None = None,
                 stop_on: Callable[[TransitionEvent], bool] | None = None):
        if horizon <= 0:
            raise ValueError('horizon must be positive')
        self.env, self.action, self.observe = env, action, observe
        self.horizon, self.initial_seed = horizon, initial_seed
        self.before_step, self.on_reset = before_step, on_reset
        self.reset_on = reset_on or (lambda e: bool(e.terminated or e.truncated))
        self.stop_on = stop_on

    def iter_steps(self, budget: int) -> Iterator[tuple[TransitionEvent, Any]]:
        if budget < 0:
            raise ValueError('budget must be nonnegative')
        observation, _ = self.env.reset(seed=self.initial_seed)
        if self.on_reset is not None:
            self.on_reset(observation, 0)
        for index in range(budget):
            episode, episode_step = divmod(index, self.horizon)
            if self.before_step is not None:
                self.before_step()
            action = self.action(observation, episode_step, episode)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            event = TransitionEvent(index, episode, episode_step, observation, action,
                                    next_observation, reward, terminated, truncated, info)
            result = self.observe(event)
            yield event, result
            if self.stop_on is not None and self.stop_on(event):
                break
            observation = next_observation
            if index + 1 < budget and self.reset_on(event):
                observation, _ = self.env.reset()
                if self.on_reset is not None:
                    self.on_reset(observation, episode + 1)

    def run(self, budget: int, record: Callable[[TransitionEvent, Any], None], *,
            checkpoint: Callable[[TransitionEvent, Any], None] | None = None,
            complete: Callable[[], Any] | None = None,
            close_env: bool = True) -> Any:
        try:
            for event, result in self.iter_steps(budget):
                record(event, result)
                if checkpoint is not None:
                    checkpoint(event, result)
            return complete() if complete is not None else None
        finally:
            if close_env:
                self.env.close()
