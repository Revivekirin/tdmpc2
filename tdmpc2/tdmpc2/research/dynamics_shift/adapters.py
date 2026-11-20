"""Transparent delegates; no replay, update, loss or checkpoint logic lives here."""

from typing import Any, Callable

from .interfaces import AdaptationState


class ExistingStrategyAdapter:
    def __init__(self, implementation: Any):
        self.implementation = implementation

    def __getattr__(self, name: str) -> Any:
        return getattr(self.implementation, name)

    def observe(self, *args: Any, **kwargs: Any) -> Any:
        return self.implementation.observe(*args, **kwargs)

    def update(self) -> Any:
        return self.implementation.update()

    def assert_fixed(self) -> None:
        self.implementation.assert_fixed()

    def state_dict(self) -> AdaptationState:
        return self.implementation.state_dict()

    def load_state_dict(self, state: AdaptationState) -> None:
        self.implementation.load_state_dict(state)


class ExistingRunnerAdapter:
    def __init__(self, implementation: Callable[..., Any]):
        self.implementation = implementation

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.implementation(*args, **kwargs)
