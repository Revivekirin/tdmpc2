"""Runner delegate retained for the existing Joint RL implementation."""

from typing import Any, Callable

class ExistingRunnerAdapter:
    def __init__(self, implementation: Callable[..., Any]):
        self.implementation = implementation

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.implementation(*args, **kwargs)
