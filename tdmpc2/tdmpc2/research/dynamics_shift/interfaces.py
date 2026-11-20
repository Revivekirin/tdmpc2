"""Small contracts for composition; legacy state and replay formats stay intact."""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, TypeAlias, TypeVar


Method: TypeAlias = Literal["frozen", "dynamics_only", "decision_ttt", "dynamics_q", "joint"]
AdaptationState: TypeAlias = Mapping[str, Any]
Transition = TypeVar("Transition")


@dataclass(frozen=True)
class AdaptationConfig:
    method: Method
    # Passed through to the original implementation without normalization or copying.
    settings: Mapping[str, Any] | None = None
    identity: Mapping[str, Any] | None = None


class AdaptationReplayBuffer(Protocol[Transition]):
    """Only the shared storage contract; sampling remains with each legacy owner."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...


class AdaptationStrategy(Protocol):
    implementation: Any

    def observe(self, *args: Any, **kwargs: Any) -> Any: ...
    def update(self) -> Any: ...
    def assert_fixed(self) -> None: ...
    def state_dict(self) -> AdaptationState: ...
    def load_state_dict(self, state: AdaptationState) -> None: ...


class AdaptationRunner(Protocol):
    implementation: Callable[..., Any]

    def run(self, *args: Any, **kwargs: Any) -> Any: ...
