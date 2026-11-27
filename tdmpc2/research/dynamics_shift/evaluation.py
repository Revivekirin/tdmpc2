"""Evaluation orchestration without simulator, optimizer, or planner imports.

Task bindings supply the historical reset, action, step and record operations.
The runner neither seeds per decision nor changes the agent's planning state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Protocol, TypeVar


Observation = TypeVar("Observation")
Action = TypeVar("Action")


@dataclass(frozen=True)
class EvaluationStep(Generic[Observation, Action]):
    index: int
    observation: Observation
    action: Action
    next_observation: Observation
    reward: Any
    terminated: bool
    truncated: bool
    info: Any


@dataclass(frozen=True)
class EpisodeResult(Generic[Observation, Action]):
    seed: int
    initial_observation: Observation
    final_observation: Observation
    steps: tuple[EvaluationStep[Observation, Action], ...]
    total_reward: float


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int
    horizon: int
    stop_on_boundary: bool = True

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("Evaluation horizon must be positive")


class EvaluationProtocol(Protocol[Observation, Action]):
    """Task-specific execution contract; no default simulator assumptions."""

    def seed(self, seed: int) -> None: ...
    def create_environment(self) -> Any: ...
    def reset(self, env: Any, seed: int) -> Observation: ...
    def action(self, observation: Observation, index: int) -> Action: ...
    def step(self, env: Any, action: Action) -> tuple[Observation, Any, bool, bool, Any]: ...
    def record_step(self, env: Any, step: EvaluationStep[Observation, Action]) -> None: ...
    def verify_unchanged(self) -> None: ...


@dataclass
class EvaluationCallbacks(Generic[Observation, Action]):
    """Bind an existing task evaluator without changing its metric functions."""

    seed: Callable[[int], None]
    create_environment: Callable[[], Any]
    reset: Callable[[Any, int], Observation]
    action: Callable[[Observation, int], Action]
    step: Callable[[Any, Action], tuple[Observation, Any, bool, bool, Any]]
    record_step: Callable[[Any, EvaluationStep[Observation, Action]], None]
    verify_unchanged: Callable[[], None]


class EvaluationRunner:
    """One seeded episode, preserving caller-specified RNG and step order."""

    def run(self, config: EvaluationConfig,
            protocol: EvaluationProtocol[Observation, Action]) -> EpisodeResult[Observation, Action]:
        protocol.seed(config.seed)
        env = protocol.create_environment()
        try:
            initial = observation = protocol.reset(env, config.seed)
            steps: list[EvaluationStep[Observation, Action]] = []
            total = 0.0
            for index in range(config.horizon):
                action = protocol.action(observation, index)
                next_observation, reward, terminated, truncated, info = protocol.step(env, action)
                event = EvaluationStep(index, observation, action, next_observation, reward,
                                       bool(terminated), bool(truncated), info)
                protocol.record_step(env, event)
                steps.append(event)
                total += float(reward)
                observation = next_observation
                if config.stop_on_boundary and (terminated or truncated):
                    break
            result = EpisodeResult(config.seed, initial, observation, tuple(steps), total)
            protocol.verify_unchanged()
            return result
        finally:
            env.close()


EvaluationKind = Literal["frozen_control", "adapted_control", "nominal_retention",
                         "fixed_bank_prediction", "critic_intervention"]


@dataclass(frozen=True)
class EvaluationIdentity:
    kind: EvaluationKind
    task: str
    checkpoint_sha256: tuple[str, ...]
    seed: int | None = None
    initial_state_sha256: str | None = None
    rng_sha256: str | None = None


def require_bank(path: Path, expected_sha256: str) -> Path:
    """Require an existing immutable bank; never generate a replacement."""
    from .checkpoints import file_sha256

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if file_sha256(source) != expected_sha256:
        raise ValueError(f"Saved transition bank identity mismatch: {source}")
    return source


def aligned_transitions(observations: Any, actions: Any) -> tuple[tuple[Any, Any, Any], ...]:
    """Return historical (observation, action, actual next observation) rows."""
    if len(observations) != len(actions) + 1:
        raise ValueError("Transition bank requires one more observation than action")
    return tuple((observations[index], actions[index], observations[index + 1])
                 for index in range(len(actions)))


@dataclass(frozen=True)
class FixedBankResult:
    bank_sha256: str
    samples: int
    latent_mse: float


def evaluate_fixed_bank(path: Path, expected_sha256: str,
                        score_batch: Callable[[Any, Any, Any], Any], *,
                        batch_size: int = 256) -> FixedBankResult:
    """Score an existing transition bank without creating simulator rollouts.

    The caller owns the exact historical encoder and one-step loss expression.
    Batch size is explicit because summation and model batch behavior matter.
    """
    import numpy as np

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    bank = require_bank(path, expected_sha256)
    with np.load(bank, allow_pickle=False) as data:
        observations = data["observations"]
        actions = data["actions"]
        if len(observations) != len(actions) + 1 or not len(actions):
            raise ValueError("Invalid saved transition alignment")
        for name in ("terminated", "truncated"):
            if name in data and (len(data[name]) != len(actions) or np.any(data[name][:-1])):
                raise ValueError(f"Bank crosses an unaligned {name} boundary")
        total = 0.0
        for start in range(0, len(actions), batch_size):
            end = min(start + batch_size, len(actions))
            errors = np.asarray(score_batch(observations[start:end], actions[start:end],
                                            observations[start + 1:end + 1]))
            if errors.shape != (end - start,) or not np.isfinite(errors).all():
                raise ValueError("Invalid per-transition prediction errors")
            total += float(errors.sum())
        result = FixedBankResult(expected_sha256, len(actions), total / len(actions))
    if require_bank(bank, expected_sha256) != bank:
        raise ValueError("Saved bank changed during evaluation")
    return result


def critic_intervention_controllers(composer: Any, dynamics_only: Any,
                                    dynamics_q: Any) -> tuple[Any, Any, Any]:
    """Construct A, B, C with identical non-Q state for the B/C comparison."""
    from .checkpoints import model_hashes

    a = composer.compose("dynamics_only", dynamics_only)
    b = composer.controlled_q_swap(dynamics_q)
    c = composer.compose("dynamics_q", dynamics_q)
    if b.component_hashes["online_q"] != a.component_hashes["online_q"]:
        raise ValueError("B must use the nominal online critic")
    for component in ("encoder", "dynamics", "reward", "target_q", "actor", "normalization"):
        if b.component_hashes[component] != c.component_hashes[component]:
            raise ValueError(f"B/C {component} differs")
    if b.component_hashes["online_q"] == c.component_hashes["online_q"]:
        raise ValueError("B/C online critics are identical")
    for controller in (a, b, c):
        if controller.model.training or any(p.requires_grad for p in controller.model.parameters()):
            raise ValueError("Evaluation controller must be frozen")
        if model_hashes(controller.model) != controller.component_hashes:
            raise ValueError("Evaluation controller changed after composition")
    return a, b, c
