"""Composition root for existing adaptation objects and stream runners."""

from dataclasses import dataclass
from typing import Any, Callable

from .adapters import ExistingRunnerAdapter
from .interfaces import AdaptationConfig, AdaptationStrategy


@dataclass(frozen=True)
class AdaptationBinding:
    strategy: AdaptationStrategy | None
    runner: ExistingRunnerAdapter | None


def compose_adaptation(
    config: AdaptationConfig,
    *,
    model: Any = None,
    agent: Any = None,
    consistency_coef: float | None = None,
    grad_clip_norm: float | None = None,
    seed: int = 0,
    run_stream: Callable[..., Any] | None = None,
) -> AdaptationBinding:
    """Select canonical adaptation without interpreting scientific settings.

    Joint's original stream function owns agent/replay construction and updates,
    so it is bound as a runner rather than fabricated as a stepwise strategy.
    A supplied run_stream is always invoked with its original arguments.
    """
    method = config.method
    runner = ExistingRunnerAdapter(run_stream) if run_stream is not None else None
    if method == "joint":
        if runner is None:
            raise ValueError("Joint requires the existing production run_stream callable")
        return AdaptationBinding(strategy=None, runner=runner)
    if method in ("frozen", "dynamics_only", "decision_ttt"):
        if model is None or config.settings is None:
            raise ValueError("Episode adaptation requires model and original settings")
        if consistency_coef is None or grad_clip_norm is None:
            raise ValueError("Episode adaptation requires original loss coefficient and clip norm")
        expected = {"frozen": "frozen", "dynamics_only": "dynamics_only",
                    "decision_ttt": "decision_ttt"}[method]
        if config.settings.get("mode") != expected:
            raise ValueError("Adaptation method and original settings mode differ")
        if method == "decision_ttt":
            from .decision import DecisionAdapter
            implementation = DecisionAdapter(model, config.settings, consistency_coef,
                                             grad_clip_norm, seed=seed)
        else:
            from .dynamics import DynamicsAdaptation
            implementation = DynamicsAdaptation(model, config.settings, consistency_coef,
                                                grad_clip_norm, seed=seed)
    elif method == "dynamics_q":
        if agent is None or config.settings is None or config.identity is None:
            raise ValueError("Dynamics+Q requires agent, original configuration and identity")
        from .dynamics_critic import DynamicsCriticAdaptation
        implementation = DynamicsCriticAdaptation(agent, config.settings, config.identity)
    else:
        raise ValueError(f"Unknown adaptation method: {method}")
    return AdaptationBinding(strategy=implementation, runner=runner)
