"""Compatibility imports for Dynamics+Q adaptation (Phase 1B-2)."""

from research.dynamics_shift.dynamics_critic import (
    DynamicsCriticAdaptation as DynamicsQAdapter,
    expected_updates,
)

__all__ = ["DynamicsQAdapter", "expected_updates"]
