"""Canonical dynamics-shift adaptation and compatibility composition."""

from .factory import AdaptationBinding, compose_adaptation
from .interfaces import (
    AdaptationConfig,
    AdaptationRunner,
    AdaptationState,
    AdaptationStrategy,
    Method,
)
from .replay import AdaptationReplayBuffer
from .schedule import UpdateSchedule
from .audit import ParameterInvarianceAudit
from .dynamics import DynamicsAdaptation
from .decision import DecisionAdapter
from .dynamics_critic import DynamicsCriticAdaptation
from .runner import AdaptationRunner, TransitionEvent
from .joint import JointAdaptation
from .checkpoints import CheckpointDescriptor, LoadedCheckpoint, load_checkpoint, model_hashes
from .composer import CheckpointComposer, ComposedController
from .evaluation import (
    EvaluationCallbacks, EvaluationConfig, EvaluationIdentity, EvaluationRunner,
    EvaluationStep, EpisodeResult, FixedBankResult, aligned_transitions,
    critic_intervention_controllers, evaluate_fixed_bank, require_bank,
)
from .planning_analysis import (
    FixedCandidateAnalysis, candidate_rank_comparison, candidate_sha256,
    controlled_q_score_difference, latent_prediction_diagnostics,
    native_trace_diagnostics, score_fixed_candidates, terminal_value_decomposition,
)

__all__ = [
    "AdaptationBinding", "AdaptationConfig", "AdaptationReplayBuffer",
    "AdaptationRunner", "AdaptationState", "AdaptationStrategy", "Method",
    "UpdateSchedule", "ParameterInvarianceAudit", "DynamicsAdaptation",
    "DecisionAdapter", "DynamicsCriticAdaptation",
    "TransitionEvent", "JointAdaptation",
    "CheckpointDescriptor", "LoadedCheckpoint", "load_checkpoint", "model_hashes",
    "CheckpointComposer", "ComposedController",
    "EvaluationCallbacks", "EvaluationConfig", "EvaluationIdentity",
    "EvaluationRunner", "EvaluationStep", "EpisodeResult", "FixedBankResult",
    "aligned_transitions", "critic_intervention_controllers", "evaluate_fixed_bank", "require_bank",
    "FixedCandidateAnalysis", "candidate_rank_comparison", "candidate_sha256",
    "controlled_q_score_difference", "latent_prediction_diagnostics",
    "native_trace_diagnostics", "score_fixed_candidates", "terminal_value_decomposition",
    "compose_adaptation",
]
