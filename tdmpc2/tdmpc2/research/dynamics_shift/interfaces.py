"""Small contracts for composition; legacy state and replay formats stay intact."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable, Literal, Mapping, Protocol, TypeAlias


Method: TypeAlias = Literal["frozen", "dynamics_only", "decision_ttt", "dynamics_q", "joint"]
AdaptationState: TypeAlias = Mapping[str, Any]
AdaptationProtocol: TypeAlias = Literal["matched_replay", "online"]
StudyStatus: TypeAlias = Literal["NOT_RUN", "PASS", "FAILED", "BLOCKED", "UNVERIFIED"]

DYNAMICS_OBJECTIVES = ("one_step_dynamics_consistency", "multi_step_imagined_dynamics_consistency")
CRITIC_OBJECTIVES = ("replay_encoded_critic", "imagined_multi_position_critic")


def _sha256(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class ExperimentIdentity:
    adaptation_protocol: AdaptationProtocol
    dynamics_objective: str
    critic_objective: str
    dynamics_shift: Mapping[str, Any]
    nominal_checkpoint_sha256: str
    adaptation_seed: int
    evaluation_seed: int | None
    interaction_budget: int
    optimizer_update_counts: Mapping[str, int]
    evaluation_checkpoint_identity: str | None

    def __post_init__(self) -> None:
        if self.adaptation_protocol not in ("matched_replay", "online"):
            raise ValueError("Unsupported adaptation protocol")
        if self.dynamics_objective not in DYNAMICS_OBJECTIVES or self.critic_objective not in CRITIC_OBJECTIVES:
            raise ValueError("Unsupported objective identity")
        if not isinstance(self.dynamics_shift, Mapping) or not self.dynamics_shift.get("parameter") or \
                type(self.dynamics_shift.get("severity")) not in (int, float) or \
                not math.isfinite(self.dynamics_shift["severity"]) or self.dynamics_shift["severity"] <= 0:
            raise ValueError("Invalid physical dynamics shift")
        if not _sha256(self.nominal_checkpoint_sha256):
            raise ValueError("Invalid nominal checkpoint SHA-256")
        if type(self.adaptation_seed) is not int or self.adaptation_seed < 0 or \
                (self.evaluation_seed is not None and
                 (type(self.evaluation_seed) is not int or self.evaluation_seed < 0 or
                  self.evaluation_seed == self.adaptation_seed)):
            raise ValueError("Invalid or overlapping adaptation/evaluation seed")
        if type(self.interaction_budget) is not int or self.interaction_budget < 0:
            raise ValueError("Invalid interaction budget")
        if not isinstance(self.optimizer_update_counts, Mapping) or any(
                type(value) is not int or value < 0 for value in self.optimizer_update_counts.values()):
            raise ValueError("Invalid optimizer update counts")
        if self.evaluation_checkpoint_identity is not None and not _sha256(self.evaluation_checkpoint_identity):
            raise ValueError("Invalid evaluation checkpoint identity")


@dataclass(frozen=True)
class ExperimentProgress:
    identity: ExperimentIdentity
    training_status: StudyStatus
    evaluation_status: StudyStatus

    def __post_init__(self) -> None:
        allowed = {"NOT_RUN", "PASS", "FAILED", "BLOCKED", "UNVERIFIED"}
        if self.training_status not in allowed or self.evaluation_status not in allowed:
            raise ValueError("Invalid operation status")
        if self.evaluation_status == "PASS" and (
                self.training_status != "PASS" or self.identity.evaluation_seed is None or
                self.identity.evaluation_checkpoint_identity is None):
            raise ValueError("Successful evaluation requires trained checkpoint and evaluation identity")


def validate_study_config(config: Mapping[str, Any]) -> None:
    """Validate planned identities without declaring an objective implemented."""
    if not isinstance(config, Mapping) or not isinstance(config.get("title"), str) or not config["title"].strip():
        raise ValueError("Study title required")
    if config.get("version") != 1 or not _sha256(config.get("nominal_checkpoint_sha256")):
        raise ValueError("Invalid study version or nominal checkpoint")
    if config.get("adaptation_protocols") != ["matched_replay", "online"] or \
            config.get("dynamics_objectives") != list(DYNAMICS_OBJECTIVES) or \
            config.get("critic_objectives") != list(CRITIC_OBJECTIVES):
        raise ValueError("Invalid study protocol/objective matrix")
    if type(config.get("interaction_budget")) is not int or config["interaction_budget"] <= 0:
        raise ValueError("Invalid study interaction budget")
    if not isinstance(config.get("historical_references"), Mapping):
        raise ValueError("Historical reference mapping required")
    paths = [config.get("nominal_checkpoint_path")]
    for reference in config["historical_references"].values():
        if not isinstance(reference, Mapping):
            raise ValueError("Invalid historical reference")
        if "checkpoint_sha256" in reference and not _sha256(reference["checkpoint_sha256"]):
            raise ValueError("Invalid historical checkpoint SHA-256")
        paths.extend(value for key, value in reference.items() if key != "checkpoint_sha256")
    for value in paths:
        if value is None:
            continue
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("Study source path must stay inside the results root")
    shift = config.get("dynamics_shift")
    if not isinstance(shift, Mapping) or not isinstance(shift.get("parameters"), list) or \
            not isinstance(shift.get("severities"), list) or not shift["parameters"] or not shift["severities"] or \
            not isinstance(shift.get("schedule"), str) or not shift["schedule"]:
        raise ValueError("Physical shift grid required")
    for parameter in shift["parameters"]:
        for severity in shift["severities"]:
            ExperimentIdentity("online", DYNAMICS_OBJECTIVES[0], CRITIC_OBJECTIVES[0],
                               {"parameter": parameter, "severity": severity},
                               config["nominal_checkpoint_sha256"], 0, None,
                               config["interaction_budget"], {}, None)


@dataclass(frozen=True)
class AdaptationConfig:
    method: Method
    # Passed through to the original implementation without normalization or copying.
    settings: Mapping[str, Any] | None = None
    identity: Mapping[str, Any] | None = None


class AdaptationStrategy(Protocol):
    def observe(self, *args: Any, **kwargs: Any) -> Any: ...
    def update(self) -> Any: ...
    def assert_fixed(self) -> None: ...
    def state_dict(self) -> AdaptationState: ...
    def load_state_dict(self, state: AdaptationState) -> None: ...


class AdaptationRunner(Protocol):
    implementation: Callable[..., Any]

    def run(self, *args: Any, **kwargs: Any) -> Any: ...
