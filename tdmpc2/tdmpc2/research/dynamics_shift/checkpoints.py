"""Read-only historical NS-Gym checkpoint loading and model integrity.

This module imports no simulator or TD-MPC2 agent. It constructs the original
WorldModel with the saved configuration and applies the agent's legacy model
key conversion before a strict state-dict load.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch

CheckpointFormat = Literal["nominal", "dynamics_only", "dynamics_q", "joint"]
COMPONENTS = {
    "encoder": "_encoder", "dynamics": "_dynamics", "reward": "_reward",
    "online_q": "_Qs", "target_q": "_target_Qs_params", "actor": "_pi",
}


class _Config(SimpleNamespace):
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: dict[str, Any]) -> str:
    """Hash state keys, tensor metadata and raw values on CPU."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def model_hashes(model: Any) -> dict[str, str]:
    result = {name: state_hash(getattr(model, attr).state_dict())
              for name, attr in COMPONENTS.items()}
    result["model"] = state_hash(model.state_dict())
    result["normalization"] = state_hash({k: v for k, v in model.state_dict().items()
                                           if k.startswith("log_std_")})
    return result


def config_identity(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CheckpointDescriptor:
    format: CheckpointFormat
    path: Path
    source_sha256: str
    nominal_sha256: str
    config_sha256: str
    task: str
    environment_id: str | None
    observation_shape: dict[str, Any]
    action_dim: int
    latent_dim: int
    adaptation_identity: dict[str, Any] | None
    adaptation_configuration: dict[str, Any] | None
    source_hashes: dict[str, str] | None
    transitions: int | None
    optimizer_keys: tuple[str, ...]
    component_hashes: dict[str, str]
    resume_marker: str | None
    exact_resume_supported: bool = False


@dataclass
class LoadedCheckpoint:
    model: Any
    descriptor: CheckpointDescriptor


def _read(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected checkpoint dictionary: {path}")
    return payload


def _require_keys(payload: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{label} missing keys: {sorted(missing)}")


def _model(config: dict[str, Any], state: dict[str, Any]) -> Any:
    from common.layers import api_model_conversion
    from common.world_model import WorldModel

    if not isinstance(state, dict):
        raise ValueError("Model state must be a dictionary")
    # Construction initializes random weights; loading must not advance caller RNG.
    with torch.random.fork_rng(devices=[]):
        model = WorldModel(_Config(**config)).to("cpu")
    converted = api_model_conversion(model.state_dict(), deepcopy(state))
    expected = model.state_dict()
    if set(converted) != set(expected):
        raise ValueError(f"Model state keys differ: missing={sorted(set(expected)-set(converted))}, "
                         f"unexpected={sorted(set(converted)-set(expected))}")
    for key, value in expected.items():
        other = converted[key]
        if torch.is_tensor(value) and (not torch.is_tensor(other) or value.shape != other.shape):
            raise ValueError(f"Model tensor shape mismatch: {key}")
    model.load_state_dict(converted, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def load_checkpoint(
    nominal_directory: str | Path,
    snapshot: str | Path | None = None,
    *,
    expected_task: str | None = None,
    expected_action_dim: int | None = None,
    expected_sha256: str | None = None,
) -> LoadedCheckpoint:
    """Load one verified nominal or adaptation snapshot into an evaluation model.

    Missing historical metadata stays ``None`` in the descriptor. No optimizer,
    replay, environment or global RNG state is restored or synthesized.
    """
    nominal_directory = Path(nominal_directory)
    metadata = json.loads((nominal_directory / "metadata.json").read_text())
    _require_keys(metadata, {"config", "model_sha256", "optimizer_sha256"}, "Nominal metadata")
    config = metadata["config"]
    _require_keys(config, {"task", "action_dim", "obs_shape", "latent_dim"}, "Model config")
    if expected_task is not None and config["task"] != expected_task:
        raise ValueError("Checkpoint task mismatch")
    if expected_action_dim is not None and config["action_dim"] != expected_action_dim:
        raise ValueError("Checkpoint action dimension mismatch")
    nominal_path = nominal_directory / "model.pt"
    nominal_hash = file_sha256(nominal_path)
    if nominal_hash != metadata["model_sha256"]:
        raise ValueError("Nominal model SHA-256 mismatch")
    optimizer_path = nominal_directory / "optimizers.pt"
    if file_sha256(optimizer_path) != metadata["optimizer_sha256"]:
        raise ValueError("Nominal optimizer SHA-256 mismatch")
    optimizers = _read(optimizer_path)
    _require_keys(optimizers, {"optim", "pi_optim", "scale"}, "Nominal optimizer")
    nominal = _read(nominal_path)
    _require_keys(nominal, {"model"}, "Nominal model")
    path = Path(snapshot) if snapshot is not None else nominal_path
    source_hash = file_sha256(path)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError("Snapshot SHA-256 mismatch")
    payload = nominal if snapshot is None else _read(path)
    identity = None
    adaptation_configuration = None
    source_hashes = None
    resume_marker = None
    transitions = None
    optimizer_keys: tuple[str, ...] = ()
    if snapshot is None:
        kind: CheckpointFormat = "nominal"
        model = _model(config, nominal["model"])
    else:
        _require_keys(payload, {"identity"}, "Adaptation snapshot")
        parent_hash = payload.get("checkpoint_sha256")
        if parent_hash is None and isinstance(payload.get("selection"), dict):
            parent_hash = payload["selection"].get("model_sha256")
            if payload["selection"].get("configuration") != config:
                raise ValueError("Selected nominal configuration mismatch")
        if parent_hash != nominal_hash:
            raise ValueError("Adaptation snapshot has another nominal parent")
        identity = payload["identity"]
        adaptation_configuration = payload.get("protocol", payload.get("configuration", payload.get("config")))
        source_hashes = payload.get("source_hashes")
        resume_marker = payload.get("resume")
        _require_keys(identity, {"parameter", "severity", "adaptation_seed", "method"},
                      "Adaptation identity")
        if "adapter" in payload and "model" not in payload:
            kind = "dynamics_only"
            _require_keys(payload, {"fixed_hash", "model_hash"}, "Dynamics-only snapshot")
            if not any(key in payload for key in ("configuration", "config")):
                raise ValueError("Dynamics-only adaptation configuration missing")
            _require_keys(payload["adapter"], {"predictor", "optimizer", "counters"},
                          "Dynamics-only adapter")
            model = _model(config, nominal["model"])
            model._dynamics.load_state_dict(payload["adapter"]["predictor"], strict=True)
            transitions = int(payload["adapter"]["counters"][1])
            optimizer_keys = ("adapter.optimizer",)
        elif "adapter" in payload and "model" in payload:
            kind = "dynamics_q"
            _require_keys(payload, {"configuration", "fixed_hash"}, "Dynamics+Q snapshot")
            if payload.get("checkpoint_config", config) != config:
                raise ValueError("Dynamics+Q model configuration mismatch")
            _require_keys(payload["adapter"], {"dynamics_optimizer", "q_optimizer", "transitions"},
                          "Dynamics+Q adapter")
            model = _model(config, payload["model"])
            transitions = int(payload["adapter"]["transitions"])
            optimizer_keys = ("adapter.dynamics_optimizer", "adapter.q_optimizer")
        elif "model" in payload and "optim" in payload:
            kind = "joint"
            _require_keys(payload, {"configuration", "pi_optim", "scale", "transitions", "updates"},
                          "Joint snapshot")
            if payload["configuration"] != config:
                raise ValueError("Joint model configuration mismatch")
            model = _model(config, payload["model"])
            transitions = int(payload["transitions"])
            optimizer_keys = ("optim", "pi_optim", "scale")
        else:
            raise ValueError("Unrecognized historical checkpoint format")
    descriptor = CheckpointDescriptor(
        format=kind, path=path, source_sha256=source_hash, nominal_sha256=nominal_hash,
        config_sha256=config_identity(config), task=config["task"],
        environment_id=config.get("environment_id"), observation_shape=config["obs_shape"],
        action_dim=int(config["action_dim"]), latent_dim=int(config["latent_dim"]),
        adaptation_identity=identity, adaptation_configuration=adaptation_configuration,
        source_hashes=source_hashes, transitions=transitions, optimizer_keys=optimizer_keys,
        component_hashes=model_hashes(model), resume_marker=resume_marker,
    )
    return LoadedCheckpoint(model, descriptor)
