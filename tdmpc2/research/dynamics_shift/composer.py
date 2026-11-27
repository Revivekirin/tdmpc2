"""In-memory evaluation-controller composition; no training or source writes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from .checkpoints import LoadedCheckpoint, model_hashes

Controller = Literal["frozen", "dynamics_only", "dynamics_q", "controlled_q_swap", "joint"]


@dataclass
class ComposedController:
    kind: Controller
    model: Any
    component_hashes: dict[str, str]
    source_sha256: tuple[str, ...]


class CheckpointComposer:
    """Construct independent, frozen models from verified checkpoint descriptors."""

    def __init__(self, nominal: LoadedCheckpoint):
        if nominal.descriptor.format != "nominal":
            raise ValueError("Composer requires a nominal checkpoint")
        self.nominal = nominal

    def _compatible(self, source: LoadedCheckpoint) -> None:
        a, b = self.nominal.descriptor, source.descriptor
        if a.nominal_sha256 != b.nominal_sha256 or a.task != b.task:
            raise ValueError("Controller task or nominal parent mismatch")
        if a.config_sha256 != b.config_sha256 or a.observation_shape != b.observation_shape:
            raise ValueError("Controller configuration or observations mismatch")
        if a.action_dim != b.action_dim or a.latent_dim != b.latent_dim:
            raise ValueError("Controller action or latent dimensions mismatch")

    @staticmethod
    def _copy(source: LoadedCheckpoint, kind: Controller) -> ComposedController:
        model = deepcopy(source.model)
        model.requires_grad_(False)
        model.eval()
        hashes = model_hashes(model)
        if hashes != source.descriptor.component_hashes:
            raise ValueError("Source model changed after checkpoint loading")
        return ComposedController(kind, model, hashes, (source.descriptor.source_sha256,))

    def compose(self, kind: Controller, source: LoadedCheckpoint | None = None) -> ComposedController:
        if kind == "frozen":
            if source is not None:
                raise ValueError("Frozen controller takes no adaptation snapshot")
            return self._copy(self.nominal, kind)
        if kind == "controlled_q_swap":
            if source is None:
                raise ValueError("Controlled Q swap requires a D+Q snapshot")
            return self.controlled_q_swap(source)
        expected = {"dynamics_only": "dynamics_only", "dynamics_q": "dynamics_q",
                    "joint": "joint"}.get(kind)
        if expected is None or source is None or source.descriptor.format != expected:
            raise ValueError("Controller and snapshot formats differ")
        self._compatible(source)
        result = self._copy(source, kind)
        nominal_hashes = self.nominal.descriptor.component_hashes
        fixed = ("encoder", "reward", "online_q", "target_q", "actor") if kind == "dynamics_only" else \
                ("encoder", "reward", "actor") if kind == "dynamics_q" else ()
        for component in fixed:
            if result.component_hashes[component] != nominal_hashes[component]:
                raise ValueError(f"Adaptation changed fixed component: {component}")
        return result

    def controlled_q_swap(self, adapted: LoadedCheckpoint) -> ComposedController:
        if adapted.descriptor.format != "dynamics_q":
            raise ValueError("Controlled Q swap requires a D+Q model")
        self._compatible(adapted)
        original = self.compose("dynamics_q", adapted)
        reference = self.nominal.descriptor.component_hashes
        before = original.component_hashes
        for component in ("encoder", "reward", "actor", "normalization"):
            if before[component] != reference[component]:
                raise ValueError(f"Nominal critic latent contract differs: {component}")
        # _Qs.params and _detach_Qs_params share tensor storage in WorldModel.
        # Loading _Qs in place therefore changes the native planning critic and
        # its detach alias. The independently stored target critic is untouched.
        original.model._Qs.load_state_dict(self.nominal.model._Qs.state_dict(), strict=True)
        after = model_hashes(original.model)
        for component in ("encoder", "dynamics", "reward", "target_q", "actor", "normalization"):
            if after[component] != before[component]:
                raise ValueError(f"Q swap changed {component}")
        if after["online_q"] != reference["online_q"]:
            raise ValueError("Nominal online critic was not restored")
        changed = {key for key, tensor in original.model.state_dict().items()
                   if not tensor.equal(adapted.model.state_dict()[key])}
        if not all(key.startswith(("_Qs.params.", "_detach_Qs_params.")) for key in changed):
            raise ValueError(f"Q swap changed non-critic state: {sorted(changed)}")
        if model_hashes(adapted.model) != adapted.descriptor.component_hashes:
            raise ValueError("Adapted source mutated")
        if model_hashes(self.nominal.model) != reference:
            raise ValueError("Nominal source mutated")
        original.kind = "controlled_q_swap"
        original.component_hashes = after
        original.source_sha256 = (adapted.descriptor.source_sha256,
                                  self.nominal.descriptor.source_sha256)
        return original
