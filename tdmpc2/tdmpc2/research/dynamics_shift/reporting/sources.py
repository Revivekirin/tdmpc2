"""Source roots and immutable-byte checks for canonical report inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from experiments.nsgym.analysis.schema import sha256


def validate_record_sources(root: Path, records: Iterable[dict]) -> dict[str, str]:
    """Reject path escape, missing inputs, and changed historical record bytes."""
    root = Path(root).resolve(strict=True)
    seen: dict[str, str] = {}
    for row in records:
        relative = Path(row["source_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Source outside explicit root: {relative}")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Source outside explicit root or missing: {relative}")
        expected = row["source_sha256"]
        key = relative.as_posix()
        if key not in seen:
            seen[key] = sha256(path)
        actual = seen[key]
        if actual != expected:
            raise ValueError(f"Historical source hash mismatch: {relative}")
    return seen


def validate_control_q_sources(root: Path):
    """Use the existing strict three-family, 36-condition/108-stream gate."""
    from experiments.nsgym.analysis.control_q.data import load_inputs
    return load_inputs(Path(root))
