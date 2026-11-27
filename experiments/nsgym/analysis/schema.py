"""Canonical records shared by task and historical-format adapters."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

IDENTITY = (
    "task", "environment_id", "method", "checkpoint_id", "training_seed",
    "adaptation_seed", "evaluation_seed", "shift_family", "shift_parameter",
    "severity_multiplier", "applied_parameter_value", "schedule", "shift_onset",
    "adaptation_budget", "optimizer_steps", "model_hash", "source_run_id",
    "source_record_id", "completion_status",
)
COMMON = IDENTITY + ("kind", "metric", "value", "unit", "sample_count", "source_path", "source_sha256")


def sha256(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with Path(path).open(newline="") as stream:
        yield from csv.DictReader(stream)


def write_csv(path: Path, data: list[dict], fields: tuple[str, ...] | list[str]):
    return save_table(path, data, fields, mode="w", ignore_extra=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def value(row: dict, *names: str):
    for name in names:
        result = row.get(name)
        if result not in (None, "", "NA", "NOT_EVALUATED"):
            return result
    return ""


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical(identity: dict, *, kind: str, metric: str, value_, source: Path,
              source_root: Path, row_number: int, source_hash: str, unit="", sample_count=""):
    result = {field: identity.get(field, "") for field in IDENTITY}
    result.update(kind=kind, metric=metric, value=value_, unit=unit,
                  sample_count=sample_count, source_path=str(source.relative_to(source_root)),
                  source_sha256=source_hash,
                  source_record_id=f"{source.relative_to(source_root)}:{row_number}")
    return result


def read_required(path: Path):
    """Strict counterpart of optional manifest reads; missing evidence is an error."""
    return json.loads(Path(path).read_text())


def list_rows(path):
    return list(rows(Path(path)))


def save_table(path, data, fields=None, *, mode='x', empty='header', ignore_extra=False):
    """Historical CSV write contracts remain explicit; existing evidence is never overwritten by default."""
    data=list(data)
    if not data and empty=='skip': return
    if not data and empty=='error': raise ValueError(f'No rows for {path}')
    fields=fields or list(dict.fromkeys(k for row in data for k in row))
    with Path(path).open(mode,newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore' if ignore_extra else 'raise')
        writer.writeheader();writer.writerows(data)
