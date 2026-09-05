"""Small immutable artifact helpers used by importable lifecycle stages."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def write_immutable_bytes(path: str | Path, content: bytes) -> Path:
    destination = Path(path)
    if destination.exists():
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable artifact conflict: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return write_immutable_bytes(destination, content)
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def write_immutable_json(path: str | Path, value: Any) -> Path:
    content = (
        json.dumps(
            jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return write_immutable_bytes(path, content)


def write_immutable_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    content = "".join(canonical_json(jsonable(row)) + "\n" for row in rows).encode(
        "utf-8"
    )
    return write_immutable_bytes(path, content)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected object")
            result.append(value)
    return result


def read_enveloped_jsonl(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    if not rows or rows[0].get("record_type") != "artifact_header":
        raise ValueError(f"ER artifact has no valid header: {path}")
    header = rows[0]
    records: list[dict[str, Any]] = []
    for line_number, envelope in enumerate(rows[1:], 2):
        record = envelope.get("record")
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: invalid record envelope")
        records.append(record)
    return header, records


__all__ = [
    "canonical_json",
    "jsonable",
    "read_enveloped_jsonl",
    "read_json",
    "read_jsonl",
    "write_immutable_bytes",
    "write_immutable_json",
    "write_immutable_jsonl",
]
