"""Canonical serialization and hashing shared by the Native runtime."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


_SLUG_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def _datetime_to_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Naive datetimes are not permitted in canonical JSON")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    """Convert common Python/Pydantic values to deterministic JSON values.

    Mapping keys must be strings. Sets are sorted by their own canonical JSON
    representation, and non-finite floats are rejected by ``canonical_json``.
    """

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floats are not permitted in canonical JSON")
        return value
    if isinstance(value, datetime):
        return _datetime_to_iso(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, bytes):
        return {"$bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Canonical JSON mapping key must be str, got {type(key)!r}"
                )
            result[key] = to_jsonable(item)
        return result
    if isinstance(value, (set, frozenset)):
        converted = [to_jsonable(item) for item in value]
        return sorted(converted, key=canonical_json)
    if isinstance(value, Sequence):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"Unsupported value for canonical JSON: {type(value)!r}")


def canonical_json(value: Any) -> str:
    """Return UTF-8-safe canonical JSON used by every content/config hash."""

    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    """Create a stable identifier from length-delimited canonical parts."""

    if not 8 <= length <= 64:
        raise ValueError("length must be between 8 and 64")
    framed = [
        {"index": index, "value": to_jsonable(part)} for index, part in enumerate(parts)
    ]
    return f"{prefix}{sha256_json(framed)[:length]}"


def slugify(value: Any, *, max_length: int = 48, fallback: str = "unknown") -> str:
    """Make a portable lowercase component for a run directory name."""

    normalized = (
        unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    )
    slug = _SLUG_SEPARATOR_RE.sub("-", normalized.lower()).strip("-")
    slug = slug[:max_length].rstrip("-")
    return slug or fallback
