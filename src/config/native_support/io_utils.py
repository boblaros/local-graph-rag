"""Native-runtime I/O primitives with explicit raw/derived artifact semantics.

Raw JSONL is append-only. Frozen JSON is create-only. Derived JSON, CSV, and
Parquet outputs are atomically replaced so they can be recomputed safely.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

try:
    from .hashes import canonical_json, to_jsonable
except ImportError:  # pragma: no cover - direct script/module bootstrap.
    from hashes import canonical_json, to_jsonable  # type: ignore


try:  # POSIX advisory lock; experiment runs target macOS/Linux.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback retains O_APPEND.
    fcntl = None  # type: ignore[assignment]


T = TypeVar("T")


class JsonlDecodeError(ValueError):
    def __init__(self, path: Path, line_number: int, message: str):
        super().__init__(f"{path}:{line_number}: {message}")
        self.path = path
        self.line_number = line_number


def _record_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        value = record.model_dump(mode="json", exclude_none=False)
    elif isinstance(record, Mapping):
        value = dict(record)
    else:
        raise TypeError("JSONL records must be Pydantic models or mappings")
    converted = to_jsonable(value)
    if not isinstance(converted, dict):
        raise TypeError("JSONL record must serialize to an object")
    return converted


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@contextmanager
def _temporary_peer(destination: Path, *, suffix: str = ".tmp") -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str | Path, value: Any) -> Path:
    """Atomically replace a derived JSON file."""

    destination = Path(path)
    payload = (
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    with _temporary_peer(destination) as temporary:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    return destination


def write_json_exclusive(path: str | Path, value: Any) -> Path:
    """Create immutable JSON without ever replacing an existing target."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    # Linking a fully fsynced peer gives create-if-absent semantics without a
    # window in which readers can observe a partially written config lock.
    with _temporary_peer(destination, suffix=".freeze") as temporary:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    return destination


def touch_append_only(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    os.close(descriptor)
    return destination


def append_jsonl(
    path: str | Path,
    record: Any,
    *,
    require_schema_version: bool = True,
    sync: bool = True,
) -> int:
    """Append exactly one JSON object and return its starting byte offset.

    A POSIX advisory lock prevents interleaving when async workers or separate
    harness processes share a run file. No update/truncate operation is
    exposed for raw artifacts.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = _record_dict(record)
    if require_schema_version and not value.get("schema_version"):
        raise ValueError("append-only records must include schema_version")
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        offset = os.lseek(descriptor, 0, os.SEEK_END)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        if sync:
            os.fsync(descriptor)
        return offset
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)


def iter_jsonl(
    path: str | Path,
    *,
    model: type[T] | None = None,
    tolerate_truncated_tail: bool = False,
) -> Iterator[dict[str, Any] | T]:
    """Stream JSONL records, optionally validating each with a Pydantic model."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if tolerate_truncated_tail:
                    remainder = handle.read()
                    if not remainder:
                        return
                raise JsonlDecodeError(source, line_number, str(exc)) from exc
            if not isinstance(value, dict):
                raise JsonlDecodeError(
                    source, line_number, "record is not a JSON object"
                )
            if model is not None:
                validator = getattr(model, "model_validate", None)
                if validator is None:
                    raise TypeError("model must expose Pydantic v2 model_validate")
                yield validator(value)
            else:
                yield value


def read_jsonl(
    path: str | Path,
    *,
    model: type[T] | None = None,
    tolerate_truncated_tail: bool = False,
) -> list[dict[str, Any] | T]:
    return list(
        iter_jsonl(
            path,
            model=model,
            tolerate_truncated_tail=tolerate_truncated_tail,
        )
    )


def write_csv_atomic(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Atomically replace a derived CSV file."""

    destination = Path(path)
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        fieldnames = list(materialized[0]) if materialized else []
    with _temporary_peer(destination, suffix=".csv.tmp") as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fieldnames), extrasaction="raise"
            )
            writer.writeheader()
            for row in materialized:
                writer.writerow({key: row.get(key) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    return destination


def write_parquet(
    path: str | Path,
    rows: Any,
    *,
    columns: Sequence[str] | None = None,
    index: bool = False,
) -> Path:
    """Atomically replace a derived Parquet table using the PyArrow engine."""

    try:
        import pandas as pd
        import pyarrow  # noqa: F401 - fail early with an actionable dependency error.
    except ImportError as exc:  # pragma: no cover - depends on environment setup.
        raise RuntimeError(
            "Parquet output requires pandas and pyarrow; install the evaluation dependencies"
        ) from exc

    destination = Path(path)
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
        if columns is not None:
            frame = frame.reindex(columns=list(columns))
    else:
        frame = pd.DataFrame.from_records(list(rows), columns=columns)

    with _temporary_peer(destination, suffix=".parquet.tmp") as temporary:
        frame.to_parquet(temporary, engine="pyarrow", index=index)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    return destination


write_parquet_atomic = write_parquet


def read_parquet(path: str | Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Parquet input requires pandas and pyarrow") from exc
    return pd.read_parquet(Path(path), engine="pyarrow")


__all__ = [
    "JsonlDecodeError",
    "append_jsonl",
    "iter_jsonl",
    "read_json",
    "read_jsonl",
    "read_parquet",
    "touch_append_only",
    "write_csv_atomic",
    "write_json_atomic",
    "write_json_exclusive",
    "write_parquet",
    "write_parquet_atomic",
]
