"""Structured append-only event logging for Native runtime stages."""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

try:
    from .io_utils import append_jsonl, touch_append_only
    from .schemas import EventRecord
except ImportError:  # pragma: no cover - direct script/module bootstrap.
    from io_utils import append_jsonl, touch_append_only  # type: ignore
    from schemas import EventRecord  # type: ignore


class EventLogger:
    def __init__(
        self,
        path: str | Path,
        run_id: str,
        *,
        default_stage: str = "harness",
        context: dict[str, Any] | None = None,
    ):
        self.path = touch_append_only(path)
        self.run_id = run_id
        self.default_stage = default_stage
        self.context = dict(context or {})
        self.logs_dir = self.path.parent
        self.pipeline_log = touch_append_only(self.logs_dir / "pipeline.log")
        self.errors_path = touch_append_only(self.logs_dir / "errors.jsonl")

    @staticmethod
    def _append_text(path: Path, line: str) -> None:
        encoded = (line.rstrip("\n") + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)

    @staticmethod
    def _stage_log_name(stage: str) -> str:
        aliases = {"answer": "answers", "workspace_export": "indexing"}
        normalized = aliases.get(stage, stage)
        safe = re.sub(r"[^a-z0-9_-]+", "_", normalized.casefold()).strip("_")
        return f"{safe or 'harness'}.log"

    def log(
        self,
        event_type: str,
        message: str,
        *,
        level: str = "INFO",
        stage: str | None = None,
        item_id: str | None = None,
        elapsed_ms: float | None = None,
        payload: dict[str, Any] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        sync: bool = True,
    ) -> EventRecord:
        merged_payload = {**self.context, **(payload or {})}
        record = EventRecord(
            event_id=f"evt_{uuid.uuid4().hex}",
            run_id=self.run_id,
            level=level.upper(),
            stage=stage or self.default_stage,
            event_type=event_type,
            message=message,
            item_id=item_id,
            elapsed_ms=elapsed_ms,
            payload=merged_payload,
            error_type=error_type,
            error_message=error_message,
        )
        append_jsonl(self.path, record, sync=sync)
        record_json = record.model_dump(mode="json", exclude_none=False)
        timestamp = str(record_json["timestamp"])
        builder = merged_payload.get("builder_model", "-")
        item = item_id or "-"
        line = (
            f"{timestamp} {record.level} run_id={self.run_id} "
            f"stage={record.stage} builder={builder} item_id={item} "
            f"event={event_type} {message}"
        )
        self._append_text(self.pipeline_log, line)
        stage_path = touch_append_only(
            self.logs_dir / self._stage_log_name(record.stage)
        )
        self._append_text(stage_path, line)
        print(line, flush=True)
        if record.level in {"ERROR", "CRITICAL"}:
            error_record = {
                **record_json,
                "builder_model": merged_payload.get("builder_model"),
                "document_or_question_id": item_id,
                "traceback": merged_payload.get("traceback"),
            }
            append_jsonl(self.errors_path, error_record, sync=sync)
        return record

    def debug(self, event_type: str, message: str, **kwargs: Any) -> EventRecord:
        return self.log(event_type, message, level="DEBUG", **kwargs)

    def info(self, event_type: str, message: str, **kwargs: Any) -> EventRecord:
        return self.log(event_type, message, level="INFO", **kwargs)

    def warning(self, event_type: str, message: str, **kwargs: Any) -> EventRecord:
        return self.log(event_type, message, level="WARNING", **kwargs)

    def error(self, event_type: str, message: str, **kwargs: Any) -> EventRecord:
        return self.log(event_type, message, level="ERROR", **kwargs)

    def exception(
        self,
        event_type: str,
        error: BaseException,
        *,
        message: str | None = None,
        **kwargs: Any,
    ) -> EventRecord:
        formatted_traceback = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        payload = dict(kwargs.pop("payload", {}) or {})
        payload["traceback"] = formatted_traceback
        return self.log(
            event_type,
            message or str(error) or type(error).__name__,
            level="ERROR",
            error_type=type(error).__name__,
            error_message=str(error),
            payload=payload,
            **kwargs,
        )

    @contextmanager
    def span(
        self,
        event_type: str,
        *,
        stage: str | None = None,
        item_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        started = time.perf_counter()
        self.info(
            f"{event_type}.started",
            f"{event_type} started",
            stage=stage,
            item_id=item_id,
            payload=payload,
        )
        try:
            yield
        except BaseException as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.exception(
                f"{event_type}.failed",
                exc,
                message=f"{event_type} failed",
                stage=stage,
                item_id=item_id,
                elapsed_ms=elapsed_ms,
                payload=payload,
            )
            raise
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.info(
                f"{event_type}.completed",
                f"{event_type} completed",
                stage=stage,
                item_id=item_id,
                elapsed_ms=elapsed_ms,
                payload=payload,
            )

    @asynccontextmanager
    async def async_span(
        self,
        event_type: str,
        *,
        stage: str | None = None,
        item_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AsyncIterator[None]:
        with self.span(
            event_type,
            stage=stage,
            item_id=item_id,
            payload=payload,
        ):
            yield


class JsonlLoggingHandler(logging.Handler):
    """Optional bridge from stdlib logging into the structured event log."""

    def __init__(self, event_logger: EventLogger, *, stage: str | None = None):
        super().__init__()
        self.event_logger = event_logger
        self.stage = stage

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            error_type = None
            error_message = None
            if record.exc_info and record.exc_info[1] is not None:
                error_type = type(record.exc_info[1]).__name__
                error_message = str(record.exc_info[1])
            self.event_logger.log(
                "python.log",
                message,
                level=record.levelname
                if record.levelname in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
                else "INFO",
                stage=self.stage,
                payload={"logger": record.name, "module": record.module},
                error_type=error_type,
                error_message=error_message,
                sync=False,
            )
        except Exception:
            self.handleError(record)


__all__ = ["EventLogger", "JsonlLoggingHandler"]
