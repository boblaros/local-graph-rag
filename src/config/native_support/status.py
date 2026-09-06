"""SQLite-backed resumable execution status for the Native runtime.

The database stores only stage state. Raw artifacts, metrics, configuration,
and model output never enter SQLite.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


StatusValue = Literal["pending", "running", "completed", "failed", "skipped"]
_VALID_STATUSES = {"pending", "running", "completed", "failed", "skipped"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class StatusRecord:
    stage: str
    item_id: str
    status: StatusValue
    attempt_count: int
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class StatusStore:
    """Small transactional status store suitable for stage/item resumption."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS execution_status (
                    stage TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'running', 'completed', 'failed', 'skipped')
                    ),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    PRIMARY KEY (stage, item_id)
                );
                CREATE INDEX IF NOT EXISTS idx_execution_status_stage_status
                    ON execution_status(stage, status);
                PRAGMA user_version = 1;
                """
            )

    @staticmethod
    def _validate_key(stage: str, item_id: str) -> tuple[str, str]:
        stage = str(stage).strip()
        item_id = str(item_id).strip()
        if not stage or not item_id:
            raise ValueError("stage and item_id must not be empty")
        return stage, item_id

    @staticmethod
    def _from_row(row: sqlite3.Row | None) -> StatusRecord | None:
        if row is None:
            return None
        return StatusRecord(**dict(row))

    def initialize_items(self, stage: str, item_ids: Iterable[str]) -> int:
        """Insert missing pending items and return the number newly inserted."""

        stage = str(stage).strip()
        if not stage:
            raise ValueError("stage must not be empty")
        now = _now_iso()
        rows = []
        seen: set[str] = set()
        for raw_item_id in item_ids:
            item_id = str(raw_item_id).strip()
            if not item_id:
                raise ValueError("item_id must not be empty")
            if item_id not in seen:
                seen.add(item_id)
                rows.append((stage, item_id, "pending", now, now))
        if not rows:
            return 0
        with self._lock:
            before = self._connection.total_changes
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO execution_status
                    (stage, item_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            return self._connection.total_changes - before

    def get(self, stage: str, item_id: str) -> StatusRecord | None:
        stage, item_id = self._validate_key(stage, item_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM execution_status WHERE stage = ? AND item_id = ?",
                (stage, item_id),
            ).fetchone()
        return self._from_row(row)

    def list(
        self,
        *,
        stage: str | None = None,
        status: StatusValue | None = None,
    ) -> list[StatusRecord]:
        clauses: list[str] = []
        parameters: list[str] = []
        if stage is not None:
            clauses.append("stage = ?")
            parameters.append(stage)
        if status is not None:
            if status not in _VALID_STATUSES:
                raise ValueError(f"invalid status: {status}")
            clauses.append("status = ?")
            parameters.append(status)
        query = "SELECT * FROM execution_status"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY stage, item_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [self._from_row(row) for row in rows if row is not None]  # type: ignore[misc]

    def claim(
        self,
        stage: str,
        item_id: str,
        *,
        retry_failed: bool = True,
        reclaim_running: bool = False,
    ) -> bool:
        """Atomically claim an item, incrementing its execution attempt.

        Completed/skipped items are never reclaimed. ``reclaim_running`` is
        intended for explicit crash recovery, not concurrent workers.
        """

        stage, item_id = self._validate_key(stage, item_id)
        now = _now_iso()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status FROM execution_status WHERE stage = ? AND item_id = ?",
                    (stage, item_id),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO execution_status (
                            stage, item_id, status, attempt_count, created_at,
                            updated_at, started_at
                        ) VALUES (?, ?, 'running', 1, ?, ?, ?)
                        """,
                        (stage, item_id, now, now, now),
                    )
                    claimed = True
                else:
                    current = str(row["status"])
                    claimable = current == "pending"
                    claimable = claimable or (current == "failed" and retry_failed)
                    claimable = claimable or (current == "running" and reclaim_running)
                    if claimable:
                        self._connection.execute(
                            """
                            UPDATE execution_status
                            SET status = 'running',
                                attempt_count = attempt_count + 1,
                                updated_at = ?, started_at = ?, completed_at = NULL,
                                error_type = NULL, error_message = NULL
                            WHERE stage = ? AND item_id = ?
                            """,
                            (now, now, stage, item_id),
                        )
                        claimed = True
                    else:
                        claimed = False
                self._connection.execute("COMMIT")
                return claimed
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def _finish(
        self,
        stage: str,
        item_id: str,
        status: StatusValue,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        stage, item_id = self._validate_key(stage, item_id)
        if status not in {"completed", "failed", "skipped"}:
            raise ValueError("finish status must be completed, failed, or skipped")
        now = _now_iso()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE execution_status
                SET status = ?, updated_at = ?, completed_at = ?,
                    error_type = ?, error_message = ?
                WHERE stage = ? AND item_id = ?
                """,
                (status, now, now, error_type, error_message, stage, item_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"status item does not exist: {stage}/{item_id}")

    def mark_completed(self, stage: str, item_id: str) -> None:
        self._finish(stage, item_id, "completed")

    def mark_failed(
        self,
        stage: str,
        item_id: str,
        error: BaseException | str,
        *,
        error_type: str | None = None,
    ) -> None:
        if isinstance(error, BaseException):
            resolved_type = error_type or type(error).__name__
            message = str(error)
        else:
            resolved_type = error_type or "ExecutionError"
            message = str(error)
        self._finish(
            stage,
            item_id,
            "failed",
            error_type=resolved_type,
            error_message=message,
        )

    def mark_skipped(self, stage: str, item_id: str, reason: str | None = None) -> None:
        self._finish(
            stage,
            item_id,
            "skipped",
            error_type="skipped" if reason else None,
            error_message=reason,
        )

    def is_completed(self, stage: str, item_id: str) -> bool:
        record = self.get(stage, item_id)
        return bool(record and record.status == "completed")

    def reset_running(self, *, stage: str | None = None) -> int:
        """Return stale running rows to pending during explicit crash recovery."""

        now = _now_iso()
        query = (
            "UPDATE execution_status SET status = 'pending', updated_at = ?, "
            "started_at = NULL WHERE status = 'running'"
        )
        parameters: list[str] = [now]
        if stage is not None:
            query += " AND stage = ?"
            parameters.append(stage)
        with self._lock:
            cursor = self._connection.execute(query, parameters)
            return cursor.rowcount

    def counts(self, *, stage: str | None = None) -> dict[str, int]:
        query = "SELECT status, COUNT(*) AS count FROM execution_status"
        parameters: list[str] = []
        if stage is not None:
            query += " WHERE stage = ?"
            parameters.append(stage)
        query += " GROUP BY status ORDER BY status"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "StatusStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["StatusRecord", "StatusStore", "StatusValue"]
