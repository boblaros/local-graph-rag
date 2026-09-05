"""Hash-aware, resumable stage state independent of executable scripts."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .lineage import ArtifactRef, artifact_maps_compatible, verify_artifact

try:  # POSIX experiment environment; retain a safe single-process fallback.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


STATUS_SCHEMA_VERSION = "2.0.0"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_BY_QUALITY_GATE = "blocked_by_quality_gate"
    SKIPPED = "skipped"
    STALE = "stale"


class StatusModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StageExpectation(StatusModel):
    lineage_id: str
    config_sha256: str
    input_artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)


class StageRecord(StatusModel):
    stage: str
    item_id: str
    status: StageStatus = StageStatus.PENDING
    attempt_count: int = Field(default=0, ge=0)
    lineage_id: str
    config_sha256: str
    input_artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    output_artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    error_type: str | None = None
    message: str | None = None


class StatusSnapshot(StatusModel):
    schema_version: Literal["2.0.0"] = STATUS_SCHEMA_VERSION
    records: dict[str, StageRecord] = Field(default_factory=dict)


class ClaimResult(StatusModel):
    claimed: bool
    reused: bool
    record: StageRecord
    reasons: list[str] = Field(default_factory=list)


def _record_key(stage: str, item_id: str) -> str:
    stage = stage.strip()
    item_id = item_id.strip()
    if not stage or not item_id:
        raise ValueError("stage and item_id must be non-empty")
    return f"{stage}\0{item_id}"


def _expectation_compatible(
    record: StageRecord, expectation: StageExpectation
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if record.lineage_id != expectation.lineage_id:
        reasons.append("lineage_id mismatch")
    if record.config_sha256 != expectation.config_sha256:
        reasons.append("config_sha256 mismatch")
    compatible, artifact_reasons = artifact_maps_compatible(
        expectation.input_artifacts, record.input_artifacts
    )
    if not compatible:
        reasons.extend(artifact_reasons)
    return not reasons, reasons


class StageStatusStore:
    """A small atomic JSON state store with artifact-aware completed reuse.

    Raw artifacts are never modified here. The mutable snapshot records only
    orchestration state and immutable artifact fingerprints.
    """

    def __init__(self, path: str | Path, *, artifact_root: str | Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = Path(artifact_root).resolve() if artifact_root else None
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._thread_lock = threading.RLock()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _read_unlocked(self) -> StatusSnapshot:
        if not self.path.exists():
            return StatusSnapshot()
        with self.path.open("r", encoding="utf-8") as handle:
            return StatusSnapshot.model_validate(json.load(handle))

    def _write_unlocked(self, snapshot: StatusSnapshot) -> None:
        encoded = (
            json.dumps(
                snapshot.model_dump(mode="json", exclude_none=False),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot(self) -> StatusSnapshot:
        with self._locked():
            return self._read_unlocked()

    def get(self, stage: str, item_id: str) -> StageRecord | None:
        key = _record_key(stage, item_id)
        return self.snapshot().records.get(key)

    def list(self, *, stage: str | None = None) -> list[StageRecord]:
        records = self.snapshot().records.values()
        return sorted(
            (record for record in records if stage is None or record.stage == stage),
            key=lambda record: (record.stage, record.item_id),
        )

    def _outputs_valid(self, record: StageRecord) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not record.output_artifacts:
            return False, ["completed stage has no output artifact refs"]
        for key, ref in sorted(record.output_artifacts.items()):
            valid, artifact_reasons = verify_artifact(ref, root=self.artifact_root)
            if not valid:
                reasons.extend(f"{key}: {reason}" for reason in artifact_reasons)
        return not reasons, reasons

    def reusable(
        self,
        stage: str,
        item_id: str,
        expectation: StageExpectation,
    ) -> tuple[bool, list[str]]:
        record = self.get(stage, item_id)
        if record is None:
            return False, ["stage has no status record"]
        if record.status != StageStatus.COMPLETED:
            return False, [f"stage status is {record.status.value}"]
        compatible, reasons = _expectation_compatible(record, expectation)
        valid_outputs, output_reasons = self._outputs_valid(record)
        return compatible and valid_outputs, [*reasons, *output_reasons]

    def claim(
        self,
        stage: str,
        item_id: str,
        expectation: StageExpectation,
        *,
        retry_failed: bool = True,
        retry_blocked: bool = True,
        reclaim_running: bool = False,
    ) -> ClaimResult:
        key = _record_key(stage, item_id)
        with self._locked():
            snapshot = self._read_unlocked()
            record = snapshot.records.get(key)
            reasons: list[str] = []
            if record is not None and record.status == StageStatus.COMPLETED:
                compatible, reasons = _expectation_compatible(record, expectation)
                valid_outputs, output_reasons = self._outputs_valid(record)
                reasons.extend(output_reasons)
                if compatible and valid_outputs:
                    return ClaimResult(
                        claimed=False, reused=True, record=record, reasons=[]
                    )
                record.status = StageStatus.STALE
                record.message = "; ".join(reasons)
                record.updated_at = _now()

            if record is None:
                record = StageRecord(
                    stage=stage.strip(),
                    item_id=item_id.strip(),
                    lineage_id=expectation.lineage_id,
                    config_sha256=expectation.config_sha256,
                    input_artifacts=dict(expectation.input_artifacts),
                )
            claimable = record.status in {StageStatus.PENDING, StageStatus.STALE}
            claimable = claimable or (
                record.status == StageStatus.FAILED and retry_failed
            )
            claimable = claimable or (
                record.status == StageStatus.BLOCKED_BY_QUALITY_GATE and retry_blocked
            )
            claimable = claimable or (
                record.status == StageStatus.RUNNING and reclaim_running
            )
            if record.status == StageStatus.SKIPPED:
                claimable = False
                reasons.append("stage is explicitly skipped")
            if record.status == StageStatus.RUNNING and not reclaim_running:
                reasons.append("stage is already running")
            if not claimable:
                snapshot.records[key] = record
                self._write_unlocked(snapshot)
                return ClaimResult(
                    claimed=False, reused=False, record=record, reasons=reasons
                )

            now = _now()
            record.status = StageStatus.RUNNING
            record.attempt_count += 1
            record.lineage_id = expectation.lineage_id
            record.config_sha256 = expectation.config_sha256
            record.input_artifacts = dict(expectation.input_artifacts)
            record.output_artifacts = {}
            record.started_at = now
            record.completed_at = None
            record.updated_at = now
            record.error_type = None
            record.message = None
            snapshot.records[key] = record
            self._write_unlocked(snapshot)
            return ClaimResult(
                claimed=True, reused=False, record=record, reasons=reasons
            )

    def _finish(
        self,
        stage: str,
        item_id: str,
        status: StageStatus,
        *,
        outputs: Mapping[str, ArtifactRef] | None = None,
        error_type: str | None = None,
        message: str | None = None,
    ) -> StageRecord:
        if status not in {
            StageStatus.COMPLETED,
            StageStatus.FAILED,
            StageStatus.BLOCKED_BY_QUALITY_GATE,
            StageStatus.SKIPPED,
            StageStatus.STALE,
        }:
            raise ValueError(f"invalid terminal/update status: {status.value}")
        key = _record_key(stage, item_id)
        with self._locked():
            snapshot = self._read_unlocked()
            record = snapshot.records.get(key)
            if record is None:
                raise KeyError(f"unknown stage item: {stage}/{item_id}")
            if (
                status
                in {
                    StageStatus.COMPLETED,
                    StageStatus.FAILED,
                    StageStatus.BLOCKED_BY_QUALITY_GATE,
                }
                and record.status != StageStatus.RUNNING
            ):
                raise RuntimeError(
                    f"cannot mark {record.status.value} stage as {status.value}"
                )
            if status == StageStatus.COMPLETED:
                if not outputs:
                    raise ValueError("completed stage requires immutable output refs")
                record.output_artifacts = dict(outputs)
                valid, reasons = self._outputs_valid(record)
                if not valid:
                    raise ValueError("invalid completed outputs: " + "; ".join(reasons))
            now = _now()
            record.status = status
            record.updated_at = now
            record.completed_at = (
                now
                if status
                in {
                    StageStatus.COMPLETED,
                    StageStatus.FAILED,
                    StageStatus.BLOCKED_BY_QUALITY_GATE,
                    StageStatus.SKIPPED,
                }
                else None
            )
            record.error_type = error_type
            record.message = message
            snapshot.records[key] = record
            self._write_unlocked(snapshot)
            return record

    def mark_completed(
        self,
        stage: str,
        item_id: str,
        outputs: Mapping[str, ArtifactRef],
    ) -> StageRecord:
        return self._finish(stage, item_id, StageStatus.COMPLETED, outputs=outputs)

    def mark_failed(
        self, stage: str, item_id: str, error: BaseException | str
    ) -> StageRecord:
        return self._finish(
            stage,
            item_id,
            StageStatus.FAILED,
            error_type=type(error).__name__
            if isinstance(error, BaseException)
            else "Error",
            message=str(error),
        )

    def mark_blocked_by_quality_gate(
        self, stage: str, item_id: str, reason: str
    ) -> StageRecord:
        return self._finish(
            stage,
            item_id,
            StageStatus.BLOCKED_BY_QUALITY_GATE,
            error_type="QualityGateViolation",
            message=reason,
        )

    def mark_skipped(
        self,
        stage: str,
        item_id: str,
        expectation: StageExpectation,
        reason: str,
    ) -> StageRecord:
        key = _record_key(stage, item_id)
        with self._locked():
            snapshot = self._read_unlocked()
            record = snapshot.records.get(key) or StageRecord(
                stage=stage.strip(),
                item_id=item_id.strip(),
                lineage_id=expectation.lineage_id,
                config_sha256=expectation.config_sha256,
                input_artifacts=dict(expectation.input_artifacts),
            )
            if record.status not in {StageStatus.PENDING, StageStatus.SKIPPED}:
                raise RuntimeError(f"cannot skip stage in status {record.status.value}")
            now = _now()
            record.status = StageStatus.SKIPPED
            record.updated_at = now
            record.completed_at = now
            record.message = reason
            snapshot.records[key] = record
            self._write_unlocked(snapshot)
            return record

    def mark_stale(self, stage: str, item_id: str, reason: str) -> StageRecord:
        return self._finish(stage, item_id, StageStatus.STALE, message=reason)


__all__ = [
    "ClaimResult",
    "STATUS_SCHEMA_VERSION",
    "StageExpectation",
    "StageRecord",
    "StageStatus",
    "StageStatusStore",
    "StatusSnapshot",
]
