"""Schemas for fixed Native configuration and saved records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


SCHEMA_VERSION = "1.0.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExtensibleModel(BaseModel):
    # Preserve additional provenance fields emitted by the pinned dependency.
    model_config = ConfigDict(extra="allow", validate_assignment=True)


class VersionedRecord(ExtensibleModel):
    schema_version: str = SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _schema_version_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("schema_version must not be empty")
        return value


class TimedRecord(VersionedRecord):
    created_at: datetime = Field(default_factory=utc_now)


class ModelIdentity(StrictModel):
    name: str
    digest: str | None = None
    quantization: str | None = None
    provider: str = "ollama"
    parameters: dict[str, Any] = Field(default_factory=dict)


class RunConfig(VersionedRecord):
    """Minimum scientific dimensions required for a deterministic run ID.

    Additional experiment parameters are accepted and become part of the
    canonical config hash, so callers can extend the config without changing
    this base schema.
    """

    builder_model: str
    quantization: str
    graph_regime: str
    retrieval_mode: str = "hybrid"
    seed: int
    answer_model: str
    keyword_model: str | None = None
    builder_model_digest: str | None = None
    keyword_model_digest: str | None = None
    answer_model_digest: str | None = None
    embedding_model: str | None = None
    embedding_model_digest: str | None = None
    documents_path: str = "../multihoprag_120/documents.jsonl"
    questions_path: str = "../multihoprag_120/questions.jsonl"
    lightrag: dict[str, Any] = Field(default_factory=dict)
    keyword: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    answer: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    _input_base_dir: Path | None = PrivateAttr(default=None)

    @field_validator(
        "builder_model",
        "quantization",
        "graph_regime",
        "retrieval_mode",
        "answer_model",
    )
    @classmethod
    def _dimension_is_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("run dimension must not be empty")
        return value

    @field_validator("keyword_model")
    @classmethod
    def _optional_dimension_is_nonempty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("keyword_model must be null or non-empty")
        return value


class FrozenConfig(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    config_hash: str
    config_hash_algorithm: Literal["sha256-canonical-json-v1"] = (
        "sha256-canonical-json-v1"
    )
    frozen_at: datetime = Field(default_factory=utc_now)
    config: RunConfig


class EnvironmentSnapshot(VersionedRecord):
    run_id: str
    config_hash: str
    captured_at: datetime = Field(default_factory=utc_now)
    python: dict[str, Any]
    platform: dict[str, Any]
    packages: dict[str, str | None]
    lightrag_checkout: dict[str, Any] = Field(default_factory=dict)
    safe_environment: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class EventRecord(VersionedRecord):
    event_id: str
    run_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    stage: str
    event_type: str
    message: str
    item_id: str | None = None
    elapsed_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class ChunkArtifact(TimedRecord):
    run_id: str
    chunk_id: str
    document_id: str
    text: str
    text_sha256: str
    token_count: int
    chunk_order: int
    technical_provenance: dict[str, Any] = Field(default_factory=dict)


class ExtractionCallArtifact(TimedRecord):
    run_id: str
    call_id: str
    document_id: str
    chunk_id: str | None = None
    chunk_sha256: str | None = None
    model_name: str
    model_digest: str | None = None
    prompt_sha256: str
    raw_response: str | None = None
    response_sha256: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float
    attempt_number: int = Field(default=1, ge=1)
    gleaning_round: int = Field(default=0, ge=0)
    json_valid: bool | None = None
    schema_valid: bool | None = None
    parse_success: bool | None = None
    entity_count: int | None = Field(default=None, ge=0)
    relationship_count: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None
    technical_provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_sha256")
    @classmethod
    def _hash_is_hex(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("chunk_sha256 must be a lowercase SHA-256 hex digest")
        return value


class GraphNodeArtifact(TimedRecord):
    run_id: str
    node_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    technical_provenance: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeArtifact(TimedRecord):
    run_id: str
    edge_id: str
    source: str
    target: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    technical_provenance: dict[str, Any] = Field(default_factory=dict)

