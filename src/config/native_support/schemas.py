"""Schemas for fixed Native configuration and saved records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

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


class RetrievalArtifact(TimedRecord):
    run_id: str
    retrieval_result_id: str
    question_id: str
    question_type: str
    retrieval_mode: str
    high_level_keywords: list[str] = Field(default_factory=list)
    low_level_keywords: list[str] = Field(default_factory=list)
    retrieved_entities: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_relationships: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    retrieved_urls: list[str] = Field(default_factory=list)
    latency_ms: float
    context_token_count: int | None = None
    context: str | None = None
    context_sha256: str | None = None
    raw_result: dict[str, Any] = Field(default_factory=dict)
    keyword_model: str | None = None
    keyword_model_digest: str | None = None
    embedding_model: str | None = None
    embedding_model_digest: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class AnswerArtifact(TimedRecord):
    run_id: str
    question_id: str
    question_type: str
    gold_answer: str
    answerable: bool
    gold_urls: list[str] = Field(default_factory=list)
    retrieval_result_id: str
    model_name: str
    model_digest: str | None = None
    raw_answer: str | None = None
    normalized_answer: str | None = None
    prompt_sha256: str
    retrieval_context_sha256: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_type: str | None = None
    error_message: str | None = None


class ExtractionPerChunkMetric(TimedRecord):
    run_id: str
    chunk_id: str
    document_id: str
    token_count: int | None = None
    chunk_order: int | None = None
    raw_json_valid: bool
    schema_valid: bool
    parse_success: bool
    final_extraction_success: bool
    repair_count: int = 0
    retry_count: int = 0
    gleaning_count: int = 0
    empty_extraction: bool
    entity_count: int = 0
    relationship_count: int = 0
    entities_per_1000_tokens: float | None = None
    relationships_per_1000_tokens: float | None = None
    latency_ms: float | None = None
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    failure_type: str | None = None


class GraphMetrics(TimedRecord):
    run_id: str
    nodes: int
    edges: int
    connected_components: int
    largest_component_share: float
    isolates: int
    average_degree: float
    normalized_density: float
    self_loops: int
    node_provenance_coverage: float
    edge_provenance_coverage: float
    duplicate_node_candidate_count: int = 0
    duplicate_edge_candidate_count: int = 0


class RetrievalPerQuestionMetric(TimedRecord):
    run_id: str
    question_id: str
    question_type: str
    answerable: bool
    metrics_available: bool
    evidence_url_recall_at_5: float | None = None
    evidence_url_recall_at_10: float | None = None
    hit_at_5: bool | None = None
    hit_at_10: bool | None = None
    complete_chain_recall_at_5: bool | None = None
    complete_chain_recall_at_10: bool | None = None
    empty_retrieval: bool
    retrieved_entity_count: int
    retrieved_relationship_count: int
    retrieved_chunk_count: int
    retrieved_document_count: int
    latency_ms: float | None = None
    context_token_count: int | None = None
    failure_type: str | None = None


class QAPerQuestionMetric(TimedRecord):
    run_id: str
    question_id: str
    question_type: str
    answerable: bool
    metrics_available: bool
    model_call_failure: bool
    normalized_exact_match: bool
    token_f1: float
    predicted_unanswerable: bool
    gold_unanswerable: bool
    null_true_positive: bool
    null_false_positive: bool
    null_false_negative: bool
    hallucination: bool
    over_abstention: bool
    normalized_gold_answer: str
    normalized_predicted_answer: str
    normalized_yes_no_gold: str | None = None
    normalized_yes_no_prediction: str | None = None
    normalized_date_gold: str | None = None
    normalized_date_prediction: str | None = None
    normalized_numeric_gold: str | None = None
    normalized_numeric_prediction: str | None = None
    failure_type: str | None = None


class RunSummary(VersionedRecord):
    generated_at: datetime
    run_id: str
    config_hash: str | None = None
    run_dimensions: dict[str, Any]
    extraction: dict[str, Any]
    graph: dict[str, Any]
    retrieval: dict[str, Any]
    qa: dict[str, Any]
    efficiency: dict[str, Any]
    by_question_type: dict[str, dict[str, Any]] = Field(default_factory=dict)


SCHEMA_MODELS: ClassVar[dict[str, type[BaseModel]]] = {
    "run_config": RunConfig,
    "config.lock": FrozenConfig,
    "environment": EnvironmentSnapshot,
    "event": EventRecord,
    "chunks": ChunkArtifact,
    "extraction_calls": ExtractionCallArtifact,
    "graph_nodes": GraphNodeArtifact,
    "graph_edges": GraphEdgeArtifact,
    "retrieval": RetrievalArtifact,
    "answers": AnswerArtifact,
    "extraction_per_chunk": ExtractionPerChunkMetric,
    "graph_metrics": GraphMetrics,
    "retrieval_per_question": RetrievalPerQuestionMetric,
    "qa_per_question": QAPerQuestionMetric,
    "summary": RunSummary,
}


def export_json_schemas(directory: str | Path, *, overwrite: bool = True) -> list[Path]:
    """Export the authoritative Pydantic schemas as deterministic JSON files."""

    try:
        from .io_utils import write_json_atomic, write_json_exclusive
    except ImportError:  # pragma: no cover - direct script execution.
        from io_utils import write_json_atomic, write_json_exclusive  # type: ignore

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in SCHEMA_MODELS.items():
        path = target / f"{name}.schema.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "record_type": name,
            "json_schema": model.model_json_schema(),
        }
        if overwrite:
            write_json_atomic(path, payload)
        else:
            write_json_exclusive(path, payload)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export experiment JSON schemas")
    parser.add_argument("directory", type=Path, help="Destination schemas directory")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail instead of replacing existing derived schema files",
    )
    args = parser.parse_args(argv)
    for path in export_json_schemas(args.directory, overwrite=not args.no_overwrite):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
