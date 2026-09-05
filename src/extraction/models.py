"""Versioned schemas for raw-to-normalized extraction staging."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EXTRACTION_SCHEMA_VERSION = "3.0.0"
EXTRACTION_VERSION = "experiment-normalizer-v4-stock-lightrag-json"
ParseStatus = Literal["strict", "recovered", "failed", "no_response"]
EndpointResolutionState = Literal["resolved", "ambiguous", "unresolved"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NormalizedEntityMention(FrozenModel):
    schema_version: str = EXTRACTION_SCHEMA_VERSION
    extraction_version: str = EXTRACTION_VERSION
    mention_id: str
    mention_index: int = Field(ge=0)
    original_name: str
    normalized_name: str
    entity_type: str | None = None
    description: str | None = None
    document_id: str
    chunk_id: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    extraction_call_id: str
    response_sha256: str | None = None
    parse_status: ParseStatus
    entity_present: bool = True
    description_present: bool
    implicit_from_relation: bool = False

    @field_validator(
        "mention_id", "original_name", "normalized_name", "document_id", "chunk_id"
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("stable IDs and entity names must be non-empty")
        return value

    @model_validator(mode="after")
    def _presence_is_literal(self) -> "NormalizedEntityMention":
        if not self.entity_present:
            raise ValueError("a mention record always represents a present entity")
        if self.description_present != bool((self.description or "").strip()):
            raise ValueError("description_present disagrees with description")
        return self


class NormalizedRelation(FrozenModel):
    schema_version: str = EXTRACTION_SCHEMA_VERSION
    extraction_version: str = EXTRACTION_VERSION
    relation_id: str
    relation_index: int = Field(ge=0)
    source_mention_id: str | None
    target_mention_id: str | None
    source_resolution_state: EndpointResolutionState
    target_resolution_state: EndpointResolutionState
    source_candidate_mention_ids: list[str] = Field(default_factory=list)
    target_candidate_mention_ids: list[str] = Field(default_factory=list)
    source_original_name: str
    target_original_name: str
    relation_type: str | None = None
    keywords: list[str] = Field(default_factory=list)
    description: str | None = None
    document_id: str
    chunk_id: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    extraction_call_id: str
    response_sha256: str | None = None
    parse_status: ParseStatus
    relation_present: bool = True
    description_present: bool

    @model_validator(mode="after")
    def _validate_presence(self) -> "NormalizedRelation":
        if not self.relation_present:
            raise ValueError("a relation record always represents a present relation")
        if self.description_present != bool((self.description or "").strip()):
            raise ValueError("description_present disagrees with relation description")
        for role in ("source", "target"):
            mention_id = getattr(self, f"{role}_mention_id")
            state = getattr(self, f"{role}_resolution_state")
            candidates = getattr(self, f"{role}_candidate_mention_ids")
            if len(candidates) != len(set(candidates)) or any(
                not item.strip() for item in candidates
            ):
                raise ValueError(f"{role} endpoint candidates must be unique IDs")
            if state == "resolved":
                if not mention_id or mention_id not in candidates:
                    raise ValueError(
                        f"resolved {role} endpoint must identify one candidate mention"
                    )
            elif mention_id is not None:
                raise ValueError(
                    f"{state} {role} endpoint must not select a mention ID"
                )
            elif state == "ambiguous" and len(candidates) < 2:
                raise ValueError(
                    f"ambiguous {role} endpoint requires at least two candidates"
                )
            elif state == "unresolved" and candidates:
                raise ValueError(
                    f"unresolved {role} endpoint must not declare candidates"
                )
        return self


class ExtractionParseRecord(FrozenModel):
    extraction_call_id: str
    response_sha256: str | None = None
    status: ParseStatus
    recovery_method: str | None = None
    error: str | None = None
    entity_records_seen: int = Field(default=0, ge=0)
    relation_records_seen: int = Field(default=0, ge=0)


class NormalizedChunkResult(FrozenModel):
    schema_version: str = EXTRACTION_SCHEMA_VERSION
    extraction_version: str = EXTRACTION_VERSION
    document_id: str
    chunk_id: str
    text: str
    chunk_sha256: str
    chunk_order: int = Field(ge=0)
    token_count: int | None = Field(default=None, ge=0)
    extraction_call_ids: list[str]
    raw_response_sha256s: list[str] = Field(default_factory=list)
    parse_records: list[ExtractionParseRecord] = Field(default_factory=list)
    entities: list[NormalizedEntityMention] = Field(default_factory=list)
    relations: list[NormalizedRelation] = Field(default_factory=list)
    entity_present: bool
    description_present: bool
    relations_present: bool
    staging_complete: bool

    @model_validator(mode="after")
    def _validate_chunk(self) -> "NormalizedChunkResult":
        import hashlib

        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.chunk_sha256:
            raise ValueError("chunk_sha256 disagrees with immutable chunk text")
        if self.entity_present != bool(self.entities):
            raise ValueError("entity_present disagrees with entity records")
        if self.description_present != any(
            item.description_present for item in self.entities
        ):
            raise ValueError("description_present disagrees with entity descriptions")
        if self.relations_present != bool(self.relations):
            raise ValueError("relations_present disagrees with relation records")
        mention_ids = [item.mention_id for item in self.entities]
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("mention IDs are not unique within chunk")
        relation_ids = [item.relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation IDs are not unique within chunk")
        known = set(mention_ids)
        dangling: list[str] = []
        for item in self.relations:
            for role in ("source", "target"):
                state = getattr(item, f"{role}_resolution_state")
                mention_id = getattr(item, f"{role}_mention_id")
                candidates = getattr(item, f"{role}_candidate_mention_ids")
                if state == "resolved" and mention_id not in known:
                    dangling.append(item.relation_id)
                if any(candidate not in known for candidate in candidates):
                    dangling.append(item.relation_id)
        if dangling:
            raise ValueError(f"relation endpoints do not resolve: {dangling[:3]}")
        if self.staging_complete and any(
            item.source_resolution_state != "resolved"
            or item.target_resolution_state != "resolved"
            for item in self.relations
        ):
            raise ValueError(
                "staging_complete cannot be true with ambiguous/unresolved endpoints"
            )
        return self


class StagingManifest(FrozenModel):
    schema_version: str = EXTRACTION_SCHEMA_VERSION
    extraction_version: str = EXTRACTION_VERSION
    base_run_id: str
    subset_id: str
    corpus_manifest_sha256: str
    documents_sha256: str
    questions_sha256: str
    chunks_sha256: str
    raw_extraction_calls_sha256: str
    normalized_chunks_sha256: str
    normalized_entities_sha256: str
    normalized_relations_sha256: str
    base_extraction_sha256: str
    expected_documents: int = Field(ge=0)
    completed_documents: int = Field(ge=0)
    expected_chunks: int = Field(ge=0)
    staged_chunks: int = Field(ge=0)
    failed_parse_chunks: list[str] = Field(default_factory=list)
    builder_model: str
    builder_model_digest: str
    extraction_prompt_version: str
    seed: int
    generation_parameters: dict[str, Any]
    input_hashes: dict[str, str]

    @property
    def corpus_complete(self) -> bool:
        return (
            self.completed_documents == self.expected_documents
            and self.staged_chunks == self.expected_chunks
        )
