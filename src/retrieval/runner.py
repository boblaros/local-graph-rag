"""Structured, resumable retrieval over an already materialized workspace."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RETRIEVAL_SCHEMA_VERSION = "3.0.0"
RetrievalOutcome = Literal["context", "empty", "failure"]


class RankedRetrievalItem(BaseModel):
    """An explicit, auditable document rank derived from LightRAG result order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    document_id: str
    source: Literal["chunk", "reference"]
    source_index: int = Field(ge=0)
    chunk_id: str | None = None
    score: float | None = None


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def _stable_id(prefix: str, *values: Any) -> str:
    digest = hashlib.sha256(_canonical(values).encode("utf-8")).hexdigest()
    return prefix + digest[:24]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class RetrievalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = RETRIEVAL_SCHEMA_VERSION
    question_id: str
    question: str
    question_type: str
    answerable: bool
    base_run_id: str
    variant_run_id: str
    base_extraction_sha256: str
    builder_model: str
    graph_regime: str
    retrieval_result_id: str
    query_model: str
    query_model_digest: str
    query_generation_parameters: dict[str, Any] = Field(default_factory=dict)
    retrieval_mode: str
    retrieval_outcome: RetrievalOutcome
    context: str
    context_sha256: str
    retrieved_entities: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_relationships: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    ranked_retrieval_items: list[RankedRetrievalItem] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    context_token_count: int | None = Field(default=None, ge=0)
    raw_result: dict[str, Any]
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_context(self) -> "RetrievalRecord":
        if self.context_sha256 != _sha256(self.context):
            raise ValueError("context hash mismatch")
        status = self.raw_result.get("status")
        metadata = self.raw_result.get("metadata")
        failure_reason = (
            metadata.get("failure_reason") if isinstance(metadata, Mapping) else None
        )
        if self.retrieval_outcome == "context":
            if status != "success" or not self.context or self.error_type:
                raise ValueError(
                    "context outcome requires a successful non-empty result"
                )
        elif self.retrieval_outcome == "empty":
            if (
                status != "failure"
                or failure_reason != "no_results"
                or self.context
                or self.retrieved_entities
                or self.retrieved_relationships
                or self.retrieved_chunks
                or self.references
                or self.error_type
            ):
                raise ValueError(
                    "empty outcome requires LightRAG's explicit no_results response"
                )
        elif (
            not self.error_type
            or not self.error_message
            or self.context
            or self.retrieved_entities
            or self.retrieved_relationships
            or self.retrieved_chunks
            or self.references
            or self.ranked_retrieval_items
            or self.retrieved_document_ids
        ):
            raise ValueError(
                "failure outcome requires an explicit error and no retrieval payload"
            )
        if [item.rank for item in self.ranked_retrieval_items] != list(
            range(1, len(self.ranked_retrieval_items) + 1)
        ):
            raise ValueError("ranked retrieval item ranks must be contiguous")
        ranked_documents = [item.document_id for item in self.ranked_retrieval_items]
        if any(not document_id.strip() for document_id in ranked_documents):
            raise ValueError("ranked retrieval document IDs must be non-empty")
        if ranked_documents and ranked_documents != self.retrieved_document_ids:
            raise ValueError("retrieved_document_ids disagree with explicit ranks")
        return self


def _read_existing(path: Path) -> dict[str, RetrievalRecord]:
    if not path.exists():
        return {}
    result: dict[str, RetrievalRecord] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = RetrievalRecord.model_validate_json(line)
        if record.retrieval_result_id in result:
            raise RuntimeError(f"duplicate retrieval_result_id at {path}:{line_number}")
        result[record.retrieval_result_id] = record
    return result


def _append(path: Path, record: RetrievalRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical(record.model_dump(mode="json")) + "\n")
        handle.flush()


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _document_ids(
    chunks: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in [*chunks, *references]:
        for field in ("document_id", "full_doc_id", "file_path", "source_id"):
            raw = item.get(field)
            if raw is None:
                continue
            for candidate in str(raw).replace("<SEP>", "|").split("|"):
                value = candidate.strip()
                if value and value not in seen:
                    seen.add(value)
                    result.append(value)
            # These fields are alternative identifiers.  Taking all populated
            # fields would falsely rank a file path/source ID as another document.
            break
    return result


def _ranked_items(
    chunks: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> list[RankedRetrievalItem]:
    """Preserve upstream list order and first occurrence of each document."""

    result: list[RankedRetrievalItem] = []
    seen: set[str] = set()
    for source, rows in (("chunk", chunks), ("reference", references)):
        for source_index, item in enumerate(rows):
            document_ids = _document_ids([item], [])
            for document_id in document_ids:
                if document_id in seen:
                    continue
                seen.add(document_id)
                score = next(
                    (
                        float(item[key])
                        for key in ("score", "similarity", "distance")
                        if isinstance(item.get(key), (int, float))
                        and not isinstance(item.get(key), bool)
                    ),
                    None,
                )
                result.append(
                    RankedRetrievalItem(
                        rank=len(result) + 1,
                        document_id=document_id,
                        source=source,
                        source_index=source_index,
                        chunk_id=str(item.get("chunk_id") or item.get("id") or "")
                        or None,
                        score=score,
                    )
                )
    return result


def _context_token_count(rag: Any, context: str) -> int | None:
    tokenizer = getattr(rag, "tokenizer", None)
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        return None
    try:
        return len(encoder(context))
    except Exception:
        return None


def _parse_lightrag_response(
    raw: Any,
    *,
    question_id: str,
) -> tuple[RetrievalOutcome, str, dict[str, Any]]:
    """Interpret the public ``aquery_llm`` response without stringifying it.

    LightRAG 1.5.2 represents a valid empty retrieval as ``status=failure`` with
    the machine-readable ``metadata.failure_reason=no_results``.  Other failure
    responses are query/runtime failures and must stop the stage.
    """

    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"LightRAG retrieval returned a non-object for {question_id}: {raw!r}"
        )
    status = raw.get("status")
    metadata = raw.get("metadata")
    failure_reason = (
        metadata.get("failure_reason") if isinstance(metadata, Mapping) else None
    )
    if status == "failure" and failure_reason == "no_results":
        payload = raw.get("data")
        if payload not in ({}, None) and not isinstance(payload, Mapping):
            raise RuntimeError(
                f"LightRAG no_results payload is malformed for {question_id}"
            )
        return "empty", "", dict(payload) if isinstance(payload, Mapping) else {}
    if status != "success":
        message = raw.get("message") or "unknown LightRAG query failure"
        raise RuntimeError(f"LightRAG retrieval failed for {question_id}: {message}")

    response = raw.get("llm_response")
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"LightRAG retrieval response lacks llm_response object for {question_id}"
        )
    content = response.get("content")
    if not isinstance(content, str) or not content:
        raise RuntimeError(
            f"LightRAG retrieval response lacks semantic context for {question_id}"
        )
    if response.get("is_streaming") is not False:
        raise RuntimeError(
            f"LightRAG retrieval unexpectedly returned a stream for {question_id}"
        )
    payload = raw.get("data")
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"LightRAG successful retrieval lacks structured data for {question_id}"
        )
    return "context", content, dict(payload)


async def run_retrieval(
    rag: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    artifact_path: str | Path,
    lineage: Mapping[str, str],
    builder_model: str,
    graph_regime: str,
    query_model: str,
    query_model_digest: str,
    query_generation_parameters: Mapping[str, Any] | None = None,
    retrieval_mode: str = "hybrid",
    retrieval_parameters: Mapping[str, Any] | None = None,
    query_param_factory: Callable[..., Any] | None = None,
) -> list[RetrievalRecord]:
    """Retrieve context only; never ask LightRAG to generate the final answer."""

    if not query_model.strip() or not query_model_digest.strip():
        raise ValueError("query role must have an explicit tag and digest")
    required_lineage = {"base_run_id", "variant_run_id", "base_extraction_sha256"}
    missing = sorted(required_lineage - set(lineage))
    if missing:
        raise ValueError(f"retrieval lineage is incomplete: {missing}")
    parameters = dict(retrieval_parameters or {})
    generation_parameters = dict(query_generation_parameters or {})
    path = Path(artifact_path)
    existing = _read_existing(path)
    results: list[RetrievalRecord] = []
    if query_param_factory is None:
        from lightrag import QueryParam

        query_param_factory = QueryParam

    for question in questions:
        question_id = str(question.get("question_id") or "").strip()
        question_text = str(question.get("question") or "").strip()
        if not question_id or not question_text:
            raise ValueError("question rows require question_id and question")
        result_id = _stable_id(
            "ret_",
            lineage["variant_run_id"],
            question_id,
            question_text,
            query_model,
            query_model_digest,
            generation_parameters,
            retrieval_mode,
            parameters,
        )
        prior = existing.get(result_id)
        if prior is not None:
            if (
                prior.variant_run_id != lineage["variant_run_id"]
                or prior.base_extraction_sha256 != lineage["base_extraction_sha256"]
            ):
                raise RuntimeError(
                    "completed retrieval artifact has incompatible lineage"
                )
            results.append(prior)
            continue

        param_kwargs = {
            **parameters,
            "mode": retrieval_mode,
            "only_need_context": True,
            "stream": False,
        }
        started = time.perf_counter()
        raw: Any = None
        error: Exception | None = None
        try:
            raw = await rag.aquery_llm(
                question_text,
                param=query_param_factory(**param_kwargs),
            )
            retrieval_outcome, context, data = _parse_lightrag_response(
                raw,
                question_id=question_id,
            )
        except Exception as caught:
            error = caught
            retrieval_outcome, context, data = "failure", "", {}
        elapsed = (time.perf_counter() - started) * 1000.0
        entities = _as_rows(data.get("entities"))
        relationships = _as_rows(data.get("relationships"))
        chunks = _as_rows(data.get("chunks"))
        references = _as_rows(data.get("references"))
        ranked_items = _ranked_items(chunks, references)
        raw_result = (
            dict(raw)
            if isinstance(raw, Mapping)
            else {"status": "exception", "data": {}, "metadata": {}}
        )
        record = RetrievalRecord(
            question_id=question_id,
            question=question_text,
            question_type=str(question.get("question_type") or "unknown"),
            answerable=bool(question.get("answerable")),
            base_run_id=lineage["base_run_id"],
            variant_run_id=lineage["variant_run_id"],
            base_extraction_sha256=lineage["base_extraction_sha256"],
            builder_model=builder_model,
            graph_regime=graph_regime,
            retrieval_result_id=result_id,
            query_model=query_model,
            query_model_digest=query_model_digest,
            query_generation_parameters=generation_parameters,
            retrieval_mode=retrieval_mode,
            retrieval_outcome=retrieval_outcome,
            context=context,
            context_sha256=_sha256(context),
            retrieved_entities=entities,
            retrieved_relationships=relationships,
            retrieved_chunks=chunks,
            references=references,
            retrieved_document_ids=[item.document_id for item in ranked_items],
            ranked_retrieval_items=ranked_items,
            latency_ms=elapsed,
            context_token_count=_context_token_count(rag, context),
            raw_result=raw_result,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )
        _append(path, record)
        existing[result_id] = record
        results.append(record)
    return results


__all__ = [
    "RETRIEVAL_SCHEMA_VERSION",
    "RetrievalOutcome",
    "RankedRetrievalItem",
    "RetrievalRecord",
    "run_retrieval",
]
