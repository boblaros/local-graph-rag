"""Public-API materialization of a rewritten ER graph in a clean workspace."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .lightrag_adapter import (
    LightRAGAdapterError,
    PublicLightRAGAdapter,
    SnapshotChunker,
    StagedChunkResult,
    WorkspaceSnapshot,
)
from .parity import ParityReport, build_parity_report
from .rewrite import GRAPH_FIELD_SEP, RewrittenGraph
from .validation import (
    ValidationReport,
    validate_rewrite_inputs,
    validate_smoke_results,
    validate_staged_chunks,
    validate_workspace_snapshot,
)


DESCRIPTION_POLICY_VERSION = "er-storage-description-v2"
ENTITY_DESCRIPTION_PLACEHOLDER = "[Description unavailable in base extraction]"
RELATION_DESCRIPTION_PLACEHOLDER = (
    "[Relationship description unavailable in base extraction]"
)
MAX_MATERIALIZED_DESCRIPTION_CHARS = 6000
DESCRIPTION_TRUNCATION_MARKER = " … [truncated]"


@dataclass(frozen=True)
class DescriptionPolicy:
    """Fixed placeholders and length limits for stored descriptions."""

    version: str = DESCRIPTION_POLICY_VERSION
    entity_placeholder: str = ENTITY_DESCRIPTION_PLACEHOLDER
    relation_placeholder: str = RELATION_DESCRIPTION_PLACEHOLDER
    maximum_chars: int = MAX_MATERIALIZED_DESCRIPTION_CHARS
    truncation_marker: str = DESCRIPTION_TRUNCATION_MARKER

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("description policy version must be non-empty")
        if not self.entity_placeholder.strip() or not self.relation_placeholder.strip():
            raise ValueError("description placeholders must be non-empty")
        if self.maximum_chars <= len(self.truncation_marker):
            raise ValueError(
                "description maximum_chars must exceed the truncation marker length"
            )
        if not self.truncation_marker.strip():
            raise ValueError("description truncation marker must be non-empty")


@dataclass
class MaterializationStageResult:
    staged_chunks: StagedChunkResult
    exact_chunk_gate: ValidationReport
    rewrite_gate: ValidationReport
    pre_finalize_workspace_gate: ValidationReport
    description_placeholders: list[dict[str, Any]]
    description_truncations: list[dict[str, Any]]
    aliases_audit: list[dict[str, Any]]
    created_entities: int
    created_relations: int


@dataclass
class MaterializationOutcome:
    stage: MaterializationStageResult
    reopened_snapshot: WorkspaceSnapshot
    reopened_workspace_gate: ValidationReport
    smoke_results: list[dict[str, Any]]
    smoke_gate: ValidationReport | None
    parity_report: ParityReport

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _document_file_paths(documents: Sequence[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for document in documents:
        if isinstance(document, Mapping):
            document_id = str(
                document.get("document_id") or document.get("id") or ""
            ).strip()
            file_path = str(document.get("file_path") or document_id).strip()
        else:
            document_id = str(
                getattr(document, "document_id", getattr(document, "id", ""))
            ).strip()
            file_path = str(
                getattr(document, "file_path", document_id) or document_id
            ).strip()
        if document_id:
            result[document_id] = file_path or document_id
    return result


def _joined(values: Sequence[Any], separator: str = GRAPH_FIELD_SEP) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return separator.join(result)


def _bounded_description(description: str, policy: DescriptionPolicy) -> str:
    """Return a deterministic prefix bounded for downstream embedding calls."""

    if len(description) <= policy.maximum_chars:
        return description
    prefix_limit = policy.maximum_chars - len(policy.truncation_marker)
    prefix = description[:prefix_limit].rstrip()
    return prefix + policy.truncation_marker


async def _resolve_factory(factory: Callable[[], Any | Awaitable[Any]]) -> Any:
    value = factory()
    return await value if inspect.isawaitable(value) else value


async def materialize_graph(
    adapter: PublicLightRAGAdapter,
    *,
    documents: Sequence[Any],
    chunks: Sequence[Any],
    graph: RewrittenGraph,
    track_id: str,
    description_policy: DescriptionPolicy = DescriptionPolicy(),
) -> MaterializationStageResult:
    """Stage exact chunks and create canonical graph objects in one clean workspace."""

    expected_chunks = SnapshotChunker(documents, chunks).expected_chunks
    rewrite_gate = validate_rewrite_inputs(graph, expected_chunks).require_valid()
    await adapter.assert_clean_workspace()
    staged = await adapter.stage_snapshot_chunks(documents, chunks, track_id=track_id)
    exact_chunk_gate = validate_staged_chunks(staged).require_valid()

    file_paths = _document_file_paths(documents)
    node_names = {str(node["entity_name"]) for node in graph.nodes}
    description_placeholders: list[dict[str, Any]] = []
    description_truncations: list[dict[str, Any]] = []
    aliases_audit: list[dict[str, Any]] = []

    for node in sorted(graph.nodes, key=lambda item: str(item["canonical_entity_id"])):
        name = str(node["entity_name"])
        aliases_audit.append(
            {
                "canonical_entity_id": str(node["canonical_entity_id"]),
                "canonical_display_name": name,
                "aliases": list(node.get("aliases", [])),
                "storage_policy": "audit_only",
                "reason": "LightRAG public entity schema has no alias metadata field",
            }
        )
        description = str(node.get("description") or "").strip()
        if not description:
            description = description_policy.entity_placeholder
            description_placeholders.append(
                {
                    "object_kind": "entity",
                    "object_id": str(node["canonical_entity_id"]),
                    "display_name": name,
                    "placeholder": description,
                    "policy_version": description_policy.version,
                    "description_present": False,
                }
            )
        elif len(description) > description_policy.maximum_chars:
            original_chars = len(description)
            description = _bounded_description(description, description_policy)
            description_truncations.append(
                {
                    "object_kind": "entity",
                    "object_id": str(node["canonical_entity_id"]),
                    "display_name": name,
                    "original_chars": original_chars,
                    "stored_chars": len(description),
                    "maximum_chars": description_policy.maximum_chars,
                    "policy_version": description_policy.version,
                    "reason": "bounded before LightRAG storage and embedding",
                }
            )
        source_chunks = [str(value) for value in node.get("source_chunk_ids", [])]
        source_documents = [str(value) for value in node.get("source_document_ids", [])]
        await adapter.create_entity(
            name,
            {
                "entity_type": str(node.get("entity_type") or "UNKNOWN"),
                "description": description,
                "source_id": _joined(source_chunks),
                "file_path": _joined(
                    [
                        file_paths.get(document_id, document_id)
                        for document_id in source_documents
                    ]
                ),
            },
        )

    for edge in sorted(graph.edges, key=lambda item: str(item["edge_id"])):
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in node_names or target not in node_names:
            raise LightRAGAdapterError(
                f"edge {edge.get('edge_id')!r} has unresolved endpoint {source!r}/{target!r}"
            )
        description = str(edge.get("description") or "").strip()
        if not description:
            description = description_policy.relation_placeholder
            description_placeholders.append(
                {
                    "object_kind": "relation",
                    "object_id": str(edge["edge_id"]),
                    "source": source,
                    "target": target,
                    "placeholder": description,
                    "policy_version": description_policy.version,
                    "description_present": False,
                }
            )
        elif len(description) > description_policy.maximum_chars:
            original_chars = len(description)
            description = _bounded_description(description, description_policy)
            description_truncations.append(
                {
                    "object_kind": "relation",
                    "object_id": str(edge["edge_id"]),
                    "source": source,
                    "target": target,
                    "original_chars": original_chars,
                    "stored_chars": len(description),
                    "maximum_chars": description_policy.maximum_chars,
                    "policy_version": description_policy.version,
                    "reason": "bounded before LightRAG storage and embedding",
                }
            )
        source_documents = [str(value) for value in edge.get("source_document_ids", [])]
        await adapter.create_relation(
            source,
            target,
            {
                "description": description,
                "keywords": str(edge.get("keywords") or ""),
                "weight": float(edge.get("weight", 1.0)),
                "source_id": _joined(
                    [str(value) for value in edge.get("source_chunk_ids", [])]
                ),
                "file_path": _joined(
                    [
                        file_paths.get(document_id, document_id)
                        for document_id in source_documents
                    ]
                ),
            },
        )

    pre_finalize_snapshot = await adapter.export_workspace(
        expected_node_count=len(graph.nodes),
        chunk_ids=[chunk.chunk_id for chunk in staged.expected_chunks],
    )
    pre_finalize_gate = validate_workspace_snapshot(
        pre_finalize_snapshot, graph, staged.expected_chunks
    ).require_valid()
    return MaterializationStageResult(
        staged_chunks=staged,
        exact_chunk_gate=exact_chunk_gate,
        rewrite_gate=rewrite_gate,
        pre_finalize_workspace_gate=pre_finalize_gate,
        description_placeholders=description_placeholders,
        description_truncations=description_truncations,
        aliases_audit=aliases_audit,
        created_entities=len(graph.nodes),
        created_relations=len(graph.edges),
    )


async def materialize_finalize_reopen(
    rag_factory: Callable[[], Any | Awaitable[Any]],
    *,
    documents: Sequence[Any],
    chunks: Sequence[Any],
    graph: RewrittenGraph,
    track_id: str,
    description_policy: DescriptionPolicy = DescriptionPolicy(),
    smoke_cases: Sequence[Mapping[str, Any]] = (),
    query_param_factory: Callable[[str], Any] | None = None,
) -> MaterializationOutcome:
    """Build a graph, reopen it, export it, and run optional retrieval checks.

    ``rag_factory`` must return a new object on each call for the same empty
    derived workspace because a finalized LightRAG instance cannot be reopened.
    """

    first_rag = await _resolve_factory(rag_factory)
    first_adapter = PublicLightRAGAdapter(first_rag)
    initialized = False
    try:
        await first_adapter.initialize()
        initialized = True
        stage = await materialize_graph(
            first_adapter,
            documents=documents,
            chunks=chunks,
            graph=graph,
            track_id=track_id,
            description_policy=description_policy,
        )
    finally:
        if initialized:
            await first_adapter.finalize()

    reopened_rag = await _resolve_factory(rag_factory)
    if reopened_rag is first_rag:
        raise LightRAGAdapterError(
            "rag_factory returned the finalized object; reopen requires a new LightRAG instance"
        )
    reopened_adapter = PublicLightRAGAdapter(reopened_rag)
    reopened_initialized = False
    try:
        await reopened_adapter.initialize()
        reopened_initialized = True
        snapshot = await reopened_adapter.export_workspace(
            expected_node_count=len(graph.nodes),
            chunk_ids=[chunk.chunk_id for chunk in stage.staged_chunks.expected_chunks],
        )
        reopened_gate = validate_workspace_snapshot(
            snapshot, graph, stage.staged_chunks.expected_chunks
        ).require_valid()
        smoke_results = (
            await reopened_adapter.smoke_retrieval(
                smoke_cases, query_param_factory=query_param_factory
            )
            if smoke_cases
            else []
        )
        smoke_gate = (
            validate_smoke_results(smoke_results).require_valid()
            if smoke_cases
            else None
        )
    finally:
        if reopened_initialized:
            await reopened_adapter.finalize()

    parity = build_parity_report(
        graph,
        chunk_read_api=snapshot.chunk_read_api,
        description_placeholder_count=len(stage.description_placeholders),
        description_truncation_count=len(stage.description_truncations),
    )
    return MaterializationOutcome(
        stage=stage,
        reopened_snapshot=snapshot,
        reopened_workspace_gate=reopened_gate,
        smoke_results=smoke_results,
        smoke_gate=smoke_gate,
        parity_report=parity,
    )


__all__ = [
    "DescriptionPolicy",
    "MaterializationOutcome",
    "MaterializationStageResult",
    "materialize_finalize_reopen",
    "materialize_graph",
]
