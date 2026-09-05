"""Validate staged chunks and reopened graph workspaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .lightrag_adapter import ChunkSnapshot, StagedChunkResult, WorkspaceSnapshot
from .rewrite import GRAPH_FIELD_SEP, RewrittenGraph


class GraphValidationError(RuntimeError):
    """Raised when graph validation fails."""


def _value(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _split_source_ids(value: Any, separator: str = GRAPH_FIELD_SEP) -> list[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [
        piece.strip() for piece in str(value or "").split(separator) if piece.strip()
    ]


def _node_id(node: Mapping[str, Any]) -> str:
    value = node.get("id") or node.get("entity_id") or node.get("name")
    return str(value).strip() if value is not None else ""


def _node_properties(node: Mapping[str, Any]) -> dict[str, Any]:
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        return dict(properties)
    return dict(node)


def _edge_endpoints(edge: Mapping[str, Any]) -> tuple[str, str]:
    source = str(edge.get("source") or edge.get("src_id") or "").strip()
    target = str(edge.get("target") or edge.get("tgt_id") or "").strip()
    return source, target


def _edge_properties(edge: Mapping[str, Any]) -> dict[str, Any]:
    properties = edge.get("properties")
    if isinstance(properties, Mapping):
        return dict(properties)
    return dict(edge)


@dataclass
class ValidationReport:
    gate: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def require_valid(self) -> "ValidationReport":
        if self.errors:
            raise GraphValidationError(
                f"{self.gate} failed with {len(self.errors)} error(s): {self.errors[0]}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def validate_rewrite_inputs(
    graph: RewrittenGraph,
    expected_chunks: Sequence[ChunkSnapshot] | Sequence[Any],
) -> ValidationReport:
    report = ValidationReport(gate="graph_rewrite")
    chunk_ids = {
        item.chunk_id
        if isinstance(item, ChunkSnapshot)
        else str(_value(item, "chunk_id", "_id") or "")
        for item in expected_chunks
    }
    chunk_ids.discard("")
    node_names = [str(node.get("entity_name") or "").strip() for node in graph.nodes]
    if any(not name for name in node_names):
        report.errors.append("rewritten graph contains a node without entity_name")
    if len(node_names) != len(set(node_names)):
        report.errors.append("rewritten graph node names are not unique")
    node_set = set(node_names)
    edge_pairs: set[tuple[str, str]] = set()
    dangling = 0
    unresolved_sources = 0
    for edge in graph.edges:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if source not in node_set or target not in node_set:
            dangling += 1
        pair = tuple(sorted((source, target)))
        if pair in edge_pairs:
            report.errors.append(f"duplicate rewritten edge endpoints {pair!r}")
        edge_pairs.add(pair)
        unresolved_sources += sum(
            chunk_id not in chunk_ids for chunk_id in edge.get("source_chunk_ids", [])
        )
    for node in graph.nodes:
        unresolved_sources += sum(
            chunk_id not in chunk_ids for chunk_id in node.get("source_chunk_ids", [])
        )
    if dangling:
        report.errors.append(f"rewritten graph has {dangling} dangling edge(s)")
    if unresolved_sources:
        report.errors.append(
            f"rewritten graph has {unresolved_sources} unresolved source chunk reference(s)"
        )
    induced_ids = {item.get("relation_id") for item in graph.induced_self_loops}
    materialized_ids = {
        relation_id
        for edge in graph.edges
        for relation_id in edge.get("source_relation_ids", [])
    }
    overlap = sorted(str(value) for value in induced_ids.intersection(materialized_ids))
    if overlap:
        report.errors.append(
            f"induced self-loop relation was retained in materialized edges: {overlap[0]}"
        )
    report.metrics = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "expected_chunks": len(chunk_ids),
        "dangling_edges": dangling,
        "unresolved_source_chunk_references": unresolved_sources,
        "induced_self_loops_removed": len(graph.induced_self_loops),
    }
    return report


def validate_staged_chunks(result: StagedChunkResult) -> ValidationReport:
    report = ValidationReport(gate="exact_chunk_snapshot")
    expected_by_id = {chunk.chunk_id: chunk for chunk in result.expected_chunks}
    stored_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in result.stored_chunks:
        chunk_id = str(raw.get("_id") or raw.get("chunk_id") or "")
        if not chunk_id:
            report.errors.append("stored chunk lacks id")
            continue
        if chunk_id in stored_by_id:
            report.errors.append(f"stored chunk id {chunk_id!r} appears more than once")
        stored_by_id[chunk_id] = raw

    missing = sorted(set(expected_by_id) - set(stored_by_id))
    unexpected = sorted(set(stored_by_id) - set(expected_by_id))
    if missing:
        report.errors.append(f"missing stored snapshot chunk {missing[0]!r}")
    if unexpected:
        report.errors.append(f"unexpected stored chunk {unexpected[0]!r}")

    content_mismatches = 0
    order_mismatches = 0
    document_mismatches = 0
    token_mismatches = 0
    for chunk_id in sorted(set(expected_by_id).intersection(stored_by_id)):
        expected = expected_by_id[chunk_id]
        stored = stored_by_id[chunk_id]
        if stored.get("content") != expected.content:
            content_mismatches += 1
        if (
            _value(stored, "chunk_order_index", "chunk_order", "order")
            != expected.chunk_order
        ):
            order_mismatches += 1
        if (
            str(stored.get("full_doc_id") or stored.get("document_id") or "")
            != expected.document_id
        ):
            document_mismatches += 1
        stored_tokens = _value(stored, "tokens", "token_count")
        if expected.token_count is not None and stored_tokens != expected.token_count:
            token_mismatches += 1

    for label, count in (
        ("content", content_mismatches),
        ("order", order_mismatches),
        ("document", document_mismatches),
        ("token count", token_mismatches),
    ):
        if count:
            report.errors.append(f"{count} stored chunk {label} mismatch(es)")

    expected_by_document: dict[str, list[str]] = {}
    for chunk in sorted(result.expected_chunks):
        expected_by_document.setdefault(chunk.document_id, []).append(chunk.chunk_id)
    status_mismatches = 0
    for document_id, expected_ids in expected_by_document.items():
        status = result.document_statuses.get(document_id, {})
        actual_ids = [str(value) for value in status.get("chunks_list", [])]
        if actual_ids != expected_ids:
            status_mismatches += 1
    if status_mismatches:
        report.errors.append(
            f"{status_mismatches} document status chunk list/order mismatch(es)"
        )

    report.metrics = {
        "expected_chunks": len(expected_by_id),
        "stored_chunks": len(stored_by_id),
        "content_mismatches": content_mismatches,
        "order_mismatches": order_mismatches,
        "document_mismatches": document_mismatches,
        "token_mismatches": token_mismatches,
        "status_mismatches": status_mismatches,
        "chunk_read_api": result.chunk_read_api,
    }
    if not result.chunk_read_api.startswith("public_facade:"):
        report.warnings.append(
            "exact chunk content verification required LightRAG's documented "
            "text_chunks storage interface because no facade chunk-read API exists"
        )
    return report


def validate_workspace_snapshot(
    snapshot: WorkspaceSnapshot,
    graph: RewrittenGraph,
    expected_chunks: Sequence[ChunkSnapshot],
) -> ValidationReport:
    report = ValidationReport(gate="reopened_workspace")
    if snapshot.graph_is_truncated:
        report.errors.append(
            "get_knowledge_graph('*') was truncated; increase LightRAG max_graph_nodes"
        )

    expected_nodes = {str(node["entity_name"]): node for node in graph.nodes}
    actual_nodes: dict[str, dict[str, Any]] = {}
    for raw in snapshot.nodes:
        node_id = _node_id(raw)
        if not node_id:
            report.errors.append("exported graph node lacks id")
            continue
        if node_id in actual_nodes:
            report.errors.append(f"exported graph node {node_id!r} is duplicated")
        actual_nodes[node_id] = raw
    missing_nodes = sorted(set(expected_nodes) - set(actual_nodes))
    unexpected_nodes = sorted(set(actual_nodes) - set(expected_nodes))
    if missing_nodes:
        report.errors.append(f"reopened graph is missing node {missing_nodes[0]!r}")
    if unexpected_nodes:
        report.errors.append(
            f"reopened graph has unexpected node {unexpected_nodes[0]!r}"
        )

    expected_pairs = {
        tuple(sorted((str(edge["source"]), str(edge["target"])))): edge
        for edge in graph.edges
    }
    actual_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    dangling = 0
    for raw in snapshot.edges:
        source, target = _edge_endpoints(raw)
        if not source or not target:
            report.errors.append("exported graph edge lacks source/target")
            continue
        if source not in actual_nodes or target not in actual_nodes:
            dangling += 1
        pair = tuple(sorted((source, target)))
        if pair in actual_pairs:
            report.errors.append(f"reopened graph has duplicate edge pair {pair!r}")
        actual_pairs[pair] = raw
    if dangling:
        report.errors.append(f"reopened graph has {dangling} dangling edge(s)")
    missing_edges = sorted(set(expected_pairs) - set(actual_pairs))
    unexpected_edges = sorted(set(actual_pairs) - set(expected_pairs))
    if missing_edges:
        report.errors.append(f"reopened graph is missing edge {missing_edges[0]!r}")
    if unexpected_edges:
        report.errors.append(
            f"reopened graph has unexpected edge {unexpected_edges[0]!r}"
        )

    expected_chunk_ids = {chunk.chunk_id for chunk in expected_chunks}
    provenance_mismatches = 0
    for name in sorted(set(expected_nodes).intersection(actual_nodes)):
        expected_sources = set(expected_nodes[name].get("source_chunk_ids", []))
        properties = _node_properties(actual_nodes[name])
        actual_sources = set(_split_source_ids(properties.get("source_id")))
        if expected_sources != actual_sources:
            provenance_mismatches += 1
    for pair in sorted(set(expected_pairs).intersection(actual_pairs)):
        expected_sources = set(expected_pairs[pair].get("source_chunk_ids", []))
        properties = _edge_properties(actual_pairs[pair])
        actual_sources = set(_split_source_ids(properties.get("source_id")))
        if expected_sources != actual_sources:
            provenance_mismatches += 1
    if provenance_mismatches:
        report.errors.append(
            f"{provenance_mismatches} graph object provenance mismatch(es) after reopen"
        )

    reopened_chunk_ids = {
        str(chunk.get("_id") or chunk.get("chunk_id") or "")
        for chunk in snapshot.chunks
    }
    if reopened_chunk_ids != expected_chunk_ids:
        report.errors.append(
            "reopened workspace chunk id set differs from base snapshot"
        )
    content_mismatches = 0
    expected_by_id = {chunk.chunk_id: chunk for chunk in expected_chunks}
    for chunk in snapshot.chunks:
        chunk_id = str(chunk.get("_id") or chunk.get("chunk_id") or "")
        expected = expected_by_id.get(chunk_id)
        if expected is not None and (
            chunk.get("content") != expected.content
            or _value(chunk, "chunk_order_index", "chunk_order", "order")
            != expected.chunk_order
            or str(chunk.get("full_doc_id") or chunk.get("document_id") or "")
            != expected.document_id
        ):
            content_mismatches += 1
    if content_mismatches:
        report.errors.append(
            f"{content_mismatches} reopened chunk content/order/document mismatch(es)"
        )

    unresolved_source_ids = 0
    for raw in snapshot.nodes:
        unresolved_source_ids += sum(
            source_id not in expected_chunk_ids
            for source_id in _split_source_ids(_node_properties(raw).get("source_id"))
        )
    for raw in snapshot.edges:
        unresolved_source_ids += sum(
            source_id not in expected_chunk_ids
            for source_id in _split_source_ids(_edge_properties(raw).get("source_id"))
        )
    if unresolved_source_ids:
        report.errors.append(
            f"reopened graph has {unresolved_source_ids} unresolved source chunk id(s)"
        )

    report.metrics = {
        "expected_nodes": len(expected_nodes),
        "reopened_nodes": len(actual_nodes),
        "expected_edges": len(expected_pairs),
        "reopened_edges": len(actual_pairs),
        "expected_chunks": len(expected_chunk_ids),
        "reopened_chunks": len(reopened_chunk_ids),
        "dangling_edges": dangling,
        "provenance_mismatches": provenance_mismatches,
        "chunk_mismatches": content_mismatches,
        "unresolved_source_ids": unresolved_source_ids,
        "chunk_read_api": snapshot.chunk_read_api,
    }
    return report


def validate_smoke_results(results: Sequence[Mapping[str, Any]]) -> ValidationReport:
    report = ValidationReport(gate="smoke_retrieval")
    failed = [result for result in results if not bool(result.get("passed"))]
    if failed:
        report.errors.append(
            f"{len(failed)} smoke retrieval case(s) failed; first={failed[0].get('query')!r}"
        )
    statuses: dict[str, int] = {}
    for result in results:
        status = str(result.get("actual_status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    report.metrics = {"cases": len(results), "statuses": statuses}
    return report


__all__ = [
    "GraphValidationError",
    "ValidationReport",
    "validate_rewrite_inputs",
    "validate_smoke_results",
    "validate_staged_chunks",
    "validate_workspace_snapshot",
]
