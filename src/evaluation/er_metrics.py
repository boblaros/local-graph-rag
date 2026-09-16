"""Entity-resolution and graph-effect metrics only."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
from typing import Any


def _row(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif hasattr(value, "model_dump"):
        result = dict(value.model_dump(mode="json"))
    elif hasattr(value, "to_dict"):
        result = dict(value.to_dict())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = dataclasses.asdict(value)
    else:
        raise TypeError(f"metric row must be mapping-like, got {type(value).__name__}")
    nested = result.get("record")
    return dict(nested) if isinstance(nested, Mapping) else result


def _rows(values: Iterable[Mapping[str, Any] | Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        row = _row(value)
        if row.get("record_type") == "artifact_header":
            continue
        result.append(row)
    return result


def _payload(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    nested = result.get("payload")
    return dict(nested) if isinstance(nested, Mapping) else result


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _difference(left: Any, right: Any) -> int | None:
    if left is None or right is None:
        return None
    return int(left) - int(right)


def _node_id(value: Mapping[str, Any] | str) -> str:
    if isinstance(value, str):
        result = value.strip()
    else:
        result = str(
            value.get("canonical_entity_id")
            or value.get("node_id")
            or value.get("entity_id")
            or value.get("id")
            or value.get("entity_name")
            or value.get("name")
            or ""
        ).strip()
    if not result:
        raise ValueError("graph node has no resolvable ID")
    return result


def _edge_endpoints(value: Mapping[str, Any] | Sequence[str]) -> tuple[str, str]:
    if isinstance(value, Mapping):
        source = str(
            value.get("source_canonical_entity_id")
            or value.get("source_entity_id")
            or value.get("source")
            or value.get("src")
            or value.get("from")
            or ""
        ).strip()
        target = str(
            value.get("target_canonical_entity_id")
            or value.get("target_entity_id")
            or value.get("target")
            or value.get("dst")
            or value.get("to")
            or ""
        ).strip()
    else:
        if len(value) != 2:
            raise ValueError("graph edge sequence must have exactly two endpoints")
        source, target = (str(item).strip() for item in value)
    if not source or not target:
        raise ValueError("graph edge has no resolvable endpoints")
    return source, target


def _has_provenance(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_provenance(item) for item in value.values())
    if isinstance(value, Iterable):
        return any(_has_provenance(item) for item in value)
    return bool(value)


def _record_has_provenance(value: Mapping[str, Any] | str) -> bool:
    if not isinstance(value, Mapping):
        return False
    attributes = value.get("attributes")
    attributes = attributes if isinstance(attributes, Mapping) else {}
    for field in (
        "source_chunk_ids",
        "source_document_ids",
        "document_ids",
        "source_mentions",
        "source_relation_ids",
        "provenance",
        "technical_provenance",
        "source_id",
        "file_path",
    ):
        if _has_provenance(value.get(field)) or _has_provenance(attributes.get(field)):
            return True
    return False


def compute_graph_topology(
    nodes_or_graph: Iterable[Mapping[str, Any] | str] | Mapping[str, Any],
    edges: Iterable[Mapping[str, Any] | Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compute topology and provenance coverage with undirected graph semantics."""

    if isinstance(nodes_or_graph, Mapping):
        if edges is not None:
            raise ValueError("edges must be omitted when a graph mapping is supplied")
        raw_nodes = nodes_or_graph.get("nodes")
        raw_edges = nodes_or_graph.get("edges")
        if not isinstance(raw_nodes, Iterable) or isinstance(
            raw_nodes, (str, bytes, Mapping)
        ):
            raise ValueError("graph mapping must contain an iterable nodes field")
        if not isinstance(raw_edges, Iterable) or isinstance(
            raw_edges, (str, bytes, Mapping)
        ):
            raise ValueError("graph mapping must contain an iterable edges field")
        nodes = list(raw_nodes)
        edge_rows = list(raw_edges)
    else:
        nodes = list(nodes_or_graph)
        if edges is None:
            raise ValueError("edges are required when nodes are supplied separately")
        edge_rows = list(edges)

    node_ids = {_node_id(node) for node in nodes}
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    edge_count = 0
    self_loops = 0
    unique_non_loop_pairs: set[tuple[str, str]] = set()
    for edge in edge_rows:
        source, target = _edge_endpoints(edge)
        dangling = {source, target} - node_ids
        if dangling:
            raise ValueError(
                f"graph contains dangling edge endpoints: {sorted(dangling)}"
            )
        if source == target:
            self_loops += 1
        else:
            unique_non_loop_pairs.add(tuple(sorted((source, target))))
            adjacency[source].add(target)
            adjacency[target].add(source)
        edge_count += 1

    seen: set[str] = set()
    component_sizes: list[int] = []
    for start in sorted(node_ids):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        component_sizes.append(size)
    largest = max(component_sizes, default=0)
    possible_edges = len(node_ids) * (len(node_ids) - 1) / 2
    mapped_nodes = [node for node in nodes if isinstance(node, Mapping)]
    mapped_edges = [edge for edge in edge_rows if isinstance(edge, Mapping)]
    return {
        "nodes": len(node_ids),
        "edges": edge_count,
        "connected_components": len(component_sizes),
        "isolates": sum(not neighbours for neighbours in adjacency.values()),
        "largest_component": largest,
        "largest_component_size": largest,
        "largest_component_share": largest / len(node_ids) if node_ids else 0.0,
        "average_degree": 2 * edge_count / len(node_ids) if node_ids else 0.0,
        "normalized_density": (
            len(unique_non_loop_pairs) / possible_edges if possible_edges else 0.0
        ),
        "self_loops": self_loops,
        "node_provenance_coverage": (
            sum(_record_has_provenance(node) for node in mapped_nodes) / len(nodes)
            if mapped_nodes
            else None
        ),
        "edge_provenance_coverage": (
            sum(_record_has_provenance(edge) for edge in mapped_edges) / len(edge_rows)
            if mapped_edges
            else None
        ),
    }


def _graph_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    graph = dict(value or {})
    raw_nodes, raw_edges = graph.get("nodes"), graph.get("edges")
    if (
        isinstance(raw_nodes, Iterable)
        and not isinstance(raw_nodes, (str, bytes, Mapping, int, float))
        and isinstance(raw_edges, Iterable)
        and not isinstance(raw_edges, (str, bytes, Mapping, int, float))
    ):
        return {**graph, **compute_graph_topology(raw_nodes, raw_edges)}
    return graph


def _strings(value: Any) -> set[str]:
    if value is None:
        return set()
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, Iterable) or isinstance(values, (bytes, Mapping)):
        values = (values,)
    return {str(item).strip().casefold() for item in values if str(item).strip()}


def compute_er_metrics(
    *,
    mentions: Iterable[Mapping[str, Any] | Any],
    canonical_entities: Iterable[Mapping[str, Any] | Any],
    candidate_pairs: Iterable[Mapping[str, Any] | Any],
    pair_decisions: Iterable[Mapping[str, Any] | Any],
    mention_to_canonical: Iterable[Mapping[str, Any] | Any],
    aliases: Iterable[Mapping[str, Any] | Any] = (),
    merge_plan: Mapping[str, Any] | None = None,
    rewrite_summary: Mapping[str, Any] | None = None,
    native_graph: Mapping[str, Any] | None = None,
    er_graph: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute ER/effect metrics from the emitted ER and rewrite schemas."""

    mention_rows = _rows(mentions)
    canonical_rows = _rows(canonical_entities)
    candidates = _rows(candidate_pairs)
    decisions = _rows(pair_decisions)
    mapping_rows = _rows(mention_to_canonical)
    alias_rows = _rows(aliases)

    actions = [str(row.get("action") or "") for row in decisions]
    invalid_actions = sorted(
        {action for action in actions if action not in {"merge", "reject", "abstain"}}
    )
    if invalid_actions:
        raise ValueError(
            f"pair decisions have invalid/missing action: {invalid_actions}"
        )
    sources = [str(row.get("source") or "") for row in decisions]
    if any(not source for source in sources):
        raise ValueError("pair decisions require source")
    decision_counts = Counter(actions)
    predicted_clusters = {
        str(row.get("mention_id")): str(row.get("canonical_entity_id"))
        for row in mapping_rows
        if row.get("mention_id") is not None
        and row.get("canonical_entity_id") is not None
    }
    sizes = Counter(predicted_clusters.values())

    source_names = {
        str(row.get("original_name") or "").strip().casefold()
        for row in mention_rows
        if str(row.get("original_name") or "").strip()
    }
    represented_names: set[str] = set()
    for row in canonical_rows:
        represented_names |= _strings(row.get("display_name"))
        represented_names |= _strings(row.get("aliases"))
    for row in alias_rows:
        represented_names |= _strings(row.get("aliases"))
        represented_names |= _strings(row.get("alias") or row.get("name"))

    native = _graph_summary(native_graph)
    advanced = _graph_summary(er_graph)
    plan = _payload(merge_plan)
    rewrite = _payload(rewrite_summary)
    if not rewrite:
        for field in ("rewrite_summary", "graph_rewrite_summary", "rewrite"):
            if isinstance(plan.get(field), Mapping):
                rewrite = dict(plan[field])
                break
    runtime_values = _payload(runtime)
    judge_calls = sum(source == "judge" for source in sources)
    planned_counts = plan.get("decision_counts")
    planned_counts = dict(planned_counts) if isinstance(planned_counts, Mapping) else {}
    blocked_merges = plan.get("blocked_merges")
    cannot_links = plan.get("cannot_links")

    deduplicated_relations = int(
        rewrite.get(
            "deduplicated_relation_count",
            rewrite.get("deduplicated_edges", plan.get("deduplicated_edges", 0)),
        )
    )
    induced_loops = int(
        rewrite.get(
            "induced_self_loops_removed",
            plan.get("induced_self_loops_removed", 0),
        )
    )
    return {
        "mentions": len(mention_rows),
        "canonical_entities": len(canonical_rows),
        "candidate_count": len(candidates),
        "merge_count": decision_counts["merge"],
        "reject_count": decision_counts["reject"],
        "abstain_count": decision_counts["abstain"],
        "abstention_rate": _ratio(decision_counts["abstain"], len(decisions)),
        "judge_rate": _ratio(judge_calls, len(decisions)),
        "cluster_size_distribution": dict(sorted(Counter(sizes.values()).items())),
        "largest_cluster": max(sizes.values(), default=0),
        "alias_coverage": _ratio(
            len(source_names & represented_names), len(source_names)
        ),
        "planned_merge_count": int(
            planned_counts.get("merge", decision_counts["merge"])
        ),
        "planned_reject_count": int(
            planned_counts.get("reject", decision_counts["reject"])
        ),
        "planned_abstain_count": int(
            planned_counts.get("abstain", decision_counts["abstain"])
        ),
        "cannot_link_count": (
            len(cannot_links) if isinstance(cannot_links, Sequence) else None
        ),
        "blocked_merge_count": (
            len(blocked_merges) if isinstance(blocked_merges, Sequence) else None
        ),
        "node_reduction": _difference(native.get("nodes"), advanced.get("nodes")),
        "edge_reduction": _difference(native.get("edges"), advanced.get("edges")),
        "edge_deduplication": deduplicated_relations,
        "deduplicated_relation_count": deduplicated_relations,
        "induced_self_loops_removed": induced_loops,
        "preexisting_self_loops_retained": rewrite.get(
            "preexisting_self_loops_retained"
        ),
        "rewrite_mention_count": rewrite.get("mention_count"),
        "rewrite_canonical_entity_count": rewrite.get("canonical_entity_count"),
        "rewrite_input_relation_count": rewrite.get("input_relation_count"),
        "rewrite_materialized_edge_count": rewrite.get("materialized_edge_count"),
        "rewrite_input_relations": rewrite.get("input_relation_count"),
        "rewrite_materialized_edges": rewrite.get("materialized_edge_count"),
        "connected_components_pre": native.get("connected_components"),
        "connected_components_post": advanced.get("connected_components"),
        "isolates_pre": native.get("isolates"),
        "isolates_post": advanced.get("isolates"),
        "largest_component_pre": native.get(
            "largest_component_size", native.get("largest_component")
        ),
        "largest_component_post": advanced.get(
            "largest_component_size", advanced.get("largest_component")
        ),
        "largest_component_share_pre": native.get("largest_component_share"),
        "largest_component_share_post": advanced.get("largest_component_share"),
        "average_degree_pre": native.get("average_degree"),
        "average_degree_post": advanced.get("average_degree"),
        "normalized_density_pre": native.get("normalized_density"),
        "normalized_density_post": advanced.get("normalized_density"),
        "self_loops_pre": native.get("self_loops"),
        "self_loops_post": advanced.get("self_loops"),
        "node_provenance_coverage_pre": native.get("node_provenance_coverage"),
        "node_provenance_coverage_post": advanced.get("node_provenance_coverage"),
        "edge_provenance_coverage_pre": native.get("edge_provenance_coverage"),
        "edge_provenance_coverage_post": advanced.get("edge_provenance_coverage"),
        "er_runtime_seconds": runtime_values.get("er_runtime_seconds"),
        "judge_calls": runtime_values.get("judge_calls", judge_calls),
        "judge_cache_hits": runtime_values.get("judge_cache_hits"),
        "judge_decisions": runtime_values.get("judge_decisions", judge_calls),
        "judge_prompt_tokens": runtime_values.get("judge_prompt_tokens"),
        "judge_output_tokens": runtime_values.get("judge_output_tokens"),
        "judge_tokens": runtime_values.get("judge_tokens"),
        "judge_cached_original_prompt_tokens": runtime_values.get(
            "judge_cached_original_prompt_tokens"
        ),
        "judge_cached_original_output_tokens": runtime_values.get(
            "judge_cached_original_output_tokens"
        ),
        "judge_cost": runtime_values.get("judge_cost"),
        "judge_cost_basis": runtime_values.get("judge_cost_basis"),
    }


__all__ = ["compute_er_metrics", "compute_graph_topology"]
