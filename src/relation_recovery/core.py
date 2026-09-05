"""Deterministic, evidence-bound relation recovery after frozen ER.

The module is intentionally independent of LightRAG and the builder.  Planning
is pure Python; the verifier receives one whole eligible chunk and may only
return positive relations between canonical entities admitted by that plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.graph.rewrite import GRAPH_FIELD_SEP, RewrittenGraph


RR_SCHEMA_VERSION = "1.0.0"
RR_PROMPT_VERSION = "rr-relation-only-v1"
RR_CANDIDATE_POLICY_VERSION = "same-chunk-missing-canonical-pair-v1"


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}{_sha256_json(parts)[:24]}"


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    left = str(left).strip()
    right = str(right).strip()
    if not left or not right:
        raise ValueError("relation endpoints must be non-empty canonical IDs")
    if left == right:
        raise ValueError("relation recovery cannot create a self-loop")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def graph_edge_pairs(edges: Iterable[Mapping[str, Any]]) -> frozenset[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for edge in edges:
        result.add(
            canonical_pair(
                str(edge.get("source_canonical_entity_id") or ""),
                str(edge.get("target_canonical_entity_id") or ""),
            )
        )
    return frozenset(result)


def iter_missing_pairs(
    canonical_entity_ids: Sequence[str],
    existing_pairs: frozenset[tuple[str, str]] | set[tuple[str, str]],
) -> Iterable[tuple[str, str]]:
    """Yield the exact frozen candidate set in canonical deterministic order."""

    identifiers = sorted(set(str(value).strip() for value in canonical_entity_ids))
    if "" in identifiers:
        raise ValueError("empty canonical entity ID in RR plan")
    for index, left in enumerate(identifiers):
        for right in identifiers[index + 1 :]:
            pair = (left, right)
            if pair not in existing_pairs:
                yield pair


def candidate_set_sha256(
    chunk_id: str,
    canonical_entity_ids: Sequence[str],
    existing_pairs: frozenset[tuple[str, str]] | set[tuple[str, str]],
) -> str:
    digest = hashlib.sha256()
    for left, right in iter_missing_pairs(canonical_entity_ids, existing_pairs):
        digest.update(_canonical_json([chunk_id, left, right]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def response_json_schema() -> dict[str, Any]:
    relation = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "entity_a_id",
            "entity_b_id",
            "relationship_description",
            "evidence_quote",
        ],
        "properties": {
            "entity_a_id": {"type": "string", "minLength": 1},
            "entity_b_id": {"type": "string", "minLength": 1},
            "relationship_description": {"type": "string", "minLength": 1},
            "evidence_quote": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["relations"],
        "properties": {"relations": {"type": "array", "items": relation}},
    }


SYSTEM_PROMPT = """You are a relation verifier, not a graph builder.
Use only the supplied source chunk. Return only meaningful relations explicitly
expressed by that text between the supplied canonical entities. Do not add
entities, use external knowledge, infer missing facts, perform link prediction,
or repeat an existing relation. An evidence_quote must be an exact literal
substring of the source chunk. Return {\"relations\": []} when no missing
relation is directly supported. Output only JSON matching the supplied schema."""


def build_user_prompt(
    *,
    chunk_text: str,
    plan_row: Mapping[str, Any],
) -> str:
    payload = {
        "task": "recover_only_explicit_missing_relations",
        "chunk_id": plan_row["chunk_id"],
        "document_id": plan_row["document_id"],
        "source_chunk": chunk_text,
        "canonical_entities": plan_row["canonical_entities"],
        "existing_relations_between_these_entities": plan_row[
            "existing_local_relations"
        ],
        "output_contract": {
            "positive_relations_only": True,
            "allowed_endpoint_ids": [
                item["canonical_entity_id"] for item in plan_row["canonical_entities"]
            ],
            "evidence_quote_must_be_literal_substring": True,
        },
    }
    return _canonical_json(payload)


def build_messages(
    *, chunk_text: str, plan_row: Mapping[str, Any]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(chunk_text=chunk_text, plan_row=plan_row),
        },
    ]


def _prompt_size(chunk_text: str, plan_row: Mapping[str, Any]) -> tuple[int, int]:
    messages = build_messages(chunk_text=chunk_text, plan_row=plan_row)
    serialized = _canonical_json(messages)
    characters = len(serialized)
    return characters, math.ceil(characters / 4)


def build_rr_plan(
    *,
    chunks: Sequence[Any],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    mention_to_canonical: Sequence[Mapping[str, Any]],
    candidate_policy_version: str = RR_CANDIDATE_POLICY_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create the complete candidate universe before any verifier call."""

    if candidate_policy_version != RR_CANDIDATE_POLICY_VERSION:
        raise ValueError(f"unsupported RR candidate policy: {candidate_policy_version}")
    node_by_id: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        canonical_id = str(node.get("canonical_entity_id") or "").strip()
        if not canonical_id or canonical_id in node_by_id:
            raise ValueError("RR requires unique non-empty canonical node IDs")
        node_by_id[canonical_id] = node

    ids_by_chunk: dict[str, set[str]] = defaultdict(set)
    aliases_by_chunk_id: dict[tuple[str, str], set[str]] = defaultdict(set)
    for canonical_id, node in node_by_id.items():
        for mention in node.get("source_mentions") or []:
            if not isinstance(mention, Mapping):
                continue
            chunk_id = str(mention.get("chunk_id") or "").strip()
            alias = str(mention.get("original_name") or "").strip()
            if chunk_id and alias:
                aliases_by_chunk_id[(chunk_id, canonical_id)].add(alias)
    for row in mention_to_canonical:
        chunk_id = str(row.get("chunk_id") or "").strip()
        canonical_id = str(row.get("canonical_entity_id") or "").strip()
        if not chunk_id or canonical_id not in node_by_id:
            raise ValueError("RR mapping references an unknown chunk/canonical entity")
        ids_by_chunk[chunk_id].add(canonical_id)
        alias = str(
            row.get("original_name")
            or row.get("mention_text")
            or row.get("canonical_display_name")
            or ""
        ).strip()
        if alias:
            aliases_by_chunk_id[(chunk_id, canonical_id)].add(alias)

    frozen_pairs = graph_edge_pairs(edges)
    edge_by_pair = {
        canonical_pair(
            str(edge.get("source_canonical_entity_id") or ""),
            str(edge.get("target_canonical_entity_id") or ""),
        ): edge
        for edge in edges
    }
    rows: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for chunk in sorted(
        chunks,
        key=lambda item: (
            str(_value(item, "document_id", "")),
            int(_value(item, "chunk_order", 0) or 0),
            str(_value(item, "chunk_id", "")),
        ),
    ):
        chunk_id = str(_value(chunk, "chunk_id", "")).strip()
        document_id = str(_value(chunk, "document_id", "")).strip()
        chunk_text = str(_value(chunk, "text", ""))
        chunk_sha256 = str(_value(chunk, "chunk_sha256", "")).strip()
        if not chunk_id or not document_id or not chunk_sha256:
            raise ValueError("RR requires chunk id, document id, and immutable hash")
        if chunk_id in seen_chunks:
            raise ValueError(f"duplicate chunk in RR plan: {chunk_id}")
        seen_chunks.add(chunk_id)
        actual_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        if actual_hash != chunk_sha256:
            raise ValueError(f"chunk hash mismatch while planning RR: {chunk_id}")

        canonical_ids = sorted(ids_by_chunk.get(chunk_id, set()))
        entities: list[dict[str, Any]] = []
        for canonical_id in canonical_ids:
            node = node_by_id[canonical_id]
            node_aliases = {
                str(value).strip()
                for value in (node.get("aliases") or [])
                if str(value).strip()
            }
            local_aliases = aliases_by_chunk_id[(chunk_id, canonical_id)]
            aliases = sorted(local_aliases or node_aliases)
            entities.append(
                {
                    "canonical_entity_id": canonical_id,
                    "canonical_name": str(node.get("entity_name") or "").strip(),
                    "local_aliases": aliases,
                }
            )
        missing_pair_count = sum(
            1 for _ in iter_missing_pairs(canonical_ids, frozen_pairs)
        )
        local_existing: list[dict[str, Any]] = []
        for pair in sorted(frozen_pairs):
            if pair[0] not in ids_by_chunk.get(chunk_id, set()) or pair[
                1
            ] not in ids_by_chunk.get(chunk_id, set()):
                continue
            edge = edge_by_pair[pair]
            local_existing.append(
                {
                    "entity_a_id": pair[0],
                    "entity_b_id": pair[1],
                    "relationship_description": str(edge.get("description") or ""),
                }
            )
        row: dict[str, Any] = {
            "schema_version": RR_SCHEMA_VERSION,
            "record_type": "rr_chunk_plan",
            "candidate_policy_version": candidate_policy_version,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "chunk_order": int(_value(chunk, "chunk_order", 0) or 0),
            "chunk_sha256": chunk_sha256,
            "canonical_entities": entities,
            "canonical_entity_count": len(canonical_ids),
            "existing_local_relations": local_existing,
            "existing_local_relation_count": len(local_existing),
            "missing_pair_count": missing_pair_count,
            "candidate_set_sha256": candidate_set_sha256(
                chunk_id, canonical_ids, frozen_pairs
            ),
            "eligible": missing_pair_count > 0,
        }
        prompt_characters, estimated_tokens = _prompt_size(chunk_text, row)
        row["prompt_characters"] = prompt_characters
        row["prompt_estimated_tokens_chars_div_4"] = estimated_tokens
        rows.append(row)

    eligible = [row for row in rows if row["eligible"]]
    prompt_sizes = sorted(int(row["prompt_characters"]) for row in eligible)
    token_estimates = sorted(
        int(row["prompt_estimated_tokens_chars_div_4"]) for row in eligible
    )

    def percentile(values: Sequence[int], fraction: float) -> int:
        if not values:
            return 0
        return values[math.ceil(len(values) * fraction) - 1]

    summary = {
        "schema_version": RR_SCHEMA_VERSION,
        "record_type": "rr_plan_summary",
        "candidate_policy_version": candidate_policy_version,
        "chunk_count": len(rows),
        "eligible_chunk_count": len(eligible),
        "ineligible_chunk_count": len(rows) - len(eligible),
        "maximum_verifier_calls": len(eligible),
        "candidate_pair_instance_count": sum(
            int(row["missing_pair_count"]) for row in rows
        ),
        "prompt_characters": {
            "total": sum(prompt_sizes),
            "minimum": min(prompt_sizes, default=0),
            "median": percentile(prompt_sizes, 0.5),
            "p95": percentile(prompt_sizes, 0.95),
            "maximum": max(prompt_sizes, default=0),
        },
        "prompt_estimated_tokens_chars_div_4": {
            "total": sum(token_estimates),
            "minimum": min(token_estimates, default=0),
            "median": percentile(token_estimates, 0.5),
            "p95": percentile(token_estimates, 0.95),
            "maximum": max(token_estimates, default=0),
        },
        "frozen_er_node_count": len(nodes),
        "frozen_er_edge_count": len(edges),
        "frozen_er_edge_set_sha256": _sha256_json(sorted(frozen_pairs)),
        "rr_plan_sha256": _sha256_json(rows),
    }
    return rows, summary


def candidate_allowed(
    plan_row: Mapping[str, Any],
    entity_a_id: str,
    entity_b_id: str,
    frozen_er_pairs: frozenset[tuple[str, str]] | set[tuple[str, str]],
) -> bool:
    try:
        pair = canonical_pair(entity_a_id, entity_b_id)
    except ValueError:
        return False
    local = {
        str(item.get("canonical_entity_id") or "")
        for item in plan_row.get("canonical_entities", [])
        if isinstance(item, Mapping)
    }
    return pair[0] in local and pair[1] in local and pair not in frozen_er_pairs


def validate_verifier_response(
    *,
    raw_content: str | Mapping[str, Any],
    chunk_text: str,
    plan_row: Mapping[str, Any],
    frozen_er_pairs: frozenset[tuple[str, str]] | set[tuple[str, str]],
    verifier_identity: Mapping[str, Any],
    request_sha256: str,
    provider_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail the whole chunk closed if any returned item violates the contract."""

    response_sha256 = hashlib.sha256(
        (
            raw_content
            if isinstance(raw_content, str)
            else _canonical_json(raw_content)
        ).encode("utf-8")
    ).hexdigest()
    try:
        payload = (
            json.loads(raw_content)
            if isinstance(raw_content, str)
            else dict(raw_content)
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        payload = None
        parse_error = f"invalid_json: {exc}"
    else:
        parse_error = ""

    errors: list[str] = []
    if not isinstance(payload, dict) or set(payload) != {"relations"}:
        errors.append(parse_error or "response must contain only the relations field")
        relations: list[Any] = []
    elif not isinstance(payload.get("relations"), list):
        errors.append("relations must be an array")
        relations = []
    else:
        relations = list(payload["relations"])

    normalized: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    required = {
        "entity_a_id",
        "entity_b_id",
        "relationship_description",
        "evidence_quote",
    }
    for index, item in enumerate(relations):
        if not isinstance(item, Mapping) or set(item) != required:
            errors.append(f"relations[{index}] has invalid fields")
            continue
        left = str(item.get("entity_a_id") or "").strip()
        right = str(item.get("entity_b_id") or "").strip()
        description = str(item.get("relationship_description") or "").strip()
        evidence = str(item.get("evidence_quote") or "").strip()
        if not candidate_allowed(plan_row, left, right, frozen_er_pairs):
            errors.append(
                f"relations[{index}] endpoints are outside the frozen candidate set"
            )
            continue
        pair = canonical_pair(left, right)
        if pair in seen_pairs:
            errors.append(f"relations[{index}] duplicates pair {pair}")
            continue
        seen_pairs.add(pair)
        if not description:
            errors.append(f"relations[{index}] has an empty description")
            continue
        if not evidence or evidence not in chunk_text:
            errors.append(
                f"relations[{index}] evidence is not a literal chunk substring"
            )
            continue
        normalized.append(
            {
                "rr_relation_id": _stable_id(
                    "rrrel_", plan_row["chunk_id"], pair[0], pair[1]
                ),
                "document_id": plan_row["document_id"],
                "chunk_id": plan_row["chunk_id"],
                "chunk_sha256": plan_row["chunk_sha256"],
                "entity_a_id": pair[0],
                "entity_b_id": pair[1],
                "relationship_description": description,
                "evidence_quote": evidence,
                "candidate_set_sha256": plan_row["candidate_set_sha256"],
                "request_sha256": request_sha256,
                "response_sha256": response_sha256,
                "verifier_identity": dict(verifier_identity),
            }
        )
    if errors:
        normalized = []
    return {
        "schema_version": RR_SCHEMA_VERSION,
        "record_type": "rr_chunk_verification",
        "document_id": plan_row["document_id"],
        "chunk_id": plan_row["chunk_id"],
        "chunk_sha256": plan_row["chunk_sha256"],
        "candidate_set_sha256": plan_row["candidate_set_sha256"],
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "status": "invalid" if errors else "valid",
        "errors": errors,
        "accepted_relations": normalized,
        "accepted_relation_count": len(normalized),
        "provider_metadata": dict(provider_metadata or {}),
    }


@dataclass(frozen=True)
class OllamaRRVerifier:
    client: Any
    model_tag: str
    model_digest: str
    prompt_version: str
    temperature: float
    seed: int
    options: Mapping[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "model_tag": self.model_tag,
            "model_digest": self.model_digest,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "seed": self.seed,
            "generation_parameters": {
                **dict(self.options),
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }

    def request_payload(
        self, *, chunk_text: str, plan_row: Mapping[str, Any]
    ) -> dict[str, Any]:
        messages = build_messages(chunk_text=chunk_text, plan_row=plan_row)
        return {
            "model": self.model_tag,
            "model_digest": self.model_digest,
            "messages": messages,
            "format": response_json_schema(),
            "options": {
                **dict(self.options),
                "temperature": self.temperature,
                "seed": self.seed,
            },
            "think": False,
        }

    def request_sha256(self, *, chunk_text: str, plan_row: Mapping[str, Any]) -> str:
        return _sha256_json(
            self.request_payload(chunk_text=chunk_text, plan_row=plan_row)
        )

    def verify(self, *, chunk_text: str, plan_row: Mapping[str, Any]) -> dict[str, Any]:
        """Make exactly one physical call; callers own immutable resume caching."""

        if self.prompt_version != RR_PROMPT_VERSION:
            raise ValueError(f"unsupported RR prompt version: {self.prompt_version}")
        request_payload = self.request_payload(chunk_text=chunk_text, plan_row=plan_row)
        messages = request_payload["messages"]
        request_sha256 = _sha256_json(request_payload)
        response = self.client.chat(
            model=self.model_tag,
            messages=messages,
            format=response_json_schema(),
            options=request_payload["options"],
            think=False,
        )
        message = _value(response, "message", {}) or {}
        content = _value(message, "content", "")
        provider_metadata = {
            key: _value(response, key)
            for key in (
                "created_at",
                "done",
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if _value(response, key) is not None
        }
        return {
            "raw_content": str(content or ""),
            "request_sha256": request_sha256,
            "provider_metadata": provider_metadata,
        }


def aggregate_rr_graph(
    er_graph: RewrittenGraph,
    accepted_relations: Sequence[Mapping[str, Any]],
    *,
    graph_field_separator: str = GRAPH_FIELD_SEP,
) -> RewrittenGraph:
    """Add one undirected edge per recovered pair while preserving raw evidence."""

    if not graph_field_separator:
        raise ValueError("graph field separator must be non-empty")
    node_name_by_id = {
        str(node["canonical_entity_id"]): str(node["entity_name"])
        for node in er_graph.nodes
    }
    existing = graph_edge_pairs(er_graph.edges)
    contributions: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen_instances: set[tuple[str, str, str]] = set()
    for item in accepted_relations:
        pair = canonical_pair(
            str(item.get("entity_a_id") or ""),
            str(item.get("entity_b_id") or ""),
        )
        if pair in existing:
            raise ValueError(f"RR accepted an edge already present after ER: {pair}")
        if pair[0] not in node_name_by_id or pair[1] not in node_name_by_id:
            raise ValueError(f"RR accepted an unknown canonical endpoint: {pair}")
        chunk_id = str(item.get("chunk_id") or "").strip()
        instance = (chunk_id, pair[0], pair[1])
        if not chunk_id or instance in seen_instances:
            raise ValueError(f"duplicate/invalid RR candidate instance: {instance}")
        seen_instances.add(instance)
        contributions[pair].append(item)

    rr_edges: list[dict[str, Any]] = []
    for (left, right), raw_items in sorted(contributions.items()):
        items = sorted(
            raw_items,
            key=lambda item: (
                str(item.get("chunk_id") or ""),
                str(item.get("rr_relation_id") or ""),
            ),
        )
        descriptions = sorted(
            {
                str(item.get("relationship_description") or "").strip()
                for item in items
                if str(item.get("relationship_description") or "").strip()
            }
        )
        relation_ids = [str(item.get("rr_relation_id") or "") for item in items]
        rr_edges.append(
            {
                "edge_id": _stable_id("edge_", left, right),
                "source_canonical_entity_id": left,
                "target_canonical_entity_id": right,
                "source": node_name_by_id[left],
                "target": node_name_by_id[right],
                "description": graph_field_separator.join(descriptions),
                "description_present": bool(descriptions),
                "keywords": "recovered relation",
                "relation_types": [],
                "weight": float(len(items)),
                "source_relation_ids": relation_ids,
                "source_chunk_ids": sorted(
                    {str(item.get("chunk_id") or "") for item in items}
                ),
                "source_document_ids": sorted(
                    {str(item.get("document_id") or "") for item in items}
                ),
                "provenance": [
                    {
                        "origin": "relation_recovery",
                        "rr_relation_id": item.get("rr_relation_id"),
                        "document_id": item.get("document_id"),
                        "chunk_id": item.get("chunk_id"),
                        "chunk_sha256": item.get("chunk_sha256"),
                        "evidence_quote": item.get("evidence_quote"),
                        "request_sha256": item.get("request_sha256"),
                        "response_sha256": item.get("response_sha256"),
                        "candidate_set_sha256": item.get("candidate_set_sha256"),
                        "verifier_identity": item.get("verifier_identity"),
                    }
                    for item in items
                ],
                "deduplicated_relation_count": len(items),
                "relation_origin": "relation_recovery",
            }
        )

    edges = sorted(
        [*deepcopy(er_graph.edges), *rr_edges],
        key=lambda edge: str(edge["edge_id"]),
    )
    summary = {
        **deepcopy(er_graph.summary),
        "er_entity_count": len(er_graph.nodes),
        "er_edge_count": len(er_graph.edges),
        "rr_candidate_instance_accept_count": len(accepted_relations),
        "rr_recovered_edge_count": len(rr_edges),
        "materialized_edge_count": len(edges),
        "er_rr_entity_count": len(er_graph.nodes),
        "er_rr_edge_count": len(edges),
    }
    return RewrittenGraph(
        nodes=deepcopy(er_graph.nodes),
        edges=edges,
        mention_to_canonical=deepcopy(er_graph.mention_to_canonical),
        induced_self_loops=deepcopy(er_graph.induced_self_loops),
        preexisting_self_loops=deepcopy(er_graph.preexisting_self_loops),
        summary=summary,
    )


__all__ = [
    "OllamaRRVerifier",
    "RR_CANDIDATE_POLICY_VERSION",
    "RR_PROMPT_VERSION",
    "RR_SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "aggregate_rr_graph",
    "build_messages",
    "build_rr_plan",
    "build_user_prompt",
    "candidate_allowed",
    "candidate_set_sha256",
    "canonical_pair",
    "graph_edge_pairs",
    "iter_missing_pairs",
    "response_json_schema",
    "validate_verifier_response",
]
