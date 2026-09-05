"""Extraction reporting with distinct entity/relation/description/recovery rates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import random
from typing import Any


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_extraction_metrics(
    normalized_chunks: Iterable[Mapping[str, Any] | Any],
) -> dict[str, Any]:
    chunks = [
        value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
        for value in normalized_chunks
    ]
    mentions = [entity for chunk in chunks for entity in chunk.get("entities", [])]
    relations = [
        relation for chunk in chunks for relation in chunk.get("relations", [])
    ]
    parse_records = [
        record for chunk in chunks for record in chunk.get("parse_records", [])
    ]
    entity_chunks = sum(bool(chunk.get("entity_present")) for chunk in chunks)
    relation_chunks = sum(bool(chunk.get("relations_present")) for chunk in chunks)
    described_mentions = sum(bool(item.get("description_present")) for item in mentions)
    described_relations = sum(
        bool(item.get("description_present")) for item in relations
    )
    recovered = sum(record.get("status") == "recovered" for record in parse_records)
    strict = sum(record.get("status") == "strict" for record in parse_records)
    failed = sum(
        record.get("status") in {"failed", "no_response"} for record in parse_records
    )
    parse_attempts = len(parse_records)
    token_count = sum(
        int(chunk["token_count"])
        for chunk in chunks
        if isinstance(chunk.get("token_count"), int)
        and not isinstance(chunk.get("token_count"), bool)
    )
    return {
        "chunks": len(chunks),
        "mentions": len(mentions),
        "relations": len(relations),
        "entity_extraction_coverage": _ratio(entity_chunks, len(chunks)),
        "relation_extraction_coverage": _ratio(relation_chunks, len(chunks)),
        "description_completeness": _ratio(described_mentions, len(mentions)),
        "relation_description_completeness": _ratio(
            described_relations, len(relations)
        ),
        "strict_json_rate": _ratio(strict, parse_attempts),
        "json_recovery_rate": _ratio(recovered, parse_attempts),
        "parse_failure_rate": _ratio(failed, parse_attempts),
        "chunks_with_entities": entity_chunks,
        "chunks_with_relations": relation_chunks,
        "mentions_without_description": len(mentions) - described_mentions,
        "relations_without_description": len(relations) - described_relations,
        "chunk_tokens": token_count or None,
        "entities_per_1000_tokens": (
            len(mentions) * 1000 / token_count if token_count else None
        ),
        "relations_per_1000_tokens": (
            len(relations) * 1000 / token_count if token_count else None
        ),
    }


def compute_extraction_efficiency(
    extraction_calls: Iterable[Mapping[str, Any] | Any],
) -> dict[str, Any]:
    """Summarize physical builder calls without double-counting exported retries."""

    physical: dict[tuple[str, int], dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for value in extraction_calls:
        row = (
            value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else dict(value)
        )
        call_id = str(row.get("call_id") or "").strip()
        attempt = int(row.get("attempt_number") or row.get("attempt") or 1)
        if call_id:
            physical[(call_id, attempt)] = row
        else:
            anonymous.append(row)
    rows = [*physical.values(), *anonymous]

    def numeric(field: str) -> list[float]:
        result: list[float] = []
        for row in rows:
            value = row.get(field)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    logical_ids = {
        str(row.get("call_id")) for row in rows if str(row.get("call_id") or "")
    }
    input_tokens = numeric("input_tokens")
    output_tokens = numeric("output_tokens")
    latency_ms = numeric("latency_ms")
    failed = sum(
        bool(row.get("error_type") or row.get("error_message")) for row in rows
    )
    cache_hits = sum(
        bool(row.get("cache_hit"))
        or (
            isinstance(row.get("technical_provenance"), Mapping)
            and bool(row["technical_provenance"].get("cache_hit"))
        )
        for row in rows
    )
    json_observed = [row for row in rows if row.get("json_valid") is not None]
    schema_observed = [row for row in rows if row.get("schema_valid") is not None]
    parse_observed = [row for row in rows if row.get("parse_success") is not None]
    operational_success = [
        row
        for row in rows
        if row.get("parse_success") is not None
        or row.get("error_type")
        or row.get("error_message")
    ]
    return {
        "physical_call_count": len(rows),
        "logical_call_count": len(logical_ids) + len(anonymous),
        "retry_count": max(0, len(rows) - len(logical_ids) - len(anonymous)),
        "gleaning_call_count": sum(
            int(row.get("gleaning_round") or 0) > 0 for row in rows
        ),
        "cache_hit_count": cache_hits,
        "failed_call_count": failed,
        "call_failure_rate": _ratio(failed, len(rows)),
        "raw_json_valid_rate": _ratio(
            sum(bool(row.get("json_valid")) for row in json_observed),
            len(json_observed),
        ),
        "strict_schema_compliance_rate": _ratio(
            sum(bool(row.get("schema_valid")) for row in schema_observed),
            len(schema_observed),
        ),
        "tolerant_parse_success_rate": _ratio(
            sum(bool(row.get("parse_success")) for row in parse_observed),
            len(parse_observed),
        ),
        "operational_extraction_success_rate": _ratio(
            sum(
                bool(row.get("parse_success"))
                and not row.get("error_type")
                and not row.get("error_message")
                for row in operational_success
            ),
            len(operational_success),
        ),
        "failure_types": dict(
            sorted(
                {
                    str(kind): sum(
                        str(row.get("error_type") or "") == str(kind) for row in rows
                    )
                    for kind in {
                        row.get("error_type") for row in rows if row.get("error_type")
                    }
                }.items()
            )
        ),
        "latency_ms_total": sum(latency_ms) if latency_ms else None,
        "latency_ms_mean": _ratio(sum(latency_ms), len(latency_ms)),
        "input_tokens_total": int(sum(input_tokens)) if input_tokens else None,
        "output_tokens_total": int(sum(output_tokens)) if output_tokens else None,
        "total_tokens": (
            int(sum(input_tokens) + sum(output_tokens))
            if input_tokens and output_tokens
            else None
        ),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_extraction_metrics(
    normalized_chunks: Iterable[Mapping[str, Any] | Any],
    extraction_calls: Iterable[Mapping[str, Any] | Any] = (),
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Cluster bootstrap: every sampled document contributes all of its chunks."""

    if bootstrap_samples <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid extraction bootstrap parameters")
    chunks = [
        value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
        for value in normalized_chunks
    ]
    calls = [
        value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
        for value in extraction_calls
    ]
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("cluster bootstrap requires document_id on every chunk")
        by_document[document_id].append(chunk)
    if not by_document:
        raise ValueError("cluster bootstrap requires at least one document")
    calls_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        document_id = str(call.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("cluster bootstrap requires document_id on every call")
        if document_id not in by_document:
            raise ValueError(
                f"extraction call references unknown document: {document_id}"
            )
        calls_by_document[document_id].append(call)

    point = {
        **compute_extraction_metrics(chunks),
        **{
            f"call_{key}": value
            for key, value in compute_extraction_efficiency(calls).items()
        },
    }
    metric_fields = sorted(
        field
        for field, value in point.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not field.endswith(("_count", "_total"))
        and field not in {"chunks", "mentions", "relations", "chunk_tokens"}
    )
    document_ids = sorted(by_document)
    generator = random.Random(
        int.from_bytes(
            hashlib.sha256(f"{seed}\0extraction".encode()).digest()[:8], "big"
        )
    )
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap_samples):
        sampled_ids = [
            document_ids[generator.randrange(len(document_ids))] for _ in document_ids
        ]
        sampled_chunks = [
            chunk for document_id in sampled_ids for chunk in by_document[document_id]
        ]
        sampled_calls: list[dict[str, Any]] = []
        for draw_index, document_id in enumerate(sampled_ids):
            for call in calls_by_document[document_id]:
                sampled = dict(call)
                call_id = str(sampled.get("call_id") or "")
                if call_id:
                    sampled["call_id"] = f"bootstrap-{draw_index}:{call_id}"
                sampled_calls.append(sampled)
        replicate = {
            **compute_extraction_metrics(sampled_chunks),
            **{
                f"call_{key}": value
                for key, value in compute_extraction_efficiency(sampled_calls).items()
            },
        }
        for field in metric_fields:
            value = replicate.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                distributions[field].append(float(value))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "method": "document_cluster_percentile",
        "cluster_unit": "document_id",
        "document_count": len(document_ids),
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "seed": seed,
        "point_estimates": {field: point.get(field) for field in metric_fields},
        "confidence_intervals": {
            field: {
                "lower": _percentile(values, alpha),
                "upper": _percentile(values, 1.0 - alpha),
                "replicate_count": len(values),
            }
            for field, values in distributions.items()
            if values
        },
    }


__all__ = [
    "bootstrap_extraction_metrics",
    "compute_extraction_efficiency",
    "compute_extraction_metrics",
]
