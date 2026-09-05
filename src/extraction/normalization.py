"""Deterministic, LLM-free normalization of captured LightRAG responses."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .models import (
    ExtractionParseRecord,
    NormalizedChunkResult,
    NormalizedEntityMention,
    NormalizedRelation,
    ParseStatus,
)


_SPACE_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return _SPACE_RE.sub(" ", text).strip().casefold()


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = _SPACE_RE.sub(" ", text).strip()
    return text or None


def _strip_wrappers(raw: str) -> str:
    value = _THINK_RE.sub("", raw).strip()
    if value.startswith("```") and value.endswith("```"):
        first_newline = value.find("\n")
        if first_newline >= 0:
            value = value[first_newline + 1 : -3].strip()
    return value


def recover_json(
    raw: str | None,
) -> tuple[dict[str, Any] | None, ParseStatus, str | None, str | None]:
    """Return payload, status, recovery method and a compact error."""

    if not isinstance(raw, str) or not raw.strip():
        return None, "no_response", None, "empty response"
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value, "strict", "json.loads", None
        return (
            None,
            "failed",
            "json.loads",
            f"top-level {type(value).__name__} is not object",
        )
    except (json.JSONDecodeError, TypeError, ValueError) as strict_error:
        cleaned = _strip_wrappers(raw)
        try:
            import json_repair

            value = json_repair.loads(cleaned)
            if isinstance(value, dict):
                return value, "recovered", "json_repair", None
            return (
                None,
                "failed",
                "json_repair",
                f"top-level {type(value).__name__} is not object",
            )
        except Exception as recovery_error:
            return (
                None,
                "failed",
                "json_repair",
                f"strict={type(strict_error).__name__}; recovery={type(recovery_error).__name__}",
            )


def _call_kind(call: Mapping[str, Any]) -> str:
    provenance = call.get("technical_provenance")
    if isinstance(provenance, Mapping) and provenance.get("call_kind"):
        return str(provenance["call_kind"])
    return "gleaning" if int(call.get("gleaning_round") or 0) else "initial"


def select_logical_calls(calls: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose one successful physical response for each logical extraction phase.

    Resumed runs can contain a second logical initial/gleaning call. The
    newest successful response wins deterministically; failed attempts remain
    auditable in the raw artifact but are not double-counted as mentions.
    """

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for value in calls:
        call = dict(value)
        key = (int(call.get("gleaning_round") or 0), _call_kind(call))
        buckets[key].append(call)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        values = buckets[key]
        successful = [
            item
            for item in values
            if not item.get("error_type") and isinstance(item.get("raw_response"), str)
        ]
        candidates = successful or values
        candidates.sort(
            key=lambda item: (
                str(item.get("created_at") or ""),
                int(item.get("attempt_number") or 0),
                str(item.get("call_id") or ""),
            )
        )
        selected.append(candidates[-1])
    return selected


def _keywords(value: Any) -> list[str]:
    values: Sequence[Any]
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
    elif value is None:
        values = []
    else:
        values = re.split(r"[,;|]", str(value))
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _clean_text(item)
        key = normalize_name(text)
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _combined_parse_status(values: Iterable[ParseStatus]) -> ParseStatus:
    statuses = list(values)
    return "recovered" if "recovered" in statuses else "strict"


def _longest_text(values: Iterable[Any]) -> str | None:
    cleaned = [text for value in values if (text := _clean_text(value))]
    if not cleaned:
        return None
    # ``max`` keeps the first value on equal lengths, giving a stable analogue
    # of LightRAG's per-chunk "keep the better/longer description" behavior.
    return max(cleaned, key=len)


def _majority_text(values: Iterable[Any]) -> str | None:
    ordered = [text for value in values if (text := _clean_text(value))]
    if not ordered:
        return None
    counts: dict[str, int] = defaultdict(int)
    display: dict[str, str] = {}
    order: list[str] = []
    for text in ordered:
        key = normalize_name(text)
        if key not in counts:
            order.append(key)
            display[key] = text
        counts[key] += 1
    winner = max(order, key=lambda key: counts[key])
    return display[winner]


def _stock_sanitize(value: Any, *, remove_inner_quotes: bool = False) -> str:
    """Use the exact sanitizer from the pinned LightRAG JSON parser."""

    from lightrag.utils import sanitize_and_normalize_extracted_text

    return sanitize_and_normalize_extracted_text(
        str(value if value is not None else ""),
        remove_inner_quotes=remove_inner_quotes,
    )


def _stock_identifier(value: Any, *, chunk_id: str, role: str) -> str:
    """Sanitize and truncate a graph identifier exactly as pinned LightRAG."""

    from lightrag.constants import DEFAULT_ENTITY_NAME_MAX_LENGTH
    from lightrag.operate import _truncate_entity_identifier

    sanitized = _stock_sanitize(value, remove_inner_quotes=True)
    if not sanitized:
        return ""
    return _truncate_entity_identifier(
        sanitized,
        DEFAULT_ENTITY_NAME_MAX_LENGTH,
        chunk_id,
        role,
    )


def _stock_entity_record(
    raw: Mapping[str, Any], *, chunk_id: str
) -> tuple[str, str, str] | None:
    """Return the values accepted by the pinned stock JSON entity parser."""

    name = _stock_identifier(
        raw.get("name", raw.get("entity_name")),
        chunk_id=chunk_id,
        role="Entity name",
    )
    entity_type = _stock_sanitize(
        raw.get("type", raw.get("entity_type")), remove_inner_quotes=True
    )
    description = _stock_sanitize(raw.get("description"))
    if (
        not name
        or not entity_type
        or not description
        or any(
            char in entity_type
            for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        )
    ):
        return None
    return name, entity_type.replace(" ", "").lower(), description


def _stock_relation_record(
    raw: Mapping[str, Any], *, chunk_id: str
) -> tuple[str, str, list[str], str] | None:
    """Return the values accepted by the pinned stock JSON relation parser."""

    source_untruncated = _stock_sanitize(
        raw.get("source", raw.get("src_id")), remove_inner_quotes=True
    )
    target_untruncated = _stock_sanitize(
        raw.get("target", raw.get("tgt_id")), remove_inner_quotes=True
    )
    description = _stock_sanitize(raw.get("description"))
    if (
        not source_untruncated
        or not target_untruncated
        or not description
        or source_untruncated == target_untruncated
    ):
        return None
    source = _stock_identifier(
        source_untruncated, chunk_id=chunk_id, role="Relation entity"
    )
    target = _stock_identifier(
        target_untruncated, chunk_id=chunk_id, role="Relation entity"
    )
    if not source or not target or source == target:
        return None
    keywords_text = _stock_sanitize(
        raw.get("keywords"), remove_inner_quotes=True
    ).replace("，", ",")
    return source, target, _keywords(keywords_text), description


def normalize_chunk_calls(
    chunk: Mapping[str, Any],
    calls: Iterable[Mapping[str, Any]],
) -> NormalizedChunkResult:
    """Normalize all initial/gleaning responses for one immutable chunk."""

    chunk_id = str(chunk.get("chunk_id") or chunk.get("_id") or "").strip()
    document_id = str(
        chunk.get("document_id") or chunk.get("full_doc_id") or ""
    ).strip()
    if not chunk_id or not document_id:
        raise ValueError("chunk must contain document_id and chunk_id")
    text = str(
        chunk.get("text")
        if chunk.get("text") is not None
        else chunk.get("content") or ""
    )
    chunk_sha256 = str(chunk.get("text_sha256") or _sha256(text))
    chunk_order = int(chunk.get("chunk_order", chunk.get("chunk_order_index", 0)))
    token_raw = chunk.get("token_count", chunk.get("tokens"))
    token_count = (
        int(token_raw) if isinstance(token_raw, int) and token_raw >= 0 else None
    )
    selected = select_logical_calls(calls)

    parsed_calls: list[tuple[dict[str, Any], dict[str, Any], ParseStatus]] = []
    parse_records: list[ExtractionParseRecord] = []
    for call in selected:
        call_id = str(call.get("call_id") or "")
        if not call_id:
            raise ValueError(f"captured extraction call for {chunk_id} lacks call_id")
        payload, status, method, error = recover_json(call.get("raw_response"))
        entities = payload.get("entities", []) if payload else []
        relations = (
            payload.get("relationships", payload.get("relations", []))
            if payload
            else []
        )
        entity_count = len(entities) if isinstance(entities, list) else 0
        relation_count = len(relations) if isinstance(relations, list) else 0
        parse_records.append(
            ExtractionParseRecord(
                extraction_call_id=call_id,
                response_sha256=call.get("response_sha256"),
                status=status,
                recovery_method=method,
                error=error,
                entity_records_seen=entity_count,
                relation_records_seen=relation_count,
            )
        )
        if payload is not None:
            parsed_calls.append((call, payload, status))

    # Stock LightRAG keys Native entities by its sanitized, case-sensitive
    # display name. Keep one ER mention per such key per chunk. The separate
    # ``normalized_name`` field remains case-folded for the later ER stage.
    entity_groups: dict[
        str,
        list[tuple[dict[str, Any], Mapping[str, Any], ParseStatus, int, int]],
    ] = {}
    pending_relations: list[
        tuple[dict[str, Any], Mapping[str, Any], ParseStatus, int, int]
    ] = []

    for call_order, (call, payload, status) in enumerate(parsed_calls):
        raw_entities = payload.get("entities", [])
        if not isinstance(raw_entities, list):
            raw_entities = []
        for raw_entity_index, raw_entity in enumerate(raw_entities):
            if not isinstance(raw_entity, Mapping):
                continue
            parsed_entity = _stock_entity_record(raw_entity, chunk_id=chunk_id)
            if parsed_entity is None:
                continue
            original_name, entity_type, description = parsed_entity
            stock_record = {
                **raw_entity,
                "name": original_name,
                "type": entity_type,
                "description": description,
            }
            entity_groups.setdefault(original_name, []).append(
                (call, stock_record, status, call_order, raw_entity_index)
            )

        raw_relations = payload.get("relationships", payload.get("relations", []))
        if not isinstance(raw_relations, list):
            raw_relations = []
        for raw_relation_index, raw_relation in enumerate(raw_relations):
            if not isinstance(raw_relation, Mapping):
                continue
            parsed_relation = _stock_relation_record(raw_relation, chunk_id=chunk_id)
            if parsed_relation is None:
                continue
            source, target, keywords, description = parsed_relation
            stock_record = {
                **raw_relation,
                "source": source,
                "target": target,
                "keywords": keywords,
                "description": description,
            }
            pending_relations.append(
                (call, stock_record, status, call_order, raw_relation_index)
            )

    mentions: list[NormalizedEntityMention] = []
    by_name: dict[str, NormalizedEntityMention] = {}
    for stock_name, records in entity_groups.items():
        first_call, first_raw, _first_status, _call_order, _raw_index = records[0]
        mention_index = len(mentions)
        original_name = stock_name
        normalized_name = normalize_name(stock_name)
        description = _longest_text(raw.get("description") for _, raw, *_ in records)
        entity_type = _majority_text(
            raw.get("type", raw.get("entity_type")) for _, raw, *_ in records
        )
        contributors = [
            {
                "extraction_call_id": str(call["call_id"]),
                "call_order": call_order,
                "raw_entity_index": raw_index,
                "gleaning_round": int(call.get("gleaning_round") or 0),
                "call_kind": _call_kind(call),
            }
            for call, _raw, _status, call_order, raw_index in records
        ]
        mention = NormalizedEntityMention(
            mention_id=f"{document_id}:{chunk_id}:mention:{mention_index:05d}",
            mention_index=mention_index,
            original_name=original_name,
            normalized_name=normalized_name,
            entity_type=entity_type,
            description=description,
            document_id=document_id,
            chunk_id=chunk_id,
            provenance={
                "normalization": "stock_lightrag_name_coalesce",
                "coalesced_record_count": len(records),
                "contributors": contributors,
            },
            extraction_call_id=str(first_call["call_id"]),
            response_sha256=first_call.get("response_sha256"),
            parse_status=_combined_parse_status(
                status for _, _, status, _, _ in records
            ),
            description_present=bool(description),
        )
        mentions.append(mention)
        by_name[stock_name] = mention

    implicit_by_name: dict[str, NormalizedEntityMention] = {}

    def endpoint_mention(
        original: str,
        call: Mapping[str, Any],
        parse_status: ParseStatus,
        endpoint_role: str,
        raw_relation_index: int,
    ) -> tuple[NormalizedEntityMention, str]:
        key = original
        if key in by_name:
            return by_name[key], "stock_exact_chunk_name"
        if key in implicit_by_name:
            return implicit_by_name[key], "reused_implicit_relation_endpoint"
        call_id = str(call["call_id"])
        mention_index = len(mentions)
        implicit = NormalizedEntityMention(
            mention_id=f"{document_id}:{chunk_id}:mention:{mention_index:05d}",
            mention_index=mention_index,
            original_name=original,
            normalized_name=normalize_name(original),
            entity_type=None,
            description=None,
            document_id=document_id,
            chunk_id=chunk_id,
            provenance={
                "normalization": "stock_lightrag_implicit_relation_endpoint",
                "created_from_relation_endpoint": endpoint_role,
                "raw_relation_index": raw_relation_index,
                "gleaning_round": int(call.get("gleaning_round") or 0),
            },
            extraction_call_id=call_id,
            response_sha256=call.get("response_sha256"),
            parse_status=parse_status,
            description_present=False,
            implicit_from_relation=True,
        )
        mentions.append(implicit)
        by_name[key] = implicit
        implicit_by_name[key] = implicit
        return implicit, "created_implicit_relation_endpoint"

    relation_groups: dict[
        tuple[str, str],
        list[
            tuple[
                dict[str, Any],
                Mapping[str, Any],
                ParseStatus,
                int,
                int,
                NormalizedEntityMention,
                NormalizedEntityMention,
                str,
                str,
            ]
        ],
    ] = {}
    for call, raw_relation, status, call_order, raw_relation_index in pending_relations:
        source_name = _clean_text(
            raw_relation.get("source", raw_relation.get("src_id"))
        )
        target_name = _clean_text(
            raw_relation.get("target", raw_relation.get("tgt_id"))
        )
        if not source_name or not target_name:
            continue
        source, source_strategy = endpoint_mention(
            source_name,
            call,
            status,
            "source",
            raw_relation_index,
        )
        target, target_strategy = endpoint_mention(
            target_name,
            call,
            status,
            "target",
            raw_relation_index,
        )
        if source.mention_id == target.mention_id:
            continue
        pair = tuple(sorted((source.mention_id, target.mention_id)))
        relation_groups.setdefault(pair, []).append(
            (
                call,
                raw_relation,
                status,
                call_order,
                raw_relation_index,
                source,
                target,
                source_strategy,
                target_strategy,
            )
        )

    relations: list[NormalizedRelation] = []
    for records in relation_groups.values():
        (
            first_call,
            first_raw,
            _first_status,
            _first_call_order,
            _first_raw_index,
            source,
            target,
            source_strategy,
            target_strategy,
        ) = records[0]
        relation_index = len(relations)
        description = _longest_text(raw.get("description") for _, raw, *_ in records)
        keywords: list[str] = []
        keyword_seen: set[str] = set()
        for _call, raw, *_rest in records:
            for keyword in _keywords(raw.get("keywords")):
                key = normalize_name(keyword)
                if key not in keyword_seen:
                    keyword_seen.add(key)
                    keywords.append(keyword)
        relation_type = _majority_text(
            raw.get("type", raw.get("relation_type")) for _, raw, *_ in records
        )
        relation = NormalizedRelation(
            relation_id=f"{document_id}:{chunk_id}:relation:{relation_index:05d}",
            relation_index=relation_index,
            source_mention_id=source.mention_id,
            target_mention_id=target.mention_id,
            source_resolution_state="resolved",
            target_resolution_state="resolved",
            source_candidate_mention_ids=[source.mention_id],
            target_candidate_mention_ids=[target.mention_id],
            source_original_name=(
                _clean_text(first_raw.get("source", first_raw.get("src_id")))
                or source.original_name
            ),
            target_original_name=(
                _clean_text(first_raw.get("target", first_raw.get("tgt_id")))
                or target.original_name
            ),
            relation_type=relation_type,
            keywords=keywords,
            description=description,
            document_id=document_id,
            chunk_id=chunk_id,
            provenance={
                "normalization": "stock_lightrag_relation_pair_coalesce",
                "coalesced_record_count": len(records),
                "source_resolution": source_strategy,
                "target_resolution": target_strategy,
                "contributors": [
                    {
                        "extraction_call_id": str(call["call_id"]),
                        "call_order": call_order,
                        "raw_relation_index": raw_index,
                        "gleaning_round": int(call.get("gleaning_round") or 0),
                        "call_kind": _call_kind(call),
                    }
                    for call, _raw, _status, call_order, raw_index, *_ in records
                ],
            },
            extraction_call_id=str(first_call["call_id"]),
            response_sha256=first_call.get("response_sha256"),
            parse_status=_combined_parse_status(status for _, _, status, *_ in records),
            description_present=bool(description),
        )
        relations.append(relation)

    return NormalizedChunkResult(
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        chunk_sha256=chunk_sha256,
        chunk_order=chunk_order,
        token_count=token_count,
        extraction_call_ids=[str(call["call_id"]) for call in selected],
        raw_response_sha256s=[
            str(call["response_sha256"])
            for call in selected
            if call.get("response_sha256")
        ],
        parse_records=parse_records,
        entities=mentions,
        relations=relations,
        entity_present=bool(mentions),
        description_present=any(item.description_present for item in mentions),
        relations_present=bool(relations),
        staging_complete=bool(selected)
        and all(
            record.status not in {"failed", "no_response"} for record in parse_records
        )
        and all(
            relation.source_resolution_state == "resolved"
            and relation.target_resolution_state == "resolved"
            for relation in relations
        ),
    )


def normalize_extraction_calls(
    chunks: Iterable[Mapping[str, Any]],
    calls: Iterable[Mapping[str, Any]],
) -> list[NormalizedChunkResult]:
    calls_by_chunk: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for call in calls:
        chunk_id = str(call.get("chunk_id") or "")
        if chunk_id:
            calls_by_chunk[chunk_id].append(call)
    results = [
        normalize_chunk_calls(
            chunk,
            calls_by_chunk.get(
                str(chunk.get("chunk_id") or chunk.get("_id") or ""), []
            ),
        )
        for chunk in chunks
    ]
    results.sort(key=lambda item: (item.document_id, item.chunk_order, item.chunk_id))
    return results


__all__ = [
    "normalize_chunk_calls",
    "normalize_extraction_calls",
    "normalize_name",
    "recover_json",
    "select_logical_calls",
]
