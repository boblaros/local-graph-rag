"""Pure mention-level graph rewriting for the corpus-level ER branch.

This module has no LightRAG dependency.  It consumes immutable normalized
mentions/relations plus the audited mention-to-canonical mapping and produces a
deterministic, LightRAG-compatible graph projection.  The projection keeps the
stable canonical id in artifacts while using the unique canonical display name
as LightRAG's node identifier (LightRAG does not expose separate id/name fields).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


GRAPH_FIELD_SEP = "<SEP>"


class GraphRewriteError(ValueError):
    """Raised when immutable extraction/ER inputs cannot be rewritten safely."""


def _value(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:24]}"


def _required_text(record: Any, names: tuple[str, ...], label: str) -> str:
    value = _value(record, *names)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise GraphRewriteError(f"{label} is required ({'/'.join(names)})")
    return text


def _optional_text(record: Any, *names: str) -> str:
    value = _value(record, *names)
    return str(value).strip() if value is not None else ""


def _surface_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _unique_strings(
            piece.strip()
            for piece in value.replace(";", ",").split(",")
            if piece.strip()
        )
    if isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        return _unique_strings(value)
    return _unique_strings([value])


def _majority_text(values: Iterable[str], default: str = "UNKNOWN") -> str:
    cleaned = [value.strip() for value in values if value and value.strip()]
    if not cleaned:
        return default
    counts = Counter(value.casefold() for value in cleaned)
    best_count = max(counts.values())
    best_keys = sorted(key for key, count in counts.items() if count == best_count)
    chosen_key = best_keys[0]
    return sorted(value for value in cleaned if value.casefold() == chosen_key)[0]


@dataclass(frozen=True, order=True)
class MentionKey:
    """Stable mention identity; surface forms are deliberately not keys."""

    document_id: str
    chunk_id: str
    mention_id: str

    @classmethod
    def from_record(cls, record: Any) -> "MentionKey":
        return cls(
            _required_text(record, ("document_id",), "mention document_id"),
            _required_text(record, ("chunk_id",), "mention chunk_id"),
            _required_text(record, ("mention_id", "entity_mention_id"), "mention_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclass
class RewrittenGraph:
    """JSON-serializable result of applying one audited ER merge plan."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    mention_to_canonical: list[dict[str, Any]]
    induced_self_loops: list[dict[str, Any]] = field(default_factory=list)
    preexisting_self_loops: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _canonical_id(record: Any) -> str:
    return _required_text(
        record,
        ("canonical_entity_id", "canonical_id", "entity_id", "id"),
        "canonical entity id",
    )


def _canonical_name(record: Any) -> str:
    return _required_text(
        record,
        ("display_name", "canonical_name", "entity_name", "name"),
        "canonical display name",
    )


def _mention_name(record: Any) -> str:
    return _required_text(
        record,
        ("original_name", "name", "entity_name", "mention_text"),
        "mention original name",
    )


def _build_resolution_index(
    resolutions: Sequence[Any] | Mapping[Any, Any],
    mentions_by_key: Mapping[MentionKey, Any],
) -> dict[MentionKey, str]:
    mention_id_index: dict[str, list[MentionKey]] = defaultdict(list)
    for key in mentions_by_key:
        mention_id_index[key.mention_id].append(key)

    def key_from_raw(raw: Any, record: Any | None = None) -> MentionKey:
        if isinstance(raw, MentionKey):
            return raw
        if isinstance(raw, tuple) and len(raw) == 3:
            return MentionKey(*(str(item) for item in raw))
        if record is not None:
            try:
                return MentionKey.from_record(record)
            except GraphRewriteError:
                pass
        text = str(raw).strip()
        candidates = mention_id_index.get(text, [])
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise GraphRewriteError(f"resolution references unknown mention {text!r}")
        raise GraphRewriteError(
            f"resolution mention_id {text!r} is ambiguous without document/chunk"
        )

    result: dict[MentionKey, str] = {}
    if isinstance(resolutions, Mapping):
        items = list(resolutions.items())
    else:
        items = [(None, record) for record in resolutions]

    for raw_key, raw_value in items:
        if raw_key is None:
            record = raw_value
            key = key_from_raw(_value(record, "mention_id", default=""), record)
            canonical = _canonical_id(record)
            state = _optional_text(record, "resolution_state", "state")
        else:
            record = raw_value if isinstance(raw_value, Mapping) else None
            key = key_from_raw(raw_key, record)
            canonical = (
                _canonical_id(raw_value)
                if isinstance(raw_value, Mapping)
                else str(raw_value).strip()
            )
            state = _optional_text(raw_value, "resolution_state", "state")
        if state and state.casefold() in {
            "abstain",
            "unresolved",
            "rejected",
            "invalid",
        }:
            raise GraphRewriteError(
                f"mention {key} has non-materializable resolution state {state!r}"
            )
        if not canonical:
            raise GraphRewriteError(f"mention {key} has an empty canonical id")
        if key not in mentions_by_key:
            raise GraphRewriteError(f"resolution references unknown mention {key}")
        previous = result.get(key)
        if previous is not None:
            raise GraphRewriteError(
                f"mention {key} has more than one resolution ({previous!r}, {canonical!r})"
            )
        result[key] = canonical

    missing = sorted(set(mentions_by_key) - set(result))
    if missing:
        raise GraphRewriteError(
            f"{len(missing)} mention(s) lack a canonical resolution; first={missing[0]}"
        )
    return result


def _resolve_endpoint(
    relation: Any,
    side: str,
    mentions_by_key: Mapping[MentionKey, Any],
    local_names: Mapping[tuple[str, str, str], list[MentionKey]],
    mention_ids: Mapping[str, list[MentionKey]],
) -> MentionKey:
    document_id = _required_text(relation, ("document_id",), "relation document_id")
    chunk_id = _required_text(relation, ("chunk_id",), "relation chunk_id")
    raw = _value(
        relation,
        f"{side}_mention_id",
        f"{side}_mention",
        f"{side}_entity_id",
        side,
    )
    if isinstance(raw, Mapping):
        if _value(raw, "document_id") is not None:
            return MentionKey.from_record(raw)
        raw = _value(raw, "mention_id", "entity_mention_id", "name", "entity_name")
    if isinstance(raw, tuple) and len(raw) == 3:
        key = MentionKey(*(str(item) for item in raw))
        if key not in mentions_by_key:
            raise GraphRewriteError(
                f"relation endpoint references unknown mention {key}"
            )
        return key
    reference = str(raw).strip() if raw is not None else ""
    if not reference:
        raise GraphRewriteError(f"relation has no resolvable {side} endpoint")

    direct = MentionKey(document_id, chunk_id, reference)
    if direct in mentions_by_key:
        return direct

    by_name = local_names.get((document_id, chunk_id, _surface_key(reference)), [])
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise GraphRewriteError(
            f"relation {side} endpoint {reference!r} is ambiguous in {document_id}/{chunk_id}"
        )

    by_id = mention_ids.get(reference, [])
    if len(by_id) == 1:
        return by_id[0]
    raise GraphRewriteError(
        f"relation {side} endpoint {reference!r} does not resolve to one mention"
    )


def _mention_provenance(key: MentionKey, mention: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        **key.to_dict(),
        "original_name": _mention_name(mention),
    }
    for name in (
        "extraction_call_id",
        "parse_status",
        "recovery_status",
        "entity_present",
        "description_present",
        "provenance",
    ):
        value = _value(mention, name)
        if value is not None:
            result[name] = _jsonable(value)
    return result


def rewrite_graph(
    mentions: Sequence[Any],
    relations: Sequence[Any],
    mention_to_canonical: Sequence[Any] | Mapping[Any, Any],
    canonical_entities: Sequence[Any],
    *,
    graph_field_separator: str = GRAPH_FIELD_SEP,
) -> RewrittenGraph:
    """Apply mention-level ER and create a deterministic LightRAG projection.

    Every mention must resolve exactly once.  Relation endpoint resolution is
    mention-local first; a global surface mapping is never constructed.  Edges
    are projected to LightRAG's undirected single-edge-per-pair model, summing
    weights and preserving all relation/provenance records.
    """

    if not graph_field_separator:
        raise GraphRewriteError("graph_field_separator must be non-empty")

    mentions_by_key: dict[MentionKey, Any] = {}
    local_names: dict[tuple[str, str, str], list[MentionKey]] = defaultdict(list)
    mention_ids: dict[str, list[MentionKey]] = defaultdict(list)
    for mention in mentions:
        key = MentionKey.from_record(mention)
        if key in mentions_by_key:
            raise GraphRewriteError(f"duplicate mention key {key}")
        mentions_by_key[key] = mention
        local_names[
            (key.document_id, key.chunk_id, _surface_key(_mention_name(mention)))
        ].append(key)
        mention_ids[key.mention_id].append(key)

    resolutions = _build_resolution_index(mention_to_canonical, mentions_by_key)

    canonical_by_id: dict[str, Any] = {}
    display_name_owner: dict[str, str] = {}
    for canonical in canonical_entities:
        canonical_id = _canonical_id(canonical)
        if canonical_id in canonical_by_id:
            raise GraphRewriteError(f"duplicate canonical entity id {canonical_id!r}")
        display_name = _canonical_name(canonical)
        name_key = _surface_key(display_name)
        previous = display_name_owner.get(name_key)
        if previous is not None:
            raise GraphRewriteError(
                "canonical display names are not uniquely disambiguated: "
                f"{display_name!r} belongs to both {previous!r} and {canonical_id!r}"
            )
        display_name_owner[name_key] = canonical_id
        canonical_by_id[canonical_id] = canonical

    used_canonical_ids = set(resolutions.values())
    missing_canonical = sorted(used_canonical_ids - set(canonical_by_id))
    if missing_canonical:
        raise GraphRewriteError(
            "resolution references missing canonical entity: " + missing_canonical[0]
        )

    members: dict[str, list[MentionKey]] = defaultdict(list)
    for key, canonical_id in resolutions.items():
        members[canonical_id].append(key)

    nodes: list[dict[str, Any]] = []
    node_name_by_id: dict[str, str] = {}
    normalized_mapping: list[dict[str, Any]] = []
    for canonical_id in sorted(used_canonical_ids):
        canonical = canonical_by_id[canonical_id]
        display_name = _canonical_name(canonical)
        node_name_by_id[canonical_id] = display_name
        member_keys = sorted(members[canonical_id])
        member_records = [mentions_by_key[key] for key in member_keys]
        aliases = _unique_strings(
            [
                *_string_list(_value(canonical, "aliases")),
                *(_mention_name(item) for item in member_records),
            ]
        )
        canonical_description = _optional_text(
            canonical, "description", "merged_description", "evidence"
        )
        descriptions = _unique_strings(
            _optional_text(item, "description", "entity_description")
            for item in member_records
        )
        description = canonical_description or graph_field_separator.join(descriptions)
        entity_type = _optional_text(
            canonical, "entity_type", "type"
        ) or _majority_text(
            _optional_text(item, "entity_type", "type") for item in member_records
        )
        source_chunks = sorted({key.chunk_id for key in member_keys})
        source_documents = sorted({key.document_id for key in member_keys})
        source_mentions = [
            _mention_provenance(key, mentions_by_key[key]) for key in member_keys
        ]
        nodes.append(
            {
                "canonical_entity_id": canonical_id,
                "entity_name": display_name,
                "entity_type": entity_type,
                "description": description,
                "description_present": bool(description.strip()),
                "aliases": aliases,
                "source_chunk_ids": source_chunks,
                "source_document_ids": source_documents,
                "source_mentions": source_mentions,
                "selection_rationale": _jsonable(
                    _value(canonical, "selection_rationale", "rationale")
                ),
            }
        )
        for key in member_keys:
            normalized_mapping.append(
                {
                    **key.to_dict(),
                    "canonical_entity_id": canonical_id,
                    "canonical_display_name": display_name,
                }
            )

    contributions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    induced_self_loops: list[dict[str, Any]] = []
    preexisting_self_loops: list[dict[str, Any]] = []
    for relation in relations:
        relation_id = _required_text(relation, ("relation_id", "id"), "relation id")
        source_key = _resolve_endpoint(
            relation, "source", mentions_by_key, local_names, mention_ids
        )
        target_key = _resolve_endpoint(
            relation, "target", mentions_by_key, local_names, mention_ids
        )
        source_canonical = resolutions[source_key]
        target_canonical = resolutions[target_key]
        source_surface = _mention_name(mentions_by_key[source_key])
        target_surface = _mention_name(mentions_by_key[target_key])
        loop_record = {
            "relation_id": relation_id,
            "source_mention": source_key.to_dict(),
            "target_mention": target_key.to_dict(),
            "source_original_name": source_surface,
            "target_original_name": target_surface,
            "canonical_entity_id": source_canonical,
            "canonical_display_name": node_name_by_id[source_canonical],
            "provenance": _jsonable(_value(relation, "provenance")),
        }
        # A loop is induced whenever two distinct mention identities collapse
        # into one canonical entity.  Surface equality is not identity: two
        # separate "Apple" records can still be merged by ER.
        if source_canonical == target_canonical and source_key != target_key:
            loop_record["reason"] = "induced_by_entity_resolution"
            induced_self_loops.append(loop_record)
            continue
        if source_canonical == target_canonical:
            loop_record["reason"] = "preexisting_same_mention_self_loop"
            preexisting_self_loops.append(loop_record)

        endpoint_ids = tuple(sorted((source_canonical, target_canonical)))
        contributions[endpoint_ids].append(
            {
                "relation_id": relation_id,
                "document_id": source_key.document_id,
                "chunk_id": source_key.chunk_id,
                "source_mention": source_key,
                "target_mention": target_key,
                "description": _optional_text(
                    relation, "description", "relation_description"
                ),
                "keywords": _string_list(
                    _value(relation, "keywords", "relation_keywords", "type")
                ),
                "relation_type": _optional_text(relation, "relation_type", "type"),
                "weight": float(_value(relation, "weight", default=1.0) or 1.0),
                "provenance": _jsonable(_value(relation, "provenance")),
            }
        )

    edges: list[dict[str, Any]] = []
    for (left_id, right_id), raw_items in sorted(contributions.items()):
        items = sorted(raw_items, key=lambda item: item["relation_id"])
        descriptions = _unique_strings(item["description"] for item in items)
        keyword_map: dict[str, str] = {}
        for item in items:
            for keyword in item["keywords"]:
                keyword_map.setdefault(keyword.casefold(), keyword)
        relation_types = _unique_strings(item["relation_type"] for item in items)
        source_chunk_ids = sorted({item["chunk_id"] for item in items})
        source_document_ids = sorted({item["document_id"] for item in items})
        provenance = [
            {
                "relation_id": item["relation_id"],
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "source_mention": item["source_mention"].to_dict(),
                "target_mention": item["target_mention"].to_dict(),
                "provenance": item["provenance"],
            }
            for item in items
        ]
        edges.append(
            {
                "edge_id": _stable_id("edge_", left_id, right_id),
                "source_canonical_entity_id": left_id,
                "target_canonical_entity_id": right_id,
                "source": node_name_by_id[left_id],
                "target": node_name_by_id[right_id],
                "description": graph_field_separator.join(descriptions),
                "description_present": bool(descriptions),
                "keywords": ",".join(keyword_map[key] for key in sorted(keyword_map)),
                "relation_types": relation_types,
                "weight": sum(item["weight"] for item in items),
                "source_relation_ids": [item["relation_id"] for item in items],
                "source_chunk_ids": source_chunk_ids,
                "source_document_ids": source_document_ids,
                "provenance": provenance,
                "deduplicated_relation_count": len(items),
            }
        )

    induced_self_loops.sort(key=lambda item: item["relation_id"])
    preexisting_self_loops.sort(key=lambda item: item["relation_id"])
    summary = {
        "mention_count": len(mentions_by_key),
        "canonical_entity_count": len(nodes),
        "input_relation_count": len(relations),
        "materialized_edge_count": len(edges),
        "deduplicated_relation_count": sum(
            max(0, edge["deduplicated_relation_count"] - 1) for edge in edges
        ),
        "induced_self_loops_removed": len(induced_self_loops),
        "preexisting_self_loops_retained": len(preexisting_self_loops),
    }
    return RewrittenGraph(
        nodes=nodes,
        edges=edges,
        mention_to_canonical=sorted(
            normalized_mapping,
            key=lambda item: (
                item["document_id"],
                item["chunk_id"],
                item["mention_id"],
            ),
        ),
        induced_self_loops=induced_self_loops,
        preexisting_self_loops=preexisting_self_loops,
        summary=summary,
    )


__all__ = [
    "GRAPH_FIELD_SEP",
    "GraphRewriteError",
    "MentionKey",
    "RewrittenGraph",
    "rewrite_graph",
]
