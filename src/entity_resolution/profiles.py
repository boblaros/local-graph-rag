"""Build mention-scoped entity profiles from a complete staged extraction."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

from .models import (
    EntityMention,
    EntityProfile,
    RelationMention,
    coerce_mentions,
    coerce_relations,
    stable_id,
)


_TYPE_ALIASES: Mapping[str, str] = {
    "person": "person",
    "people": "person",
    "human": "person",
    "individual": "person",
    "organization": "organization",
    "organisation": "organization",
    "org": "organization",
    "company": "organization",
    "corporation": "organization",
    "business": "organization",
    "institution": "organization",
    "agency": "organization",
    "university": "organization",
    "location": "location",
    "place": "location",
    "city": "location",
    "country": "location",
    "region": "location",
    "facility": "location",
    "event": "event",
    "product": "product",
    "work": "product",
    "technology": "product",
    "natural": "natural",
    "fruit": "natural",
    "animal": "natural",
    "plant": "natural",
    "species": "natural",
}


def normalize_text(value: str) -> str:
    """Return one lossless canonical form for identity-bearing text.

    Unicode compatibility, case and whitespace differences are normalized, but
    punctuation and other semantic symbols are preserved.  Matchers that need
    punctuation-insensitive comparison must explicitly request ``lexical_view``.
    """

    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(text.split())


def lexical_view(value: str) -> str:
    """Derive a punctuation-insensitive view from canonical text for fuzzy use."""

    text = re.sub(r"[^\w]+", " ", normalize_text(value), flags=re.UNICODE)
    return " ".join(text.split())


def normalize_entity_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = lexical_view(value)
    return normalized or None


def entity_type_family(value: str | None) -> str | None:
    normalized = normalize_entity_type(value)
    if normalized is None:
        return None
    if normalized in _TYPE_ALIASES:
        return _TYPE_ALIASES[normalized]
    for token in normalized.split():
        if token in _TYPE_ALIASES:
            return _TYPE_ALIASES[token]
    # An unknown extracted type remains useful for exact-type agreement without
    # inventing a broader family.
    return normalized


def _normalized_context(values: Iterable[str]) -> tuple[str, ...] | None:
    result = tuple(
        sorted({item for value in values if (item := lexical_view(value))})
    )
    return result or None


def native_entity_id(original_name: str) -> str:
    """Return the stable ER-side identity of one exact Native graph node."""

    if not original_name.strip():
        raise ValueError("Native entity name must be non-empty")
    return stable_id("native_entity_", original_name)


def _majority(values: Iterable[str | None]) -> str | None:
    available = [value for value in values if value]
    if not available:
        return None
    counts = Counter(available)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _representative_description(mentions: Sequence[EntityMention]) -> str | None:
    descriptions = {mention.description for mention in mentions if mention.description}
    if not descriptions:
        return None
    return sorted(
        descriptions,
        key=lambda value: (-len(value), value.casefold(), value),
    )[0]


def build_native_entity_embedding_records(
    mentions: Sequence[EntityMention | Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build one deterministic embedding input per exact Native entity name."""

    mention_records = coerce_mentions(mentions)
    groups: dict[str, list[EntityMention]] = defaultdict(list)
    seen_ids: set[str] = set()
    for mention in mention_records:
        if mention.mention_id in seen_ids:
            raise ValueError(f"duplicate mention_id: {mention.mention_id}")
        seen_ids.add(mention.mention_id)
        groups[mention.original_name].append(mention)

    records: list[dict[str, object]] = []
    for original_name, members in sorted(groups.items()):
        members = sorted(members, key=lambda item: item.mention_id)
        records.append(
            {
                "mention_id": native_entity_id(original_name),
                "original_name": original_name,
                "entity_type": _majority(
                    normalize_entity_type(item.entity_type) for item in members
                ),
                "description": _representative_description(members),
                "source_mention_ids": [item.mention_id for item in members],
            }
        )
    return records


def build_native_entity_profiles(
    mentions: Sequence[EntityMention | Mapping[str, object]],
    relations: Sequence[RelationMention | Mapping[str, object]],
    *,
    embedded_native_entities: Sequence[Mapping[str, object]] | None = None,
) -> list[EntityProfile]:
    """Aggregate exact Native graph nodes into indivisible ER profiles.

    ``original_name`` is the actual Native LightRAG node key preserved by the
    immutable staging normalizer.  Different exact names remain separate even
    when their normalized forms are equal, so ER may still merge spelling or
    capitalization variants.
    """

    mention_records = coerce_mentions(mentions)
    relation_records = coerce_relations(relations)
    by_mention: dict[str, EntityMention] = {}
    groups: dict[str, list[EntityMention]] = defaultdict(list)
    for mention in mention_records:
        if mention.mention_id in by_mention:
            raise ValueError(f"duplicate mention_id: {mention.mention_id}")
        by_mention[mention.mention_id] = mention
        groups[mention.original_name].append(mention)

    group_id_by_mention = {
        mention.mention_id: native_entity_id(mention.original_name)
        for mention in mention_records
    }
    neighbours: dict[str, list[str]] = defaultdict(list)
    contexts: dict[str, list[str]] = defaultdict(list)
    relation_ids: set[str] = set()
    for relation in relation_records:
        if relation.relation_id in relation_ids:
            raise ValueError(f"duplicate relation_id: {relation.relation_id}")
        relation_ids.add(relation.relation_id)
        missing = [
            endpoint
            for endpoint in (relation.source_mention_id, relation.target_mention_id)
            if endpoint not in by_mention
        ]
        if missing:
            raise ValueError(
                f"relation {relation.relation_id} has unresolved mention endpoints: {missing}"
            )
        source = by_mention[relation.source_mention_id]
        target = by_mention[relation.target_mention_id]
        source_group = group_id_by_mention[source.mention_id]
        target_group = group_id_by_mention[target.mention_id]
        if source_group != target_group:
            neighbours[source_group].append(target.original_name)
            neighbours[target_group].append(source.original_name)
        relation_evidence = [
            *(relation.keywords or ()),
            *([relation.relation_type] if relation.relation_type else []),
            *([relation.description] if relation.description else []),
        ]
        if relation_evidence:
            contexts[source_group].extend(relation_evidence)
            contexts[target_group].extend(relation_evidence)

    embedded_by_id: dict[str, Mapping[str, object]] = {}
    for record in embedded_native_entities or ():
        group_id = str(record.get("mention_id") or "").strip()
        if not group_id or group_id in embedded_by_id:
            raise ValueError("embedded Native profiles require unique non-empty IDs")
        embedded_by_id[group_id] = record
    expected_group_ids = {native_entity_id(name) for name in groups}
    if embedded_by_id and set(embedded_by_id) != expected_group_ids:
        missing = sorted(expected_group_ids - set(embedded_by_id))
        extra = sorted(set(embedded_by_id) - expected_group_ids)
        raise ValueError(
            f"embedded Native profile coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )

    profiles: list[EntityProfile] = []
    for original_name, members in sorted(groups.items()):
        members = sorted(members, key=lambda item: item.mention_id)
        group_id = native_entity_id(original_name)
        embedded = embedded_by_id.get(group_id, {})
        raw_embedding = embedded.get("embedding")
        provenance = {
            "native_entity_group": {
                "native_node_id": original_name,
                "source_mention_count": len(members),
            }
        }
        embedding_lineage = {
            key: embedded[key]
            for key in (
                "embedding_cache_key",
                "embedding_dimension",
                "embedding_model_digest",
                "embedding_model_tag",
                "embedding_profile_text_sha256",
                "embedding_profile_text_version",
            )
            if key in embedded
        }
        if embedding_lineage:
            provenance["er_embedding"] = embedding_lineage
        normalized_types = [normalize_entity_type(item.entity_type) for item in members]
        profiles.append(
            EntityProfile(
                mention_id=group_id,
                document_id=min(item.document_id for item in members),
                chunk_id=min(item.chunk_id for item in members),
                original_name=original_name,
                normalized_name=normalize_text(original_name),
                entity_type=_majority(normalized_types),
                type_family=_majority(
                    entity_type_family(item) for item in normalized_types
                ),
                description=_representative_description(members),
                neighbours=_normalized_context(neighbours.get(group_id, ())),
                relation_context=_normalized_context(contexts.get(group_id, ())),
                mention_frequency=len(members),
                source_diversity=len({item.document_id for item in members}),
                provenance=provenance,
                extraction_call_id=next(
                    (
                        item.extraction_call_id
                        for item in members
                        if item.extraction_call_id
                    ),
                    None,
                ),
                embedding=(
                    tuple(float(value) for value in raw_embedding)
                    if isinstance(raw_embedding, Sequence)
                    and not isinstance(raw_embedding, (str, bytes, bytearray))
                    else None
                ),
                source_mentions=tuple(
                    {
                        "mention_id": item.mention_id,
                        "document_id": item.document_id,
                        "chunk_id": item.chunk_id,
                        "original_name": item.original_name,
                        "extraction_call_id": item.extraction_call_id,
                    }
                    for item in members
                ),
            )
        )
    return profiles


def build_entity_profiles(
    mentions: Sequence[EntityMention | Mapping[str, object]],
    relations: Sequence[RelationMention | Mapping[str, object]],
) -> list[EntityProfile]:
    """Build one immutable profile per mention.

    Relations are resolved through mention IDs, never through surface names.
    Missing relation/description/embedding evidence stays ``None`` so scoring
    can mark it unavailable instead of treating it as negative evidence.
    """

    mention_records = coerce_mentions(mentions)
    relation_records = coerce_relations(relations)
    by_id: dict[str, EntityMention] = {}
    for mention in mention_records:
        if mention.mention_id in by_id:
            raise ValueError(f"duplicate mention_id: {mention.mention_id}")
        by_id[mention.mention_id] = mention

    relation_ids: set[str] = set()
    neighbours: dict[str, list[str]] = defaultdict(list)
    contexts: dict[str, list[str]] = defaultdict(list)
    for relation in relation_records:
        if relation.relation_id in relation_ids:
            raise ValueError(f"duplicate relation_id: {relation.relation_id}")
        relation_ids.add(relation.relation_id)
        missing = [
            endpoint
            for endpoint in (relation.source_mention_id, relation.target_mention_id)
            if endpoint not in by_id
        ]
        if missing:
            raise ValueError(
                f"relation {relation.relation_id} has unresolved mention endpoints: {missing}"
            )
        source = by_id[relation.source_mention_id]
        target = by_id[relation.target_mention_id]
        neighbours[source.mention_id].append(target.original_name)
        neighbours[target.mention_id].append(source.original_name)
        relation_evidence = [
            *(relation.keywords or ()),
            *([relation.relation_type] if relation.relation_type else []),
            *([relation.description] if relation.description else []),
        ]
        if relation_evidence:
            contexts[source.mention_id].extend(relation_evidence)
            contexts[target.mention_id].extend(relation_evidence)

    normalized_names = {
        item.mention_id: normalize_text(item.original_name) for item in mention_records
    }
    name_frequency = Counter(normalized_names.values())
    name_documents: dict[str, set[str]] = defaultdict(set)
    for mention in mention_records:
        name_documents[normalized_names[mention.mention_id]].add(mention.document_id)

    profiles = [
        EntityProfile(
            mention_id=mention.mention_id,
            document_id=mention.document_id,
            chunk_id=mention.chunk_id,
            original_name=mention.original_name,
            normalized_name=normalized_names[mention.mention_id],
            entity_type=normalize_entity_type(mention.entity_type),
            type_family=entity_type_family(mention.entity_type),
            description=mention.description,
            neighbours=_normalized_context(neighbours.get(mention.mention_id, ())),
            relation_context=_normalized_context(contexts.get(mention.mention_id, ())),
            mention_frequency=name_frequency[normalized_names[mention.mention_id]],
            source_diversity=len(name_documents[normalized_names[mention.mention_id]]),
            provenance=mention.provenance,
            extraction_call_id=mention.extraction_call_id,
            embedding=mention.embedding,
        )
        for mention in mention_records
    ]
    return sorted(profiles, key=lambda item: item.mention_id)
