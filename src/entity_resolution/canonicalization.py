"""Deterministic canonical entities and mention-level resolution mapping."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .models import (
    CanonicalEntity,
    Cluster,
    EntityProfile,
    MentionResolution,
    stable_id,
)


@dataclass(frozen=True)
class CanonicalizationResult:
    canonical_entities: tuple[CanonicalEntity, ...]
    mention_to_canonical: tuple[MentionResolution, ...]


def _choose_display_name(profiles: Sequence[EntityProfile]) -> tuple[str, str]:
    counts: Counter[str] = Counter()
    for profile in profiles:
        counts[profile.original_name] += profile.mention_frequency
    # Prefer a frequent extracted surface; ties choose the more informative
    # extracted form (e.g. "Apple Inc." over "Apple") deterministically.
    candidates = sorted(
        counts,
        key=lambda name: (
            -counts[name],
            -len(name.split()),
            -len(name),
            name.casefold(),
            name,
        ),
    )
    selected = candidates[0]
    rationale = (
        "selected from extracted names by frequency, then specificity and "
        "deterministic lexical order"
    )
    return selected, rationale


def _choose_type(profiles: Sequence[EntityProfile]) -> str | None:
    available = [profile.entity_type or profile.type_family for profile in profiles]
    values = [value for value in available if value]
    if not values:
        return None
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _merged_description(profiles: Sequence[EntityProfile]) -> str | None:
    descriptions = sorted(
        {profile.description for profile in profiles if profile.description},
        key=lambda value: (value.casefold(), value),
    )
    return " | ".join(descriptions) if descriptions else None


def _unique_display_names(
    entities: Sequence[CanonicalEntity],
) -> tuple[CanonicalEntity, ...]:
    by_name: dict[str, list[CanonicalEntity]] = defaultdict(list)
    for entity in entities:
        by_name[entity.display_name.casefold()].append(entity)
    used = {entity.display_name.casefold() for entity in entities}
    output: list[CanonicalEntity] = []
    for entity in sorted(entities, key=lambda item: item.canonical_entity_id):
        group = by_name[entity.display_name.casefold()]
        if len(group) == 1:
            output.append(entity)
            continue

        # An extracted, more specific alias is preferable to an invented label.
        alternatives = sorted(
            (
                alias
                for alias in entity.aliases
                if alias.casefold() not in used
                and alias.casefold() != entity.display_name.casefold()
            ),
            key=lambda name: (-len(name.split()), -len(name), name.casefold(), name),
        )
        if alternatives:
            display = alternatives[0]
            aliases = tuple(
                sorted(
                    {
                        *entity.aliases,
                        entity.display_name,
                    }
                    - {display},
                    key=str.casefold,
                )
            )
            rationale = (
                entity.selection_rationale + "; disambiguated with extracted alias"
            )
        else:
            qualifier = entity.entity_type or "entity"
            display = f"{entity.display_name} ({qualifier})"
            if display.casefold() in used:
                display = f"{display} [{entity.canonical_entity_id[-8:]}]"
            aliases = tuple(
                sorted({*entity.aliases, entity.display_name}, key=str.casefold)
            )
            rationale = (
                entity.selection_rationale + "; added type qualifier for uniqueness"
            )
        used.add(display.casefold())
        output.append(
            replace(
                entity,
                display_name=display,
                aliases=aliases,
                selection_rationale=rationale,
            )
        )
    names = [entity.display_name.casefold() for entity in output]
    if len(names) != len(set(names)):
        raise ValueError("canonical display names are not unique after disambiguation")
    return tuple(sorted(output, key=lambda item: item.canonical_entity_id))


def canonicalize_clusters(
    clusters: Sequence[Cluster],
    profiles: Sequence[EntityProfile],
    *,
    namespace: str,
) -> CanonicalizationResult:
    """Canonicalize clusters while retaining every source mention and document."""

    if not namespace.strip():
        raise ValueError("canonicalization namespace must be non-empty")
    by_id = {profile.mention_id: profile for profile in profiles}
    seen_profiles: set[str] = set()
    entities: list[CanonicalEntity] = []
    mapping_seed: list[tuple[Mapping[str, str | None], str]] = []
    for cluster in sorted(clusters, key=lambda item: item.mention_ids):
        unknown = sorted(set(cluster.mention_ids) - set(by_id))
        if unknown:
            raise ValueError(f"cluster contains unknown mentions: {unknown}")
        overlap = seen_profiles & set(cluster.mention_ids)
        if overlap:
            raise ValueError(f"clusters overlap at profiles: {sorted(overlap)}")
        seen_profiles.update(cluster.mention_ids)
        members = [by_id[mention_id] for mention_id in cluster.mention_ids]
        source_mentions = tuple(
            source for profile in members for source in profile.source_mentions
        )
        source_mention_ids = tuple(
            sorted(str(source["mention_id"]) for source in source_mentions)
        )
        display_name, rationale = _choose_display_name(members)
        canonical_id = stable_id("entity_", namespace, source_mention_ids)
        aliases = tuple(
            sorted(
                {profile.original_name for profile in members} - {display_name},
                key=str.casefold,
            )
        )
        evidence = tuple(
            {
                "native_entity_id": profile.mention_id,
                "original_name": profile.original_name,
                "entity_type": profile.entity_type,
                "description": profile.description,
                "document_id": profile.document_id,
                "chunk_id": profile.chunk_id,
                "extraction_call_id": profile.extraction_call_id,
                "source_mention_ids": profile.source_mention_ids,
                "provenance": dict(profile.provenance),
            }
            for profile in sorted(members, key=lambda item: item.mention_id)
        )
        entities.append(
            CanonicalEntity(
                canonical_entity_id=canonical_id,
                display_name=display_name,
                aliases=aliases,
                entity_type=_choose_type(members),
                merged_description=_merged_description(members),
                evidence=evidence,
                source_mention_ids=source_mention_ids,
                source_document_ids=tuple(
                    str(source["document_id"]) for source in source_mentions
                ),
                selection_rationale=rationale,
            )
        )
        mapping_seed.extend((source, canonical_id) for source in source_mentions)

    expected_mentions = set(by_id)
    if seen_profiles != expected_mentions:
        missing = sorted(expected_mentions - seen_profiles)
        raise ValueError(f"profiles missing from clusters: {missing}")
    canonical_entities = _unique_display_names(entities)
    mapping = tuple(
        sorted(
            (
                MentionResolution(
                    document_id=str(source["document_id"]),
                    chunk_id=str(source["chunk_id"]),
                    mention_id=str(source["mention_id"]),
                    canonical_entity_id=canonical_id,
                )
                for source, canonical_id in mapping_seed
            ),
            key=lambda item: (item.document_id, item.chunk_id, item.mention_id),
        )
    )
    return CanonicalizationResult(
        canonical_entities=canonical_entities,
        mention_to_canonical=mapping,
    )
