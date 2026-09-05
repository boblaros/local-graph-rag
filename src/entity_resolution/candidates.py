"""Deterministic, blocked candidate generation for mention-level ER."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from .embeddings import EmbeddingNeighborEvidence
from .models import CandidatePair, ERConfig, EntityProfile, stable_id
from .profiles import lexical_view


def name_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in value.split() if token)


def acronym(value: str) -> str | None:
    tokens = name_tokens(value)
    if len(tokens) < 2:
        return None
    result = "".join(token[0] for token in tokens if token)
    return result if len(result) >= 2 else None


def _pair_key(left: str, right: str) -> tuple[str, str]:
    first, second = sorted((left, right))
    if first == second:
        raise ValueError("candidate endpoints must differ")
    return first, second


def _bounded_pairs(
    ids: Iterable[str], max_block_size: int
) -> Iterable[tuple[str, str]]:
    members = sorted(set(ids))
    if len(members) <= max_block_size:
        yield from combinations(members, 2)
        return
    # Very common tokens (for example, "university") must not turn candidate
    # generation back into a corpus-wide quadratic scan.  A deterministic
    # neighbourhood still allows multiple routes in the union to recover a pair.
    radius = max(2, min(20, max_block_size // 10))
    for index, left in enumerate(members):
        for right in members[index + 1 : index + 1 + radius]:
            yield left, right


def generate_candidate_pairs(
    profiles: Sequence[EntityProfile],
    config: ERConfig | None = None,
    *,
    embedding_neighbors: Mapping[
        str, Iterable[str | EmbeddingNeighborEvidence]
    ]
    | None = None,
) -> list[CandidatePair]:
    """Generate a sparse union of identity-bearing candidate routes.

    Embedding candidates are accepted as a precomputed nearest-neighbour map;
    this module deliberately does not compute every embedding pair. Type,
    provenance, neighbourhood and relation context remain scoring evidence
    only: they are too weak to create a pair by themselves. All comparisons are
    made inside bounded indexes, not over the corpus Cartesian product.
    """

    config = config or ERConfig()
    by_id = {profile.mention_id: profile for profile in profiles}
    if len(by_id) != len(profiles):
        raise ValueError("duplicate mention_id in profiles")
    methods: dict[tuple[str, str], set[str]] = defaultdict(set)
    route_evidence: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    precomputed_signals: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)

    def add(
        left: str,
        right: str,
        method: str,
        *,
        evidence: Mapping[str, Any] | None = None,
        signals: Mapping[str, float] | None = None,
    ) -> None:
        if left == right:
            return
        if left not in by_id or right not in by_id:
            raise ValueError(
                f"candidate neighbour references unknown mention: {left}, {right}"
            )
        key = _pair_key(left, right)
        methods[key].add(method)
        payload = dict(evidence or {})
        previous_evidence = route_evidence[key].get(method)
        if previous_evidence is not None and previous_evidence != payload:
            raise ValueError(f"conflicting evidence for candidate route {method}: {key}")
        route_evidence[key][method] = payload
        for signal, raw_value in (signals or {}).items():
            value = float(raw_value)
            previous_signal = precomputed_signals[key].get(signal)
            if previous_signal is not None and previous_signal != value:
                raise ValueError(
                    f"conflicting precomputed candidate signal {signal}: {key}"
                )
            precomputed_signals[key][signal] = value

    exact_index: dict[str, list[str]] = defaultdict(list)
    token_index: dict[str, list[str]] = defaultdict(list)
    fuzzy_blocks: dict[tuple[str, int], list[str]] = defaultdict(list)
    acronym_index: dict[str, list[str]] = defaultdict(list)
    lexical_names = {
        profile.mention_id: lexical_view(profile.normalized_name) for profile in profiles
    }
    for profile in profiles:
        exact_index[profile.normalized_name].append(profile.mention_id)
        lexical_name = lexical_names[profile.mention_id]
        tokens = name_tokens(lexical_name)
        for token in set(tokens):
            token_index[token].append(profile.mention_id)
        first = lexical_name[:1]
        length_bucket = len(lexical_name) // 4
        for bucket in {length_bucket - 1, length_bucket, length_bucket + 1}:
            if bucket >= 0:
                fuzzy_blocks[(first, bucket)].append(profile.mention_id)
        generated = acronym(lexical_name)
        if generated:
            acronym_index[generated].append(profile.mention_id)
        compact = lexical_name.replace(" ", "")
        if 2 <= len(compact) <= 10:
            acronym_index[compact].append(profile.mention_id)

    for ids in exact_index.values():
        for left, right in _bounded_pairs(ids, config.max_block_size):
            add(
                left,
                right,
                "exact_name",
                evidence={"canonical_name": by_id[left].normalized_name},
            )

    # Token overlap supplies containment candidates without checking all names.
    for ids in token_index.values():
        for left, right in _bounded_pairs(ids, config.max_block_size):
            left_name = lexical_names[left]
            right_name = lexical_names[right]
            if min(len(left_name), len(right_name)) >= config.containment_min_chars:
                if left_name in right_name or right_name in left_name:
                    contained, container = sorted(
                        (left_name, right_name), key=lambda value: (len(value), value)
                    )
                    add(
                        left,
                        right,
                        "containment",
                        evidence={
                            "contained_name": contained,
                            "container_name": container,
                            "minimum_chars": config.containment_min_chars,
                        },
                    )

    seen_fuzzy: set[tuple[str, str]] = set()
    for ids in fuzzy_blocks.values():
        for left, right in _bounded_pairs(ids, config.max_block_size):
            key = _pair_key(left, right)
            if key in seen_fuzzy:
                continue
            seen_fuzzy.add(key)
            ratio = SequenceMatcher(
                None, lexical_names[left], lexical_names[right]
            ).ratio()
            if ratio >= config.fuzzy_candidate_threshold:
                add(
                    left,
                    right,
                    "fuzzy_name",
                    evidence={
                        "lexical_similarity": ratio,
                        "threshold": config.fuzzy_candidate_threshold,
                    },
                    signals={"lexical": ratio},
                )

    for ids in acronym_index.values():
        for left, right in _bounded_pairs(ids, config.max_block_size):
            add(left, right, "acronym", evidence={"index_match": True})

    if embedding_neighbors:
        normalized_neighbours: dict[str, dict[str, dict[str, Any]]] = {}
        for mention_id, neighbours in embedding_neighbors.items():
            records: dict[str, dict[str, Any]] = {}
            for fallback_rank, neighbour in enumerate(neighbours, start=1):
                if isinstance(neighbour, EmbeddingNeighborEvidence):
                    neighbour_id = neighbour.mention_id
                    record = neighbour.to_dict()
                else:
                    neighbour_id = str(neighbour)
                    record = {
                        "mention_id": neighbour_id,
                        "raw_cosine": None,
                        "rank": fallback_rank,
                        "scoring_signal": None,
                    }
                previous = records.get(neighbour_id)
                if previous is not None and previous != record:
                    raise ValueError(
                        "duplicate embedding neighbour has conflicting evidence: "
                        f"{mention_id}, {neighbour_id}"
                    )
                records[neighbour_id] = record
            normalized_neighbours[str(mention_id)] = records

        embedding_pairs = {
            _pair_key(left, right)
            for left, neighbours in normalized_neighbours.items()
            for right in neighbours
            if left != right
        }
        for left, right in sorted(embedding_pairs):
            left_record = normalized_neighbours.get(left, {}).get(right)
            right_record = normalized_neighbours.get(right, {}).get(left)
            cosine_values = {
                float(record["raw_cosine"])
                for record in (left_record, right_record)
                if record is not None and record["raw_cosine"] is not None
            }
            if len(cosine_values) > 1:
                raise ValueError(
                    f"embedding neighbour cosine is not symmetric: {(left, right)}"
                )
            raw_cosine = next(iter(cosine_values), None)
            scoring_signal = (
                max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0))
                if raw_cosine is not None
                else None
            )
            embedding_evidence = {
                "left_selected_right": left_record is not None,
                "left_rank": left_record["rank"] if left_record else None,
                "right_selected_left": right_record is not None,
                "right_rank": right_record["rank"] if right_record else None,
                "raw_cosine": raw_cosine,
                "scoring_signal": scoring_signal,
            }
            add(
                left,
                right,
                "embedding_neighbour",
                evidence=embedding_evidence,
                signals=(
                    {"embedding": scoring_signal}
                    if scoring_signal is not None
                    else None
                ),
            )
            if left_record is not None and right_record is not None:
                add(
                    left,
                    right,
                    "reciprocal_embedding_neighbour",
                    evidence={
                        "left_rank": left_record["rank"],
                        "right_rank": right_record["rank"],
                        "raw_cosine": raw_cosine,
                    },
                )

    result: list[CandidatePair] = []
    for (left, right), pair_methods in sorted(methods.items()):
        result.append(
            CandidatePair(
                pair_id=stable_id("pair_", left, right),
                left_mention_id=left,
                right_mention_id=right,
                methods=tuple(pair_methods),
                route_evidence=route_evidence[(left, right)],
                precomputed_signals=precomputed_signals[(left, right)],
            )
        )
    return result
