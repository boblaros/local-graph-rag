"""Multi-signal pair scoring with explicit unavailable evidence."""

from __future__ import annotations

from difflib import SequenceMatcher
from math import sqrt
from typing import Iterable, Mapping, Sequence

from .candidates import acronym, name_tokens
from .models import CandidatePair, ERConfig, EntityProfile, PairScore
from .profiles import lexical_view


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _context_tokens(value: tuple[str, ...]) -> set[str]:
    return {token for item in value for token in name_tokens(item)}


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def pair_signals(
    left: EntityProfile,
    right: EntityProfile,
    *,
    precomputed: Mapping[str, float] | None = None,
) -> dict[str, float | None]:
    """Compute signals; ``None`` means unavailable, never implicit mismatch."""

    reused = dict(precomputed or {})
    unexpected = sorted(set(reused) - {"lexical", "embedding"})
    if unexpected:
        raise ValueError(f"unsupported precomputed pair signals: {unexpected}")
    left_canonical, right_canonical = left.normalized_name, right.normalized_name
    exact = left_canonical == right_canonical
    left_name, right_name = lexical_view(left_canonical), lexical_view(right_canonical)

    left_compact = left_name.replace(" ", "")
    right_compact = right_name.replace(" ", "")
    left_acronym, right_acronym = acronym(left_name), acronym(right_name)
    acronym_match = (
        (left_acronym is not None and left_acronym == right_compact)
        or (right_acronym is not None and right_acronym == left_compact)
        or (
            left_acronym is not None
            and right_acronym is not None
            and left_acronym == right_acronym
        )
    )
    containment = left_name in right_name or right_name in left_name

    if left.description is None or right.description is None:
        description_score: float | None = None
    else:
        description_score = _jaccard(
            name_tokens(lexical_view(left.description)),
            name_tokens(lexical_view(right.description)),
        )

    if "embedding" in reused:
        embedding_score = float(reused["embedding"])
    elif left.embedding is None or right.embedding is None:
        embedding_score: float | None = None
    else:
        embedding_score = _cosine(left.embedding, right.embedding)

    if left.type_family is None or right.type_family is None:
        type_score: float | None = None
    elif left.type_family == right.type_family:
        type_score = 1.0
    elif left.entity_type is not None and left.entity_type == right.entity_type:
        type_score = 1.0
    else:
        type_score = 0.0

    if left.neighbours is None or right.neighbours is None:
        neighbourhood_score: float | None = None
    else:
        neighbourhood_score = _jaccard(
            _context_tokens(left.neighbours), _context_tokens(right.neighbours)
        )

    if left.relation_context is None or right.relation_context is None:
        relation_score: float | None = None
    else:
        relation_score = _jaccard(
            _context_tokens(left.relation_context),
            _context_tokens(right.relation_context),
        )

    if left.chunk_id == right.chunk_id and left.document_id == right.document_id:
        provenance_score = 1.0
    elif left.document_id == right.document_id:
        provenance_score = 0.5
    else:
        provenance_score = 0.0

    return {
        "name_exact": 1.0 if exact else 0.0,
        "lexical": (
            float(reused["lexical"])
            if "lexical" in reused
            else SequenceMatcher(None, left_name, right_name).ratio()
        ),
        "acronym": 1.0 if acronym_match else 0.0,
        "containment": 1.0 if containment else 0.0,
        "description": description_score,
        "embedding": embedding_score,
        "type": type_score,
        "neighbourhood": neighbourhood_score,
        "relation_context": relation_score,
        "provenance": provenance_score,
        "frequency": min(left.mention_frequency, right.mention_frequency)
        / max(left.mention_frequency, right.mention_frequency),
        "source_diversity": min(left.source_diversity, right.source_diversity)
        / max(left.source_diversity, right.source_diversity),
    }


def score_pair(
    candidate: CandidatePair,
    left: EntityProfile,
    right: EntityProfile,
    config: ERConfig | None = None,
) -> PairScore:
    config = config or ERConfig()
    if {candidate.left_mention_id, candidate.right_mention_id} != {
        left.mention_id,
        right.mention_id,
    }:
        raise ValueError("candidate endpoints do not match profiles")
    signals = pair_signals(
        left,
        right,
        precomputed=candidate.precomputed_signals,
    )
    available_weights = {
        signal: weight
        for signal, weight in config.scoring_weights.items()
        if weight > 0 and signals.get(signal) is not None
    }
    weight_sum = sum(available_weights.values())
    if weight_sum:
        effective = {
            signal: weight / weight_sum
            for signal, weight in sorted(available_weights.items())
        }
        aggregate: float | None = sum(
            effective[signal] * float(signals[signal]) for signal in effective
        )
    else:
        effective = {}
        aggregate = None
    unavailable = tuple(
        sorted(
            signal for signal in config.scoring_weights if signals.get(signal) is None
        )
    )
    return PairScore(
        pair_id=candidate.pair_id,
        left_mention_id=candidate.left_mention_id,
        right_mention_id=candidate.right_mention_id,
        aggregate_score=aggregate,
        signals=signals,
        effective_weights=effective,
        unavailable_signals=unavailable,
    )


def score_candidate_pairs(
    candidates: Sequence[CandidatePair],
    profiles: Sequence[EntityProfile],
    config: ERConfig | None = None,
) -> list[PairScore]:
    config = config or ERConfig()
    by_id = {profile.mention_id: profile for profile in profiles}
    return [
        score_pair(
            candidate,
            by_id[candidate.left_mention_id],
            by_id[candidate.right_mention_id],
            config,
        )
        for candidate in sorted(candidates, key=lambda item: item.pair_id)
    ]
