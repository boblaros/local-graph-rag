from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entity_resolution import (  # noqa: E402
    CannotLink,
    EntityMention,
    PairDecision,
    build_entity_profiles,
    canonicalize_clusters,
    constrained_cluster,
)


def _decision(left: str, right: str, action: str, score: float) -> PairDecision:
    left, right = sorted((left, right))
    return PairDecision(
        pair_id=f"pair_{left}_{right}",
        left_mention_id=left,
        right_mention_id=right,
        action=action,  # type: ignore[arg-type]
        source=(
            "score_policy_merge"
            if action == "merge"
            else "score_policy_reject"
            if action == "reject"
            else "no_judge"
        ),
        score=score,
        rationale="fixture",
    )


def _chain_profiles():
    mentions = [
        EntityMention("a", "d1", "c1", "Alpha", "organization"),
        EntityMention("b", "d2", "c2", "Alpha Labs", "organization"),
        EntityMention("c", "d3", "c3", "Alpha Fruit", "organization"),
    ]
    return build_entity_profiles(mentions, [])


def test_explicit_cannot_link_prevents_transitive_false_merge() -> None:
    profiles = _chain_profiles()
    decisions = [
        _decision("a", "b", "merge", 0.95),
        _decision("b", "c", "merge", 0.90),
        _decision("a", "c", "reject", 0.10),
    ]
    cannot = [CannotLink("a", "c", "contradictory profiles")]

    result = constrained_cluster(profiles, decisions, cannot_links=cannot)

    assert {cluster.mention_ids for cluster in result.clusters} == {("a", "b"), ("c",)}
    assert len(result.blocked_merges) == 1
    assert "cannot_link" in result.blocked_merges[0].reason


def test_missing_and_abstained_cross_pairs_are_neutral() -> None:
    profiles = _chain_profiles()
    chain_only = [
        _decision("a", "b", "merge", 0.95),
        _decision("b", "c", "merge", 0.90),
    ]
    result = constrained_cluster(
        profiles,
        [*chain_only, _decision("a", "c", "abstain", 0.60)],
    )

    assert [cluster.mention_ids for cluster in result.clusters] == [("a", "b", "c")]
    assert result.blocked_merges == ()


def test_cluster_size_has_no_hard_limit() -> None:
    mentions = [
        EntityMention(f"m{index:02d}", "d1", "c1", f"Alias {index}", "person")
        for index in range(30)
    ]
    profiles = build_entity_profiles(mentions, [])
    decisions = [
        _decision("m00", f"m{index:02d}", "merge", 0.99 - index / 1000)
        for index in range(1, 30)
    ]

    result = constrained_cluster(profiles, decisions)

    assert len(result.clusters) == 1
    assert len(result.clusters[0].mention_ids) == 30
    assert result.blocked_merges == ()


def test_type_mismatch_does_not_block_an_explicit_merge() -> None:
    profiles = build_entity_profiles(
        [
            EntityMention("person", "d1", "c1", "Jordan", "person"),
            EntityMention("place", "d2", "c2", "Jordan", "location"),
        ],
        [],
    )

    result = constrained_cluster(
        profiles,
        [_decision("person", "place", "merge", 0.90)],
    )

    assert [cluster.mention_ids for cluster in result.clusters] == [("person", "place")]
    assert result.blocked_merges == ()


def test_canonicalization_is_deterministic_and_mention_level() -> None:
    mentions = [
        EntityMention("company_long", "d1", "c1", "Apple Inc.", "organization"),
        EntityMention("company_short", "d2", "c2", "Apple", "company"),
        EntityMention("fruit", "d3", "c3", "Apple", "fruit"),
    ]
    profiles = build_entity_profiles(mentions, [])
    clustering = constrained_cluster(
        profiles,
        [
            _decision("company_long", "company_short", "merge", 0.95),
            _decision("company_short", "fruit", "reject", 0.05),
            _decision("company_long", "fruit", "reject", 0.05),
        ],
        cannot_links=[CannotLink("company_short", "fruit", "different senses")],
    )

    first = canonicalize_clusters(clustering.clusters, profiles, namespace="base_hash")
    second = canonicalize_clusters(
        list(reversed(clustering.clusters)),
        list(reversed(profiles)),
        namespace="base_hash",
    )

    assert first == second
    assert {entity.display_name for entity in first.canonical_entities} == {
        "Apple Inc.",
        "Apple (fruit)",
    }
    company = next(
        entity
        for entity in first.canonical_entities
        if entity.display_name == "Apple Inc."
    )
    assert company.aliases == ("Apple",)
    assert len(first.mention_to_canonical) == 3
    assert len({item.mention_id for item in first.mention_to_canonical}) == 3
    assert len({item.canonical_entity_id for item in first.mention_to_canonical}) == 2
