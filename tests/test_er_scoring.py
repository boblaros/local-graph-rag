from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import entity_resolution.scoring as scoring_module  # noqa: E402
from entity_resolution import (  # noqa: E402
    CandidatePair,
    EmbeddingNeighborEvidence,
    ERConfig,
    ERDecisionPolicy,
    EntityMention,
    JudgeCache,
    JudgeBudgetExceededError,
    JudgeIdentity,
    JudgeResult,
    PairScore,
    build_entity_profiles,
    decide_pairs,
    generate_candidate_pairs,
    score_candidate_pairs,
    lexical_view,
    normalize_text,
)


def _named_mentions(*names: str) -> list[EntityMention]:
    return [
        EntityMention(
            mention_id=f"m{index}",
            document_id=f"doc{index}",
            chunk_id=f"chunk{index}",
            original_name=name,
            extraction_call_id=f"call{index}",
            description_present=False,
        )
        for index, name in enumerate(names, start=1)
    ]


def _mentions() -> list[EntityMention]:
    return [
        EntityMention(
            mention_id="m_company_long",
            document_id="doc_1",
            chunk_id="chunk_1",
            original_name="Apple Inc.",
            entity_type="organization",
            description="Consumer technology company",
            extraction_call_id="call_1",
        ),
        EntityMention(
            mention_id="m_company_short",
            document_id="doc_2",
            chunk_id="chunk_2",
            original_name="Apple",
            entity_type="company",
            description="Consumer technology company",
            extraction_call_id="call_2",
        ),
        EntityMention(
            mention_id="m_fruit",
            document_id="doc_3",
            chunk_id="chunk_3",
            original_name="Apple",
            entity_type="fruit",
            description=None,
            extraction_call_id="call_3",
            description_present=False,
        ),
    ]


def test_missing_signals_are_unavailable_and_weights_are_renormalized() -> None:
    profiles = build_entity_profiles(_mentions(), [])
    candidates = generate_candidate_pairs(profiles)
    target = next(
        pair
        for pair in candidates
        if {pair.left_mention_id, pair.right_mention_id}
        == {"m_company_long", "m_fruit"}
    )
    score = score_candidate_pairs([target], profiles)[0]

    assert score.signals["description"] is None
    assert score.signals["embedding"] is None
    assert score.signals["neighbourhood"] is None
    assert "description" in score.unavailable_signals
    assert "description" not in score.effective_weights
    assert sum(score.effective_weights.values()) == pytest.approx(1.0)


def test_candidate_union_and_type_mismatch_is_scoring_only() -> None:
    profiles = build_entity_profiles(_mentions(), [])
    candidates = generate_candidate_pairs(
        profiles,
        embedding_neighbors={"m_company_long": ["m_company_short"]},
    )
    by_members = {
        frozenset((pair.left_mention_id, pair.right_mention_id)): pair
        for pair in candidates
    }
    company_pair = by_members[frozenset(("m_company_long", "m_company_short"))]
    fruit_pair = by_members[frozenset(("m_company_short", "m_fruit"))]

    assert {"containment", "embedding_neighbour"} <= set(company_pair.methods)
    assert "compatible_types" not in company_pair.methods
    scores = score_candidate_pairs([fruit_pair], profiles)
    assert scores[0].signals["type"] == 0.0


def test_large_exact_name_block_is_bounded() -> None:
    mention_count = 205
    mentions = [
        EntityMention(
            mention_id=f"m_{index:04d}",
            document_id=f"doc_{index:04d}",
            chunk_id=f"chunk_{index:04d}",
            original_name="Common Surface",
            entity_type=None,
            description=None,
            extraction_call_id=f"call_{index:04d}",
            description_present=False,
        )
        for index in range(mention_count)
    ]
    profiles = build_entity_profiles(mentions, [])

    candidates = generate_candidate_pairs(
        profiles,
        ERConfig(max_block_size=10),
    )

    assert candidates
    assert len(candidates) < mention_count * 3
    assert all("exact_name" in candidate.methods for candidate in candidates)


def test_context_type_and_shared_provenance_are_scoring_only() -> None:
    mentions = [
        EntityMention(
            mention_id=f"m{index}",
            document_id="doc",
            chunk_id="chunk",
            original_name=name,
            entity_type="organization",
            description="same generic context",
            extraction_call_id="call",
        )
        for index, name in enumerate(("Alpha", "Beta"), start=1)
    ]
    profiles = build_entity_profiles(mentions, [])

    assert generate_candidate_pairs(profiles) == []


def test_fuzzy_candidate_threshold_is_point_seven_and_persists_similarity() -> None:
    profiles = build_entity_profiles(_named_mentions("abcdefghij", "abcdefz"), [])

    included = generate_candidate_pairs(profiles)
    excluded = generate_candidate_pairs(
        profiles,
        ERConfig(fuzzy_candidate_threshold=0.71),
    )

    assert len(included) == 1
    candidate = included[0]
    expected = 0.7058823529411765
    assert candidate.methods == ("fuzzy_name",)
    assert candidate.route_evidence["fuzzy_name"] == {
        "lexical_similarity": pytest.approx(expected),
        "threshold": 0.70,
    }
    assert candidate.precomputed_signals["lexical"] == pytest.approx(expected)
    assert excluded == []


def test_exact_name_preserves_symbols_while_fuzzy_uses_lexical_view() -> None:
    profiles = build_entity_profiles(_named_mentions("$30", "30%"), [])

    assert normalize_text("  $30  ") == "$30"
    assert normalize_text("30%") == "30%"
    assert lexical_view("$30") == lexical_view("30%") == "30"

    candidate = generate_candidate_pairs(profiles)[0]
    assert "exact_name" not in candidate.methods
    assert "fuzzy_name" in candidate.methods

    score = score_candidate_pairs([candidate], profiles)[0]
    assert score.signals["name_exact"] == 0.0
    assert score.signals["lexical"] == 1.0


def test_exact_name_normalizes_case_unicode_and_whitespace_only() -> None:
    profiles = build_entity_profiles(_named_mentions("  BetMGM  ", "ＢＥＴＭＧＭ"), [])

    candidate = generate_candidate_pairs(profiles)[0]
    assert "exact_name" in candidate.methods
    assert candidate.route_evidence["exact_name"] == {"canonical_name": "betmgm"}
    assert score_candidate_pairs([candidate], profiles)[0].signals["name_exact"] == 1.0


def test_scorer_reuses_candidate_lexical_similarity(monkeypatch) -> None:  # noqa: ANN001
    profiles = build_entity_profiles(_named_mentions("abcdefghij", "abcdefz"), [])
    candidate = generate_candidate_pairs(profiles)[0]

    def explode(*_: object, **__: object) -> None:
        raise AssertionError("lexical similarity must not be recomputed")

    monkeypatch.setattr(scoring_module, "SequenceMatcher", explode)
    score = score_candidate_pairs([candidate], profiles)[0]

    assert score.signals["lexical"] == pytest.approx(
        candidate.precomputed_signals["lexical"]
    )


def test_scorer_reuses_embedding_cosine_from_candidate_route(monkeypatch) -> None:  # noqa: ANN001
    mentions = _named_mentions("Alpha", "Beta")
    mentions = [
        EntityMention(**{**mention.to_dict(), "embedding": embedding})
        for mention, embedding in zip(
            mentions,
            ((1.0, 0.0), (0.8, 0.6)),
            strict=True,
        )
    ]
    profiles = build_entity_profiles(mentions, [])
    neighbours = {
        "m1": [EmbeddingNeighborEvidence("m2", raw_cosine=0.8, rank=1)],
        "m2": [EmbeddingNeighborEvidence("m1", raw_cosine=0.8, rank=1)],
    }
    candidate = generate_candidate_pairs(
        profiles,
        embedding_neighbors=neighbours,
    )[0]

    def explode(*_: object, **__: object) -> None:
        raise AssertionError("embedding cosine must not be recomputed")

    monkeypatch.setattr(scoring_module, "_cosine", explode)
    score = score_candidate_pairs([candidate], profiles)[0]

    assert candidate.precomputed_signals["embedding"] == pytest.approx(0.9)
    assert score.signals["embedding"] == pytest.approx(0.9)
    assert candidate.route_evidence["embedding_neighbour"]["raw_cosine"] == 0.8
    assert "reciprocal_embedding_neighbour" in candidate.methods


def test_embedding_candidates_mark_only_reciprocal_neighbours() -> None:
    profiles = build_entity_profiles(_mentions(), [])
    one_way = generate_candidate_pairs(
        profiles,
        embedding_neighbors={"m_company_long": ["m_company_short"]},
    )
    reciprocal = generate_candidate_pairs(
        profiles,
        embedding_neighbors={
            "m_company_long": ["m_company_short"],
            "m_company_short": ["m_company_long"],
        },
    )
    one_way_pair = next(
        item
        for item in one_way
        if {item.left_mention_id, item.right_mention_id}
        == {"m_company_long", "m_company_short"}
    )
    reciprocal_pair = next(
        item
        for item in reciprocal
        if {item.left_mention_id, item.right_mention_id}
        == {"m_company_long", "m_company_short"}
    )

    assert "reciprocal_embedding_neighbour" not in one_way_pair.methods
    assert "reciprocal_embedding_neighbour" in reciprocal_pair.methods


class _CountingJudge:
    identity = JudgeIdentity(
        model_tag="judge:fixed",
        model_digest="sha256:judge",
        prompt_version="er-pair-v1",
        temperature=0.0,
        seed=17,
    )

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, left, right, score):  # noqa: ANN001
        self.calls += 1
        assert {left.original_name, right.original_name} == {"Apple", "Apple Inc."}
        return JudgeResult(
            relationship="same_entity",
            rationale="same company and compatible context",
        )


def _fixed_pair_score(candidate: CandidatePair, value: float | None) -> PairScore:
    return PairScore(
        pair_id=candidate.pair_id,
        left_mention_id=candidate.left_mention_id,
        right_mention_id=candidate.right_mention_id,
        aggregate_score=value,
        signals={"name_exact": value},
        effective_weights={"name_exact": 1.0} if value is not None else {},
        unavailable_signals=() if value is not None else ("name_exact",),
    )


def test_fixed_policy_boundaries() -> None:
    profiles = build_entity_profiles(_mentions()[:2], [])
    candidates = generate_candidate_pairs(
        profiles,
        embedding_neighbors={
            "m_company_long": ["m_company_short"],
            "m_company_short": ["m_company_long"],
        },
    )
    candidate = candidates[0]
    judge = _CountingJudge()

    below = decide_pairs(
        [_fixed_pair_score(candidate, 0.4999)],
        profiles,
        judge=judge,
        candidates=candidates,
    ).decisions[0]
    at_lower = decide_pairs(
        [_fixed_pair_score(candidate, 0.50)],
        profiles,
        judge=judge,
        candidates=candidates,
    ).decisions[0]
    below_upper = decide_pairs(
        [_fixed_pair_score(candidate, 0.7599)],
        profiles,
        judge=judge,
        candidates=candidates,
    ).decisions[0]
    at_upper = decide_pairs(
        [_fixed_pair_score(candidate, 0.76)],
        profiles,
        judge=judge,
        candidates=candidates,
    ).decisions[0]
    unavailable = decide_pairs(
        [_fixed_pair_score(candidate, None)],
        profiles,
        judge=judge,
        candidates=candidates,
    ).decisions[0]

    assert (below.action, below.source) == ("reject", "score_policy_reject")
    assert (at_lower.action, at_lower.source) == ("merge", "judge")
    assert (below_upper.action, below_upper.source) == ("merge", "judge")
    assert (at_upper.action, at_upper.source) == ("merge", "score_policy_merge")
    assert (unavailable.action, unavailable.source) == ("abstain", "no_judge")
    assert judge.calls == 2


def test_mutual_top_one_admits_only_the_pair_best_for_both_mentions() -> None:
    mentions = [
        EntityMention(
            mention_id=mention_id,
            document_id=f"doc-{mention_id}",
            chunk_id=f"chunk-{mention_id}",
            original_name=f"Entity {mention_id}",
            entity_type="organization",
        )
        for mention_id in ("a", "b", "c")
    ]
    profiles = build_entity_profiles(mentions, [])
    pairs = [
        CandidatePair("p-ab", "a", "b", ("reciprocal_embedding_neighbour",)),
        CandidatePair("p-ac", "a", "c", ("reciprocal_embedding_neighbour",)),
        CandidatePair("p-bc", "b", "c", ("reciprocal_embedding_neighbour",)),
    ]
    scores = [
        _fixed_pair_score(pairs[0], 0.68),
        _fixed_pair_score(pairs[1], 0.70),
        _fixed_pair_score(pairs[2], 0.72),
    ]

    batch = decide_pairs(scores, profiles, candidates=pairs)
    by_pair = {item.pair_id: item for item in batch.decisions}

    assert by_pair["p-bc"].source == "no_judge"
    assert by_pair["p-ab"].source == "judge_admission"
    assert by_pair["p-ac"].source == "judge_admission"
    assert batch.judge_budget["admitted_ambiguous_pairs"] == 1


def test_ambiguous_pair_uses_fixed_judge_and_content_addressed_cache() -> None:
    profiles = build_entity_profiles(_mentions()[:2], [])
    candidates = generate_candidate_pairs(
        profiles,
        embedding_neighbors={
            "m_company_long": ["m_company_short"],
            "m_company_short": ["m_company_long"],
        },
    )
    candidate = candidates[0]
    config = ERConfig(
        decision_policy=ERDecisionPolicy(
            reject_below_score=0.05,
            auto_merge_at_score=0.99,
        )
    )
    score = score_candidate_pairs([candidate], profiles, config)[0]
    assert (
        config.decision_policy.reject_below_score
        <= score.aggregate_score
        < config.decision_policy.auto_merge_at_score
    )
    judge = _CountingJudge()
    cache = JudgeCache()

    first = decide_pairs(
        [score],
        profiles,
        config,
        judge=judge,
        judge_cache=cache,
        candidates=candidates,
    )
    second = decide_pairs(
        [score],
        profiles,
        config,
        judge=judge,
        judge_cache=cache,
        candidates=candidates,
    )

    assert first.decisions[0].action == "merge"
    assert first.decisions[0].judge_cache_hit is False
    assert second.decisions[0].judge_cache_hit is True
    assert first.decisions[0].judge_cache_key == second.decisions[0].judge_cache_key
    assert judge.calls == 1


def test_ambiguous_pair_abstains_when_no_judge_is_supplied() -> None:
    profiles = build_entity_profiles(_mentions()[:2], [])
    candidates = generate_candidate_pairs(
        profiles,
        embedding_neighbors={
            "m_company_long": ["m_company_short"],
            "m_company_short": ["m_company_long"],
        },
    )
    candidate = candidates[0]
    config = ERConfig(
        decision_policy=ERDecisionPolicy(
            reject_below_score=0.05,
            auto_merge_at_score=0.99,
        )
    )
    score = score_candidate_pairs([candidate], profiles, config)[0]
    assert (
        config.decision_policy.reject_below_score
        <= score.aggregate_score
        < config.decision_policy.auto_merge_at_score
    )
    judge = _CountingJudge()

    batch = decide_pairs([score], profiles, config, candidates=candidates)

    assert batch.decisions[0].action == "abstain"
    assert batch.decisions[0].source == "no_judge"
    assert batch.decisions[0].judge_cache_key is None
    assert judge.calls == 0


def test_ambiguous_pair_without_candidate_route_is_not_admitted() -> None:
    profiles = build_entity_profiles(_mentions()[:2], [])
    candidate = generate_candidate_pairs(profiles)[0]
    score = _fixed_pair_score(candidate, 0.60)
    judge = _CountingJudge()

    batch = decide_pairs([score], profiles, judge=judge)

    assert batch.decisions[0].action == "abstain"
    assert batch.decisions[0].source == "judge_admission"
    assert batch.judge_budget["admitted_ambiguous_pairs"] == 0
    assert judge.calls == 0


def test_nonreciprocal_embedding_only_pair_abstains_before_judge() -> None:
    mentions = [
        EntityMention(
            mention_id=f"m{index}",
            document_id=f"doc{index}",
            chunk_id=f"chunk{index}",
            original_name=name,
            entity_type="organization",
            description="technology company",
            extraction_call_id=f"call{index}",
        )
        for index, name in enumerate(("Alpha", "Beta"), start=1)
    ]
    profiles = build_entity_profiles(mentions, [])
    candidates = generate_candidate_pairs(profiles, embedding_neighbors={"m1": ["m2"]})
    config = ERConfig(
        decision_policy=ERDecisionPolicy(
            reject_below_score=0.0,
            auto_merge_at_score=0.99,
        )
    )
    scores = score_candidate_pairs(candidates, profiles, config)
    judge = _CountingJudge()

    batch = decide_pairs(
        scores,
        profiles,
        config,
        judge=judge,
        candidates=candidates,
    )

    assert batch.decisions[0].action == "abstain"
    assert batch.decisions[0].source == "judge_admission"
    assert judge.calls == 0

    high_score = _fixed_pair_score(candidates[0], 0.80)
    high_batch = decide_pairs(
        [high_score], profiles, ERConfig(), judge=judge, candidates=candidates
    )
    assert high_batch.decisions[0].action == "merge"
    assert high_batch.decisions[0].source == "score_policy_merge"
    assert judge.calls == 0


def test_judge_budget_fails_before_first_model_call() -> None:
    profiles = build_entity_profiles(_mentions()[:2], [])
    candidates = generate_candidate_pairs(
        profiles,
        embedding_neighbors={
            "m_company_long": ["m_company_short"],
            "m_company_short": ["m_company_long"],
        },
    )
    config = ERConfig(
        decision_policy=ERDecisionPolicy(
            reject_below_score=0.05,
            auto_merge_at_score=0.99,
            max_judge_calls_per_run=0,
        )
    )
    scores = score_candidate_pairs(candidates, profiles, config)
    judge = _CountingJudge()

    with pytest.raises(JudgeBudgetExceededError) as raised:
        decide_pairs(
            scores,
            profiles,
            config,
            judge=judge,
            candidates=candidates,
        )

    assert raised.value.report["admitted_ambiguous_pairs"] == 1
    assert raised.value.report["passed"] is False
    assert judge.calls == 0


def test_operational_budget_override_allows_the_same_admitted_pair() -> None:
    profiles = build_entity_profiles(_mentions()[:2], [])
    candidates = generate_candidate_pairs(
        profiles,
        embedding_neighbors={
            "m_company_long": ["m_company_short"],
            "m_company_short": ["m_company_long"],
        },
    )
    config = ERConfig(
        decision_policy=ERDecisionPolicy(
            reject_below_score=0.05,
            auto_merge_at_score=0.99,
            max_judge_calls_per_run=0,
        )
    )
    scores = score_candidate_pairs(candidates, profiles, config)
    judge = _CountingJudge()

    batch = decide_pairs(
        scores,
        profiles,
        config,
        judge=judge,
        candidates=candidates,
        operational_max_judge_calls_per_run=1,
    )

    assert batch.decisions[0].source == "judge"
    assert batch.judge_budget["planned_max_judge_calls_per_run"] == 0
    assert batch.judge_budget["max_judge_calls_per_run"] == 1
    assert batch.judge_budget["operational_budget_overridden"] is True
    assert batch.judge_budget["passed"] is True
    assert judge.calls == 1
