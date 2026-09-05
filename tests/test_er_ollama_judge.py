from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entity_resolution import (  # noqa: E402
    ERConfig,
    ERDecisionPolicy,
    EntityMention,
    OllamaERJudge,
    build_entity_profiles,
    generate_candidate_pairs,
    score_candidate_pairs,
)


class _FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        return {"message": {"content": self.content}}


class _SequenceOllamaClient:
    def __init__(self, responses: list[str | dict]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def chat(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, dict):
            return response
        return {"message": {"content": response}}


def _ambiguous_pair():
    profiles = build_entity_profiles(
        [
            EntityMention("m1", "d1", "c1", "Apple Inc.", "organization"),
            EntityMention("m2", "d2", "c2", "Apple", "company"),
        ],
        [],
    )
    config = ERConfig(
        decision_policy=ERDecisionPolicy(
            reject_below_score=0.05,
            auto_merge_at_score=0.99,
        )
    )
    candidate = generate_candidate_pairs(profiles, config)[0]
    score = score_candidate_pairs([candidate], profiles, config)[0]
    return profiles, score


def test_sync_ollama_judge_uses_only_frozen_config_and_strict_json() -> None:
    client = _FakeOllamaClient(
        '{"relationship":"same_entity","rationale":"same company with compatible evidence"}'
    )
    judge = OllamaERJudge(
        client=client,
        model_tag="resolved-judge:tag",
        model_digest="sha256:fixed-digest",
        prompt_version="er-pair-v1",
        temperature=0.0,
        seed=41,
        options={"num_predict": 128},
    )
    profiles, score = _ambiguous_pair()

    result = judge.decide(profiles[0], profiles[1], score)

    assert result.relationship == "same_entity"
    assert result.action == "merge"
    assert judge.identity.model_digest == "sha256:fixed-digest"
    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["model"] == "resolved-judge:tag"
    assert request["stream"] is False
    assert request["options"] == {
        "num_predict": 128,
        "temperature": 0.0,
        "seed": 41,
    }
    assert request["format"]["additionalProperties"] is False
    assert "sha256:fixed-digest" not in str(request["messages"])


def test_sync_judge_prompt_omits_raw_embedding_vectors() -> None:
    client = _FakeOllamaClient(
        '{"relationship":"uncertain","rationale":"insufficient"}'
    )
    judge = OllamaERJudge(
        client=client,
        model_tag="resolved-judge:tag",
        model_digest="sha256:fixed-digest",
        prompt_version="er-pair-v2-no-vector-payload",
        temperature=0.0,
        seed=42,
        options={},
    )
    profiles, score = _ambiguous_pair()
    left = profiles[0].__class__(
        **{**profiles[0].to_dict(), "embedding": (0.1, 0.2, 0.3)}
    )
    right = profiles[1].__class__(
        **{**profiles[1].to_dict(), "embedding": (0.4, 0.5, 0.6)}
    )

    judge.decide(left, right, score)

    user_payload = json.loads(client.calls[0]["messages"][1]["content"])
    system_prompt = client.calls[0]["messages"][0]["content"]
    assert "embedding" not in user_payload["left_profile"]
    assert "embedding" not in user_payload["right_profile"]
    assert user_payload["left_profile"]["embedding_available"] is True
    assert user_payload["left_profile"]["embedding_dimension"] == 3
    assert "not an automatic veto" in system_prompt
    assert "exact same real-world referent" in system_prompt
    assert "version_or_variant for a version, release, edition" in system_prompt
    assert "Return only a JSON object with exactly relationship" in system_prompt


def test_sync_judge_retries_invalid_json_without_changing_request() -> None:
    client = _SequenceOllamaClient(
        [
            '{"relationship":"same_entity",',
            '{"relationship":"same_entity","rationale":"same entity"}',
        ]
    )
    judge = OllamaERJudge(
        client=client,
        model_tag="resolved-judge:tag",
        model_digest="sha256:fixed-digest",
        prompt_version="er-pair-v1",
        temperature=0.0,
        seed=41,
        options={},
    )
    profiles, score = _ambiguous_pair()

    result = judge.decide(profiles[0], profiles[1], score)

    assert result.action == "merge"
    assert result.metadata["physical_attempts"] == 2
    assert client.calls[0] == client.calls[1]


def test_sync_judge_compacts_only_after_confirmed_context_overflow() -> None:
    client = _SequenceOllamaClient(
        [
            {
                "message": {"content": "{"},
                "prompt_eval_count": 8191,
                "eval_count": 1,
            },
            {
                "message": {
                    "content": (
                        '{"relationship":"same_entity",'
                        '"rationale":"same company"}'
                    )
                },
                "prompt_eval_count": 1800,
                "eval_count": 20,
            },
        ]
    )
    judge = OllamaERJudge(
        client=client,
        model_tag="resolved-judge:tag",
        model_digest="sha256:fixed-digest",
        prompt_version="er-pair-v5-relationship-classification",
        temperature=0.0,
        seed=42,
        options={"num_ctx": 8192, "num_predict": 256},
    )
    profiles, score = _ambiguous_pair()

    result = judge.decide(profiles[0], profiles[1], score)

    full = json.loads(client.calls[0]["messages"][1]["content"])
    compact = json.loads(client.calls[1]["messages"][1]["content"])
    assert "source_mentions" in full["left_profile"]
    assert "source_mentions" not in compact["left_profile"]
    assert "provenance" not in compact["left_profile"]
    assert compact["left_profile"]["original_name"] == profiles[0].original_name
    assert compact["left_profile"]["description"] == profiles[0].description
    assert compact["pair_score"] == full["pair_score"]
    assert result.action == "merge"
    assert result.metadata["physical_attempts"] == 2
    assert result.metadata["context_overflow_recovered"] is True
    assert result.metadata["prompt_mode"] == "semantic_compact"
    assert result.metadata["overflow_recovery_version"].endswith("-v1")
    assert len(result.metadata["compact_request_sha256"]) == 64
    assert result.metadata["overflow_events"] == [
        {
            "physical_attempt": 1,
            "prompt_mode": "full",
            "prompt_eval_count": 8191,
        }
    ]


def test_sync_judge_does_not_compact_non_overflow_parse_retry() -> None:
    client = _SequenceOllamaClient(
        [
            {
                "message": {"content": "{"},
                "prompt_eval_count": 1200,
                "eval_count": 1,
            },
            {
                "message": {
                    "content": (
                        '{"relationship":"uncertain",'
                        '"rationale":"insufficient"}'
                    )
                },
                "prompt_eval_count": 1200,
                "eval_count": 20,
            },
        ]
    )
    judge = OllamaERJudge(
        client=client,
        model_tag="resolved-judge:tag",
        model_digest="sha256:fixed-digest",
        prompt_version="er-pair-v5-relationship-classification",
        temperature=0.0,
        seed=42,
        options={"num_ctx": 8192, "num_predict": 256},
    )
    profiles, score = _ambiguous_pair()

    result = judge.decide(profiles[0], profiles[1], score)

    assert client.calls[0] == client.calls[1]
    assert result.metadata["physical_attempts"] == 2
    assert "context_overflow_recovered" not in result.metadata


def test_sync_judge_uses_bounded_compaction_after_second_overflow() -> None:
    client = _SequenceOllamaClient(
        [
            {
                "message": {"content": "{"},
                "prompt_eval_count": 8191,
            },
            {
                "message": {"content": "{"},
                "prompt_eval_count": 8191,
            },
            {
                "message": {
                    "content": (
                        '{"relationship":"different_entity",'
                        '"rationale":"different"}'
                    )
                },
                "prompt_eval_count": 2000,
            },
        ]
    )
    judge = OllamaERJudge(
        client=client,
        model_tag="resolved-judge:tag",
        model_digest="sha256:fixed-digest",
        prompt_version="er-pair-v5-relationship-classification",
        temperature=0.0,
        seed=42,
        options={"num_ctx": 8192, "num_predict": 256},
    )
    profiles, score = _ambiguous_pair()
    left = profiles[0].__class__(
        **{
            **profiles[0].to_dict(),
            "neighbours": tuple(f"neighbour-{index}" for index in range(40)),
            "relation_context": tuple(
                f"relation-context-{index}" for index in range(40)
            ),
        }
    )

    result = judge.decide(left, profiles[1], score)

    semantic = json.loads(client.calls[1]["messages"][1]["content"])
    bounded = json.loads(client.calls[2]["messages"][1]["content"])
    assert len(semantic["left_profile"]["neighbours"]) == 40
    assert len(semantic["left_profile"]["relation_context"]) == 40
    assert len(bounded["left_profile"]["neighbours"]) == 24
    assert len(bounded["left_profile"]["relation_context"]) == 24
    assert result.action == "reject"
    assert result.metadata["prompt_mode"] == "bounded_compact"
    assert result.metadata["physical_attempts"] == 3
    assert result.metadata["bounded_profile_limits"]["neighbour_items"] == 24


@pytest.mark.parametrize(
    ("relationship", "expected_action"),
    [
        ("same_entity", "merge"),
        ("version_or_variant", "reject"),
        ("role_or_character", "reject"),
        ("related_or_broader_narrower", "reject"),
        ("different_entity", "reject"),
        ("uncertain", "abstain"),
    ],
)
def test_judge_relationship_deterministically_maps_to_action(
    relationship: str,
    expected_action: str,
) -> None:
    client = _FakeOllamaClient(
        json.dumps({"relationship": relationship, "rationale": "fixed rationale"})
    )
    judge = OllamaERJudge(
        client=client,
        model_tag="resolved-judge:tag",
        model_digest="sha256:fixed-digest",
        prompt_version="er-pair-v5-relationship-classification",
        temperature=0.0,
        seed=41,
        options={},
    )
    profiles, score = _ambiguous_pair()

    result = judge.decide(profiles[0], profiles[1], score)

    assert result.relationship == relationship
    assert result.action == expected_action


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"relationship":"same_entity","rationale":"x"}\n```',
        '{"relationship":"maybe","rationale":"x"}',
        '{"relationship":"different_entity","rationale":"x","confidence":0.9}',
        '{"relationship":"uncertain","rationale":""}',
    ],
)
def test_sync_ollama_judge_rejects_non_strict_responses(content: str) -> None:
    client = _FakeOllamaClient(content)
    judge = OllamaERJudge(
        client=client,
        model_tag="resolved-judge:tag",
        model_digest="sha256:fixed-digest",
        prompt_version="er-pair-v1",
        temperature=0.0,
        seed=41,
        options={},
    )
    profiles, score = _ambiguous_pair()

    with pytest.raises(ValueError, match="Ollama judge"):
        judge.decide(profiles[0], profiles[1], score)
