from __future__ import annotations

import asyncio
import hashlib
import json
import random
from pathlib import Path

import pytest

from src.entity_resolution.embeddings import (
    EMBEDDING_PROFILE_TEXT_VERSION,
    EmbeddingCacheError,
    EmbeddingDimensionError,
    OllamaEmbeddingIdentity,
    augment_mentions_with_ollama_embeddings,
    build_lsh_cosine_neighbor_evidence,
    build_lsh_cosine_neighbor_map,
    embedding_cache_key,
    embedding_cache_path,
    mention_embedding_profile_text,
    profile_text_sha256,
)
from src.entity_resolution.profiles import build_entity_profiles
from src.extraction.models import NormalizedEntityMention


class _FakeAsyncOllamaClient:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.calls: list[dict[str, object]] = []

    async def embed(self, *, model: str, input: list[str]):  # noqa: A002
        self.calls.append({"model": model, "input": list(input)})
        vectors = []
        for text in input:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            vectors.append(
                [float((seed + index) % 997) / 997.0 for index in range(self.dimension)]
            )
        return {"embeddings": vectors}


class _ExplodingAsyncOllamaClient:
    async def embed(self, **_: object):
        raise AssertionError("cache hit must not call Ollama")


class _TransientAsyncOllamaClient(_FakeAsyncOllamaClient):
    def __init__(self, dimension: int, failures: int) -> None:
        super().__init__(dimension)
        self.failures = failures
        self.attempts = 0

    async def embed(self, *, model: str, input: list[str]):  # noqa: A002
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("backend embeddings request: EOF")
        return await super().embed(model=model, input=input)


def _mention(index: int) -> dict[str, object]:
    return {
        "mention_id": f"mention-{index}",
        "mention_index": index,
        "document_id": f"document-{index}",
        "chunk_id": f"document-{index}-chunk-000",
        "original_name": f"Entity {index}",
        "normalized_name": f"entity {index}",
        "entity_type": "organization",
        "description": f"Description {index}",
        "description_present": True,
        "entity_present": True,
    }


def test_profile_text_is_versioned_deterministic_and_uses_only_profile_fields():
    left = {
        **_mention(1),
        "original_name": "  Ａpple\n Inc. ",
        "entity_type": " Organization ",
        "description": None,
        "description_present": False,
        "provenance": {"row": 1},
    }
    right = {
        **left,
        "document_id": "another-document",
        "chunk_id": "another-document-chunk-007",
        "provenance": {"row": 999},
    }

    left_text = mention_embedding_profile_text(left)
    assert left_text == mention_embedding_profile_text(right)
    assert json.loads(left_text) == {
        "description": None,
        "name": "Apple Inc.",
        "profile_text_version": EMBEDDING_PROFILE_TEXT_VERSION,
        "type": "Organization",
    }


def test_async_ollama_embedding_batches_and_augments_normalized_models(tmp_path: Path):
    client = _FakeAsyncOllamaClient(dimension=3)
    identity = OllamaEmbeddingIdentity(
        resolved_tag="resolved-embed:tag",
        resolved_digest="sha256:resolved-digest",
        dimension=3,
    )
    pydantic_mention = NormalizedEntityMention(
        mention_id="mention-pydantic",
        mention_index=99,
        original_name="Pydantic Entity",
        normalized_name="pydantic entity",
        entity_type=None,
        description=None,
        document_id="document-pydantic",
        chunk_id="document-pydantic-chunk-000",
        provenance={},
        extraction_call_id="call-pydantic",
        parse_status="strict",
        entity_present=True,
        description_present=False,
    )
    mentions = [*_mention_sequence(4), pydantic_mention]

    augmented = asyncio.run(
        augment_mentions_with_ollama_embeddings(
            mentions,
            client=client,
            identity=identity,
            cache_dir=tmp_path / "cache",
            batch_size=2,
        )
    )

    assert [len(call["input"]) for call in client.calls] == [2, 2, 1]
    assert {call["model"] for call in client.calls} == {"resolved-embed:tag"}
    assert [item["mention_id"] for item in augmented] == [
        "mention-0",
        "mention-1",
        "mention-2",
        "mention-3",
        "mention-pydantic",
    ]
    assert all(len(item["embedding"]) == 3 for item in augmented)
    assert all(
        item["embedding_model_digest"] == identity.resolved_digest for item in augmented
    )
    assert all(item["embedding_dimension"] == 3 for item in augmented)
    assert all(item["embedding_cache_hit"] is False for item in augmented)

    # The augmented mappings feed the existing profile/scoring path directly.
    profiles = build_entity_profiles(augmented, [])
    assert all(profile.embedding is not None for profile in profiles)
    assert all(len(profile.embedding or ()) == 3 for profile in profiles)


def test_transient_embedding_eof_is_retried_with_identical_input(tmp_path: Path):
    client = _TransientAsyncOllamaClient(dimension=2, failures=2)

    augmented = asyncio.run(
        augment_mentions_with_ollama_embeddings(
            [_mention(1)],
            client=client,
            identity=OllamaEmbeddingIdentity("embed:tag", "digest", 2),
            cache_dir=tmp_path / "cache",
            batch_size=1,
        )
    )

    assert client.attempts == 3
    assert len(client.calls) == 1
    assert len(augmented[0]["embedding"]) == 2


def test_transient_embedding_eof_fails_after_bounded_attempts(tmp_path: Path):
    client = _TransientAsyncOllamaClient(dimension=2, failures=3)

    with pytest.raises(RuntimeError, match="EOF"):
        asyncio.run(
            augment_mentions_with_ollama_embeddings(
                [_mention(1)],
                client=client,
                identity=OllamaEmbeddingIdentity("embed:tag", "digest", 2),
                cache_dir=tmp_path / "cache",
                batch_size=1,
            )
        )

    assert client.attempts == 3
    assert not list((tmp_path / "cache").rglob("*.json"))


def _mention_sequence(count: int) -> list[dict[str, object]]:
    return [_mention(index) for index in range(count)]


def test_cache_is_keyed_by_digest_and_text_hash_not_mutable_tag(tmp_path: Path):
    mention = _mention(1)
    cache_dir = tmp_path / "cache"
    first_identity = OllamaEmbeddingIdentity("alias-one", "sha256:same", 2)
    first_client = _FakeAsyncOllamaClient(dimension=2)
    first = asyncio.run(
        augment_mentions_with_ollama_embeddings(
            [mention],
            client=first_client,
            identity=first_identity,
            cache_dir=cache_dir,
            batch_size=8,
        )
    )

    second_identity = OllamaEmbeddingIdentity("alias-two", "sha256:same", 2)
    second = asyncio.run(
        augment_mentions_with_ollama_embeddings(
            [mention],
            client=_ExplodingAsyncOllamaClient(),
            identity=second_identity,
            cache_dir=cache_dir,
            batch_size=8,
        )
    )

    assert len(first_client.calls) == 1
    assert first[0]["embedding"] == second[0]["embedding"]
    assert first[0]["embedding_cache_key"] == second[0]["embedding_cache_key"]
    assert first[0]["embedding_cache_hit"] is False
    assert second[0]["embedding_cache_hit"] is True
    assert second[0]["embedding_model_tag"] == "alias-two"
    assert len(list(cache_dir.rglob("*.json"))) == 1


def test_wrong_ollama_dimension_fails_before_cache_write(tmp_path: Path):
    identity = OllamaEmbeddingIdentity("resolved-tag", "sha256:digest", 3)
    client = _FakeAsyncOllamaClient(dimension=2)

    with pytest.raises(EmbeddingDimensionError, match="dimension 2 != expected 3"):
        asyncio.run(
            augment_mentions_with_ollama_embeddings(
                [_mention(1)],
                client=client,
                identity=identity,
                cache_dir=tmp_path / "cache",
                batch_size=1,
            )
        )

    assert list((tmp_path / "cache").rglob("*.json")) == []


def test_tampered_immutable_cache_fails_closed_without_ollama_call(tmp_path: Path):
    mention = _mention(1)
    identity = OllamaEmbeddingIdentity("resolved-tag", "sha256:digest", 3)
    text = mention_embedding_profile_text(mention)
    text_hash = profile_text_sha256(text)
    key = embedding_cache_key(identity.resolved_digest, text_hash)
    path = embedding_cache_path(tmp_path / "cache", key)
    path.parent.mkdir(parents=True)
    path.write_text('{"dimension":999}\n', encoding="utf-8")

    with pytest.raises(EmbeddingCacheError, match="metadata mismatch"):
        asyncio.run(
            augment_mentions_with_ollama_embeddings(
                [mention],
                client=_ExplodingAsyncOllamaClient(),
                identity=identity,
                cache_dir=tmp_path / "cache",
                batch_size=4,
            )
        )


@pytest.mark.parametrize(
    ("tag", "digest", "dimension"),
    [("", "sha256:x", 3), ("model", "", 3), ("model", "sha256:x", 0)],
)
def test_embedding_identity_must_be_explicit(tag: str, digest: str, dimension: int):
    with pytest.raises(ValueError):
        OllamaEmbeddingIdentity(tag, digest, dimension)


def _embedded_mention(mention_id: str, vector: list[float]) -> dict[str, object]:
    return {
        "mention_id": mention_id,
        "embedding": vector,
        "embedding_dimension": len(vector),
        "embedding_model_digest": "sha256:fixed-embedding-model",
    }


def test_lsh_cosine_map_finds_near_vectors_and_excludes_opposite_vector():
    mentions = [
        _embedded_mention("alpha", [1.0, 0.0, 0.0]),
        _embedded_mention("alpha-alias", [0.999, 0.02, 0.0]),
        _embedded_mention("orthogonal", [0.0, 1.0, 0.0]),
        _embedded_mention("opposite", [-1.0, 0.0, 0.0]),
    ]

    neighbours = build_lsh_cosine_neighbor_map(
        mentions,
        tables=12,
        bits=4,
        k=1,
        seed=17,
        max_bucket=8,
    )

    assert neighbours["alpha"] == ["alpha-alias"]
    assert neighbours["alpha-alias"] == ["alpha"]
    assert "opposite" not in neighbours["alpha"]


def test_lsh_evidence_preserves_rank_and_symmetric_cosine_for_scoring():
    mentions = [
        _embedded_mention("alpha", [1.0, 0.0, 0.0]),
        _embedded_mention("alpha-alias", [0.8, 0.6, 0.0]),
    ]

    evidence = build_lsh_cosine_neighbor_evidence(
        mentions,
        tables=4,
        bits=1,
        k=1,
        seed=17,
        max_bucket=8,
    )

    left = evidence["alpha"][0]
    right = evidence["alpha-alias"][0]
    assert (left.mention_id, left.rank) == ("alpha-alias", 1)
    assert (right.mention_id, right.rank) == ("alpha", 1)
    assert left.raw_cosine == pytest.approx(0.8)
    assert right.raw_cosine == pytest.approx(left.raw_cosine)
    assert left.scoring_signal == pytest.approx(0.9)
    assert build_lsh_cosine_neighbor_map(
        mentions,
        tables=4,
        bits=1,
        k=1,
        seed=17,
        max_bucket=8,
    ) == {"alpha": ["alpha-alias"], "alpha-alias": ["alpha"]}


def test_lsh_is_deterministic_under_input_order_and_ties():
    mentions = [
        _embedded_mention(f"mention-{index:02d}", [1.0, 1.0, 0.0]) for index in range(9)
    ]
    expected = build_lsh_cosine_neighbor_map(
        mentions,
        tables=3,
        bits=2,
        k=4,
        seed=991,
        max_bucket=5,
    )
    shuffled = list(mentions)
    random.Random(23).shuffle(shuffled)
    actual = build_lsh_cosine_neighbor_map(
        shuffled,
        tables=3,
        bits=2,
        k=4,
        seed=991,
        max_bucket=5,
    )

    assert actual == expected
    assert list(actual) == sorted(actual)
    assert all(values == sorted(values) for values in actual.values())


def test_lsh_bucket_cap_prevents_full_cartesian_candidates():
    count = 30
    mentions = [
        _embedded_mention(f"mention-{index:02d}", [1.0, 0.0]) for index in range(count)
    ]

    neighbours = build_lsh_cosine_neighbor_map(
        mentions,
        tables=1,
        bits=1,
        k=count - 1,
        seed=5,
        max_bucket=3,
    )

    # All vectors share one raw LSH bucket. Deterministic partitioning caps
    # each comparison group at three instead of evaluating 30 choose 2 pairs.
    assert max(len(values) for values in neighbours.values()) <= 2
    assert sum(len(values) for values in neighbours.values()) <= count * 2
    assert sum(len(values) for values in neighbours.values()) < count * (count - 1)


def test_lsh_rejects_mixed_dimensions_and_model_lineage():
    with pytest.raises(EmbeddingDimensionError, match="dimension"):
        build_lsh_cosine_neighbor_map(
            [
                _embedded_mention("one", [1.0, 0.0]),
                _embedded_mention("two", [1.0, 0.0, 0.0]),
            ],
            tables=2,
            bits=2,
            k=1,
            seed=1,
            max_bucket=4,
        )

    mixed = [
        _embedded_mention("one", [1.0, 0.0]),
        {
            **_embedded_mention("two", [0.9, 0.1]),
            "embedding_model_digest": "sha256:different-model",
        },
    ]
    with pytest.raises(ValueError, match="multiple embedding model digests"):
        build_lsh_cosine_neighbor_map(
            mixed,
            tables=2,
            bits=2,
            k=1,
            seed=1,
            max_bucket=4,
        )
