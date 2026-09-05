"""Add deterministic Ollama embeddings to normalized entity mentions.

Callers provide the fixed model tag, digest, and vector dimension. The module
uses Ollama's public embedding API and a content-addressed cache.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

from .models import canonical_json


EMBEDDING_CACHE_SCHEMA_VERSION = "1.0.0"
EMBEDDING_PROFILE_TEXT_VERSION = "er-mention-embedding-text-v1"
EMBEDDING_TRANSIENT_MAX_ATTEMPTS = 3


class EmbeddingAugmentationError(RuntimeError):
    """Base error for mention embedding."""


class EmbeddingCacheError(EmbeddingAugmentationError):
    """Raised when an immutable cache record is invalid or conflicts."""


class EmbeddingDimensionError(EmbeddingAugmentationError):
    """Raised when Ollama or cache data has the wrong vector dimension."""


class AsyncOllamaEmbedClient(Protocol):
    """Subset of the public ``ollama.AsyncClient`` used by this module."""

    async def embed(self, *, model: str, input: Sequence[str]) -> Any: ...


@dataclass(frozen=True)
class OllamaEmbeddingIdentity:
    """Fixed embedding model identity supplied by the experiment config."""

    resolved_tag: str
    resolved_digest: str
    dimension: int

    def __post_init__(self) -> None:
        for field_name in ("resolved_tag", "resolved_digest"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be explicitly resolved")
            object.__setattr__(self, field_name, value)
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise ValueError("embedding dimension must be an integer")
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _normalized_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(value))
    result = " ".join(normalized.split())
    return result or None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="python", exclude_none=False))
    raise TypeError(
        "normalized mention must be a mapping, dataclass, or Pydantic model"
    )


def mention_embedding_profile_text(mention: Any) -> str:
    """Return versioned deterministic text using only name/type/description."""

    record = _as_mapping(mention)
    name = _normalized_optional_text(
        record.get("original_name") or record.get("name") or record.get("entity_name")
    )
    if name is None:
        raise ValueError("normalized mention has no non-empty original name")
    payload = {
        "description": _normalized_optional_text(record.get("description")),
        "name": name,
        "profile_text_version": EMBEDDING_PROFILE_TEXT_VERSION,
        "type": _normalized_optional_text(
            record.get("entity_type", record.get("type"))
        ),
    }
    return canonical_json(payload)


def profile_text_sha256(profile_text: str) -> str:
    return hashlib.sha256(profile_text.encode("utf-8")).hexdigest()


def embedding_cache_key(model_digest: str, text_sha256: str) -> str:
    """Address one vector by immutable model digest plus profile-text hash."""

    digest = str(model_digest).strip()
    text_digest = str(text_sha256).strip().casefold()
    if not digest or not text_digest:
        raise ValueError("model digest and text sha256 must be non-empty")
    if len(text_digest) != 64 or any(
        character not in "0123456789abcdef" for character in text_digest
    ):
        raise ValueError("profile text sha256 must be a 64-character hex digest")
    return hashlib.sha256(f"{digest}\x00{text_digest}".encode("utf-8")).hexdigest()


def embedding_cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    key = str(cache_key).strip().casefold()
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise ValueError("embedding cache key must be a sha256 hex digest")
    root = Path(cache_dir)
    return root / key[:2] / f"{key}.json"


def _validated_vector(
    value: Any,
    *,
    expected_dimension: int,
    source: str,
) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EmbeddingDimensionError(f"{source} embedding is not a numeric sequence")
    if len(value) != expected_dimension:
        raise EmbeddingDimensionError(
            f"{source} embedding dimension {len(value)} != expected "
            f"{expected_dimension}"
        )
    vector: list[float] = []
    for index, raw in enumerate(value):
        try:
            number = float(raw)
        except (TypeError, ValueError) as error:
            raise EmbeddingDimensionError(
                f"{source} embedding value {index} is not numeric"
            ) from error
        if not math.isfinite(number):
            raise EmbeddingDimensionError(
                f"{source} embedding value {index} is non-finite"
            )
        vector.append(number)
    return vector


def _cache_payload(
    *,
    cache_key: str,
    model_tag: str,
    model_digest: str,
    text_sha256: str,
    dimension: int,
    embedding: Sequence[float],
) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "dimension": dimension,
        "embedding": list(embedding),
        "model_digest": model_digest,
        "model_tag_at_creation": model_tag,
        "profile_text_sha256": text_sha256,
        "profile_text_version": EMBEDDING_PROFILE_TEXT_VERSION,
        "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
    }


def _load_cache(
    path: Path,
    *,
    cache_key: str,
    identity: OllamaEmbeddingIdentity,
    text_sha256: str,
) -> list[float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EmbeddingCacheError(f"invalid embedding cache record: {path}") from error
    if not isinstance(payload, dict):
        raise EmbeddingCacheError(f"embedding cache record is not an object: {path}")
    expected_metadata = {
        "cache_key": cache_key,
        "dimension": identity.dimension,
        "model_digest": identity.resolved_digest,
        "profile_text_sha256": text_sha256,
        "profile_text_version": EMBEDDING_PROFILE_TEXT_VERSION,
        "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
    }
    for name, expected in expected_metadata.items():
        if payload.get(name) != expected:
            raise EmbeddingCacheError(
                f"embedding cache metadata mismatch for {name!r}: {path}"
            )
    if set(payload) != {*expected_metadata, "embedding", "model_tag_at_creation"}:
        raise EmbeddingCacheError(f"embedding cache schema mismatch: {path}")
    cached_tag = payload["model_tag_at_creation"]
    if not isinstance(cached_tag, str) or not cached_tag.strip():
        raise EmbeddingCacheError(f"embedding cache has no creation model tag: {path}")
    return _validated_vector(
        payload["embedding"],
        expected_dimension=identity.dimension,
        source="cached",
    )


def _write_cache_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    content = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise EmbeddingCacheError(
                f"cannot verify concurrent embedding cache write: {path}"
            ) from error
        if existing != content:
            raise EmbeddingCacheError(f"immutable embedding cache conflict: {path}")


def _response_embeddings(response: Any) -> Sequence[Any]:
    if isinstance(response, Mapping):
        embeddings = response.get("embeddings")
    else:
        embeddings = getattr(response, "embeddings", None)
    if not isinstance(embeddings, Sequence) or isinstance(
        embeddings, (str, bytes, bytearray)
    ):
        raise EmbeddingAugmentationError(
            "Ollama embed response has no embeddings sequence"
        )
    return embeddings


def _transient_embedding_error(error: Exception) -> bool:
    """Recognize transport/backend failures without retrying semantic 4xx errors."""

    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    message = str(error).casefold()
    return "eof" in message or "connection reset" in message


async def _embed_with_transient_retry(
    client: AsyncOllamaEmbedClient,
    *,
    model: str,
    inputs: Sequence[str],
) -> Any:
    for attempt in range(1, EMBEDDING_TRANSIENT_MAX_ATTEMPTS + 1):
        try:
            return await client.embed(model=model, input=inputs)
        except Exception as error:
            if (
                attempt == EMBEDDING_TRANSIENT_MAX_ATTEMPTS
                or not _transient_embedding_error(error)
            ):
                raise
            await asyncio.sleep(0.25 * attempt)
    raise AssertionError("embedding retry loop exhausted without returning")


async def augment_mentions_with_ollama_embeddings(
    mentions: Sequence[Any],
    *,
    client: AsyncOllamaEmbedClient,
    identity: OllamaEmbeddingIdentity,
    cache_dir: str | Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Return mention mappings augmented with frozen, auditable embeddings.

    Identical profile texts are embedded once per call. Cache reuse depends on
    the resolved model digest, not its mutable tag. Input and output ordering is
    preserved, while remote batches are formed in deterministic cache-key order.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("embedding batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("embedding batch_size must be positive")
    if isinstance(cache_dir, str) and not cache_dir.strip():
        raise ValueError("embedding cache_dir must be non-empty")

    records: list[dict[str, Any]] = []
    metadata_by_key: dict[str, dict[str, Any]] = {}
    mention_ids: set[str] = set()
    for mention in mentions:
        record = _as_mapping(mention)
        mention_id = str(record.get("mention_id") or "").strip()
        if not mention_id:
            raise ValueError("normalized mention has no non-empty mention_id")
        if mention_id in mention_ids:
            raise ValueError(f"duplicate mention_id for embedding: {mention_id}")
        mention_ids.add(mention_id)
        text = mention_embedding_profile_text(record)
        text_hash = profile_text_sha256(text)
        cache_key = embedding_cache_key(identity.resolved_digest, text_hash)
        records.append(record)
        metadata_by_key.setdefault(
            cache_key,
            {
                "profile_text": text,
                "profile_text_sha256": text_hash,
            },
        )

    embeddings_by_key: dict[str, list[float]] = {}
    cache_hits: dict[str, bool] = {}
    missing_keys: list[str] = []
    for cache_key in sorted(metadata_by_key):
        text_hash = metadata_by_key[cache_key]["profile_text_sha256"]
        path = embedding_cache_path(cache_dir, cache_key)
        cached = _load_cache(
            path,
            cache_key=cache_key,
            identity=identity,
            text_sha256=text_hash,
        )
        if cached is None:
            missing_keys.append(cache_key)
            cache_hits[cache_key] = False
        else:
            embeddings_by_key[cache_key] = cached
            cache_hits[cache_key] = True

    for offset in range(0, len(missing_keys), batch_size):
        batch_keys = missing_keys[offset : offset + batch_size]
        batch_texts = [metadata_by_key[key]["profile_text"] for key in batch_keys]
        response = await _embed_with_transient_retry(
            client,
            model=identity.resolved_tag,
            inputs=batch_texts,
        )
        raw_embeddings = _response_embeddings(response)
        if len(raw_embeddings) != len(batch_keys):
            raise EmbeddingAugmentationError(
                "Ollama embed response cardinality mismatch: "
                f"expected {len(batch_keys)}, found {len(raw_embeddings)}"
            )
        for cache_key, raw_vector in zip(batch_keys, raw_embeddings, strict=True):
            vector = _validated_vector(
                raw_vector,
                expected_dimension=identity.dimension,
                source="Ollama",
            )
            text_hash = metadata_by_key[cache_key]["profile_text_sha256"]
            payload = _cache_payload(
                cache_key=cache_key,
                model_tag=identity.resolved_tag,
                model_digest=identity.resolved_digest,
                text_sha256=text_hash,
                dimension=identity.dimension,
                embedding=vector,
            )
            _write_cache_immutable(embedding_cache_path(cache_dir, cache_key), payload)
            embeddings_by_key[cache_key] = vector

    result: list[dict[str, Any]] = []
    for record in records:
        text = mention_embedding_profile_text(record)
        text_hash = profile_text_sha256(text)
        cache_key = embedding_cache_key(identity.resolved_digest, text_hash)
        result.append(
            {
                **record,
                "embedding": list(embeddings_by_key[cache_key]),
                "embedding_cache_hit": cache_hits[cache_key],
                "embedding_cache_key": cache_key,
                "embedding_dimension": identity.dimension,
                "embedding_model_digest": identity.resolved_digest,
                "embedding_model_tag": identity.resolved_tag,
                "embedding_profile_text_sha256": text_hash,
                "embedding_profile_text_version": EMBEDDING_PROFILE_TEXT_VERSION,
            }
        )
    return result


def _lsh_normal_coordinate(
    *, seed: int, table: int, bit: int, coordinate: int
) -> float:
    """Generate one stateless deterministic standard-normal coordinate."""

    digest = hashlib.sha256(
        f"{seed}\x1f{table}\x1f{bit}\x1f{coordinate}".encode("utf-8")
    ).digest()
    scale = float(1 << 64)
    # Half-unit offsets keep both Box-Muller inputs strictly inside (0, 1).
    uniform_one = (int.from_bytes(digest[:8], "big") + 0.5) / scale
    uniform_two = (int.from_bytes(digest[8:16], "big") + 0.5) / scale
    return math.sqrt(-2.0 * math.log(uniform_one)) * math.cos(math.tau * uniform_two)


def _lsh_bucket_rank(
    *, seed: int, table: int, signature: tuple[bool, ...], mention_id: str
) -> tuple[str, str]:
    signature_text = "".join("1" if value else "0" for value in signature)
    digest = hashlib.sha256(
        f"{seed}\x1f{table}\x1f{signature_text}\x1f{mention_id}".encode("utf-8")
    ).hexdigest()
    return digest, mention_id


def _normalized_lsh_vectors(
    mentions: Sequence[Any],
) -> tuple[list[str], dict[str, tuple[float, ...]]]:
    records: dict[str, tuple[float, ...]] = {}
    expected_dimension: int | None = None
    model_digest: str | None = None
    saw_model_digest = False
    saw_missing_model_digest = False
    for mention in mentions:
        record = _as_mapping(mention)
        mention_id = str(record.get("mention_id") or "").strip()
        if not mention_id:
            raise ValueError("augmented mention has no non-empty mention_id")
        if mention_id in records:
            raise ValueError(f"duplicate mention_id for LSH: {mention_id}")
        raw_vector = record.get("embedding")
        if not isinstance(raw_vector, Sequence) or isinstance(
            raw_vector, (str, bytes, bytearray)
        ):
            raise EmbeddingDimensionError(
                f"mention {mention_id!r} has no embedding sequence"
            )
        if expected_dimension is None:
            expected_dimension = len(raw_vector)
            if expected_dimension <= 0:
                raise EmbeddingDimensionError("LSH embeddings must be non-empty")
        vector = _validated_vector(
            raw_vector,
            expected_dimension=expected_dimension,
            source=f"mention {mention_id!r}",
        )
        declared_dimension = record.get("embedding_dimension")
        if declared_dimension is not None and declared_dimension != expected_dimension:
            raise EmbeddingDimensionError(
                f"mention {mention_id!r} declares embedding dimension "
                f"{declared_dimension!r}, expected {expected_dimension}"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise EmbeddingDimensionError(
                f"mention {mention_id!r} has an all-zero embedding"
            )
        records[mention_id] = tuple(value / norm for value in vector)

        raw_digest = record.get("embedding_model_digest")
        if raw_digest is None or not str(raw_digest).strip():
            saw_missing_model_digest = True
        else:
            saw_model_digest = True
            current_digest = str(raw_digest).strip()
            if model_digest is None:
                model_digest = current_digest
            elif model_digest != current_digest:
                raise ValueError("LSH inputs contain multiple embedding model digests")

    if saw_model_digest and saw_missing_model_digest:
        raise ValueError(
            "LSH inputs mix augmented model lineage with unversioned embeddings"
        )
    return sorted(records), records


@dataclass(frozen=True)
class EmbeddingNeighborEvidence:
    """One ranked directed neighbour backed by one unordered cosine value."""

    mention_id: str
    raw_cosine: float
    rank: int

    def __post_init__(self) -> None:
        mention_id = str(self.mention_id).strip()
        if not mention_id:
            raise ValueError("embedding neighbour mention_id must be non-empty")
        cosine = float(self.raw_cosine)
        if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
            raise ValueError("embedding neighbour cosine must be finite and in [-1, 1]")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("embedding neighbour rank must be a positive integer")
        object.__setattr__(self, "mention_id", mention_id)
        object.__setattr__(self, "raw_cosine", cosine)

    @property
    def scoring_signal(self) -> float:
        return max(0.0, min(1.0, (self.raw_cosine + 1.0) / 2.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "raw_cosine": self.raw_cosine,
            "rank": self.rank,
            "scoring_signal": self.scoring_signal,
        }


def build_lsh_cosine_neighbor_evidence(
    mentions: Sequence[Any],
    *,
    tables: int,
    bits: int,
    k: int,
    seed: int,
    max_bucket: int,
) -> dict[str, tuple[EmbeddingNeighborEvidence, ...]]:
    """Build deterministic ranked LSH evidence with one cosine per pair.

    Each table uses ``bits`` SHA-derived random hyperplanes. Buckets larger
    than ``max_bucket`` are deterministically ranked and partitioned before
    pair generation, so candidate cosine evaluations are bounded by
    ``tables * n * (max_bucket - 1) / 2`` rather than the corpus Cartesian
    product. Final neighbours are ordered by descending cosine, then mention
    ID, and truncated independently to ``k``.
    """

    for name, value in (
        ("tables", tables),
        ("bits", bits),
        ("k", k),
        ("max_bucket", max_bucket),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"LSH {name} must be an integer")
        if value <= 0:
            raise ValueError(f"LSH {name} must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("LSH seed must be an integer")

    mention_ids, vectors = _normalized_lsh_vectors(mentions)
    if not mention_ids:
        return {}
    dimension = len(vectors[mention_ids[0]])
    candidates: dict[str, set[str]] = {mention_id: set() for mention_id in mention_ids}

    for table in range(tables):
        planes = [
            tuple(
                _lsh_normal_coordinate(
                    seed=seed,
                    table=table,
                    bit=bit,
                    coordinate=coordinate,
                )
                for coordinate in range(dimension)
            )
            for bit in range(bits)
        ]
        buckets: dict[tuple[bool, ...], list[str]] = {}
        for mention_id in mention_ids:
            vector = vectors[mention_id]
            signature = tuple(
                sum(
                    value * plane_value
                    for value, plane_value in zip(vector, plane, strict=True)
                )
                >= 0.0
                for plane in planes
            )
            buckets.setdefault(signature, []).append(mention_id)

        for signature, members in sorted(buckets.items()):
            ranked = sorted(
                members,
                key=lambda mention_id: _lsh_bucket_rank(
                    seed=seed,
                    table=table,
                    signature=signature,
                    mention_id=mention_id,
                ),
            )
            for offset in range(0, len(ranked), max_bucket):
                bounded_group = ranked[offset : offset + max_bucket]
                for left, right in combinations(bounded_group, 2):
                    candidates[left].add(right)
                    candidates[right].add(left)

    # Cosine is symmetric. Compute each unordered pair once and feed the same
    # value into both directed top-k rankings.
    scored_by_mention: dict[str, list[tuple[float, str]]] = {
        mention_id: [] for mention_id in mention_ids
    }
    for left in mention_ids:
        for right in sorted(candidates[left]):
            if left >= right:
                continue
            cosine = sum(
                left_value * right_value
                for left_value, right_value in zip(
                    vectors[left], vectors[right], strict=True
                )
            )
            cosine = max(-1.0, min(1.0, cosine))
            scored_by_mention[left].append((cosine, right))
            scored_by_mention[right].append((cosine, left))

    result: dict[str, tuple[EmbeddingNeighborEvidence, ...]] = {}
    for mention_id in mention_ids:
        scored = sorted(
            scored_by_mention[mention_id], key=lambda item: (-item[0], item[1])
        )[:k]
        result[mention_id] = tuple(
            EmbeddingNeighborEvidence(
                mention_id=candidate_id,
                raw_cosine=cosine,
                rank=rank,
            )
            for rank, (cosine, candidate_id) in enumerate(scored, start=1)
        )
    return result


def build_lsh_cosine_neighbor_map(
    mentions: Sequence[Any],
    *,
    tables: int,
    bits: int,
    k: int,
    seed: int,
    max_bucket: int,
) -> dict[str, list[str]]:
    """Compatibility view exposing only neighbour IDs."""

    evidence = build_lsh_cosine_neighbor_evidence(
        mentions,
        tables=tables,
        bits=bits,
        k=k,
        seed=seed,
        max_bucket=max_bucket,
    )
    return {
        mention_id: [item.mention_id for item in neighbours]
        for mention_id, neighbours in evidence.items()
    }


__all__ = [
    "EMBEDDING_CACHE_SCHEMA_VERSION",
    "EMBEDDING_PROFILE_TEXT_VERSION",
    "AsyncOllamaEmbedClient",
    "EmbeddingAugmentationError",
    "EmbeddingCacheError",
    "EmbeddingDimensionError",
    "EmbeddingNeighborEvidence",
    "OllamaEmbeddingIdentity",
    "augment_mentions_with_ollama_embeddings",
    "build_lsh_cosine_neighbor_evidence",
    "build_lsh_cosine_neighbor_map",
    "embedding_cache_key",
    "embedding_cache_path",
    "mention_embedding_profile_text",
    "profile_text_sha256",
]
