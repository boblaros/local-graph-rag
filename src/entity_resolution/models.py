"""Typed contracts for corpus-level merge-only entity resolution.

Native LightRAG entities are immutable starting groups.  ER may merge those
groups through alias evidence, but it never splits the exact-name aggregation
already performed by the Native baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Any, Literal, Mapping, Sequence


SCHEMA_VERSION = "1.3.0"
DEFAULT_ER_VERSION = "corpus-native-merge-only-er-v8"
DecisionAction = Literal["merge", "reject", "abstain"]


def canonical_json(value: Any) -> str:
    """Return deterministic JSON without importing orchestration utilities."""

    import json

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    digest = sha256(
        "\x1f".join(canonical_json(part) for part in parts).encode("utf-8")
    ).hexdigest()
    return f"{prefix}{digest[:length]}"


def stable_mention_id(document_id: str, chunk_id: str, mention_index: int) -> str:
    if mention_index < 0:
        raise ValueError("mention_index must be non-negative")
    return stable_id("mention_", document_id, chunk_id, mention_index)


def stable_relation_id(document_id: str, chunk_id: str, relation_index: int) -> str:
    if relation_index < 0:
        raise ValueError("relation_index must be non-negative")
    return stable_id("relation_", document_id, chunk_id, relation_index)


def _nonempty(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _string_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    else:
        values = tuple(value)
    result = tuple(
        dict.fromkeys(str(item).strip() for item in values if str(item).strip())
    )
    return result


@dataclass(frozen=True)
class EntityMention:
    mention_id: str
    document_id: str
    chunk_id: str
    original_name: str
    entity_type: str | None = None
    description: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extraction_call_id: str | None = None
    parse_status: str = "parsed"
    entity_present: bool = True
    description_present: bool | None = None
    embedding: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mention_id", _nonempty(self.mention_id, "mention_id"))
        object.__setattr__(
            self, "document_id", _nonempty(self.document_id, "document_id")
        )
        object.__setattr__(self, "chunk_id", _nonempty(self.chunk_id, "chunk_id"))
        object.__setattr__(
            self, "original_name", _nonempty(self.original_name, "original_name")
        )
        object.__setattr__(self, "entity_type", _optional_text(self.entity_type))
        object.__setattr__(self, "description", _optional_text(self.description))
        object.__setattr__(
            self, "extraction_call_id", _optional_text(self.extraction_call_id)
        )
        object.__setattr__(
            self, "parse_status", _nonempty(self.parse_status, "parse_status")
        )
        object.__setattr__(self, "provenance", dict(self.provenance))
        if not self.entity_present:
            raise ValueError(
                "ER profiles may only be built from entity_present=True records"
            )
        actual_description = self.description is not None
        if self.description_present is None:
            object.__setattr__(self, "description_present", actual_description)
        elif bool(self.description_present) != actual_description:
            raise ValueError("description_present does not match description")
        if self.embedding is not None:
            vector = tuple(float(item) for item in self.embedding)
            if not vector:
                raise ValueError("embedding must be non-empty when supplied")
            object.__setattr__(self, "embedding", vector)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EntityMention":
        return cls(
            mention_id=str(value["mention_id"]),
            document_id=str(value["document_id"]),
            chunk_id=str(value["chunk_id"]),
            original_name=str(value.get("original_name") or value.get("name") or ""),
            entity_type=value.get("entity_type", value.get("type")),
            description=value.get("description"),
            provenance=dict(value.get("provenance") or {}),
            extraction_call_id=value.get("extraction_call_id"),
            parse_status=str(value.get("parse_status") or "parsed"),
            entity_present=bool(value.get("entity_present", True)),
            description_present=value.get("description_present"),
            embedding=(
                tuple(value["embedding"])
                if value.get("embedding") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationMention:
    relation_id: str
    source_mention_id: str
    target_mention_id: str
    document_id: str
    chunk_id: str
    keywords: tuple[str, ...] | None = None
    relation_type: str | None = None
    description: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extraction_call_id: str | None = None
    relations_present: bool = True

    def __post_init__(self) -> None:
        for name in (
            "relation_id",
            "source_mention_id",
            "target_mention_id",
            "document_id",
            "chunk_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(self, "keywords", _string_tuple(self.keywords))
        object.__setattr__(self, "relation_type", _optional_text(self.relation_type))
        object.__setattr__(self, "description", _optional_text(self.description))
        object.__setattr__(
            self, "extraction_call_id", _optional_text(self.extraction_call_id)
        )
        object.__setattr__(self, "provenance", dict(self.provenance))
        if not self.relations_present:
            raise ValueError("ER relations must have relations_present=True")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RelationMention":
        source_state = value.get("source_resolution_state", "resolved")
        target_state = value.get("target_resolution_state", "resolved")
        if source_state != "resolved" or target_state != "resolved":
            raise ValueError(
                "ER cannot consume a relation with ambiguous/unresolved endpoints"
            )
        raw_keywords = value.get("keywords")
        if isinstance(raw_keywords, str):
            raw_keywords = tuple(
                part.strip()
                for part in raw_keywords.replace(";", ",").split(",")
                if part.strip()
            )
        return cls(
            relation_id=str(value["relation_id"]),
            source_mention_id=str(value["source_mention_id"]),
            target_mention_id=str(value["target_mention_id"]),
            document_id=str(value["document_id"]),
            chunk_id=str(value["chunk_id"]),
            keywords=tuple(raw_keywords) if raw_keywords is not None else None,
            relation_type=value.get("relation_type", value.get("type")),
            description=value.get("description"),
            provenance=dict(value.get("provenance") or {}),
            extraction_call_id=value.get("extraction_call_id"),
            relations_present=bool(value.get("relations_present", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntityProfile:
    mention_id: str
    document_id: str
    chunk_id: str
    original_name: str
    normalized_name: str
    entity_type: str | None
    type_family: str | None
    description: str | None
    neighbours: tuple[str, ...] | None
    relation_context: tuple[str, ...] | None
    mention_frequency: int
    source_diversity: int
    provenance: Mapping[str, Any]
    extraction_call_id: str | None
    embedding: tuple[float, ...] | None = None
    source_mentions: tuple[Mapping[str, str | None], ...] = ()

    def __post_init__(self) -> None:
        if self.mention_frequency < 1 or self.source_diversity < 1:
            raise ValueError("profile frequency/diversity must be positive")
        object.__setattr__(self, "neighbours", _string_tuple(self.neighbours))
        object.__setattr__(
            self, "relation_context", _string_tuple(self.relation_context)
        )
        object.__setattr__(self, "provenance", dict(self.provenance))
        source_mentions = self.source_mentions or (
            {
                "mention_id": self.mention_id,
                "document_id": self.document_id,
                "chunk_id": self.chunk_id,
                "original_name": self.original_name,
                "extraction_call_id": self.extraction_call_id,
            },
        )
        normalized_sources: list[dict[str, str | None]] = []
        source_ids: set[str] = set()
        for source in source_mentions:
            record = {
                "mention_id": _nonempty(
                    str(source.get("mention_id") or ""), "source mention_id"
                ),
                "document_id": _nonempty(
                    str(source.get("document_id") or ""), "source document_id"
                ),
                "chunk_id": _nonempty(
                    str(source.get("chunk_id") or ""), "source chunk_id"
                ),
                "original_name": _nonempty(
                    str(source.get("original_name") or ""), "source original_name"
                ),
                "extraction_call_id": _optional_text(source.get("extraction_call_id")),
            }
            if record["mention_id"] in source_ids:
                raise ValueError("source mention IDs must be unique within a profile")
            if record["original_name"] != self.original_name:
                raise ValueError(
                    "a Native entity profile may contain only one exact original name"
                )
            source_ids.add(str(record["mention_id"]))
            normalized_sources.append(record)
        object.__setattr__(
            self,
            "source_mentions",
            tuple(sorted(normalized_sources, key=lambda item: str(item["mention_id"]))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_hash(self) -> str:
        return content_hash(self.to_dict())

    @property
    def source_mention_ids(self) -> tuple[str, ...]:
        return tuple(str(item["mention_id"]) for item in self.source_mentions)

    @property
    def source_document_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({str(item["document_id"]) for item in self.source_mentions})
        )

    @property
    def source_chunk_ids(self) -> tuple[str, ...]:
        return tuple(sorted({str(item["chunk_id"]) for item in self.source_mentions}))


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    left_mention_id: str
    right_mention_id: str
    methods: tuple[str, ...]
    route_evidence: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    precomputed_signals: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.left_mention_id >= self.right_mention_id:
            raise ValueError(
                "candidate endpoints must be stored in deterministic order"
            )
        methods = tuple(sorted(set(self.methods)))
        evidence = {
            str(method): dict(payload)
            for method, payload in sorted(self.route_evidence.items())
        }
        unexpected_evidence = sorted(set(evidence) - set(methods))
        if unexpected_evidence:
            raise ValueError(
                "candidate route evidence lacks a matching method: "
                f"{unexpected_evidence}"
            )
        signals: dict[str, float] = {}
        for name, raw_value in sorted(self.precomputed_signals.items()):
            if name not in {"lexical", "embedding"}:
                raise ValueError(f"unsupported precomputed candidate signal: {name}")
            value = float(raw_value)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"precomputed candidate signal {name} must be finite and in [0, 1]"
                )
            signals[name] = value
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "route_evidence", evidence)
        object.__setattr__(self, "precomputed_signals", signals)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairScore:
    pair_id: str
    left_mention_id: str
    right_mention_id: str
    aggregate_score: float | None
    signals: Mapping[str, float | None]
    effective_weights: Mapping[str, float]
    unavailable_signals: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.aggregate_score is not None and not 0.0 <= self.aggregate_score <= 1.0:
            raise ValueError("aggregate_score must be in [0, 1]")
        object.__setattr__(self, "signals", dict(self.signals))
        object.__setattr__(self, "effective_weights", dict(self.effective_weights))
        object.__setattr__(
            self, "unavailable_signals", tuple(sorted(set(self.unavailable_signals)))
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairDecision:
    pair_id: str
    left_mention_id: str
    right_mention_id: str
    action: DecisionAction
    source: Literal[
        "score_policy_merge",
        "score_policy_reject",
        "judge",
        "judge_admission",
        "no_judge",
    ]
    score: float | None
    rationale: str
    judge_cache_key: str | None = None
    judge_cache_hit: bool | None = None
    judge_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in {"merge", "reject", "abstain"}:
            raise ValueError(f"invalid decision action: {self.action}")
        if self.source not in {
            "score_policy_merge",
            "score_policy_reject",
            "judge",
            "judge_admission",
            "no_judge",
        }:
            raise ValueError(f"invalid decision source: {self.source}")
        object.__setattr__(self, "judge_metadata", dict(self.judge_metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CannotLink:
    left_mention_id: str
    right_mention_id: str
    reason: str
    pair_id: str | None = None
    source: str = "decision"

    def __post_init__(self) -> None:
        left, right = sorted((self.left_mention_id, self.right_mention_id))
        if left == right:
            raise ValueError("cannot-link endpoints must differ")
        object.__setattr__(self, "left_mention_id", left)
        object.__setattr__(self, "right_mention_id", right)
        object.__setattr__(self, "reason", _nonempty(self.reason, "reason"))

    @property
    def key(self) -> tuple[str, str]:
        return self.left_mention_id, self.right_mention_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    mention_ids: tuple[str, ...]
    merge_evidence_pair_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        members = tuple(sorted(set(self.mention_ids)))
        if not members:
            raise ValueError("cluster must contain at least one mention")
        object.__setattr__(self, "mention_ids", members)
        object.__setattr__(
            self,
            "merge_evidence_pair_ids",
            tuple(sorted(set(self.merge_evidence_pair_ids))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalEntity:
    canonical_entity_id: str
    display_name: str
    aliases: tuple[str, ...]
    entity_type: str | None
    merged_description: str | None
    evidence: tuple[Mapping[str, Any], ...]
    source_mention_ids: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    selection_rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "display_name", _nonempty(self.display_name, "display_name")
        )
        object.__setattr__(
            self, "aliases", tuple(sorted(set(self.aliases), key=str.casefold))
        )
        object.__setattr__(
            self, "evidence", tuple(dict(item) for item in self.evidence)
        )
        object.__setattr__(
            self, "source_mention_ids", tuple(sorted(set(self.source_mention_ids)))
        )
        object.__setattr__(
            self, "source_document_ids", tuple(sorted(set(self.source_document_ids)))
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MentionResolution:
    document_id: str
    chunk_id: str
    mention_id: str
    canonical_entity_id: str
    resolution_state: Literal["resolved"] = "resolved"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusSnapshot:
    expected_document_ids: tuple[str, ...]
    completed_document_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...] = ()
    completed_chunk_ids: tuple[str, ...] = ()

    def validate_complete(self) -> None:
        expected_docs = tuple(sorted(set(self.expected_document_ids)))
        completed_docs = tuple(sorted(set(self.completed_document_ids)))
        if not expected_docs:
            raise ValueError("corpus snapshot has no expected documents")
        if expected_docs != completed_docs:
            missing = sorted(set(expected_docs) - set(completed_docs))
            extra = sorted(set(completed_docs) - set(expected_docs))
            raise IncompleteCorpusError(
                f"ER requires a complete corpus snapshot; missing={missing}, extra={extra}"
            )
        if self.expected_chunk_ids:
            expected_chunks = tuple(sorted(set(self.expected_chunk_ids)))
            completed_chunks = tuple(sorted(set(self.completed_chunk_ids)))
            if expected_chunks != completed_chunks:
                missing = sorted(set(expected_chunks) - set(completed_chunks))
                extra = sorted(set(completed_chunks) - set(expected_chunks))
                raise IncompleteCorpusError(
                    f"ER requires all staged chunks; missing={missing}, extra={extra}"
                )


class IncompleteCorpusError(RuntimeError):
    """Raised when ER is attempted before the complete corpus is staged."""


@dataclass(frozen=True)
class ERLineage:
    base_run_id: str
    base_extraction_hash: str
    corpus_manifest_hash: str
    input_hashes: Mapping[str, str]
    er_version: str = DEFAULT_ER_VERSION
    er_config_hash: str | None = None
    judge_tag: str | None = None
    judge_digest: str | None = None
    judge_prompt_version: str | None = None
    judge_config_hash: str | None = None
    merge_plan_hash: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "base_run_id",
            "base_extraction_hash",
            "corpus_manifest_hash",
            "er_version",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        input_hashes = {
            str(key).strip(): str(value).strip()
            for key, value in self.input_hashes.items()
        }
        if not input_hashes or any(
            not key or not value for key, value in input_hashes.items()
        ):
            raise ValueError(
                "input_hashes must contain non-empty artifact hash lineage"
            )
        object.__setattr__(self, "input_hashes", dict(sorted(input_hashes.items())))
        judge_fields = (
            self.judge_tag,
            self.judge_digest,
            self.judge_prompt_version,
            self.judge_config_hash,
        )
        if any(judge_fields) and not all(judge_fields):
            raise ValueError(
                "judge lineage must include tag, digest, prompt version, and config hash"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ERDecisionPolicy:
    """Frozen, expert-defined pair decision policy for the main experiment."""

    version: str = "expert-fixed-v1"
    reject_below_score: float = 0.50
    auto_merge_at_score: float = 0.76
    judge_candidate_method: str = "reciprocal_embedding_neighbour"
    judge_mutual_top_k: int = 1
    max_judge_calls_per_run: int = 650

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("decision policy version must be non-empty")
        if not 0.0 <= self.reject_below_score < self.auto_merge_at_score <= 1.0:
            raise ValueError(
                "decision policy scores must satisfy "
                "0 <= reject_below_score < auto_merge_at_score <= 1"
            )
        if not self.judge_candidate_method.strip():
            raise ValueError("judge_candidate_method must be non-empty")
        if self.judge_mutual_top_k < 1:
            raise ValueError("judge_mutual_top_k must be positive")
        if self.max_judge_calls_per_run < 0:
            raise ValueError("max_judge_calls_per_run must be non-negative")


@dataclass(frozen=True)
class ERConfig:
    decision_policy: ERDecisionPolicy = field(default_factory=ERDecisionPolicy)
    fuzzy_candidate_threshold: float = 0.70
    containment_min_chars: int = 4
    max_block_size: int = 200
    er_version: str = DEFAULT_ER_VERSION
    scoring_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "name_exact": 0.24,
            "lexical": 0.15,
            "acronym": 0.09,
            "containment": 0.08,
            "description": 0.12,
            "embedding": 0.12,
            "type": 0.08,
            "neighbourhood": 0.05,
            "relation_context": 0.04,
            "provenance": 0.01,
            "frequency": 0.01,
            "source_diversity": 0.01,
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.decision_policy, ERDecisionPolicy):
            raise TypeError("decision_policy must be an ERDecisionPolicy")
        if not 0.0 <= self.fuzzy_candidate_threshold <= 1.0:
            raise ValueError("fuzzy_candidate_threshold must be in [0, 1]")
        if self.containment_min_chars < 1 or self.max_block_size < 2:
            raise ValueError("candidate limits must be positive")
        weights = {
            str(key): float(value) for key, value in self.scoring_weights.items()
        }
        if (
            not weights
            or any(value < 0 for value in weights.values())
            or sum(weights.values()) <= 0
        ):
            raise ValueError(
                "scoring_weights must contain non-negative weights with positive sum"
            )
        object.__setattr__(self, "scoring_weights", dict(sorted(weights.items())))

    @property
    def config_hash(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_mapping(value: Any, *, record_type: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", exclude_none=False)
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"{record_type} must be an ER record, mapping, or Pydantic model")


def coerce_mentions(
    values: Sequence[EntityMention | Mapping[str, Any] | Any],
) -> list[EntityMention]:
    return [
        value
        if isinstance(value, EntityMention)
        else EntityMention.from_mapping(_model_mapping(value, record_type="mention"))
        for value in values
    ]


def coerce_relations(
    values: Sequence[RelationMention | Mapping[str, Any] | Any],
) -> list[RelationMention]:
    return [
        value
        if isinstance(value, RelationMention)
        else RelationMention.from_mapping(_model_mapping(value, record_type="relation"))
        for value in values
    ]
