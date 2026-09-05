"""Corpus-level ER lifecycle and immutable, auditable artifact emission."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidates import generate_candidate_pairs
from .canonicalization import canonicalize_clusters
from .clustering import BlockedMerge, constrained_cluster
from .decisions import decide_pairs
from .embeddings import EmbeddingNeighborEvidence
from .judge import FixedJudge, JudgeCache
from .models import (
    CandidatePair,
    CannotLink,
    CanonicalEntity,
    Cluster,
    CorpusSnapshot,
    ERConfig,
    ERLineage,
    EntityMention,
    EntityProfile,
    MentionResolution,
    PairDecision,
    PairScore,
    RelationMention,
    canonical_json,
    coerce_mentions,
    coerce_relations,
    content_hash,
)
from .profiles import build_native_entity_profiles
from .scoring import score_candidate_pairs


JSONL_ARTIFACTS = (
    "entity_profiles.jsonl",
    "candidate_pairs.jsonl",
    "pair_scores.jsonl",
    "pair_decisions.jsonl",
    "cannot_links.jsonl",
    "clusters.jsonl",
    "canonical_entities.jsonl",
    "mention_to_canonical.jsonl",
    "aliases.jsonl",
)
JSON_ARTIFACTS = ("merge_plan.json", "summary.json")
ER_ARTIFACTS = JSONL_ARTIFACTS + JSON_ARTIFACTS


class ArtifactConflictError(RuntimeError):
    """An immutable artifact path already contains different content."""


@dataclass(frozen=True)
class ERResult:
    lineage: ERLineage
    profiles: tuple[EntityProfile, ...]
    candidates: tuple[CandidatePair, ...]
    scores: tuple[PairScore, ...]
    decisions: tuple[PairDecision, ...]
    cannot_links: tuple[CannotLink, ...]
    clusters: tuple[Cluster, ...]
    blocked_merges: tuple[BlockedMerge, ...]
    canonical_entities: tuple[CanonicalEntity, ...]
    mention_to_canonical: tuple[MentionResolution, ...]
    merge_plan: Mapping[str, Any]
    summary: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]


def _complete_snapshot_gate(
    snapshot: CorpusSnapshot,
    mentions: Sequence[EntityMention],
    relations: Sequence[RelationMention],
) -> None:
    snapshot.validate_complete()
    expected_docs = set(snapshot.expected_document_ids)
    expected_chunks = set(snapshot.expected_chunk_ids)
    for record in (*mentions, *relations):
        if record.document_id not in expected_docs:
            raise ValueError(
                f"staged ER record references document outside corpus snapshot: {record.document_id}"
            )
        if expected_chunks and record.chunk_id not in expected_chunks:
            raise ValueError(
                f"staged ER record references chunk outside corpus snapshot: {record.chunk_id}"
            )


def _validated_lineage(
    lineage: ERLineage,
    config: ERConfig,
    judge: FixedJudge | None,
) -> ERLineage:
    if lineage.er_version != config.er_version:
        raise ValueError(
            f"ER version mismatch: lineage={lineage.er_version}, config={config.er_version}"
        )
    if lineage.er_config_hash not in (None, config.config_hash):
        raise ValueError("ER config hash does not match frozen lineage")
    result = replace(lineage, er_config_hash=config.config_hash)
    if judge is None:
        return result
    identity = judge.identity
    values = {
        "judge_tag": identity.model_tag,
        "judge_digest": identity.model_digest,
        "judge_prompt_version": identity.prompt_version,
        "judge_config_hash": identity.config_hash,
    }
    for key, value in values.items():
        frozen = getattr(result, key)
        if frozen not in (None, value):
            raise ValueError(f"{key} does not match the fixed judge")
    return replace(result, **values)


def _record_envelope(
    artifact_type: str,
    lineage: ERLineage,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": artifact_type,
        "schema_version": lineage.schema_version,
        "er_version": lineage.er_version,
        "lineage": lineage.to_dict(),
        "record_hash": content_hash(record),
        "record": dict(record),
    }


def _jsonl_content(
    artifact_type: str,
    lineage: ERLineage,
    records: Iterable[Mapping[str, Any]],
) -> str:
    # The header retains lineage even when a valid artifact has zero records.
    header = {
        "record_type": "artifact_header",
        "artifact_type": artifact_type,
        "schema_version": lineage.schema_version,
        "er_version": lineage.er_version,
        "lineage": lineage.to_dict(),
    }
    lines = [canonical_json(header)]
    lines.extend(
        canonical_json(_record_envelope(artifact_type, lineage, record))
        for record in records
    )
    return "\n".join(lines) + "\n"


def _json_content(
    artifact_type: str,
    lineage: ERLineage,
    payload: Mapping[str, Any],
) -> str:
    envelope = {
        "artifact_type": artifact_type,
        "schema_version": lineage.schema_version,
        "er_version": lineage.er_version,
        "lineage": lineage.to_dict(),
        "payload_hash": content_hash(payload),
        "payload": dict(payload),
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _write_immutable_artifacts(
    directory: Path,
    contents: Mapping[str, str],
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    unexpected = set(contents) - set(ER_ARTIFACTS)
    missing = set(ER_ARTIFACTS) - set(contents)
    if unexpected or missing:
        raise ValueError(
            f"artifact set mismatch: missing={sorted(missing)}, extra={sorted(unexpected)}"
        )

    # Check all collisions before writing anything, preserving an existing run.
    for filename, content in contents.items():
        path = directory / filename
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ArtifactConflictError(
                f"refusing to overwrite immutable ER artifact: {path}"
            )

    hashes: dict[str, str] = {}
    for filename, content in sorted(contents.items()):
        path = directory / filename
        if not path.exists():
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                if path.read_text(encoding="utf-8") != content:
                    raise ArtifactConflictError(
                        f"concurrent immutable artifact conflict: {path}"
                    )
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(content)
        hashes[filename] = sha256(content.encode("utf-8")).hexdigest()
    return hashes


def _merge_plan_payload(
    decisions: Sequence[PairDecision],
    cannot_links: Sequence[CannotLink],
    clusters: Sequence[Cluster],
    blocked_merges: Sequence[BlockedMerge],
    canonical_entities: Sequence[CanonicalEntity],
    mapping: Sequence[MentionResolution],
) -> dict[str, Any]:
    return {
        "decision_counts": {
            action: sum(decision.action == action for decision in decisions)
            for action in ("merge", "reject", "abstain")
        },
        "cannot_links": [link.to_dict() for link in cannot_links],
        "clusters": [cluster.to_dict() for cluster in clusters],
        "blocked_merges": [blocked.to_dict() for blocked in blocked_merges],
        "canonical_entities": [
            {
                "canonical_entity_id": entity.canonical_entity_id,
                "display_name": entity.display_name,
                "aliases": entity.aliases,
                "source_mention_ids": entity.source_mention_ids,
            }
            for entity in canonical_entities
        ],
        "mention_to_canonical": [item.to_dict() for item in mapping],
    }


def _summary_payload(
    *,
    profiles: Sequence[EntityProfile],
    mention_count: int,
    relations: Sequence[RelationMention],
    candidates: Sequence[CandidatePair],
    decisions: Sequence[PairDecision],
    cannot_links: Sequence[CannotLink],
    clusters: Sequence[Cluster],
    blocked_merges: Sequence[BlockedMerge],
    canonical_entities: Sequence[CanonicalEntity],
    merge_plan_hash: str,
    judge_budget: Mapping[str, Any],
    runtime_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    judge_decisions = [item for item in decisions if item.source == "judge"]
    native_cluster_sizes = Counter(len(cluster.mention_ids) for cluster in clusters)
    mention_cluster_sizes = Counter(
        len(entity.source_mention_ids) for entity in canonical_entities
    )
    aliases = sum(len(entity.aliases) for entity in canonical_entities)
    payload: dict[str, Any] = {
        "mentions": mention_count,
        "native_entities": len(profiles),
        "relations": len(relations),
        "canonical_entities": len(canonical_entities),
        "candidate_count": len(candidates),
        "merge_count": sum(item.action == "merge" for item in decisions),
        "reject_count": sum(item.action == "reject" for item in decisions),
        "abstain_count": sum(item.action == "abstain" for item in decisions),
        "abstention_rate": (
            sum(item.action == "abstain" for item in decisions) / len(decisions)
            if decisions
            else 0.0
        ),
        "judge_count": len(judge_decisions),
        "judge_rate": len(judge_decisions) / len(decisions) if decisions else 0.0,
        "judge_budget": dict(judge_budget),
        # Cache hits are an operational property of a particular invocation,
        # not part of the scientific ER result.  Recording them here would
        # make a resume conflict with the immutable first-run artifacts.
        # Actual calls, cache hits, and token counts are saved in runtime.json.
        "judge_cache_hits": None,
        "judge_cache_misses": None,
        "cannot_link_count": len(cannot_links),
        "blocked_merge_count": len(blocked_merges),
        "native_entity_cluster_size_distribution": {
            str(size): count for size, count in sorted(native_cluster_sizes.items())
        },
        "mention_cluster_size_distribution": {
            str(size): count for size, count in sorted(mention_cluster_sizes.items())
        },
        "alias_count": aliases,
        "alias_coverage": (
            sum(bool(entity.aliases) for entity in canonical_entities)
            / len(canonical_entities)
            if canonical_entities
            else 0.0
        ),
        "merge_plan_hash": merge_plan_hash,
        "er_runtime_seconds": None,
        "judge_calls": None,
        "judge_tokens": None,
        "judge_cost": None,
    }
    if runtime_metrics:
        payload.update(runtime_metrics)
    return payload


class EntityResolutionPipeline:
    """Reusable offline ER pipeline that refuses partial-corpus execution."""

    def __init__(self, config: ERConfig | None = None) -> None:
        self.config = config or ERConfig()

    def run(
        self,
        *,
        snapshot: CorpusSnapshot,
        mentions: Sequence[EntityMention | Mapping[str, Any]],
        relations: Sequence[RelationMention | Mapping[str, Any]],
        lineage: ERLineage,
        artifact_dir: str | Path,
        judge: FixedJudge | None = None,
        judge_cache: JudgeCache | None = None,
        embedding_neighbors: Mapping[str, Iterable[str | EmbeddingNeighborEvidence]]
        | None = None,
        embedded_native_entities: Sequence[Mapping[str, Any]] | None = None,
        runtime_metrics: Mapping[str, Any] | None = None,
        operational_max_judge_calls_per_run: int | None = None,
    ) -> ERResult:
        mention_records = coerce_mentions(mentions)
        relation_records = coerce_relations(relations)
        _complete_snapshot_gate(snapshot, mention_records, relation_records)
        frozen_lineage = _validated_lineage(lineage, self.config, judge)

        profiles = build_native_entity_profiles(
            mention_records,
            relation_records,
            embedded_native_entities=embedded_native_entities,
        )
        candidates = generate_candidate_pairs(
            profiles,
            self.config,
            embedding_neighbors=embedding_neighbors,
        )
        scores = score_candidate_pairs(candidates, profiles, self.config)
        decision_batch = decide_pairs(
            scores,
            profiles,
            self.config,
            judge=judge,
            judge_cache=judge_cache,
            candidates=candidates,
            operational_max_judge_calls_per_run=(
                operational_max_judge_calls_per_run
            ),
        )
        cannot_links = decision_batch.cannot_links
        cluster_result = constrained_cluster(
            profiles,
            decision_batch.decisions,
            cannot_links=cannot_links,
        )
        canonical = canonicalize_clusters(
            cluster_result.clusters,
            profiles,
            namespace=frozen_lineage.base_extraction_hash,
        )
        merge_plan = _merge_plan_payload(
            decision_batch.decisions,
            cannot_links,
            cluster_result.clusters,
            cluster_result.blocked_merges,
            canonical.canonical_entities,
            canonical.mention_to_canonical,
        )
        merge_plan_hash = content_hash(merge_plan)
        if frozen_lineage.merge_plan_hash not in (None, merge_plan_hash):
            raise ValueError("merge plan hash does not match frozen lineage")
        frozen_lineage = replace(frozen_lineage, merge_plan_hash=merge_plan_hash)
        summary = _summary_payload(
            profiles=profiles,
            mention_count=len(mention_records),
            relations=relation_records,
            candidates=candidates,
            decisions=decision_batch.decisions,
            cannot_links=cannot_links,
            clusters=cluster_result.clusters,
            blocked_merges=cluster_result.blocked_merges,
            canonical_entities=canonical.canonical_entities,
            merge_plan_hash=merge_plan_hash,
            judge_budget=decision_batch.judge_budget,
            runtime_metrics=runtime_metrics,
        )

        alias_records = [
            {
                "canonical_entity_id": entity.canonical_entity_id,
                "display_name": entity.display_name,
                "aliases": entity.aliases,
                "source_mention_ids": entity.source_mention_ids,
            }
            for entity in canonical.canonical_entities
        ]
        record_sets: dict[str, list[Mapping[str, Any]]] = {
            "entity_profiles.jsonl": [
                {**profile.to_dict(), "profile_hash": profile.profile_hash}
                for profile in profiles
            ],
            "candidate_pairs.jsonl": [item.to_dict() for item in candidates],
            "pair_scores.jsonl": [item.to_dict() for item in scores],
            "pair_decisions.jsonl": [
                {
                    **item.to_dict(),
                    # See _summary_payload: cache-hit state belongs to the
                    # mutable runtime report, while the cache key and frozen
                    # judge/profile lineage remain auditable here.
                    "judge_cache_hit": None,
                }
                for item in decision_batch.decisions
            ],
            "cannot_links.jsonl": [item.to_dict() for item in cannot_links],
            "clusters.jsonl": [item.to_dict() for item in cluster_result.clusters],
            "canonical_entities.jsonl": [
                item.to_dict() for item in canonical.canonical_entities
            ],
            "mention_to_canonical.jsonl": [
                item.to_dict() for item in canonical.mention_to_canonical
            ],
            "aliases.jsonl": alias_records,
        }
        contents = {
            filename: _jsonl_content(
                filename.removesuffix(".jsonl"), frozen_lineage, records
            )
            for filename, records in record_sets.items()
        }
        contents["merge_plan.json"] = _json_content(
            "merge_plan", frozen_lineage, merge_plan
        )
        contents["summary.json"] = _json_content("summary", frozen_lineage, summary)
        artifact_hashes = _write_immutable_artifacts(Path(artifact_dir), contents)
        return ERResult(
            lineage=frozen_lineage,
            profiles=tuple(profiles),
            candidates=tuple(candidates),
            scores=tuple(scores),
            decisions=decision_batch.decisions,
            cannot_links=cannot_links,
            clusters=cluster_result.clusters,
            blocked_merges=cluster_result.blocked_merges,
            canonical_entities=canonical.canonical_entities,
            mention_to_canonical=canonical.mention_to_canonical,
            merge_plan=merge_plan,
            summary=summary,
            artifact_hashes=artifact_hashes,
        )
