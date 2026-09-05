"""Serializable validation reports for experiment stages.

Validators accept mappings, dataclasses, or Pydantic models so saved records
can be checked without depending on a specific stage implementation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import ExperimentConfig
from src.relation_recovery import (
    build_rr_plan,
    canonical_pair,
    graph_edge_pairs,
)

from .lineage import sha256_json


QUALITY_GATE_SCHEMA_VERSION = "2.0.0"
_VALID_RESOLUTION_STATES = {"resolved"}


class GateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GateCheck(GateModel):
    name: str
    passed: bool
    detail: str
    severity: Literal["error", "warning"] = "error"


class QualityGateReport(GateModel):
    schema_version: Literal["2.0.0"] = QUALITY_GATE_SCHEMA_VERSION
    gate: str
    checks: tuple[GateCheck, ...]
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed or check.severity == "warning" for check in self.checks)

    @property
    def errors(self) -> list[str]:
        return [
            f"{check.name}: {check.detail}"
            for check in self.checks
            if not check.passed and check.severity == "error"
        ]

    @property
    def warnings(self) -> list[str]:
        return [
            f"{check.name}: {check.detail}"
            for check in self.checks
            if not check.passed and check.severity == "warning"
        ]

    def require_passed(self) -> "QualityGateReport":
        if not self.passed:
            raise QualityGateViolation(self)
        return self


class QualityGateViolation(RuntimeError):
    def __init__(self, report: QualityGateReport):
        self.report = report
        first = report.errors[0] if report.errors else "unknown failure"
        super().__init__(
            f"{report.gate} quality gate failed "
            f"({len(report.errors)} error(s)): {first}"
        )


class _GateBuilder:
    def __init__(self, gate: str):
        self.gate = gate
        self.checks: list[GateCheck] = []
        self.metrics: dict[str, Any] = {}

    def check(
        self,
        name: str,
        passed: bool,
        detail: str,
        *,
        severity: Literal["error", "warning"] = "error",
    ) -> None:
        self.checks.append(
            GateCheck(
                name=name,
                passed=bool(passed),
                detail=detail,
                severity=severity,
            )
        )

    def finish(self, *, fail_closed: bool) -> QualityGateReport:
        report = QualityGateReport(
            gate=self.gate, checks=tuple(self.checks), metrics=self.metrics
        )
        return report.require_passed() if fail_closed else report


def _value(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _record_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json", exclude_none=False)
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return dataclasses.asdict(record)
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError(f"quality gate record is not mapping-like: {type(record)!r}")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _duplicate_values(values: Sequence[str]) -> list[str]:
    return sorted(
        value for value, count in Counter(values).items() if value and count > 1
    )


def _normalized_display_name(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).casefold()


def _raw_response_sha256(response: Any) -> str:
    if isinstance(response, bytes):
        content = response
    else:
        content = str(response).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def validate_extraction_quality_gate(
    *,
    expected_document_ids: Sequence[str],
    manifest: Any,
    chunks: Sequence[Any],
    extraction_calls: Sequence[Any],
    entities: Sequence[Any],
    relations: Sequence[Any],
    native_capture_version: str | None = None,
    fail_closed: bool = True,
) -> QualityGateReport:
    """Validate the complete raw-to-normalized base extraction snapshot.

    A missing description is measured, never treated as a missing entity.
    """

    gate = _GateBuilder("extraction_snapshot")
    expected_docs = {_text(value) for value in expected_document_ids if _text(value)}
    gate.check(
        "expected_document_set", bool(expected_docs), "expected documents are explicit"
    )

    chunk_ids = [_text(_value(row, "chunk_id", "_id")) for row in chunks]
    chunk_docs = [_text(_value(row, "document_id", "full_doc_id")) for row in chunks]
    duplicate_chunks = _duplicate_values(chunk_ids)
    gate.check(
        "stable_chunk_ids_unique",
        bool(chunk_ids)
        and not any(not value for value in chunk_ids)
        and not duplicate_chunks,
        f"chunks={len(chunk_ids)}, duplicates={duplicate_chunks[:3]}",
    )
    staged_docs = {value for value in chunk_docs if value}
    gate.check(
        "complete_document_snapshot",
        staged_docs == expected_docs,
        f"completed={len(staged_docs)}, expected={len(expected_docs)}, "
        f"missing={sorted(expected_docs - staged_docs)[:3]}, "
        f"unexpected={sorted(staged_docs - expected_docs)[:3]}",
    )
    chunk_to_doc = dict(zip(chunk_ids, chunk_docs, strict=False))
    chunk_complete_failures = [
        chunk_id
        for chunk_id, row in zip(chunk_ids, chunks, strict=False)
        if _value(row, "staging_complete", default=True) is not True
    ]
    gate.check(
        "chunks_staging_complete",
        not chunk_complete_failures,
        f"incomplete chunks={chunk_complete_failures[:3]}",
    )

    manifest_expected_docs = int(_value(manifest, "expected_documents", default=-1))
    manifest_completed_docs = int(_value(manifest, "completed_documents", default=-1))
    manifest_expected_chunks = int(_value(manifest, "expected_chunks", default=-1))
    manifest_staged_chunks = int(_value(manifest, "staged_chunks", default=-1))
    failed_parse_chunks = _strings(_value(manifest, "failed_parse_chunks", default=[]))
    gate.check(
        "manifest_document_counts",
        manifest_expected_docs == len(expected_docs)
        and manifest_completed_docs == manifest_expected_docs,
        f"completed_documents={manifest_completed_docs}, "
        f"expected_documents={manifest_expected_docs}",
    )
    gate.check(
        "manifest_chunk_counts",
        manifest_expected_chunks == len(chunks)
        and manifest_staged_chunks == manifest_expected_chunks,
        f"staged_chunks={manifest_staged_chunks}, expected_chunks={manifest_expected_chunks}, "
        f"records={len(chunks)}",
    )
    gate.check(
        "normalization_parse_success",
        not failed_parse_chunks,
        f"failed parse chunks={failed_parse_chunks[:3]}",
    )

    calls_by_id: dict[str, list[Any]] = defaultdict(list)
    raw_call_errors: list[str] = []
    raw_hashes_by_call: dict[str, set[str]] = defaultdict(set)
    for row in extraction_calls:
        call_id = _text(_value(row, "call_id", "extraction_call_id"))
        chunk_id = _text(_value(row, "chunk_id"))
        document_id = _text(_value(row, "document_id"))
        if not call_id or chunk_id not in chunk_to_doc:
            raw_call_errors.append(call_id or "<missing-call-id>")
            continue
        if document_id and document_id != chunk_to_doc[chunk_id]:
            raw_call_errors.append(call_id)
        response = _value(row, "raw_response", default=None)
        if response is None:
            raw_call_errors.append(call_id)
        else:
            calculated = _raw_response_sha256(response)
            declared = _text(_value(row, "response_sha256"))
            if declared and declared != calculated:
                raw_call_errors.append(call_id)
            raw_hashes_by_call[call_id].add(declared or calculated)
        calls_by_id[call_id].append(row)
    physical_attempt_keys = [
        (
            _text(_value(row, "call_id", "extraction_call_id")),
            int(_value(row, "attempt_number", default=1)),
        )
        for row in extraction_calls
    ]
    duplicate_attempts = [
        f"{call_id}:{attempt}"
        for (call_id, attempt), count in Counter(physical_attempt_keys).items()
        if count > 1
    ]
    gate.check(
        "raw_extraction_responses",
        bool(extraction_calls) and not raw_call_errors and not duplicate_attempts,
        f"calls={len(extraction_calls)}, invalid={raw_call_errors[:3]}, "
        f"duplicate attempts={duplicate_attempts[:3]}",
    )
    if native_capture_version is not None:
        capture_errors: list[str] = []
        for row in extraction_calls:
            if _value(row, "error_type", default=None) is not None:
                continue
            technical = _value(row, "technical_provenance", default={})
            provider = (
                technical.get("provider") if isinstance(technical, Mapping) else None
            )
            capture = (
                provider.get("native_capture")
                if isinstance(provider, Mapping)
                else None
            )
            if (
                not isinstance(capture, Mapping)
                or capture.get("version") != native_capture_version
                or capture.get("response_mutated") is not False
            ):
                capture_errors.append(
                    _text(_value(row, "call_id", default="<missing-call-id>"))
                )
        gate.check(
            "passive_native_capture",
            not capture_errors,
            f"version={native_capture_version}, invalid calls={capture_errors[:3]}",
        )

    chunk_call_errors: list[str] = []
    for row in chunks:
        chunk_id = _text(_value(row, "chunk_id", "_id"))
        document_id = _text(_value(row, "document_id", "full_doc_id"))
        call_ids = _strings(_value(row, "extraction_call_ids", "call_ids", default=[]))
        if not call_ids:
            chunk_call_errors.append(chunk_id)
            continue
        for call_id in call_ids:
            candidates = calls_by_id.get(call_id, [])
            if not candidates or not any(
                _text(_value(item, "chunk_id")) == chunk_id
                and _text(_value(item, "document_id")) in {"", document_id}
                for item in candidates
            ):
                chunk_call_errors.append(chunk_id)
    gate.check(
        "raw_normalized_chunk_lineage",
        not chunk_call_errors,
        f"invalid chunk lineage={sorted(set(chunk_call_errors))[:3]}",
    )

    mention_ids = [_text(_value(row, "mention_id")) for row in entities]
    duplicate_mentions = _duplicate_values(mention_ids)
    entity_errors: list[str] = []
    description_present = 0
    for row, mention_id in zip(entities, mention_ids, strict=False):
        chunk_id = _text(_value(row, "chunk_id"))
        document_id = _text(_value(row, "document_id"))
        call_id = _text(_value(row, "extraction_call_id", "call_id"))
        entity_present = _value(row, "entity_present", default=True)
        declared_response_hash = _text(_value(row, "response_sha256"))
        declared_description_present = bool(
            _value(row, "description_present", default=False)
        )
        actual_description_present = bool(_text(_value(row, "description")))
        if (
            not mention_id
            or entity_present is not True
            or chunk_id not in chunk_to_doc
            or document_id != chunk_to_doc.get(chunk_id)
            or call_id not in calls_by_id
            or not any(
                _text(_value(item, "chunk_id")) == chunk_id
                for item in calls_by_id.get(call_id, [])
            )
            or (
                declared_response_hash
                and declared_response_hash not in raw_hashes_by_call.get(call_id, set())
            )
            or declared_description_present != actual_description_present
        ):
            entity_errors.append(mention_id or "<missing-mention-id>")
        if bool(_value(row, "description_present", default=False)):
            description_present += 1
    gate.check(
        "mention_ids_and_lineage",
        not any(not value for value in mention_ids)
        and not duplicate_mentions
        and not entity_errors,
        f"mentions={len(mention_ids)}, duplicates={duplicate_mentions[:3]}, "
        f"invalid={entity_errors[:3]}",
    )

    known_mentions = set(mention_ids)
    mention_lineage = {
        mention_id: (
            _text(_value(row, "document_id")),
            _text(_value(row, "chunk_id")),
        )
        for mention_id, row in zip(mention_ids, entities, strict=False)
    }
    relation_ids = [_text(_value(row, "relation_id")) for row in relations]
    duplicate_relations = _duplicate_values(relation_ids)
    relation_errors: list[str] = []
    for row, relation_id in zip(relations, relation_ids, strict=False):
        source = _text(_value(row, "source_mention_id"))
        target = _text(_value(row, "target_mention_id"))
        source_state = _text(_value(row, "source_resolution_state"))
        target_state = _text(_value(row, "target_resolution_state"))
        source_candidates = _strings(
            _value(row, "source_candidate_mention_ids", default=[])
        )
        target_candidates = _strings(
            _value(row, "target_candidate_mention_ids", default=[])
        )
        chunk_id = _text(_value(row, "chunk_id"))
        document_id = _text(_value(row, "document_id"))
        call_id = _text(_value(row, "extraction_call_id", "call_id"))
        expected_lineage = (document_id, chunk_id)
        declared_description_present = bool(
            _value(row, "description_present", default=False)
        )
        actual_description_present = bool(_text(_value(row, "description")))
        if (
            not relation_id
            or source_state != "resolved"
            or target_state != "resolved"
            or source not in known_mentions
            or target not in known_mentions
            or source_candidates != [source]
            or target_candidates != [target]
            or chunk_id not in chunk_to_doc
            or document_id != chunk_to_doc.get(chunk_id)
            or call_id not in calls_by_id
            or not any(
                _text(_value(item, "chunk_id")) == chunk_id
                for item in calls_by_id.get(call_id, [])
            )
            or mention_lineage.get(source) != expected_lineage
            or mention_lineage.get(target) != expected_lineage
            or _value(row, "relation_present", "relations_present", default=True)
            is not True
            or declared_description_present != actual_description_present
        ):
            relation_errors.append(relation_id or "<missing-relation-id>")
    gate.check(
        "relation_ids_endpoints_and_lineage",
        not duplicate_relations and not relation_errors,
        f"relations={len(relations)}, duplicates={duplicate_relations[:3]}, "
        f"invalid={relation_errors[:3]}",
    )

    chunk_summary_errors: list[str] = []
    for row in chunks:
        chunk_id = _text(_value(row, "chunk_id", "_id"))
        chunk_entities = [
            item for item in entities if _text(_value(item, "chunk_id")) == chunk_id
        ]
        chunk_relations = [
            item for item in relations if _text(_value(item, "chunk_id")) == chunk_id
        ]
        described = any(
            bool(_value(item, "description_present", default=False))
            for item in chunk_entities
        )
        if (
            bool(_value(row, "entity_present", default=bool(chunk_entities)))
            != bool(chunk_entities)
            or bool(_value(row, "description_present", default=described)) != described
            or bool(_value(row, "relations_present", default=bool(chunk_relations)))
            != bool(chunk_relations)
        ):
            chunk_summary_errors.append(chunk_id)
    gate.check(
        "chunk_presence_summaries",
        not chunk_summary_errors,
        f"inconsistent chunk presence flags={chunk_summary_errors[:3]}",
    )

    gate.metrics.update(
        {
            "expected_documents": len(expected_docs),
            "staged_documents": len(staged_docs),
            "chunks": len(chunks),
            "raw_extraction_attempts": len(extraction_calls),
            "mentions": len(entities),
            "relations": len(relations),
            "description_complete_mentions": description_present,
            "description_completeness": (
                description_present / len(entities) if entities else None
            ),
        }
    )
    return gate.finish(fail_closed=fail_closed)


def _profile_hash(profile: Any) -> str:
    explicit = _value(profile, "profile_hash", default=None)
    if explicit:
        return _text(explicit)
    return sha256_json(_record_dict(profile))


def _judge_identity_hash(identity: Any) -> tuple[dict[str, Any], str]:
    payload = _record_dict(identity)
    payload.pop("config_hash", None)
    explicit_hash = _value(identity, "config_hash", default=None)
    return payload, _text(explicit_hash) or sha256_json(payload)


def validate_er_quality_gate(
    *,
    extraction_report: QualityGateReport,
    config: ExperimentConfig,
    base_extraction_sha256: str,
    profiles: Sequence[Any],
    candidate_pairs: Sequence[Any],
    pair_scores: Sequence[Any],
    pair_decisions: Sequence[Any],
    cannot_links: Sequence[Any],
    clusters: Sequence[Any],
    canonical_entities: Sequence[Any],
    mention_to_canonical: Sequence[Any],
    lineage: Any,
    judge_identity: Any | None = None,
    rewritten_relations: Sequence[Any] = (),
    expected_er_config_hash: str | None = None,
    operational_max_judge_calls_per_run: int | None = None,
    fail_closed: bool = True,
) -> QualityGateReport:
    """Validate a complete corpus-level, mention-level ER plan."""

    gate = _GateBuilder("corpus_entity_resolution")
    gate.check(
        "complete_extraction_prerequisite",
        extraction_report.passed and extraction_report.gate == "extraction_snapshot",
        f"upstream gate={extraction_report.gate}, passed={extraction_report.passed}",
    )
    policy = config.entity_resolution.decision_policy
    effective_judge_budget = (
        policy.max_judge_calls_per_run
        if operational_max_judge_calls_per_run is None
        else operational_max_judge_calls_per_run
    )
    gate.check(
        "decision_policy_frozen",
        policy.version == "expert-fixed-v1"
        and policy.reject_below_score == 0.50
        and policy.auto_merge_at_score == 0.76
        and policy.judge_candidate_method == "reciprocal_embedding_neighbour"
        and policy.judge_mutual_top_k == 1
        and policy.max_judge_calls_per_run == 650,
        str(policy.model_dump(mode="json")),
    )
    gate.check(
        "operational_judge_budget",
        isinstance(effective_judge_budget, int)
        and not isinstance(effective_judge_budget, bool)
        and effective_judge_budget >= policy.max_judge_calls_per_run,
        f"planned={policy.max_judge_calls_per_run}, "
        f"effective={effective_judge_budget}",
    )
    judge = config.roles.er_judge
    gate.check(
        "judge_frozen",
        judge is not None and not judge.unresolved_fields("roles.er_judge"),
        "ER judge tag, resolved name, digest, prompt, temperature and seed are explicit",
    )

    profile_ids = [_text(_value(row, "mention_id")) for row in profiles]
    duplicate_profiles = _duplicate_values(profile_ids)
    profile_set = set(profile_ids)
    gate.check(
        "profile_mentions_unique",
        bool(profile_ids) and not duplicate_profiles and "" not in profile_set,
        f"profiles={len(profiles)}, duplicates={duplicate_profiles[:3]}",
    )
    profile_by_id = {
        mention_id: row
        for mention_id, row in zip(profile_ids, profiles, strict=False)
        if mention_id
    }
    source_mention_by_id: dict[str, Any] = {}
    profile_source_ids: dict[str, frozenset[str]] = {}
    invalid_native_profiles: list[str] = []
    duplicate_source_mentions: list[str] = []
    for profile_id, row in profile_by_id.items():
        raw_sources = _value(row, "source_mentions", default=None)
        sources = (
            list(raw_sources)
            if isinstance(raw_sources, (list, tuple)) and raw_sources
            else [row]
        )
        source_ids: set[str] = set()
        profile_name = _text(_value(row, "original_name"))
        for source in sources:
            source_id = _text(_value(source, "mention_id"))
            source_name = _text(_value(source, "original_name", default=profile_name))
            if (
                not source_id
                or not _text(_value(source, "document_id"))
                or not _text(_value(source, "chunk_id"))
                or source_name != profile_name
            ):
                invalid_native_profiles.append(profile_id)
                continue
            if source_id in source_mention_by_id:
                duplicate_source_mentions.append(source_id)
            source_mention_by_id[source_id] = source
            source_ids.add(source_id)
        if not source_ids:
            invalid_native_profiles.append(profile_id)
        profile_source_ids[profile_id] = frozenset(source_ids)
    source_mention_set = set(source_mention_by_id)
    gate.check(
        "native_entity_groups_exact_and_disjoint",
        not invalid_native_profiles and not duplicate_source_mentions,
        f"native_entities={len(profiles)}, source_mentions={len(source_mention_set)}, "
        f"invalid={invalid_native_profiles[:3]}, "
        f"duplicates={duplicate_source_mentions[:3]}",
    )

    allowed_candidate_methods = {
        "exact_name",
        "containment",
        "fuzzy_name",
        "acronym",
        "embedding_neighbour",
        "reciprocal_embedding_neighbour",
    }
    candidate_ids = [_text(_value(row, "pair_id")) for row in candidate_pairs]
    duplicate_candidate_ids = _duplicate_values(candidate_ids)
    candidate_by_pair = {
        pair_id: row
        for pair_id, row in zip(candidate_ids, candidate_pairs, strict=False)
        if pair_id
    }
    candidate_errors: list[str] = []
    fuzzy_threshold = config.entity_resolution.fuzzy_candidate_threshold
    for row, pair_id in zip(candidate_pairs, candidate_ids, strict=False):
        left = _text(_value(row, "left_mention_id"))
        right = _text(_value(row, "right_mention_id"))
        methods = set(_strings(_value(row, "methods", default=[])))
        route_evidence = _value(row, "route_evidence", default={})
        precomputed = _value(row, "precomputed_signals", default={})
        route_evidence = route_evidence if isinstance(route_evidence, Mapping) else {}
        precomputed = precomputed if isinstance(precomputed, Mapping) else {}
        invalid = (
            not pair_id
            or left not in profile_set
            or right not in profile_set
            or left >= right
            or not methods
            or not methods.issubset(allowed_candidate_methods)
            or set(route_evidence) != methods
            or not set(precomputed).issubset({"lexical", "embedding"})
        )
        parsed_signals: dict[str, float] = {}
        for signal, raw_value in precomputed.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                invalid = True
                continue
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                invalid = True
            parsed_signals[str(signal)] = value

        fuzzy_evidence = route_evidence.get("fuzzy_name", {})
        if "fuzzy_name" in methods:
            try:
                lexical = float(_value(fuzzy_evidence, "lexical_similarity"))
                recorded_threshold = float(_value(fuzzy_evidence, "threshold"))
            except (TypeError, ValueError):
                invalid = True
            else:
                invalid = invalid or (
                    "lexical" not in parsed_signals
                    or abs(parsed_signals["lexical"] - lexical) > 1e-12
                    or abs(recorded_threshold - fuzzy_threshold) > 1e-12
                    or lexical < recorded_threshold
                )
        elif "lexical" in parsed_signals:
            invalid = True

        embedding_evidence = route_evidence.get("embedding_neighbour", {})
        if "embedding_neighbour" in methods:
            try:
                raw_cosine = float(_value(embedding_evidence, "raw_cosine"))
                scoring_signal = float(_value(embedding_evidence, "scoring_signal"))
            except (TypeError, ValueError):
                invalid = True
            else:
                invalid = invalid or (
                    "embedding" not in parsed_signals
                    or abs(parsed_signals["embedding"] - scoring_signal) > 1e-12
                    or abs(scoring_signal - max(0.0, min(1.0, (raw_cosine + 1) / 2)))
                    > 1e-12
                )
        elif "embedding" in parsed_signals:
            invalid = True

        if "reciprocal_embedding_neighbour" in methods:
            reciprocal_evidence = route_evidence.get(
                "reciprocal_embedding_neighbour", {}
            )
            invalid = invalid or (
                "embedding_neighbour" not in methods
                or not bool(_value(embedding_evidence, "left_selected_right"))
                or not bool(_value(embedding_evidence, "right_selected_left"))
                or _value(reciprocal_evidence, "left_rank", default=None) is None
                or _value(reciprocal_evidence, "right_rank", default=None) is None
            )
        if invalid:
            candidate_errors.append(pair_id or "<missing-pair-id>")
    gate.check(
        "candidate_route_evidence_auditable",
        not duplicate_candidate_ids and not candidate_errors,
        f"candidates={len(candidate_pairs)}, duplicates={duplicate_candidate_ids[:3]}, "
        f"invalid={candidate_errors[:3]}",
    )

    mapping_ids = [_text(_value(row, "mention_id")) for row in mention_to_canonical]
    duplicate_mappings = _duplicate_values(mapping_ids)
    invalid_states = sorted(
        {
            _text(_value(row, "resolution_state", default="resolved"))
            for row in mention_to_canonical
            if _text(_value(row, "resolution_state", default="resolved"))
            not in _VALID_RESOLUTION_STATES
        }
    )
    gate.check(
        "one_resolution_per_mention",
        set(mapping_ids) == source_mention_set
        and not duplicate_mappings
        and not invalid_states,
        f"mapping={len(mapping_ids)}, source_mentions={len(source_mention_set)}, "
        f"duplicates={duplicate_mappings[:3]}, invalid_states={invalid_states}",
    )

    cluster_membership: dict[str, str] = {}
    cluster_duplicates: list[str] = []
    empty_clusters: list[str] = []
    cluster_sets: list[frozenset[str]] = []
    for row in clusters:
        cluster_id = _text(_value(row, "cluster_id", "canonical_entity_id"))
        members = _strings(_value(row, "mention_ids", "source_mention_ids", default=[]))
        if not cluster_id or not members:
            empty_clusters.append(cluster_id or "<missing-cluster-id>")
        cluster_sets.append(
            frozenset(
                source_id
                for profile_id in members
                for source_id in profile_source_ids.get(profile_id, frozenset())
            )
        )
        for mention_id in members:
            if mention_id in cluster_membership:
                cluster_duplicates.append(mention_id)
            cluster_membership[mention_id] = cluster_id
    gate.check(
        "clusters_disjoint_complete",
        set(cluster_membership) == profile_set
        and not cluster_duplicates
        and not empty_clusters,
        f"clusters={len(clusters)}, duplicate_members={cluster_duplicates[:3]}, "
        f"invalid={empty_clusters[:3]}",
    )

    canonical_ids = [
        _text(_value(row, "canonical_entity_id")) for row in canonical_entities
    ]
    canonical_names = [
        _normalized_display_name(_value(row, "display_name", "entity_name", "name"))
        for row in canonical_entities
    ]
    duplicate_canonical_ids = _duplicate_values(canonical_ids)
    duplicate_canonical_names = _duplicate_values(canonical_names)
    canonical_sources: dict[str, frozenset[str]] = {}
    canonical_errors: list[str] = []
    for row, canonical_id in zip(canonical_entities, canonical_ids, strict=False):
        sources = frozenset(
            _strings(_value(row, "source_mention_ids", "mentions", default=[]))
        )
        documents = _strings(
            _value(row, "source_document_ids", "source_documents", default=[])
        )
        aliases_value = _value(row, "aliases", default=None)
        aliases_are_a_sequence = isinstance(aliases_value, (list, tuple))
        rationale = _text(_value(row, "selection_rationale", "rationale"))
        if (
            not canonical_id
            or not sources
            or not documents
            or not aliases_are_a_sequence
            or not rationale
            or not sources.issubset(source_mention_set)
        ):
            canonical_errors.append(canonical_id or "<missing-canonical-id>")
        canonical_sources[canonical_id] = sources
    gate.check(
        "canonical_entities_auditable_unique",
        len(canonical_entities) == len(clusters)
        and not duplicate_canonical_ids
        and not duplicate_canonical_names
        and not any(not value for value in canonical_names)
        and not canonical_errors
        and sorted(canonical_sources.values(), key=lambda value: sorted(value))
        == sorted(cluster_sets, key=lambda value: sorted(value)),
        f"canonical={len(canonical_entities)}, duplicate_ids={duplicate_canonical_ids[:3]}, "
        f"duplicate_names={duplicate_canonical_names[:3]}, invalid={canonical_errors[:3]}",
    )
    canonical_id_set = set(canonical_ids)
    mapping_errors: list[str] = []
    for row in mention_to_canonical:
        mention_id = _text(_value(row, "mention_id"))
        canonical_id = _text(_value(row, "canonical_entity_id"))
        source_mention = source_mention_by_id.get(mention_id)
        if (
            canonical_id not in canonical_id_set
            or mention_id not in canonical_sources.get(canonical_id, frozenset())
            or source_mention is None
            or _text(_value(row, "document_id"))
            != _text(_value(source_mention, "document_id"))
            or _text(_value(row, "chunk_id"))
            != _text(_value(source_mention, "chunk_id"))
        ):
            mapping_errors.append(mention_id)
    gate.check(
        "mention_level_mapping",
        not mapping_errors,
        f"invalid mention mappings={mapping_errors[:3]}",
    )
    canonical_by_mention = {
        _text(_value(row, "mention_id")): _text(_value(row, "canonical_entity_id"))
        for row in mention_to_canonical
    }
    split_native_groups = [
        profile_id
        for profile_id, source_ids in profile_source_ids.items()
        if len({canonical_by_mention.get(source_id) for source_id in source_ids}) != 1
    ]
    gate.check(
        "native_entities_are_never_split",
        not split_native_groups,
        f"split_native_entities={split_native_groups[:3]}",
    )
    gate.check(
        "merge_only_entity_count_nonincreasing",
        len(canonical_entities) <= len(profiles),
        f"native_entities={len(profiles)}, canonical_entities={len(canonical_entities)}",
    )

    cannot_link_violations: list[str] = []
    invalid_cannot_links: list[str] = []
    cannot_link_pairs: set[tuple[str, str]] = set()
    for row in cannot_links:
        left = _text(_value(row, "left_mention_id"))
        right = _text(_value(row, "right_mention_id"))
        pair = tuple(sorted((left, right)))
        cannot_link_pairs.add(pair)
        if left not in profile_set or right not in profile_set or left == right:
            invalid_cannot_links.append(f"{left}|{right}")
        if cluster_membership.get(left) == cluster_membership.get(right):
            cannot_link_violations.append(f"{left}|{right}")
    gate.check(
        "cannot_links_respected",
        not invalid_cannot_links and not cannot_link_violations,
        f"invalid={invalid_cannot_links[:3]}, violated={cannot_link_violations[:3]}",
    )

    score_ids = [_text(_value(row, "pair_id")) for row in pair_scores]
    score_by_pair = {
        pair_id: row for pair_id, row in zip(score_ids, pair_scores, strict=False)
    }
    duplicate_score_ids = _duplicate_values(score_ids)
    candidate_score_errors: list[str] = []
    for pair_id, score_row in score_by_pair.items():
        candidate = candidate_by_pair.get(pair_id)
        signals = _value(score_row, "signals", default={})
        precomputed = _value(candidate, "precomputed_signals", default={})
        signals = signals if isinstance(signals, Mapping) else {}
        precomputed = precomputed if isinstance(precomputed, Mapping) else {}
        signal_mismatch = False
        for signal, raw_value in precomputed.items():
            try:
                score_value = float(signals[signal])
                candidate_value = float(raw_value)
            except (KeyError, TypeError, ValueError):
                signal_mismatch = True
                break
            if (
                not isfinite(score_value)
                or not isfinite(candidate_value)
                or abs(score_value - candidate_value) > 1e-12
            ):
                signal_mismatch = True
                break
        if (
            candidate is None
            or _text(_value(candidate, "left_mention_id"))
            != _text(_value(score_row, "left_mention_id"))
            or _text(_value(candidate, "right_mention_id"))
            != _text(_value(score_row, "right_mention_id"))
            or signal_mismatch
        ):
            candidate_score_errors.append(pair_id)
    gate.check(
        "candidate_scores_reuse_route_signals",
        set(candidate_ids) == set(score_ids)
        and not duplicate_score_ids
        and not candidate_score_errors,
        f"candidates={len(candidate_pairs)}, scores={len(pair_scores)}, "
        f"invalid={candidate_score_errors[:3]}",
    )
    decision_ids = [_text(_value(row, "pair_id")) for row in pair_decisions]
    duplicate_decisions = _duplicate_values(decision_ids)
    decision_errors: list[str] = []
    judge_decisions: list[Any] = []
    reject_below = policy.reject_below_score
    auto_merge_at = policy.auto_merge_at_score
    for row in pair_decisions:
        pair_id = _text(_value(row, "pair_id"))
        left = _text(_value(row, "left_mention_id"))
        right = _text(_value(row, "right_mention_id"))
        action = _text(_value(row, "action"))
        source = _text(_value(row, "source"))
        score_row = score_by_pair.get(pair_id)
        score = _value(score_row, "aggregate_score", "score", default=None)
        if (
            score_row is None
            or left not in profile_set
            or right not in profile_set
            or action not in {"merge", "reject", "abstain"}
            or source
            not in {
                "score_policy_merge",
                "score_policy_reject",
                "judge",
                "judge_admission",
                "no_judge",
            }
            or (
                source == "score_policy_merge"
                and (action != "merge" or score is None or score < auto_merge_at)
            )
            or (
                source == "score_policy_reject"
                and (action != "reject" or score is None or score >= reject_below)
            )
            or (
                source == "judge"
                and (score is None or not reject_below <= score < auto_merge_at)
            )
            or (
                source == "judge_admission"
                and (
                    action != "abstain"
                    or score is None
                    or not reject_below <= score < auto_merge_at
                )
            )
            or (
                source == "no_judge"
                and score is not None
                and not reject_below <= score < auto_merge_at
            )
            or (
                action == "reject"
                and tuple(sorted((left, right))) not in cannot_link_pairs
            )
        ):
            decision_errors.append(pair_id or "<missing-pair-id>")
        if source == "judge":
            judge_decisions.append(row)
    gate.check(
        "pair_decisions_policy",
        set(decision_ids) == set(score_ids)
        and not duplicate_score_ids
        and not duplicate_decisions
        and not decision_errors,
        f"scores={len(pair_scores)}, decisions={len(pair_decisions)}, "
        f"invalid={decision_errors[:3]}, duplicate_decisions={duplicate_decisions[:3]}",
    )
    gate.check(
        "judge_usage_within_budget",
        len(judge_decisions) <= effective_judge_budget,
        f"judge decisions={len(judge_decisions)}, "
        f"planned_max_judge_calls_per_run={policy.max_judge_calls_per_run}, "
        f"effective_max_judge_calls_per_run={effective_judge_budget}",
    )

    judge_cache_errors: list[str] = []
    expected_identity_payload = judge.identity_payload() if judge is not None else None
    expected_identity_hash = (
        sha256_json(expected_identity_payload)
        if expected_identity_payload is not None
        else ""
    )
    if judge_decisions and judge_identity is None:
        judge_cache_errors.append("judge identity used for decisions was not supplied")
    elif judge_decisions and judge_identity is not None:
        identity_payload, identity_hash = _judge_identity_hash(judge_identity)
        configured_tags = {
            value
            for value in (
                judge.requested_tag if judge else None,
                judge.resolved_name if judge else None,
            )
            if value
        }
        if (
            _text(identity_payload.get("model_tag")) not in configured_tags
            or not judge
            or _text(identity_payload.get("model_digest")) != judge.digest
            or _text(identity_payload.get("prompt_version")) != judge.prompt_version
            or identity_payload.get("temperature") != judge.temperature
            or identity_payload.get("seed") != judge.seed
            or identity_hash != expected_identity_hash
        ):
            judge_cache_errors.append(
                "runtime judge identity differs from frozen config"
            )
        generation = identity_payload.get("generation_parameters")
        generation = generation if isinstance(generation, Mapping) else {}
        context_window = generation.get("context_window", generation.get("num_ctx"))
        output_tokens = generation.get("output_tokens", generation.get("num_predict"))
        if (
            context_window != judge.context_window
            or output_tokens != judge.output_tokens
            or generation != expected_identity_payload["generation_parameters"]
        ):
            judge_cache_errors.append(
                "runtime judge generation parameters differ from frozen config"
            )
        for row in judge_decisions:
            pair_id = _text(_value(row, "pair_id"))
            left = _text(_value(row, "left_mention_id"))
            right = _text(_value(row, "right_mention_id"))
            metadata = _value(row, "judge_metadata", default={}) or {}
            score_row = score_by_pair.get(pair_id)
            expected_key = sha256_json(
                {
                    "judge": identity_payload,
                    "judge_config_hash": identity_hash,
                    "profiles": sorted(
                        (
                            {
                                "mention_id": left,
                                "profile_hash": _profile_hash(profile_by_id[left]),
                            },
                            {
                                "mention_id": right,
                                "profile_hash": _profile_hash(profile_by_id[right]),
                            },
                        ),
                        key=lambda item: item["mention_id"],
                    ),
                    "pair_score": _record_dict(score_row),
                }
            )
            if (
                _text(_value(row, "judge_cache_key")) != expected_key
                or _text(_value(metadata, "judge_digest"))
                != (judge.digest if judge else "")
                or _text(_value(metadata, "judge_prompt_version"))
                != (judge.prompt_version if judge else "")
                or _text(_value(metadata, "judge_config_hash"))
                != expected_identity_hash
            ):
                judge_cache_errors.append(pair_id)
    gate.check(
        "judge_cache_profile_lineage",
        not judge_cache_errors,
        f"judge decisions={len(judge_decisions)}, invalid={judge_cache_errors[:3]}",
    )

    lineage_errors: list[str] = []
    if lineage is None:
        lineage_errors.append("missing ER lineage")
    else:
        expected_er_hash = expected_er_config_hash or sha256_json(
            config.entity_resolution.pipeline_payload()
        )
        if (
            _text(_value(lineage, "base_extraction_hash", "base_extraction_sha256"))
            != base_extraction_sha256
        ):
            lineage_errors.append("base extraction hash")
        if (
            _text(_value(lineage, "corpus_manifest_hash", "corpus_manifest_sha256"))
            != config.corpus.manifest_sha256
        ):
            lineage_errors.append("corpus manifest hash")
        if _text(_value(lineage, "er_version")) != config.entity_resolution.version:
            lineage_errors.append("ER version")
        if (
            _text(_value(lineage, "er_config_hash", "er_config_sha256"))
            != expected_er_hash
        ):
            lineage_errors.append("ER config hash")
        if judge is not None:
            configured_tags = {judge.requested_tag, judge.resolved_name}
            if _text(_value(lineage, "judge_tag")) not in configured_tags:
                lineage_errors.append("judge tag")
            if _text(_value(lineage, "judge_digest")) != judge.digest:
                lineage_errors.append("judge digest")
            if _text(_value(lineage, "judge_prompt_version")) != judge.prompt_version:
                lineage_errors.append("judge prompt version")
            if _text(_value(lineage, "judge_config_hash")) != expected_identity_hash:
                lineage_errors.append("judge config hash")
        input_hashes = _value(lineage, "input_hashes", default={})
        if not isinstance(input_hashes, Mapping) or not input_hashes:
            lineage_errors.append("input hashes")
        else:
            required_input_hashes = {
                "normalized_chunks",
                "normalized_entities",
                "normalized_relations",
                "embedding_model_digest",
                "embedding_candidate_config",
            }
            missing_input_hashes = sorted(required_input_hashes - set(input_hashes))
            if missing_input_hashes:
                lineage_errors.append(
                    "missing input hashes: " + ", ".join(missing_input_hashes)
                )
            invalid_hashes = sorted(
                str(key)
                for key, value in input_hashes.items()
                if re.fullmatch(r"[0-9a-f]{64}", _text(value)) is None
            )
            if invalid_hashes:
                lineage_errors.append(
                    "invalid input SHA-256 values: " + ", ".join(invalid_hashes)
                )
            if (
                _text(input_hashes.get("embedding_model_digest"))
                != config.roles.embedding.digest
            ):
                lineage_errors.append("embedding model digest")
            embedding_options = config.entity_resolution.embedding_candidate_payload()
            expected_embedding_candidate_hash = sha256_json(
                {
                    "k": embedding_options["embedding_neighbor_k"],
                    "tables": embedding_options["embedding_lsh_tables"],
                    "bits": embedding_options["embedding_lsh_bits"],
                    "max_bucket": embedding_options["embedding_lsh_max_bucket"],
                    "seed": config.extraction.seed,
                }
            )
            if (
                _text(input_hashes.get("embedding_candidate_config"))
                != expected_embedding_candidate_hash
            ):
                lineage_errors.append("embedding candidate config")
    gate.check(
        "er_lineage_frozen",
        not lineage_errors,
        f"invalid lineage fields={lineage_errors}",
    )

    valid_endpoints = canonical_id_set.union(
        _text(_value(row, "display_name", "entity_name", "name"))
        for row in canonical_entities
    )
    rewritten_errors: list[str] = []
    for row in rewritten_relations:
        source = _text(
            _value(row, "source_canonical_entity_id", "source_entity_id", "source")
        )
        target = _text(
            _value(row, "target_canonical_entity_id", "target_entity_id", "target")
        )
        if source not in valid_endpoints or target not in valid_endpoints:
            rewritten_errors.append(
                _text(_value(row, "relation_id", default="<unknown>"))
            )
    gate.check(
        "rewritten_relation_endpoints",
        not rewritten_errors,
        f"unresolved rewritten relations={rewritten_errors[:3]}",
    )

    gate.metrics.update(
        {
            "mentions": len(source_mention_set),
            "native_entities": len(profiles),
            "canonical_entities": len(canonical_entities),
            "candidate_pairs": len(candidate_pairs),
            "candidate_scores": len(pair_scores),
            "pair_decisions": len(pair_decisions),
            "cannot_links": len(cannot_links),
            "clusters": len(clusters),
            "judge_decisions": len(judge_decisions),
            "abstentions": sum(
                _text(_value(row, "action")) == "abstain" for row in pair_decisions
            ),
        }
    )
    return gate.finish(fail_closed=fail_closed)


def _node_id(row: Any) -> str:
    return _text(
        _value(row, "id", "entity_id", "canonical_entity_id", "entity_name", "name")
    )


def validate_rr_quality_gate(
    *,
    config: ExperimentConfig,
    chunks: Sequence[Any],
    er_nodes: Sequence[Mapping[str, Any]],
    er_edges: Sequence[Mapping[str, Any]],
    mention_to_canonical: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
    plan_summary: Mapping[str, Any],
    chunk_results: Sequence[Mapping[str, Any]],
    rr_nodes: Sequence[Mapping[str, Any]] | None = None,
    rr_edges: Sequence[Mapping[str, Any]] | None = None,
    fail_closed: bool = True,
) -> QualityGateReport:
    """Recompute the frozen RR universe and validate cached verifier results."""

    gate = _GateBuilder("relation_recovery")
    expected_rows, expected_summary = build_rr_plan(
        chunks=chunks,
        nodes=er_nodes,
        edges=er_edges,
        mention_to_canonical=mention_to_canonical,
        candidate_policy_version=config.relation_recovery.candidate_policy_version,
    )
    gate.check(
        "candidate_plan_deterministic",
        list(plan_rows) == expected_rows
        and _text(plan_summary.get("rr_plan_sha256"))
        == _text(expected_summary.get("rr_plan_sha256")),
        f"saved={plan_summary.get('rr_plan_sha256')}, "
        f"recomputed={expected_summary.get('rr_plan_sha256')}",
    )
    eligible_by_chunk = {
        _text(row.get("chunk_id")): row for row in plan_rows if row.get("eligible")
    }
    result_ids = [_text(row.get("chunk_id")) for row in chunk_results]
    gate.check(
        "one_result_per_eligible_chunk",
        set(result_ids) == set(eligible_by_chunk)
        and not _duplicate_values(result_ids)
        and "" not in result_ids,
        f"eligible={len(eligible_by_chunk)}, results={len(chunk_results)}, "
        f"duplicates={_duplicate_values(result_ids)[:3]}",
    )

    verifier = config.roles.rr_verifier
    expected_identity = verifier.identity_payload() if verifier is not None else {}
    frozen_pairs = graph_edge_pairs(er_edges)
    invalid_results: list[str] = []
    accepted: list[Mapping[str, Any]] = []
    seen_instances: set[tuple[str, str, str]] = set()
    for result in chunk_results:
        chunk_id = _text(result.get("chunk_id"))
        plan = eligible_by_chunk.get(chunk_id)
        verification = result.get("verification", result)
        if not isinstance(verification, Mapping) or plan is None:
            invalid_results.append(chunk_id or "<missing-chunk-id>")
            continue
        status = _text(verification.get("status"))
        relations = verification.get("accepted_relations")
        errors = verification.get("errors")
        if not isinstance(relations, list) or not isinstance(errors, list):
            invalid_results.append(chunk_id)
            continue
        # Invalid output is allowed only when it contributes no relation. It
        # remains visible in the validation metrics.
        if (
            status not in {"valid", "invalid"}
            or (status == "invalid" and (relations or not errors))
            or (status == "valid" and errors)
        ):
            invalid_results.append(chunk_id)
            continue
        for relation in relations:
            if not isinstance(relation, Mapping):
                invalid_results.append(chunk_id)
                continue
            try:
                pair = canonical_pair(
                    _text(relation.get("entity_a_id")),
                    _text(relation.get("entity_b_id")),
                )
            except ValueError:
                invalid_results.append(chunk_id)
                continue
            local_ids = {
                _text(item.get("canonical_entity_id"))
                for item in plan.get("canonical_entities", [])
                if isinstance(item, Mapping)
            }
            evidence = _text(relation.get("evidence_quote"))
            chunk_text = next(
                (
                    str(_value(chunk, "text", default=""))
                    for chunk in chunks
                    if _text(_value(chunk, "chunk_id")) == chunk_id
                ),
                "",
            )
            instance = (chunk_id, pair[0], pair[1])
            if (
                pair in frozen_pairs
                or pair[0] not in local_ids
                or pair[1] not in local_ids
                or not _text(relation.get("relationship_description"))
                or not evidence
                or evidence not in chunk_text
                or relation.get("verifier_identity") != expected_identity
                or instance in seen_instances
            ):
                invalid_results.append(chunk_id)
                continue
            seen_instances.add(instance)
            accepted.append(relation)
    gate.check(
        "verifier_results_evidence_bound",
        not invalid_results,
        f"invalid_chunks={sorted(set(invalid_results))[:3]}",
    )

    if rr_nodes is not None or rr_edges is not None:
        rr_node_ids = {
            _text(row.get("canonical_entity_id")) for row in (rr_nodes or [])
        }
        er_node_ids = {_text(row.get("canonical_entity_id")) for row in er_nodes}
        rr_pairs = graph_edge_pairs(rr_edges or [])
        gate.check(
            "rr_preserves_er_nodes",
            rr_node_ids == er_node_ids,
            f"er={len(er_node_ids)}, er_rr={len(rr_node_ids)}",
        )
        gate.check(
            "rr_graph_is_er_edge_superset",
            frozen_pairs.issubset(rr_pairs),
            f"er={len(frozen_pairs)}, er_rr={len(rr_pairs)}, "
            f"missing={sorted(frozen_pairs - rr_pairs)[:3]}",
        )
    gate.metrics.update(
        {
            "chunks": len(plan_rows),
            "eligible_chunks": len(eligible_by_chunk),
            "verifier_results": len(chunk_results),
            "invalid_model_outputs_rejected": sum(
                _text((row.get("verification", row) or {}).get("status")) == "invalid"
                for row in chunk_results
                if isinstance(row.get("verification", row), Mapping)
            ),
            "accepted_candidate_instances": len(accepted),
            "recovered_pairs": len(
                {
                    canonical_pair(
                        _text(row.get("entity_a_id")),
                        _text(row.get("entity_b_id")),
                    )
                    for row in accepted
                }
            ),
        }
    )
    return gate.finish(fail_closed=fail_closed)


def _edge_endpoints(row: Any) -> tuple[str, str]:
    return _text(_value(row, "source", "src_id")), _text(
        _value(row, "target", "tgt_id")
    )


def validate_final_workspace_quality_gate(
    *,
    graph_regime: Literal[
        "native_lightrag", "advanced_lightrag_er", "advanced_lightrag_er_rr"
    ],
    base_extraction_sha256: str,
    expected_base_extraction_sha256: str,
    nodes: Sequence[Any],
    edges: Sequence[Any],
    chunks: Sequence[Any],
    expected_chunk_ids: Sequence[str],
    induced_self_loops: Sequence[Any],
    preexisting_self_loops: Sequence[Any] = (),
    workspace_clean: bool,
    persisted: bool,
    finalized: bool,
    reopened: bool,
    export_succeeded: bool,
    smoke_retrieval_answerable: bool,
    smoke_retrieval_unanswerable: bool,
    native_workspace_sha256_before: str | None = None,
    native_workspace_sha256_after: str | None = None,
    fail_closed: bool = True,
) -> QualityGateReport:
    """Validate a persisted/reopened graph workspace without touching storage."""

    gate = _GateBuilder("final_workspace")
    gate.check(
        "shared_base_extraction",
        bool(base_extraction_sha256)
        and base_extraction_sha256 == expected_base_extraction_sha256,
        f"actual={base_extraction_sha256}, expected={expected_base_extraction_sha256}",
    )
    gate.check("clean_workspace", workspace_clean, f"workspace_clean={workspace_clean}")
    gate.check(
        "persistence_lifecycle",
        persisted and finalized and reopened and export_succeeded,
        f"persisted={persisted}, finalized={finalized}, reopened={reopened}, "
        f"export={export_succeeded}",
    )

    node_ids = [_node_id(row) for row in nodes]
    duplicate_nodes = _duplicate_values(node_ids)
    gate.check(
        "exported_nodes_unique",
        bool(node_ids)
        and not duplicate_nodes
        and not any(not value for value in node_ids),
        f"nodes={len(nodes)}, duplicates={duplicate_nodes[:3]}",
    )
    node_set = set(node_ids)
    dangling: list[str] = []
    self_loops: list[str] = []
    duplicate_edges: list[str] = []
    edge_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(edges):
        source, target = _edge_endpoints(row)
        if source not in node_set or target not in node_set:
            dangling.append(str(index))
        if source == target:
            self_loops.append(str(index))
        key = tuple(sorted((source, target)))
        if key in edge_keys:
            duplicate_edges.append(f"{source}|{target}")
        edge_keys.add(key)
    gate.check(
        "no_dangling_or_duplicate_edges",
        not dangling and not duplicate_edges,
        f"edges={len(edges)}, dangling={dangling[:3]}, duplicate={duplicate_edges[:3]}",
    )
    allowed_preexisting_names = {
        _text(_value(row, "canonical_display_name", "entity_name", "name"))
        for row in preexisting_self_loops
    }
    allowed_preexisting_names.discard("")
    retained_self_loop_names = {
        _edge_endpoints(edges[int(index)])[0] for index in self_loops
    }
    unlogged_self_loops = sorted(retained_self_loop_names - allowed_preexisting_names)
    gate.check(
        "er_induced_self_loops_removed",
        graph_regime == "native_lightrag" or not unlogged_self_loops,
        f"retained={sorted(retained_self_loop_names)[:3]}, "
        f"unlogged={unlogged_self_loops[:3]}",
    )
    invalid_loop_logs = [
        str(index)
        for index, row in enumerate(induced_self_loops)
        if not _text(_value(row, "relation_id", "source_relation_id"))
        or not _text(
            _value(row, "canonical_entity_id", "source", "target", "entity_name")
        )
    ]
    gate.check(
        "induced_self_loops_audited",
        not invalid_loop_logs,
        f"logged={len(induced_self_loops)}, invalid logs={invalid_loop_logs[:3]}",
    )
    invalid_preexisting_logs = [
        str(index)
        for index, row in enumerate(preexisting_self_loops)
        if not _text(_value(row, "relation_id", "source_relation_id"))
        or not _text(
            _value(
                row,
                "canonical_entity_id",
                "canonical_display_name",
                "source",
                "target",
            )
        )
    ]
    gate.check(
        "preexisting_self_loops_retained_only",
        not invalid_preexisting_logs
        and (
            graph_regime == "native_lightrag"
            or allowed_preexisting_names == retained_self_loop_names
        ),
        f"logged_relations={len(preexisting_self_loops)}, "
        f"retained_nodes={len(retained_self_loop_names)}, "
        f"invalid_logs={invalid_preexisting_logs[:3]}",
    )

    expected_chunks = {_text(value) for value in expected_chunk_ids if _text(value)}
    actual_chunks = {_text(_value(row, "chunk_id", "_id")) for row in chunks}
    actual_chunks.discard("")
    gate.check(
        "exact_chunk_snapshot_reopened",
        bool(expected_chunks) and actual_chunks == expected_chunks,
        f"actual={len(actual_chunks)}, expected={len(expected_chunks)}, "
        f"missing={sorted(expected_chunks - actual_chunks)[:3]}, "
        f"unexpected={sorted(actual_chunks - expected_chunks)[:3]}",
    )
    if graph_regime != "native_lightrag":
        native_unchanged = (
            bool(native_workspace_sha256_before)
            and native_workspace_sha256_before == native_workspace_sha256_after
        )
        detail = (
            f"before={native_workspace_sha256_before}, "
            f"after={native_workspace_sha256_after}"
        )
    else:
        native_unchanged = True
        detail = "native build is the baseline workspace"
    gate.check("native_workspace_unchanged", native_unchanged, detail)
    gate.check(
        "smoke_retrieval",
        smoke_retrieval_answerable and smoke_retrieval_unanswerable,
        f"answerable={smoke_retrieval_answerable}, "
        f"unanswerable={smoke_retrieval_unanswerable}",
    )

    gate.metrics.update(
        {
            "nodes": len(nodes),
            "edges": len(edges),
            "chunks": len(chunks),
            "dangling_edges": len(dangling),
            "duplicate_edges": len(duplicate_edges),
            "induced_self_loops_removed": len(induced_self_loops),
        }
    )
    return gate.finish(fail_closed=fail_closed)


__all__ = [
    "GateCheck",
    "QUALITY_GATE_SCHEMA_VERSION",
    "QualityGateReport",
    "QualityGateViolation",
    "validate_er_quality_gate",
    "validate_extraction_quality_gate",
    "validate_final_workspace_quality_gate",
    "validate_rr_quality_gate",
]
