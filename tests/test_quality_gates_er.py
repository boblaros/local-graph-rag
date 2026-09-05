from __future__ import annotations

from copy import deepcopy

from src.orchestration.lineage import sha256_json
from src.orchestration.quality_gates import (
    GateCheck,
    QualityGateReport,
    validate_er_quality_gate,
)

from ._config_helpers import resolved_config


def _passed_extraction() -> QualityGateReport:
    return QualityGateReport(
        gate="extraction_snapshot",
        checks=(GateCheck(name="complete", passed=True, detail="complete"),),
    )


def _valid_er_payload(
    *, score: float = 0.9, source: str = "score_policy_merge"
) -> dict:
    config = resolved_config()
    embedding_options = config.entity_resolution.embedding_candidate_payload()
    embedding_candidate_hash = sha256_json(
        {
            "k": embedding_options["embedding_neighbor_k"],
            "tables": embedding_options["embedding_lsh_tables"],
            "bits": embedding_options["embedding_lsh_bits"],
            "max_bucket": embedding_options["embedding_lsh_max_bucket"],
            "seed": config.extraction.seed,
        }
    )
    profiles = [
        {
            "mention_id": "d1:c1:0",
            "document_id": "d1",
            "chunk_id": "c1",
            "original_name": "Apple Inc.",
            "profile_hash": "1" * 64,
        },
        {
            "mention_id": "d2:c2:0",
            "document_id": "d2",
            "chunk_id": "c2",
            "original_name": "Apple",
            "profile_hash": "2" * 64,
        },
    ]
    pair_score = {
        "pair_id": "p1",
        "left_mention_id": "d1:c1:0",
        "right_mention_id": "d2:c2:0",
        "aggregate_score": score,
        "signals": {"name_exact": 0.9},
        "effective_weights": {"name_exact": 1.0},
        "unavailable_signals": [],
    }
    return {
        "extraction_report": _passed_extraction(),
        "config": config,
        "base_extraction_sha256": "a" * 64,
        "profiles": profiles,
        "candidate_pairs": [
            {
                "pair_id": "p1",
                "left_mention_id": "d1:c1:0",
                "right_mention_id": "d2:c2:0",
                "methods": ["containment"],
                "route_evidence": {
                    "containment": {
                        "contained_name": "apple",
                        "container_name": "apple inc",
                        "minimum_chars": config.entity_resolution.containment_min_chars,
                    }
                },
                "precomputed_signals": {},
            }
        ],
        "pair_scores": [pair_score],
        "pair_decisions": [
            {
                "pair_id": "p1",
                "left_mention_id": "d1:c1:0",
                "right_mention_id": "d2:c2:0",
                "action": "merge",
                "source": source,
                "score": score,
                "rationale": "same company",
            }
        ],
        "cannot_links": [],
        "clusters": [
            {"cluster_id": "cluster-1", "mention_ids": ["d1:c1:0", "d2:c2:0"]}
        ],
        "canonical_entities": [
            {
                "canonical_entity_id": "ce-apple-company",
                "display_name": "Apple Inc.",
                "aliases": ["Apple", "Apple Inc."],
                "source_mention_ids": ["d1:c1:0", "d2:c2:0"],
                "source_document_ids": ["d1", "d2"],
                "selection_rationale": "most explicit extracted company name",
            }
        ],
        "mention_to_canonical": [
            {
                "document_id": "d1",
                "chunk_id": "c1",
                "mention_id": "d1:c1:0",
                "canonical_entity_id": "ce-apple-company",
                "resolution_state": "resolved",
            },
            {
                "document_id": "d2",
                "chunk_id": "c2",
                "mention_id": "d2:c2:0",
                "canonical_entity_id": "ce-apple-company",
                "resolution_state": "resolved",
            },
        ],
        "lineage": {
            "base_run_id": "base-1",
            "base_extraction_hash": "a" * 64,
            "corpus_manifest_hash": config.corpus.manifest_sha256,
            "input_hashes": {
                "normalized_chunks": "b" * 64,
                "normalized_entities": "c" * 64,
                "normalized_relations": "d" * 64,
                "embedding_model_digest": config.roles.embedding.digest,
                "embedding_candidate_config": embedding_candidate_hash,
            },
            "er_version": config.entity_resolution.version,
            "er_config_hash": sha256_json(config.entity_resolution.pipeline_payload()),
            "judge_tag": config.roles.er_judge.requested_tag,
            "judge_digest": config.roles.er_judge.digest,
            "judge_prompt_version": config.roles.er_judge.prompt_version,
            "judge_config_hash": sha256_json(config.roles.er_judge.identity_payload()),
        },
    }


def test_valid_mention_level_er_plan_passes() -> None:
    report = validate_er_quality_gate(**_valid_er_payload())
    assert report.passed
    assert report.metrics["mentions"] == 2
    assert report.metrics["canonical_entities"] == 1


def test_operational_judge_budget_override_does_not_change_frozen_policy() -> None:
    report = validate_er_quality_gate(
        **_valid_er_payload(),
        operational_max_judge_calls_per_run=1000,
    )

    assert report.passed
    budget_check = next(
        check for check in report.checks if check.name == "operational_judge_budget"
    )
    assert budget_check.passed
    assert "planned=650" in budget_check.detail
    assert "effective=1000" in budget_check.detail


def test_valid_native_group_expands_to_all_source_mentions() -> None:
    payload = _valid_er_payload()
    payload["profiles"][0]["source_mentions"] = [
        {
            "mention_id": "d1:c1:0",
            "document_id": "d1",
            "chunk_id": "c1",
            "original_name": "Apple Inc.",
        },
        {
            "mention_id": "d3:c3:0",
            "document_id": "d3",
            "chunk_id": "c3",
            "original_name": "Apple Inc.",
        },
    ]
    payload["canonical_entities"][0]["source_mention_ids"].append("d3:c3:0")
    payload["canonical_entities"][0]["source_document_ids"].append("d3")
    payload["mention_to_canonical"].append(
        {
            "document_id": "d3",
            "chunk_id": "c3",
            "mention_id": "d3:c3:0",
            "canonical_entity_id": "ce-apple-company",
            "resolution_state": "resolved",
        }
    )

    report = validate_er_quality_gate(**payload)

    assert report.passed
    assert report.metrics["native_entities"] == 2
    assert report.metrics["mentions"] == 3


def test_merge_only_gate_rejects_more_canonical_than_native_entities() -> None:
    payload = _valid_er_payload()
    payload["canonical_entities"].extend(
        [
            {
                **deepcopy(payload["canonical_entities"][0]),
                "canonical_entity_id": f"extra-{index}",
                "display_name": f"Extra {index}",
            }
            for index in range(2)
        ]
    )

    report = validate_er_quality_gate(**payload, fail_closed=False)

    assert not report.passed
    assert any(
        "merge_only_entity_count_nonincreasing" in error for error in report.errors
    )


def test_candidate_route_evidence_must_match_frozen_threshold() -> None:
    payload = _valid_er_payload()
    payload["candidate_pairs"][0] = {
        **payload["candidate_pairs"][0],
        "methods": ["fuzzy_name"],
        "route_evidence": {
            "fuzzy_name": {"lexical_similarity": 0.72, "threshold": 0.72}
        },
        "precomputed_signals": {"lexical": 0.72},
    }
    payload["pair_scores"][0]["signals"]["lexical"] = 0.72

    report = validate_er_quality_gate(**payload, fail_closed=False)

    assert not report.passed
    assert any("candidate_route_evidence_auditable" in error for error in report.errors)


def test_singleton_canonical_entity_may_have_no_aliases() -> None:
    payload = _valid_er_payload()
    payload["profiles"] = payload["profiles"][:1]
    payload["candidate_pairs"] = []
    payload["pair_scores"] = []
    payload["pair_decisions"] = []
    payload["clusters"] = [{"cluster_id": "cluster-1", "mention_ids": ["d1:c1:0"]}]
    payload["canonical_entities"] = [
        {
            "canonical_entity_id": "ce-apple-company",
            "display_name": "Apple Inc.",
            "aliases": [],
            "source_mention_ids": ["d1:c1:0"],
            "source_document_ids": ["d1"],
            "selection_rationale": "only extracted mention",
        }
    ]
    payload["mention_to_canonical"] = payload["mention_to_canonical"][:1]

    assert validate_er_quality_gate(**payload).passed


def test_cannot_link_prevents_transitive_or_direct_false_merge() -> None:
    payload = _valid_er_payload()
    payload["cannot_links"] = [
        {
            "left_mention_id": "d1:c1:0",
            "right_mention_id": "d2:c2:0",
            "reason": "company versus fruit",
        }
    ]
    report = validate_er_quality_gate(**payload, fail_closed=False)
    assert not report.passed
    assert any("cannot_links_respected" in error for error in report.errors)


def test_judge_cache_key_must_bind_profiles_score_prompt_and_digest() -> None:
    payload = _valid_er_payload(score=0.6, source="judge")
    config = payload["config"]
    judge = config.roles.er_judge
    identity_payload = judge.identity_payload()
    identity_hash = sha256_json(identity_payload)
    identity = {**identity_payload, "config_hash": identity_hash}
    expected_key = sha256_json(
        {
            "judge": identity_payload,
            "judge_config_hash": identity_hash,
            "profiles": [
                {"mention_id": "d1:c1:0", "profile_hash": "1" * 64},
                {"mention_id": "d2:c2:0", "profile_hash": "2" * 64},
            ],
            "pair_score": payload["pair_scores"][0],
        }
    )
    payload["pair_decisions"][0].update(
        judge_cache_key=expected_key,
        judge_metadata={
            "judge_tag": judge.resolved_name,
            "judge_digest": judge.digest,
            "judge_prompt_version": judge.prompt_version,
            "judge_config_hash": identity_hash,
        },
    )
    assert validate_er_quality_gate(**payload, judge_identity=identity).passed

    corrupted = deepcopy(payload)
    corrupted["pair_decisions"][0]["judge_cache_key"] = "0" * 64
    report = validate_er_quality_gate(
        **corrupted,
        judge_identity=identity,
        fail_closed=False,
    )
    assert not report.passed
    assert any("judge_cache_profile_lineage" in error for error in report.errors)


def test_judge_decision_is_rejected_above_fixed_call_budget() -> None:
    payload = _valid_er_payload(score=0.6, source="judge")
    config_payload = payload["config"].model_dump(mode="json", exclude_none=False)
    config_payload["entity_resolution"]["decision_policy"][
        "max_judge_calls_per_run"
    ] = 0
    payload["config"] = type(payload["config"]).model_validate(config_payload)
    payload["lineage"]["er_config_hash"] = sha256_json(
        payload["config"].entity_resolution.pipeline_payload()
    )

    judge = payload["config"].roles.er_judge
    identity_payload = judge.identity_payload()
    identity_hash = sha256_json(identity_payload)
    identity = {**identity_payload, "config_hash": identity_hash}
    payload["pair_decisions"][0].update(
        judge_cache_key=sha256_json(
            {
                "judge": identity_payload,
                "judge_config_hash": identity_hash,
                "profiles": [
                    {"mention_id": "d1:c1:0", "profile_hash": "1" * 64},
                    {"mention_id": "d2:c2:0", "profile_hash": "2" * 64},
                ],
                "pair_score": payload["pair_scores"][0],
            }
        ),
        judge_metadata={
            "judge_tag": judge.resolved_name,
            "judge_digest": judge.digest,
            "judge_prompt_version": judge.prompt_version,
            "judge_config_hash": identity_hash,
        },
    )

    report = validate_er_quality_gate(
        **payload,
        judge_identity=identity,
        fail_closed=False,
    )
    assert not report.passed
    assert any("judge_usage_within_budget" in error for error in report.errors)
