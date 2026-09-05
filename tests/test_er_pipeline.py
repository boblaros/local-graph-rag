from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entity_resolution import (  # noqa: E402
    ER_ARTIFACTS,
    ERConfig,
    ERDecisionPolicy,
    ERLineage,
    EntityMention,
    EntityResolutionPipeline,
    IncompleteCorpusError,
    JudgeCache,
    JudgeIdentity,
    JudgeResult,
    native_entity_id,
    RelationMention,
    CorpusSnapshot,
    content_hash,
)
from entity_resolution.profiles import build_entity_profiles  # noqa: E402
from extraction.models import (  # noqa: E402
    NormalizedEntityMention,
    NormalizedRelation,
)


class _MergeJudge:
    identity = JudgeIdentity(
        model_tag="fixed-er-judge",
        model_digest="sha256:fixed-er-judge",
        prompt_version="er-pair-v1",
        temperature=0.0,
        seed=42,
    )

    def decide(self, left, right, score):
        del left, right, score
        return JudgeResult(relationship="same_entity", rationale="same company")


def _records():
    mentions = [
        EntityMention(
            "m1",
            "doc1",
            "chunk1",
            "Apple Inc.",
            "organization",
            "Technology company",
            extraction_call_id="call1",
        ),
        EntityMention(
            "m2",
            "doc2",
            "chunk2",
            "Apple",
            "company",
            "Technology company",
            extraction_call_id="call2",
        ),
        EntityMention(
            "m3",
            "doc3",
            "chunk3",
            "Apple",
            "fruit",
            "Edible fruit",
            extraction_call_id="call3",
        ),
    ]
    relations = [
        RelationMention(
            "r1",
            "m1",
            "m2",
            "doc1",
            "chunk1",
            keywords=("technology", "company"),
            description="refers to the same corporation",
            extraction_call_id="call1",
        )
    ]
    return mentions, relations


def _lineage() -> ERLineage:
    return ERLineage(
        base_run_id="base_run_1",
        base_extraction_hash="base_extraction_sha256",
        corpus_manifest_hash="corpus_sha256",
        input_hashes={
            "normalized_extraction": "normalized_sha256",
            "raw_extraction": "raw_sha256",
        },
    )


def test_pipeline_refuses_incomplete_corpus_before_writing(tmp_path: Path) -> None:
    mentions, relations = _records()
    snapshot = CorpusSnapshot(
        expected_document_ids=("doc1", "doc2", "doc3"),
        completed_document_ids=("doc1", "doc2"),
    )

    with pytest.raises(IncompleteCorpusError, match="complete corpus"):
        EntityResolutionPipeline().run(
            snapshot=snapshot,
            mentions=mentions,
            relations=relations,
            lineage=_lineage(),
            artifact_dir=tmp_path / "er",
        )
    assert not (tmp_path / "er").exists()


def test_pipeline_writes_complete_lineaged_immutable_artifact_set(
    tmp_path: Path,
) -> None:
    mentions, relations = _records()
    snapshot = CorpusSnapshot(
        expected_document_ids=("doc1", "doc2", "doc3"),
        completed_document_ids=("doc3", "doc1", "doc2"),
        expected_chunk_ids=("chunk1", "chunk2", "chunk3"),
        completed_chunk_ids=("chunk3", "chunk2", "chunk1"),
    )
    artifact_dir = tmp_path / "er"
    pipeline = EntityResolutionPipeline()

    first = pipeline.run(
        snapshot=snapshot,
        mentions=mentions,
        relations=relations,
        lineage=_lineage(),
        artifact_dir=artifact_dir,
    )
    second = pipeline.run(
        snapshot=snapshot,
        mentions=list(reversed(mentions)),
        relations=relations,
        lineage=_lineage(),
        artifact_dir=artifact_dir,
    )

    assert {path.name for path in artifact_dir.iterdir()} == set(ER_ARTIFACTS)
    assert len(ER_ARTIFACTS) == 11
    assert first.artifact_hashes == second.artifact_hashes
    assert first.lineage.merge_plan_hash == second.lineage.merge_plan_hash
    assert first.lineage.er_config_hash == pipeline.config.config_hash
    assert len(first.mention_to_canonical) == len(mentions)
    assert len({item.mention_id for item in first.mention_to_canonical}) == len(
        mentions
    )
    assert first.summary["mentions"] == 3
    assert first.summary["native_entities"] == 2
    assert first.summary["canonical_entities"] == 2
    assert len(first.profiles) == 2
    apple_profile = next(
        profile for profile in first.profiles if profile.original_name == "Apple"
    )
    assert set(apple_profile.source_mention_ids) == {"m2", "m3"}

    profile_lines = (
        (artifact_dir / "entity_profiles.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    header = json.loads(profile_lines[0])
    record = json.loads(profile_lines[1])
    assert header["lineage"]["base_extraction_hash"] == "base_extraction_sha256"
    assert header["lineage"]["merge_plan_hash"] == first.lineage.merge_plan_hash
    assert record["record_hash"] == content_hash(record["record"])
    assert len(record["record"]["profile_hash"]) == 64
    assert record["lineage"]["input_hashes"]["raw_extraction"] == "raw_sha256"

    candidate_records = [
        json.loads(line)["record"]
        for line in (artifact_dir / "candidate_pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    ]
    assert candidate_records
    assert all(record["route_evidence"] for record in candidate_records)
    fuzzy_records = [
        record for record in candidate_records if "fuzzy_name" in record["methods"]
    ]
    assert fuzzy_records
    assert all(
        record["precomputed_signals"]["lexical"]
        == record["route_evidence"]["fuzzy_name"]["lexical_similarity"]
        for record in fuzzy_records
    )


def test_cached_judge_resume_keeps_immutable_artifact_hashes(tmp_path: Path) -> None:
    mentions, relations = _records()
    snapshot = CorpusSnapshot(
        expected_document_ids=("doc1", "doc2", "doc3"),
        completed_document_ids=("doc1", "doc2", "doc3"),
        expected_chunk_ids=("chunk1", "chunk2", "chunk3"),
        completed_chunk_ids=("chunk1", "chunk2", "chunk3"),
    )
    pipeline = EntityResolutionPipeline(
        ERConfig(
            decision_policy=ERDecisionPolicy(
                reject_below_score=0.0,
                auto_merge_at_score=1.0,
            )
        )
    )
    cache = JudgeCache(tmp_path / "judge-cache")
    artifact_dir = tmp_path / "er"

    first = pipeline.run(
        snapshot=snapshot,
        mentions=mentions,
        relations=relations,
        lineage=_lineage(),
        artifact_dir=artifact_dir,
        judge=_MergeJudge(),
        judge_cache=cache,
        embedding_neighbors={
            native_entity_id("Apple Inc."): [native_entity_id("Apple")],
            native_entity_id("Apple"): [native_entity_id("Apple Inc.")],
        },
    )
    second = pipeline.run(
        snapshot=snapshot,
        mentions=mentions,
        relations=relations,
        lineage=_lineage(),
        artifact_dir=artifact_dir,
        judge=_MergeJudge(),
        judge_cache=cache,
        embedding_neighbors={
            native_entity_id("Apple Inc."): [native_entity_id("Apple")],
            native_entity_id("Apple"): [native_entity_id("Apple Inc.")],
        },
    )

    first_judged = [item for item in first.decisions if item.source == "judge"]
    second_judged = [item for item in second.decisions if item.source == "judge"]
    assert first_judged and second_judged
    assert all(item.judge_cache_hit is False for item in first_judged)
    assert all(item.judge_cache_hit is True for item in second_judged)
    assert first.artifact_hashes == second.artifact_hashes
    stored_decisions = [
        json.loads(line)["record"]
        for line in (artifact_dir / "pair_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    ]
    assert all(item["judge_cache_hit"] is None for item in stored_decisions)


def test_profiles_accept_immutable_staging_pydantic_records() -> None:
    left = NormalizedEntityMention(
        mention_id="mention-left",
        mention_index=0,
        original_name="Apple Inc.",
        normalized_name="apple inc",
        entity_type="organization",
        description="Technology company",
        document_id="doc-1",
        chunk_id="doc-1-chunk-000",
        provenance={},
        extraction_call_id="call-1",
        parse_status="strict",
        entity_present=True,
        description_present=True,
    )
    right = NormalizedEntityMention(
        mention_id="mention-right",
        mention_index=1,
        original_name="Apple",
        normalized_name="apple",
        entity_type="organization",
        description="The company",
        document_id="doc-1",
        chunk_id="doc-1-chunk-000",
        provenance={},
        extraction_call_id="call-1",
        parse_status="strict",
        entity_present=True,
        description_present=True,
    )
    relation = NormalizedRelation(
        relation_id="relation-1",
        relation_index=0,
        source_mention_id="mention-left",
        target_mention_id="mention-right",
        source_resolution_state="resolved",
        target_resolution_state="resolved",
        source_candidate_mention_ids=["mention-left"],
        target_candidate_mention_ids=["mention-right"],
        source_original_name="Apple Inc.",
        target_original_name="Apple",
        keywords=["alias"],
        relation_type=None,
        description="Names the same organization",
        description_present=True,
        document_id="doc-1",
        chunk_id="doc-1-chunk-000",
        provenance={},
        extraction_call_id="call-1",
        parse_status="strict",
        relation_present=True,
    )

    profiles = build_entity_profiles([left, right], [relation])

    assert [profile.mention_id for profile in profiles] == [
        "mention-left",
        "mention-right",
    ]
    assert profiles[0].neighbours == ("apple",)
