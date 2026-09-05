from __future__ import annotations

import hashlib

from src.orchestration.quality_gates import validate_extraction_quality_gate


def _valid_snapshot() -> dict[str, object]:
    raw_response = '{"entities":[],"relationships":[]}'
    response_sha = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    return {
        "expected_document_ids": ["d1"],
        "manifest": {
            "expected_documents": 1,
            "completed_documents": 1,
            "expected_chunks": 1,
            "staged_chunks": 1,
            "failed_parse_chunks": [],
        },
        "chunks": [
            {
                "document_id": "d1",
                "chunk_id": "c1",
                "extraction_call_ids": ["call1"],
                "staging_complete": True,
            }
        ],
        "extraction_calls": [
            {
                "call_id": "call1",
                "attempt_number": 1,
                "document_id": "d1",
                "chunk_id": "c1",
                "raw_response": raw_response,
                "response_sha256": response_sha,
            }
        ],
        "entities": [
            {
                "mention_id": "m1",
                "document_id": "d1",
                "chunk_id": "c1",
                "extraction_call_id": "call1",
                "response_sha256": response_sha,
                "entity_present": True,
                "description": None,
                "description_present": False,
            },
            {
                "mention_id": "m2",
                "document_id": "d1",
                "chunk_id": "c1",
                "extraction_call_id": "call1",
                "response_sha256": response_sha,
                "entity_present": True,
                "description": "available",
                "description_present": True,
            },
        ],
        "relations": [
            {
                "relation_id": "r1",
                "source_mention_id": "m1",
                "target_mention_id": "m2",
                "source_resolution_state": "resolved",
                "target_resolution_state": "resolved",
                "source_candidate_mention_ids": ["m1"],
                "target_candidate_mention_ids": ["m2"],
                "document_id": "d1",
                "chunk_id": "c1",
                "extraction_call_id": "call1",
                "relation_present": True,
            }
        ],
    }


def test_missing_description_is_completeness_metric_not_missing_entity() -> None:
    report = validate_extraction_quality_gate(**_valid_snapshot())
    assert report.passed
    assert report.metrics["mentions"] == 2
    assert report.metrics["description_complete_mentions"] == 1
    assert report.metrics["description_completeness"] == 0.5


def test_incomplete_corpus_blocks_er_prerequisite() -> None:
    snapshot = _valid_snapshot()
    snapshot["expected_document_ids"] = ["d1", "d2"]
    report = validate_extraction_quality_gate(
        **snapshot,
        fail_closed=False,
    )
    assert not report.passed
    assert any("complete_document_snapshot" in error for error in report.errors)
    assert any("manifest_document_counts" in error for error in report.errors)


def test_unresolved_relation_endpoint_fails_closed() -> None:
    snapshot = _valid_snapshot()
    snapshot["relations"][0]["target_mention_id"] = "unknown"
    report = validate_extraction_quality_gate(
        **snapshot,
        fail_closed=False,
    )
    assert not report.passed
    assert any("relation_ids_endpoints_and_lineage" in error for error in report.errors)


def test_ambiguous_relation_endpoint_fails_closed_without_arbitrary_choice() -> None:
    snapshot = _valid_snapshot()
    relation = snapshot["relations"][0]
    relation["source_mention_id"] = None
    relation["source_resolution_state"] = "ambiguous"
    relation["source_candidate_mention_ids"] = ["m1", "m2"]
    report = validate_extraction_quality_gate(
        **snapshot,
        fail_closed=False,
    )
    assert not report.passed
    assert any("relation_ids_endpoints_and_lineage" in error for error in report.errors)


def test_presence_flags_must_be_literal_but_missing_description_remains_valid() -> None:
    snapshot = _valid_snapshot()
    snapshot["entities"][0]["description_present"] = True
    report = validate_extraction_quality_gate(
        **snapshot,
        fail_closed=False,
    )
    assert not report.passed
    assert any("mention_ids_and_lineage" in error for error in report.errors)


def test_required_passive_native_capture_fails_closed() -> None:
    snapshot = _valid_snapshot()
    report = validate_extraction_quality_gate(
        **snapshot,
        native_capture_version="capture-v1",
        fail_closed=False,
    )
    assert not report.passed
    assert any("passive_native_capture" in error for error in report.errors)

    snapshot["extraction_calls"][0]["technical_provenance"] = {
        "provider": {
            "native_capture": {
                "version": "capture-v1",
                "response_mutated": False,
            }
        }
    }
    passed = validate_extraction_quality_gate(
        **snapshot,
        native_capture_version="capture-v1",
    )
    assert passed.passed


def test_native_capture_rejects_any_response_mutation_flag() -> None:
    snapshot = _valid_snapshot()
    snapshot["extraction_calls"][0]["technical_provenance"] = {
        "provider": {
            "native_capture": {
                "version": "capture-v1",
                "response_mutated": True,
            }
        }
    }

    report = validate_extraction_quality_gate(
        **snapshot,
        native_capture_version="capture-v1",
        fail_closed=False,
    )

    assert not report.passed
    assert any("passive_native_capture" in error for error in report.errors)
