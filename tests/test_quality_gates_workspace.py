from __future__ import annotations

from src.orchestration.quality_gates import validate_final_workspace_quality_gate


def _valid_workspace() -> dict:
    return {
        "graph_regime": "advanced_lightrag_er",
        "base_extraction_sha256": "a" * 64,
        "expected_base_extraction_sha256": "a" * 64,
        "nodes": [
            {"id": "Apple Inc.", "source_chunk_ids": ["c1", "c2"]},
            {"id": "Cupertino", "source_chunk_ids": ["c1"]},
        ],
        "edges": [
            {
                "source": "Apple Inc.",
                "target": "Cupertino",
                "source_chunk_ids": ["c1"],
            }
        ],
        "chunks": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        "expected_chunk_ids": ["c1", "c2"],
        "induced_self_loops": [
            {"relation_id": "r-merged-loop", "canonical_entity_id": "ce-apple"}
        ],
        "workspace_clean": True,
        "persisted": True,
        "finalized": True,
        "reopened": True,
        "export_succeeded": True,
        "smoke_retrieval_answerable": True,
        "smoke_retrieval_unanswerable": True,
        "native_workspace_sha256_before": "b" * 64,
        "native_workspace_sha256_after": "b" * 64,
    }


def test_final_er_workspace_persists_reopens_exports_and_preserves_native() -> None:
    report = validate_final_workspace_quality_gate(**_valid_workspace())
    assert report.passed
    assert report.metrics["induced_self_loops_removed"] == 1


def test_dangling_edges_and_native_mutation_block_workspace() -> None:
    payload = _valid_workspace()
    payload["edges"][0]["target"] = "missing-node"
    payload["native_workspace_sha256_after"] = "c" * 64
    report = validate_final_workspace_quality_gate(**payload, fail_closed=False)
    assert not report.passed
    assert any("no_dangling_or_duplicate_edges" in error for error in report.errors)
    assert any("native_workspace_unchanged" in error for error in report.errors)


def test_workspace_requires_both_answerable_and_unanswerable_smoke_retrieval() -> None:
    payload = _valid_workspace()
    payload["smoke_retrieval_unanswerable"] = False
    report = validate_final_workspace_quality_gate(**payload, fail_closed=False)
    assert not report.passed
    assert any("smoke_retrieval" in error for error in report.errors)


def test_preexisting_self_loop_is_allowed_only_when_separately_audited() -> None:
    payload = _valid_workspace()
    payload["edges"].append(
        {
            "source": "Apple Inc.",
            "target": "Apple Inc.",
            "source_chunk_ids": ["c2"],
        }
    )
    payload["preexisting_self_loops"] = [
        {
            "relation_id": "r-preexisting-loop",
            "canonical_entity_id": "ce-apple",
            "canonical_display_name": "Apple Inc.",
        }
    ]
    assert validate_final_workspace_quality_gate(**payload).passed

    payload["preexisting_self_loops"] = []
    report = validate_final_workspace_quality_gate(**payload, fail_closed=False)
    assert not report.passed
    assert any("er_induced_self_loops_removed" in error for error in report.errors)
