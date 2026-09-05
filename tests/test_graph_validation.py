from __future__ import annotations

from copy import deepcopy

from src.graph.lightrag_adapter import ChunkSnapshot, StagedChunkResult
from src.graph.validation import validate_staged_chunks


def _stage_result() -> StagedChunkResult:
    expected = [
        ChunkSnapshot("doc-1", 0, "doc-1-chunk-000", "first", 1),
        ChunkSnapshot("doc-1", 1, "doc-1-chunk-001", "second", 1),
    ]
    stored = [
        {
            "_id": item.chunk_id,
            "content": item.content,
            "full_doc_id": item.document_id,
            "chunk_order_index": item.chunk_order,
            "tokens": item.token_count,
        }
        for item in expected
    ]
    return StagedChunkResult(
        track_id="track",
        expected_chunks=expected,
        stored_chunks=stored,
        document_statuses={
            "doc-1": {
                "status": "processed",
                "chunks_list": [item.chunk_id for item in expected],
            }
        },
        chunk_read_api="storage_interface:text_chunks.get_by_ids",
    )


def test_exact_chunk_gate_checks_ids_content_order_document_and_status():
    report = validate_staged_chunks(_stage_result())
    assert report.passed
    assert report.metrics["content_mismatches"] == 0
    assert report.warnings


def test_exact_chunk_gate_detects_content_and_status_order_mismatch():
    result = deepcopy(_stage_result())
    result.stored_chunks[0]["content"] = "changed"
    result.document_statuses["doc-1"]["chunks_list"].reverse()
    report = validate_staged_chunks(result)
    assert not report.passed
    assert report.metrics["content_mismatches"] == 1
    assert report.metrics["status_mismatches"] == 1
