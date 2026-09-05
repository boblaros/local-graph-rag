from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from src.extraction.normalization import normalize_chunk_calls, recover_json
from src.extraction.staging import load_staged_snapshot, stage_native_artifacts


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_recovery_preserves_a_stock_complete_entity_and_relation() -> None:
    raw = """```json
    {'entities':[{'name':'Apple','type':'Organization','description':'Company'}],
     'relationships':[{'source':'Apple','target':'Cupertino',
     'keywords':'based in','description':'Apple is based in Cupertino.'}]}
    ```"""
    payload, status, method, error = recover_json(raw)
    assert payload is not None
    assert status == "recovered"
    assert method == "json_repair"
    assert error is None
    chunk = {
        "chunk_id": "doc-1-chunk-000",
        "document_id": "doc-1",
        "text": "Apple is based in Cupertino.",
        "chunk_order": 0,
    }
    call = {
        "call_id": "call-1",
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "raw_response": raw,
        "response_sha256": _hash(raw),
        "gleaning_round": 0,
        "attempt_number": 1,
        "technical_provenance": {"call_kind": "initial"},
    }
    result = normalize_chunk_calls(chunk, [call])
    apple = next(item for item in result.entities if item.original_name == "Apple")
    cupertino = next(
        item for item in result.entities if item.original_name == "Cupertino"
    )
    assert apple.entity_present is True
    assert apple.description_present is True
    assert cupertino.implicit_from_relation is True
    assert result.relations[0].target_mention_id == cupertino.mention_id
    assert result.entity_present is True
    assert result.description_present is True
    assert result.relations_present is True


def test_stock_duplicate_surface_rows_coalesce_per_chunk() -> None:
    raw = json.dumps(
        {
            "entities": [
                {"name": "Apple", "type": "Organization", "description": "Company"},
                {"name": "Apple", "type": "Organization", "description": "Company"},
            ],
            "relationships": [
                {
                    "source": "Apple",
                    "target": "iPhone",
                    "keywords": "manufactures",
                    "description": "Makes",
                }
            ],
        }
    )
    chunk = {
        "chunk_id": "doc-1-chunk-000",
        "document_id": "doc-1",
        "text": "x",
        "chunk_order": 0,
    }
    call = {
        "call_id": "call-1",
        "raw_response": raw,
        "response_sha256": _hash(raw),
        "gleaning_round": 0,
        "technical_provenance": {"call_kind": "initial"},
    }
    first = normalize_chunk_calls(chunk, [call])
    second = normalize_chunk_calls(chunk, [call])
    assert first == second
    assert [item.original_name for item in first.entities] == ["Apple", "iPhone"]
    apple = first.entities[0]
    assert apple.provenance["normalization"] == "stock_lightrag_name_coalesce"
    assert apple.provenance["coalesced_record_count"] == 2
    relation = first.relations[0]
    assert relation.source_mention_id == apple.mention_id
    assert relation.source_resolution_state == "resolved"
    assert first.staging_complete is True


def test_stock_normalizer_ignores_experiment_local_ids() -> None:
    raw = json.dumps(
        {
            "entities": [
                {
                    "id": "e-company",
                    "name": "Apple",
                    "type": "Organization",
                    "description": "Technology company",
                },
                {
                    "id": "e-fruit",
                    "name": "Apple",
                    "type": "NaturalObject",
                    "description": "Fruit",
                },
            ],
            "relationships": [
                {
                    "source_id": "e-company",
                    "source": "Apple",
                    "target_id": "e-fruit",
                    "target": "Apple",
                    "description": "Shares a surface name",
                }
            ],
        }
    )
    chunk = {
        "chunk_id": "doc-1-chunk-000",
        "document_id": "doc-1",
        "text": "Apple and apple.",
        "chunk_order": 0,
    }
    call = {
        "call_id": "call-1",
        "raw_response": raw,
        "response_sha256": _hash(raw),
        "gleaning_round": 0,
        "technical_provenance": {"call_kind": "initial"},
    }

    result = normalize_chunk_calls(chunk, [call])

    assert result.staging_complete is True
    assert len(result.entities) == 1
    assert result.entities[0].original_name == "Apple"
    # The stock parser sees this as a self-edge after name coalescing and drops it.
    assert result.relations == []


def test_stock_name_resolution_spans_initial_and_gleaning_calls() -> None:
    initial_raw = json.dumps(
        {
            "entities": [
                {
                    "name": "Apple",
                    "type": "Organization",
                    "description": "Company",
                },
                {
                    "name": "iPhone",
                    "type": "Artifact",
                    "description": "Phone",
                },
            ],
            "relationships": [],
        }
    )
    gleaning_raw = json.dumps(
        {
            "entities": [
                {
                    "name": "Tim Cook",
                    "type": "Person",
                    "description": "Executive",
                }
            ],
            "relationships": [
                {
                    "source": "Tim Cook",
                    "target": "iPhone",
                    "description": "Discusses iPhone",
                }
            ],
        }
    )
    chunk = {
        "chunk_id": "doc-1-chunk-000",
        "document_id": "doc-1",
        "text": "Apple makes iPhone.",
        "chunk_order": 0,
    }
    calls = [
        {
            "call_id": "call-initial",
            "raw_response": initial_raw,
            "response_sha256": _hash(initial_raw),
            "gleaning_round": 0,
            "technical_provenance": {"call_kind": "initial"},
        },
        {
            "call_id": "call-gleaning",
            "raw_response": gleaning_raw,
            "response_sha256": _hash(gleaning_raw),
            "gleaning_round": 1,
            "technical_provenance": {"call_kind": "gleaning"},
        },
    ]

    result = normalize_chunk_calls(chunk, calls)

    assert result.staging_complete is True
    relation = result.relations[0]
    names = {item.mention_id: item.original_name for item in result.entities}
    assert names[relation.source_mention_id] == "Tim Cook"
    assert names[relation.target_mention_id] == "iPhone"
    assert relation.provenance["target_resolution"] == "stock_exact_chunk_name"


def test_stock_keys_remain_case_sensitive_until_er() -> None:
    raw = json.dumps(
        {
            "entities": [
                {"name": "Apple", "type": "Organization", "description": "Company"},
                {"name": "apple", "type": "Food", "description": "Fruit"},
            ],
            "relationships": [],
        }
    )
    chunk = {
        "chunk_id": "doc-1-chunk-000",
        "document_id": "doc-1",
        "text": "Apple and apple.",
        "chunk_order": 0,
    }
    call = {
        "call_id": "call-1",
        "raw_response": raw,
        "response_sha256": _hash(raw),
        "gleaning_round": 0,
        "technical_provenance": {"call_kind": "initial"},
    }

    result = normalize_chunk_calls(chunk, [call])

    assert [item.original_name for item in result.entities] == ["Apple", "apple"]
    assert {item.normalized_name for item in result.entities} == {"apple"}


def test_offline_stock_inclusion_matches_pinned_lightrag_json_parser() -> None:
    from lightrag.operate import _process_json_extraction_result

    raw = json.dumps(
        {
            "entities": [
                {"name": '"Apple"', "type": "Organization", "description": "Company"},
                {"name": "apple", "type": "Food", "description": "Fruit"},
                {"name": "Dropped", "type": "Other", "description": ""},
            ],
            "relationships": [
                {
                    "source": '"Apple"',
                    "target": "iPhone",
                    "keywords": "manufactures, product",
                    "description": "Apple makes iPhone.",
                },
                {
                    "source": "apple",
                    "target": "apple",
                    "keywords": "identity",
                    "description": "Self edge.",
                },
                {
                    "source": "Apple",
                    "target": "Dropped",
                    "keywords": "invalid",
                    "description": "",
                },
            ],
        }
    )
    chunk = {
        "chunk_id": "doc-1-chunk-000",
        "document_id": "doc-1",
        "text": "Apple makes iPhone.",
        "chunk_order": 0,
    }
    call = {
        "call_id": "call-1",
        "raw_response": raw,
        "response_sha256": _hash(raw),
        "gleaning_round": 0,
        "technical_provenance": {"call_kind": "initial"},
    }

    native_nodes, native_edges = asyncio.run(
        _process_json_extraction_result(raw, chunk["chunk_id"], 0)
    )
    staged = normalize_chunk_calls(chunk, [call])

    staged_explicit_names = {
        item.original_name
        for item in staged.entities
        if not item.implicit_from_relation
    }
    staged_edges = {
        tuple(sorted((item.source_original_name, item.target_original_name)))
        for item in staged.relations
    }
    assert staged_explicit_names == set(native_nodes)
    assert staged_edges == {tuple(sorted(pair)) for pair in native_edges}


def test_stock_incomplete_relation_is_dropped() -> None:
    raw = json.dumps(
        {
            "entities": [
                {"name": "Apple", "type": "Organization", "description": "Company"},
                {"name": "iPhone", "type": "Artifact", "description": "Phone"},
            ],
            "relationships": [
                {
                    "source_id": "e1",
                    "source": "Apple",
                    "target": "iPhone",
                }
            ],
        }
    )
    chunk = {
        "chunk_id": "doc-1-chunk-000",
        "document_id": "doc-1",
        "text": "Apple makes iPhone.",
        "chunk_order": 0,
    }
    call = {
        "call_id": "call-1",
        "raw_response": raw,
        "response_sha256": _hash(raw),
        "gleaning_round": 0,
        "technical_provenance": {"call_kind": "initial"},
    }

    result = normalize_chunk_calls(chunk, [call])

    assert result.staging_complete is True
    assert result.relations == []
    assert result.parse_records[0].relation_records_seen == 1


def test_staging_is_immutable_and_hash_verified(tmp_path: Path) -> None:
    run = tmp_path / "native"
    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True)
    text = "Apple is a company."
    chunk = {
        "run_id": "legacy",
        "chunk_id": "doc-1-chunk-000",
        "document_id": "doc-1",
        "text": text,
        "text_sha256": _hash(text),
        "chunk_order": 0,
    }
    raw = json.dumps(
        {
            "entities": [
                {"name": "Apple", "type": "Organization", "description": "Company"}
            ],
            "relationships": [],
        }
    )
    call = {
        "call_id": "call-1",
        "chunk_id": chunk["chunk_id"],
        "document_id": "doc-1",
        "raw_response": raw,
        "response_sha256": _hash(raw),
        "gleaning_round": 0,
        "attempt_number": 1,
        "technical_provenance": {"call_kind": "initial"},
    }
    (artifacts / "chunks.jsonl").write_text(json.dumps(chunk) + "\n")
    (artifacts / "extraction_calls.jsonl").write_text(json.dumps(call) + "\n")
    documents = tmp_path / "documents.jsonl"
    questions = tmp_path / "questions.jsonl"
    manifest = tmp_path / "manifest.json"
    documents.write_text(json.dumps({"document_id": "doc-1", "text": text}) + "\n")
    questions.write_text(json.dumps({"question_id": "q-1", "question": "What?"}) + "\n")
    manifest.write_text(json.dumps({"subset_id": "test-subset"}))
    output = tmp_path / "staged"
    first = stage_native_artifacts(
        run,
        documents_path=documents,
        questions_path=questions,
        corpus_manifest_path=manifest,
        base_run_id="base_123",
        builder_model="builder:tag",
        builder_model_digest="a" * 64,
        extraction_prompt_version="extract-v1",
        seed=42,
        generation_parameters={"temperature": 0},
        output_dir=output,
    )
    second = stage_native_artifacts(
        run,
        documents_path=documents,
        questions_path=questions,
        corpus_manifest_path=manifest,
        base_run_id="base_123",
        builder_model="builder:tag",
        builder_model_digest="a" * 64,
        extraction_prompt_version="extract-v1",
        seed=42,
        generation_parameters={"temperature": 0},
        output_dir=output,
    )
    assert first == second
    assert first.corpus_complete
    loaded, chunks, mentions, relations = load_staged_snapshot(output)
    assert loaded.base_extraction_sha256 == first.base_extraction_sha256
    assert len(chunks) == 1 and len(mentions) == 1 and relations == []
    with (output / "normalized_entities.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_staged_snapshot(output)
