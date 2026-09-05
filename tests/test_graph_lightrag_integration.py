from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.graph.materialization import materialize_finalize_reopen
from src.graph.rewrite import rewrite_graph
from src.retrieval.runner import run_retrieval


pytestmark = pytest.mark.integration


class _CharacterTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(character) for character in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _graph():
    mentions = [
        {
            "document_id": "doc-real",
            "chunk_id": "doc-real-chunk-000",
            "mention_id": "mention-alpha",
            "original_name": "Alpha",
            "entity_type": "concept",
            "description": "",
            "description_present": False,
        },
        {
            "document_id": "doc-real",
            "chunk_id": "doc-real-chunk-000",
            "mention_id": "mention-beta",
            "original_name": "Beta",
            "entity_type": "concept",
            "description": "Beta is a test concept.",
            "description_present": True,
        },
    ]
    relations = [
        {
            "relation_id": "relation-alpha-beta",
            "document_id": "doc-real",
            "chunk_id": "doc-real-chunk-000",
            "source_mention_id": "mention-alpha",
            "target_mention_id": "mention-beta",
            "description": "Alpha is related to Beta.",
            "keywords": ["related"],
        }
    ]
    resolutions = [
        {
            "document_id": "doc-real",
            "chunk_id": "doc-real-chunk-000",
            "mention_id": "mention-alpha",
            "canonical_entity_id": "canonical-alpha",
        },
        {
            "document_id": "doc-real",
            "chunk_id": "doc-real-chunk-000",
            "mention_id": "mention-beta",
            "canonical_entity_id": "canonical-beta",
        },
    ]
    canonical = [
        {
            "canonical_entity_id": "canonical-alpha",
            "display_name": "Alpha",
            "aliases": ["A", "Alpha"],
            "entity_type": "concept",
            "merged_description": "",
        },
        {
            "canonical_entity_id": "canonical-beta",
            "display_name": "Beta",
            "aliases": ["Beta"],
            "entity_type": "concept",
            "merged_description": "Beta is a test concept.",
        },
    ]
    return rewrite_graph(mentions, relations, resolutions, canonical)


def test_real_lightrag_public_materialization_reopen_export_and_smoke(tmp_path: Path):
    """Exercise the local LightRAG package without network or model calls."""

    numpy = pytest.importorskip("numpy")
    lightrag = pytest.importorskip("lightrag")
    from lightrag.utils import EmbeddingFunc, Tokenizer

    llm_calls: list[str] = []

    async def deterministic_embedding(texts: list[str]):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "unanswerable" in lowered or "zeta" in lowered:
                vectors.append([-1.0, 0.0, 0.0, 0.0])
            elif any(term in lowered for term in ("alpha", "beta", "related")):
                vectors.append([1.0, 0.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0, 0.0])
        return numpy.asarray(vectors, dtype=float)

    async def deterministic_llm(prompt: str, **_: object) -> str:
        llm_calls.append(prompt)
        if "unanswerable" in prompt.casefold() or "zeta" in prompt.casefold():
            high, low = ["zeta"], ["unanswerable"]
        else:
            high, low = ["related"], ["Alpha", "Beta"]
        return json.dumps({"high_level_keywords": high, "low_level_keywords": low})

    workspace = f"graph_real_{uuid4().hex}"
    working_dir = tmp_path / "lightrag"

    def rag_factory():
        return lightrag.LightRAG(
            working_dir=str(working_dir),
            workspace=workspace,
            llm_model_func=deterministic_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=4,
                max_token_size=4096,
                func=deterministic_embedding,
            ),
            tokenizer=Tokenizer("graph-integration-tokenizer", _CharacterTokenizer()),
            max_graph_nodes=100,
            embedding_cache_config={
                "enabled": False,
                "similarity_threshold": 0.95,
                "use_llm_check": False,
            },
            enable_llm_cache=False,
        )

    async def exercise():
        from lightrag import QueryParam

        outcome = await materialize_finalize_reopen(
            rag_factory,
            documents=[
                {
                    "document_id": "doc-real",
                    "text": "Alpha is related to Beta.",
                    "file_path": "corpus/doc-real.txt",
                }
            ],
            chunks=[
                {
                    "document_id": "doc-real",
                    "chunk_id": "doc-real-chunk-000",
                    "chunk_order": 0,
                    "text": "Alpha is related to Beta.",
                    # LightRAG's pre-embedding hard gate recomputes this with
                    # the configured tokenizer, just as in the native run.
                    "token_count": 25,
                }
            ],
            graph=_graph(),
            track_id="real-local-materialization",
            smoke_cases=[
                {
                    "query": "How are Alpha and Beta related?",
                    "expected_status": "success",
                },
                {
                    "query": "unanswerable zeta",
                    "expected_status": "failure",
                },
            ],
            query_param_factory=lambda mode: QueryParam(
                mode=mode,
                only_need_context=True,
                stream=False,
                top_k=5,
                chunk_top_k=5,
                enable_rerank=False,
            ),
        )
        retrieval_rag = rag_factory()
        await retrieval_rag.initialize_storages()
        try:
            records = await run_retrieval(
                retrieval_rag,
                [
                    {
                        "question_id": "q-real-context",
                        "question": "How are Alpha and Beta related?",
                        "question_type": "integration",
                        "answerable": True,
                    },
                    {
                        "question_id": "q-real-empty",
                        "question": "unanswerable zeta",
                        "question_type": "integration",
                        "answerable": False,
                    },
                ],
                artifact_path=tmp_path / "real_retrieval.jsonl",
                lineage={
                    "base_run_id": "base_real",
                    "variant_run_id": "variant_real",
                    "base_extraction_sha256": "a" * 64,
                },
                builder_model="builder-real",
                graph_regime="advanced_lightrag_er",
                query_model="query-real",
                query_model_digest="b" * 64,
                retrieval_parameters={
                    "top_k": 5,
                    "chunk_top_k": 5,
                    "enable_rerank": False,
                },
            )
        finally:
            await retrieval_rag.finalize_storages()
        return outcome, records

    outcome, retrieval = asyncio.run(exercise())

    assert outcome.stage.exact_chunk_gate.passed
    assert outcome.stage.pre_finalize_workspace_gate.passed
    assert outcome.reopened_workspace_gate.passed
    assert outcome.stage.pre_finalize_workspace_gate.metrics[
        "provenance_mismatches"
    ] == 0
    assert outcome.stage.pre_finalize_workspace_gate.metrics[
        "unresolved_source_ids"
    ] == 0
    assert outcome.reopened_workspace_gate.metrics["provenance_mismatches"] == 0
    assert outcome.reopened_workspace_gate.metrics["unresolved_source_ids"] == 0
    assert outcome.smoke_gate is not None and outcome.smoke_gate.passed
    assert len(outcome.reopened_snapshot.nodes) == 2
    assert len(outcome.reopened_snapshot.edges) == 1
    assert len(outcome.reopened_snapshot.chunks) == 1
    assert all(node.get("source_id") for node in outcome.reopened_snapshot.nodes)
    assert all(edge.get("source_id") for edge in outcome.reopened_snapshot.edges)
    assert outcome.reopened_snapshot.chunk_read_api == (
        "storage_interface:text_chunks.get_by_ids"
    )
    assert len(outcome.stage.description_placeholders) == 1
    assert [item["actual_status"] for item in outcome.smoke_results] == [
        "success",
        "failure",
    ]
    assert [item.retrieval_outcome for item in retrieval] == ["context", "empty"]
    assert retrieval[0].context
    assert retrieval[0].context_token_count is not None
    assert retrieval[0].context_token_count > 0
    assert not retrieval[0].context.lstrip().startswith("{'content':")
    assert retrieval[0].retrieved_document_ids == ["doc-real.txt"]
    assert [item.rank for item in retrieval[0].ranked_retrieval_items] == [1]
    assert retrieval[0].ranked_retrieval_items[0].source == "chunk"
    assert retrieval[1].context == ""
    assert retrieval[1].context_token_count == 0
    assert retrieval[1].raw_result["metadata"]["failure_reason"] == "no_results"
    assert len(llm_calls) == 4
