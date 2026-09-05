"""Tiny network-free smoke of the real public LightRAG materialization path."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import uuid4

from src.graph import materialize_finalize_reopen, rewrite_graph


class _CharacterTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(character) for character in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _smoke_graph():
    mentions = [
        {
            "document_id": "smoke-doc",
            "chunk_id": "smoke-doc-chunk-000",
            "mention_id": "smoke-alpha",
            "original_name": "Alpha",
            "entity_type": "concept",
            "description": None,
            "description_present": False,
        },
        {
            "document_id": "smoke-doc",
            "chunk_id": "smoke-doc-chunk-000",
            "mention_id": "smoke-beta",
            "original_name": "Beta",
            "entity_type": "concept",
            "description": "Beta is a smoke-test concept.",
            "description_present": True,
        },
    ]
    relations = [
        {
            "relation_id": "smoke-relation",
            "document_id": "smoke-doc",
            "chunk_id": "smoke-doc-chunk-000",
            "source_mention_id": "smoke-alpha",
            "target_mention_id": "smoke-beta",
            "description": "Alpha is related to Beta.",
            "keywords": ["related"],
        }
    ]
    mapping = [
        {
            "document_id": mention["document_id"],
            "chunk_id": mention["chunk_id"],
            "mention_id": mention["mention_id"],
            "canonical_entity_id": f"canonical-{mention['mention_id']}",
        }
        for mention in mentions
    ]
    canonical = [
        {
            "canonical_entity_id": item["canonical_entity_id"],
            "display_name": mention["original_name"],
            "aliases": [mention["original_name"]],
            "entity_type": mention["entity_type"],
            "merged_description": mention["description"],
        }
        for mention, item in zip(mentions, mapping, strict=True)
    ]
    return rewrite_graph(mentions, relations, mapping, canonical)


async def run_local_public_api_smoke() -> dict[str, object]:
    """Materialize/reopen/query one synthetic document without Ollama calls."""

    import numpy
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc, Tokenizer

    llm_calls: list[str] = []

    async def embedding(texts: list[str]):
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

    async def keyword_llm(prompt: str, **_: object) -> str:
        llm_calls.append(prompt)
        if "unanswerable" in prompt.casefold() or "zeta" in prompt.casefold():
            high, low = ["zeta"], ["unanswerable"]
        else:
            high, low = ["related"], ["Alpha", "Beta"]
        return json.dumps({"high_level_keywords": high, "low_level_keywords": low})

    with tempfile.TemporaryDirectory(prefix="experiment-lightrag-smoke-") as temporary:
        working_dir = Path(temporary) / "workspace"
        workspace = f"experiment_smoke_{uuid4().hex}"

        def factory():
            return LightRAG(
                working_dir=str(working_dir),
                workspace=workspace,
                llm_model_func=keyword_llm,
                embedding_func=EmbeddingFunc(
                    embedding_dim=4,
                    max_token_size=4096,
                    func=embedding,
                ),
                tokenizer=Tokenizer(
                    "experiment-character-smoke", _CharacterTokenizer()
                ),
                max_graph_nodes=100,
                embedding_cache_config={
                    "enabled": False,
                    "similarity_threshold": 0.95,
                    "use_llm_check": False,
                },
            )

        outcome = await materialize_finalize_reopen(
            factory,
            documents=[
                {
                    "document_id": "smoke-doc",
                    "text": "Alpha is related to Beta.",
                    "file_path": "smoke-doc",
                }
            ],
            chunks=[
                {
                    "document_id": "smoke-doc",
                    "chunk_id": "smoke-doc-chunk-000",
                    "chunk_order": 0,
                    "text": "Alpha is related to Beta.",
                    "token_count": 25,
                }
            ],
            graph=_smoke_graph(),
            track_id="experiment-public-api-smoke",
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
    return {
        "passed": bool(
            outcome.stage.exact_chunk_gate.passed
            and outcome.reopened_workspace_gate.passed
            and outcome.smoke_gate is not None
            and outcome.smoke_gate.passed
        ),
        "nodes": len(outcome.reopened_snapshot.nodes),
        "edges": len(outcome.reopened_snapshot.edges),
        "chunks": len(outcome.reopened_snapshot.chunks),
        "smoke_statuses": [item["actual_status"] for item in outcome.smoke_results],
        "llm_calls": len(llm_calls),
        "chunk_read_api": outcome.reopened_snapshot.chunk_read_api,
        "parity": outcome.parity_report.to_dict(),
    }


__all__ = ["run_local_public_api_smoke"]
