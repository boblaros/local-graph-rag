from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.graph.lightrag_adapter import PublicLightRAGAdapter
from src.graph.materialization import DescriptionPolicy, materialize_finalize_reopen
from src.graph.rewrite import rewrite_graph


@dataclass
class _PersistentWorkspace:
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)
    statuses: dict[str, dict[str, Any]] = field(default_factory=dict)
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    instances: int = 0
    initializes: int = 0
    finalizes: int = 0
    enqueue_calls: list[dict[str, Any]] = field(default_factory=list)
    query_calls: list[tuple[str, Any]] = field(default_factory=list)


class _FakeTextChunks:
    def __init__(self, state: _PersistentWorkspace):
        self._state = state

    async def get_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any] | None]:
        return [self._state.chunks.get(chunk_id) for chunk_id in chunk_ids]


class _FakeGraphStorage:
    def __init__(self, state: _PersistentWorkspace):
        self._state = state

    async def get_all_nodes(self) -> list[dict[str, Any]]:
        return [
            {"id": name, "labels": [], "properties": dict(properties)}
            for name, properties in sorted(self._state.nodes.items())
        ]

    async def get_all_edges(self) -> list[dict[str, Any]]:
        return [
            {
                "source": pair[0],
                "target": pair[1],
                "properties": dict(properties),
            }
            for pair, properties in sorted(self._state.edges.items())
        ]


class _FakeLightRAG:
    """Small persistent fake with only the LightRAG APIs used by the adapter."""

    def __init__(self, state: _PersistentWorkspace):
        self._state = state
        self._state.instances += 1
        self.text_chunks = _FakeTextChunks(state)
        self.chunk_entity_relation_graph = _FakeGraphStorage(state)
        self.chunking_func = None
        self._pending: dict[str, Any] | None = None

    async def initialize_storages(self) -> None:
        self._state.initializes += 1

    async def finalize_storages(self) -> None:
        self._state.finalizes += 1

    async def get_graph_labels(self) -> list[str]:
        return sorted(self._state.nodes)

    async def get_processing_status(self) -> dict[str, int]:
        return {"processed": len(self._state.statuses)} if self._state.statuses else {}

    async def apipeline_enqueue_documents(
        self,
        texts: list[str],
        *,
        ids: list[str],
        file_paths: list[str],
        track_id: str,
        process_options: str,
    ) -> str:
        self._state.enqueue_calls.append(
            {
                "ids": list(ids),
                "file_paths": list(file_paths),
                "track_id": track_id,
                "process_options": process_options,
            }
        )
        self._pending = {
            "texts": list(texts),
            "ids": list(ids),
            "file_paths": list(file_paths),
            "track_id": track_id,
        }
        return track_id

    async def apipeline_process_enqueue_documents(self) -> None:
        assert self._pending is not None
        assert callable(self.chunking_func)
        for text, document_id in zip(
            self._pending["texts"], self._pending["ids"], strict=True
        ):
            chunks = self.chunking_func(None, text, None, False, 100, 1200)
            chunk_ids: list[str] = []
            for chunk in chunks:
                chunk_id = str(chunk["chunk_id"])
                chunk_ids.append(chunk_id)
                self._state.chunks[chunk_id] = {
                    "_id": chunk_id,
                    "content": chunk["content"],
                    "tokens": chunk["tokens"],
                    "chunk_order_index": chunk["chunk_order_index"],
                    "full_doc_id": document_id,
                }
            self._state.statuses[document_id] = {
                "status": "processed",
                "chunks_list": chunk_ids,
                "track_id": self._pending["track_id"],
            }
        self._pending = None

    async def aget_docs_by_track_id(self, track_id: str) -> dict[str, dict[str, Any]]:
        return {
            document_id: dict(status)
            for document_id, status in self._state.statuses.items()
            if status["track_id"] == track_id
        }

    async def acreate_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        assert entity_data["description"]
        self._state.nodes[entity_name] = dict(entity_data)
        return {"entity_name": entity_name, **entity_data}

    async def acreate_relation(
        self,
        source_entity: str,
        target_entity: str,
        relation_data: dict[str, Any],
    ) -> dict[str, Any]:
        assert source_entity in self._state.nodes
        assert target_entity in self._state.nodes
        assert relation_data["description"]
        pair = tuple(sorted((source_entity, target_entity)))
        self._state.edges[pair] = dict(relation_data)
        return {
            "src_id": source_entity,
            "tgt_id": target_entity,
            **relation_data,
        }

    async def get_knowledge_graph(
        self, node_label: str, *, max_depth: int = 3, max_nodes: int = 1000
    ) -> dict[str, Any]:
        del max_depth
        assert node_label == "*"
        nodes = [
            {"id": name, "labels": [], "properties": dict(properties)}
            for name, properties in sorted(self._state.nodes.items())
        ]
        edges = [
            {
                "source": pair[0],
                "target": pair[1],
                "properties": dict(properties),
            }
            for pair, properties in sorted(self._state.edges.items())
        ]
        return {
            "nodes": nodes[:max_nodes],
            "edges": edges,
            "is_truncated": len(nodes) > max_nodes,
        }

    async def aquery_data(self, query: str, query_param: Any) -> dict[str, Any]:
        self._state.query_calls.append((query, query_param))
        return {
            "status": "failure" if query == "unanswerable" else "success",
            "data": {"context": "Alpha is related to Beta."},
        }


def _rewritten_graph():
    mentions = [
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-alpha",
            "original_name": "Alpha",
            "entity_type": "concept",
            "description": "",
            "description_present": False,
        },
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-beta",
            "original_name": "Beta",
            "entity_type": "concept",
            "description": "Beta description.",
            "description_present": True,
        },
    ]
    relations = [
        {
            "relation_id": "r-alpha-beta",
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "source_mention_id": "m-alpha",
            "target_mention_id": "m-beta",
            "description": "",
            "keywords": "related",
        }
    ]
    resolutions = [
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-alpha",
            "canonical_entity_id": "c-alpha",
        },
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-beta",
            "canonical_entity_id": "c-beta",
        },
    ]
    canonical = [
        {
            "canonical_entity_id": "c-alpha",
            "display_name": "Alpha",
            "aliases": ["A", "Alpha"],
            "entity_type": "concept",
            "description": "",
        },
        {
            "canonical_entity_id": "c-beta",
            "display_name": "Beta",
            "aliases": ["Beta"],
            "entity_type": "concept",
            "description": "Beta description.",
        },
    ]
    return rewrite_graph(mentions, relations, resolutions, canonical)


def test_workspace_export_is_not_capped_at_one_thousand_nodes() -> None:
    state = _PersistentWorkspace(
        nodes={f"node-{index:04d}": {} for index in range(1001)}
    )
    adapter = PublicLightRAGAdapter(_FakeLightRAG(state))

    snapshot = asyncio.run(
        adapter.export_workspace(expected_node_count=1001, chunk_ids=[])
    )

    assert len(snapshot.nodes) == 1001
    assert snapshot.graph_is_truncated is False


def test_public_api_materialization_finalize_reopen_export_and_smoke():
    state = _PersistentWorkspace()

    async def exercise():
        return await materialize_finalize_reopen(
            lambda: _FakeLightRAG(state),
            documents=[
                {
                    "document_id": "doc-1",
                    "text": "Alpha is related to Beta.",
                    "file_path": "corpus/doc-1.txt",
                }
            ],
            chunks=[
                {
                    "document_id": "doc-1",
                    "chunk_id": "doc-1-chunk-000",
                    "chunk_order_index": 0,
                    "content": "Alpha is related to Beta.",
                    "tokens": 6,
                }
            ],
            graph=_rewritten_graph(),
            track_id="variant-run-1",
            smoke_cases=[
                {"query": "answerable", "expected_status": "success"},
                {"query": "unanswerable", "expected_status": "failure"},
            ],
            query_param_factory=lambda mode: {"mode": mode, "only_need_context": True},
        )

    outcome = asyncio.run(exercise())

    assert state.instances == 2
    assert state.initializes == 2
    assert state.finalizes == 2
    assert state.enqueue_calls == [
        {
            "ids": ["doc-1"],
            "file_paths": ["corpus/doc-1.txt"],
            "track_id": "variant-run-1",
            "process_options": "!",
        }
    ]
    assert outcome.stage.exact_chunk_gate.passed
    assert outcome.stage.pre_finalize_workspace_gate.passed
    assert outcome.reopened_workspace_gate.passed
    assert outcome.smoke_gate is not None and outcome.smoke_gate.passed
    assert len(outcome.stage.description_placeholders) == 2
    assert outcome.stage.description_truncations == []
    assert outcome.stage.aliases_audit[0]["storage_policy"] == "audit_only"
    assert "aliases" not in state.nodes["Alpha"]
    assert outcome.parity_report.exact_chunk_identity
    assert outcome.parity_report.extraction_reused_without_builder_call
    assert not outcome.parity_report.custom_kg_chunk_path_used
    assert outcome.parity_report.description_truncations == 0
    assert outcome.parity_report.chunk_read_api == (
        "storage_interface:text_chunks.get_by_ids"
    )


def test_materialization_bounds_long_descriptions_and_audits_truncation():
    state = _PersistentWorkspace()
    graph = _rewritten_graph()
    beta = next(node for node in graph.nodes if node["entity_name"] == "Beta")
    beta["description"] = "word " * 20
    edge = graph.edges[0]
    edge["description"] = "relation " * 20
    policy = DescriptionPolicy(maximum_chars=48)

    async def exercise():
        return await materialize_finalize_reopen(
            lambda: _FakeLightRAG(state),
            documents=[
                {
                    "document_id": "doc-1",
                    "text": "Alpha is related to Beta.",
                    "file_path": "corpus/doc-1.txt",
                }
            ],
            chunks=[
                {
                    "document_id": "doc-1",
                    "chunk_id": "doc-1-chunk-000",
                    "chunk_order_index": 0,
                    "content": "Alpha is related to Beta.",
                    "tokens": 6,
                }
            ],
            graph=graph,
            track_id="variant-run-long-description",
            description_policy=policy,
        )

    outcome = asyncio.run(exercise())

    assert len(state.nodes["Beta"]["description"]) <= 48
    assert state.nodes["Beta"]["description"].endswith(" … [truncated]")
    assert len(state.edges[("Alpha", "Beta")]["description"]) <= 48
    assert len(outcome.stage.description_truncations) == 2
    assert {item["object_kind"] for item in outcome.stage.description_truncations} == {
        "entity",
        "relation",
    }
    assert all(
        item["stored_chars"] <= item["maximum_chars"]
        for item in outcome.stage.description_truncations
    )
    assert outcome.parity_report.description_truncations == 2
