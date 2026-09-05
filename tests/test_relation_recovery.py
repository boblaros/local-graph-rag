from __future__ import annotations

import hashlib

from src.graph import RewrittenGraph
from src.relation_recovery import (
    OllamaRRVerifier,
    aggregate_rr_graph,
    build_rr_plan,
    candidate_allowed,
    graph_edge_pairs,
    validate_verifier_response,
)
from src.orchestration.quality_gates import validate_rr_quality_gate

from ._config_helpers import resolved_config


def _chunk(chunk_id: str, text: str, order: int = 0) -> dict[str, object]:
    return {
        "document_id": "doc-1",
        "chunk_id": chunk_id,
        "chunk_order": order,
        "text": text,
        "chunk_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _graph_inputs():
    nodes = [
        {
            "canonical_entity_id": "A",
            "entity_name": "Alpha",
            "aliases": ["Alpha"],
            "source_mentions": [
                {"document_id": "doc-1", "chunk_id": "c1", "original_name": "A"},
                {"document_id": "doc-1", "chunk_id": "c2", "original_name": "Alpha"},
            ],
        },
        {
            "canonical_entity_id": "B",
            "entity_name": "Beta",
            "aliases": ["Beta"],
            "source_mentions": [
                {"document_id": "doc-1", "chunk_id": "c1", "original_name": "B"},
                {"document_id": "doc-1", "chunk_id": "c2", "original_name": "Beta"},
            ],
        },
        {
            "canonical_entity_id": "C",
            "entity_name": "Gamma",
            "aliases": ["Gamma"],
            "source_mentions": [
                {"document_id": "doc-1", "chunk_id": "c1", "original_name": "C"}
            ],
        },
    ]
    mapping = [
        {
            "document_id": "doc-1",
            "chunk_id": "c1",
            "mention_id": "m1",
            "canonical_entity_id": "A",
        },
        {
            "document_id": "doc-1",
            "chunk_id": "c1",
            "mention_id": "m2",
            "canonical_entity_id": "B",
        },
        {
            "document_id": "doc-1",
            "chunk_id": "c1",
            "mention_id": "m3",
            "canonical_entity_id": "C",
        },
        {
            "document_id": "doc-1",
            "chunk_id": "c2",
            "mention_id": "m4",
            "canonical_entity_id": "A",
        },
        {
            "document_id": "doc-1",
            "chunk_id": "c2",
            "mention_id": "m5",
            "canonical_entity_id": "B",
        },
    ]
    edges = [
        {
            "edge_id": "edge-ac",
            "source_canonical_entity_id": "A",
            "target_canonical_entity_id": "C",
            "source": "Alpha",
            "target": "Gamma",
            "description": "Alpha knows Gamma",
        }
    ]
    return nodes, edges, mapping


def test_plan_is_deterministic_and_excludes_self_and_frozen_er_edges() -> None:
    nodes, edges, mapping = _graph_inputs()
    chunks = [
        _chunk("c2", "Alpha acquired Beta.", 1),
        _chunk("c1", "A works with B and C.", 0),
    ]
    first = build_rr_plan(
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        mention_to_canonical=mapping,
    )
    second = build_rr_plan(
        chunks=list(reversed(chunks)),
        nodes=list(reversed(nodes)),
        edges=list(reversed(edges)),
        mention_to_canonical=list(reversed(mapping)),
    )
    assert first == second
    rows, summary = first
    assert [row["chunk_id"] for row in rows] == ["c1", "c2"]
    assert rows[0]["missing_pair_count"] == 2  # AB and BC; AC already exists.
    assert rows[1]["missing_pair_count"] == 1  # AB remains a separate instance.
    frozen = graph_edge_pairs(edges)
    assert candidate_allowed(rows[0], "A", "B", frozen)
    assert not candidate_allowed(rows[0], "A", "A", frozen)
    assert not candidate_allowed(rows[0], "A", "C", frozen)
    assert not candidate_allowed(rows[0], "A", "unknown", frozen)
    assert summary["eligible_chunk_count"] == 2
    assert summary["candidate_pair_instance_count"] == 3
    assert summary["maximum_verifier_calls"] == 2


def test_response_validation_is_exact_evidence_bound_and_fails_chunk_closed() -> None:
    nodes, edges, mapping = _graph_inputs()
    text = "Alpha acquired Beta in 2024."
    rows, _ = build_rr_plan(
        chunks=[_chunk("c1", text)],
        nodes=nodes[:2],
        edges=[],
        mention_to_canonical=mapping[:2],
    )
    valid = validate_verifier_response(
        raw_content={
            "relations": [
                {
                    "entity_a_id": "B",
                    "entity_b_id": "A",
                    "relationship_description": "Alpha acquired Beta in 2024.",
                    "evidence_quote": "Alpha acquired Beta",
                }
            ]
        },
        chunk_text=text,
        plan_row=rows[0],
        frozen_er_pairs=frozenset(),
        verifier_identity={"model_digest": "digest"},
        request_sha256="request",
    )
    assert valid["status"] == "valid"
    assert valid["accepted_relations"][0]["entity_a_id"] == "A"
    assert valid["accepted_relations"][0]["entity_b_id"] == "B"

    for bad_content in (
        "not json",
        {"relations": [{**valid["accepted_relations"][0], "entity_a_id": "X"}]},
        {
            "relations": [
                {
                    "entity_a_id": "A",
                    "entity_b_id": "B",
                    "relationship_description": "acquisition",
                    "evidence_quote": "not present verbatim",
                }
            ]
        },
    ):
        invalid = validate_verifier_response(
            raw_content=bad_content,
            chunk_text=text,
            plan_row=rows[0],
            frozen_er_pairs=frozenset(),
            verifier_identity={},
            request_sha256="request",
        )
        assert invalid["status"] == "invalid"
        assert invalid["accepted_relations"] == []


def test_same_pair_from_multiple_chunks_aggregates_to_one_provenance_rich_edge() -> (
    None
):
    nodes, _, mapping = _graph_inputs()
    er_graph = RewrittenGraph(
        nodes=nodes[:2],
        edges=[],
        mention_to_canonical=mapping[:2] + mapping[3:],
        summary={"canonical_entity_count": 2, "materialized_edge_count": 0},
    )
    accepted = [
        {
            "rr_relation_id": "rr-1",
            "document_id": "doc-1",
            "chunk_id": "c1",
            "chunk_sha256": "h1",
            "entity_a_id": "A",
            "entity_b_id": "B",
            "relationship_description": "Alpha acquired Beta.",
            "evidence_quote": "Alpha acquired Beta",
            "candidate_set_sha256": "p1",
            "request_sha256": "q1",
            "response_sha256": "r1",
            "verifier_identity": {"digest": "d"},
        },
        {
            "rr_relation_id": "rr-2",
            "document_id": "doc-1",
            "chunk_id": "c2",
            "chunk_sha256": "h2",
            "entity_a_id": "B",
            "entity_b_id": "A",
            "relationship_description": "Alpha acquired Beta.",
            "evidence_quote": "acquired Beta",
            "candidate_set_sha256": "p2",
            "request_sha256": "q2",
            "response_sha256": "r2",
            "verifier_identity": {"digest": "d"},
        },
    ]
    graph = aggregate_rr_graph(er_graph, accepted)
    assert graph.nodes == er_graph.nodes
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge["description"] == "Alpha acquired Beta."
    assert edge["weight"] == 2.0
    assert edge["source_chunk_ids"] == ["c1", "c2"]
    assert len(edge["provenance"]) == 2
    assert graph.summary["rr_candidate_instance_accept_count"] == 2
    assert graph.summary["rr_recovered_edge_count"] == 1


def test_verifier_adapter_makes_exactly_one_call() -> None:
    class Client:
        calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return {"message": {"content": '{"relations":[]}'}, "eval_count": 5}

    client = Client()
    verifier = OllamaRRVerifier(
        client=client,
        model_tag="model",
        model_digest="digest",
        prompt_version="rr-relation-only-v1",
        temperature=0.0,
        seed=42,
        options={"num_ctx": 8192, "num_predict": 128},
    )
    verifier.verify(
        chunk_text="Alpha and Beta.",
        plan_row={
            "chunk_id": "c1",
            "document_id": "d1",
            "canonical_entities": [],
            "existing_local_relations": [],
        },
    )
    assert client.calls == 1


def test_rr_quality_gate_recomputes_plan_and_graph_invariants() -> None:
    config = resolved_config()
    nodes, _, mapping = _graph_inputs()
    text = "Alpha acquired Beta."
    chunks = [_chunk("c1", text)]
    plan, summary = build_rr_plan(
        chunks=chunks,
        nodes=nodes[:2],
        edges=[],
        mention_to_canonical=mapping[:2],
    )
    relation = validate_verifier_response(
        raw_content={
            "relations": [
                {
                    "entity_a_id": "A",
                    "entity_b_id": "B",
                    "relationship_description": "Alpha acquired Beta.",
                    "evidence_quote": "Alpha acquired Beta",
                }
            ]
        },
        chunk_text=text,
        plan_row=plan[0],
        frozen_er_pairs=frozenset(),
        verifier_identity=config.roles.rr_verifier.identity_payload(),
        request_sha256="request",
    )
    er_graph = RewrittenGraph(
        nodes=nodes[:2], edges=[], mention_to_canonical=mapping[:2]
    )
    rr_graph = aggregate_rr_graph(er_graph, relation["accepted_relations"])
    gate = validate_rr_quality_gate(
        config=config,
        chunks=chunks,
        er_nodes=er_graph.nodes,
        er_edges=er_graph.edges,
        mention_to_canonical=er_graph.mention_to_canonical,
        plan_rows=plan,
        plan_summary=summary,
        chunk_results=[{"chunk_id": "c1", "verification": relation}],
        rr_nodes=rr_graph.nodes,
        rr_edges=rr_graph.edges,
    )
    assert gate.passed
    assert gate.metrics["accepted_candidate_instances"] == 1
    assert gate.metrics["recovered_pairs"] == 1
