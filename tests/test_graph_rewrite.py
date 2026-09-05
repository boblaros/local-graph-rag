from __future__ import annotations

import random

import pytest

from src.graph.rewrite import GraphRewriteError, rewrite_graph


def _inputs():
    mentions = [
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-company-full",
            "original_name": "Apple Inc.",
            "entity_type": "organization",
            "description": "A technology company.",
            "extraction_call_id": "call-1",
        },
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-company-short",
            "original_name": "Apple",
            "entity_type": "organization",
            "description": "The company also called Apple.",
            "extraction_call_id": "call-1",
        },
        {
            "document_id": "doc-2",
            "chunk_id": "doc-2-chunk-000",
            "mention_id": "m-fruit",
            "original_name": "Apple",
            "entity_type": "food",
            "description": "A fruit.",
            "extraction_call_id": "call-2",
        },
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-phone",
            "original_name": "iPhone",
            "entity_type": "product",
            "description": "A smartphone.",
            "extraction_call_id": "call-1",
        },
    ]
    resolutions = [
        {
            **{key: mention[key] for key in ("document_id", "chunk_id", "mention_id")},
            "canonical_entity_id": canonical,
        }
        for mention, canonical in zip(
            mentions, ["c-company", "c-company", "c-fruit", "c-phone"], strict=True
        )
    ]
    canonical = [
        {
            "canonical_entity_id": "c-company",
            "display_name": "Apple Inc.",
            "aliases": ["Apple", "Apple Inc."],
            "entity_type": "organization",
            "description": "Apple Inc. is a technology company.",
            "selection_rationale": "most explicit extracted name",
        },
        {
            "canonical_entity_id": "c-fruit",
            "display_name": "Apple (fruit)",
            "aliases": ["Apple"],
            "entity_type": "food",
            "description": "An apple fruit.",
        },
        {
            "canonical_entity_id": "c-phone",
            "display_name": "iPhone",
            "aliases": ["iPhone"],
            "entity_type": "product",
            "description": "A smartphone.",
        },
    ]
    relations = [
        {
            "relation_id": "r-induced-loop",
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "source_mention_id": "m-company-full",
            "target_mention_id": "m-company-short",
            "description": "Apple Inc. is also called Apple.",
            "keywords": "alias",
            "weight": 1.0,
        },
        {
            "relation_id": "r-company-phone-1",
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "source_mention_id": "m-company-full",
            "target_mention_id": "m-phone",
            "description": "Apple Inc. makes the iPhone.",
            "keywords": "makes, product",
            "weight": 1.0,
            "provenance": {"row": 1},
        },
        {
            "relation_id": "r-company-phone-2",
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "source_mention_id": "m-company-short",
            "target_mention_id": "m-phone",
            "description": "The iPhone is an Apple product.",
            "keywords": "product",
            "weight": 2.0,
            "provenance": {"row": 2},
        },
        {
            "relation_id": "r-fruit-phone",
            "document_id": "doc-2",
            "chunk_id": "doc-2-chunk-000",
            "source_mention_id": "m-fruit",
            "target_mention_id": "m-phone",
            "description": "The two terms appear in a comparison.",
            "keywords": "comparison",
            "weight": 1.0,
        },
        {
            "relation_id": "r-existing-loop",
            "document_id": "doc-2",
            "chunk_id": "doc-2-chunk-000",
            "source_mention_id": "m-fruit",
            "target_mention_id": "m-fruit",
            "description": "An explicit source self-reference.",
            "keywords": "self",
            "weight": 1.0,
        },
    ]
    return mentions, relations, resolutions, canonical


def test_mention_level_rewrite_disambiguates_apple_and_removes_only_induced_loop():
    graph = rewrite_graph(*_inputs())

    names = {node["canonical_entity_id"]: node["entity_name"] for node in graph.nodes}
    assert names == {
        "c-company": "Apple Inc.",
        "c-fruit": "Apple (fruit)",
        "c-phone": "iPhone",
    }
    mapping = {
        (item["document_id"], item["chunk_id"], item["mention_id"]): item[
            "canonical_entity_id"
        ]
        for item in graph.mention_to_canonical
    }
    assert mapping[("doc-1", "doc-1-chunk-000", "m-company-short")] == "c-company"
    assert mapping[("doc-2", "doc-2-chunk-000", "m-fruit")] == "c-fruit"

    assert [item["relation_id"] for item in graph.induced_self_loops] == [
        "r-induced-loop"
    ]
    assert [item["relation_id"] for item in graph.preexisting_self_loops] == [
        "r-existing-loop"
    ]
    all_relation_ids = {
        relation_id
        for edge in graph.edges
        for relation_id in edge["source_relation_ids"]
    }
    assert "r-induced-loop" not in all_relation_ids
    assert "r-existing-loop" in all_relation_ids


def test_edge_dedup_sums_weight_and_preserves_relation_provenance():
    graph = rewrite_graph(*_inputs())
    edge = next(
        item
        for item in graph.edges
        if {item["source"], item["target"]} == {"Apple Inc.", "iPhone"}
    )

    assert edge["weight"] == 3.0
    assert edge["source_relation_ids"] == [
        "r-company-phone-1",
        "r-company-phone-2",
    ]
    assert edge["deduplicated_relation_count"] == 2
    assert [item["provenance"] for item in edge["provenance"]] == [
        {"row": 1},
        {"row": 2},
    ]
    assert edge["keywords"] == "makes,product"


def test_same_surface_distinct_mentions_still_create_er_induced_loop():
    mentions, relations, resolutions, canonical = _inputs()
    mentions.append(
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-company-same-surface",
            "original_name": "Apple",
            "entity_type": "organization",
            "description": "A second extracted Apple mention.",
        }
    )
    resolutions.append(
        {
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "mention_id": "m-company-same-surface",
            "canonical_entity_id": "c-company",
        }
    )
    relations.append(
        {
            "relation_id": "r-induced-same-surface",
            "document_id": "doc-1",
            "chunk_id": "doc-1-chunk-000",
            "source_mention_id": "m-company-short",
            "target_mention_id": "m-company-same-surface",
            "description": "Two records with the same extracted name.",
        }
    )

    graph = rewrite_graph(mentions, relations, resolutions, canonical)

    assert [item["relation_id"] for item in graph.induced_self_loops] == [
        "r-induced-loop",
        "r-induced-same-surface",
    ]
    assert graph.summary["induced_self_loops_removed"] == 2


def test_rewrite_is_deterministic_under_input_order_changes():
    inputs = list(_inputs())
    expected = rewrite_graph(*inputs).to_dict()
    randomizer = random.Random(17)
    for values in inputs:
        randomizer.shuffle(values)
    assert rewrite_graph(*inputs).to_dict() == expected


def test_rewrite_fails_closed_when_a_mention_lacks_resolution():
    mentions, relations, resolutions, canonical = _inputs()
    resolutions.pop()
    with pytest.raises(GraphRewriteError, match="lack a canonical resolution"):
        rewrite_graph(mentions, relations, resolutions, canonical)


def test_canonical_names_must_be_disambiguated():
    mentions, relations, resolutions, canonical = _inputs()
    canonical[1]["display_name"] = "apple inc."
    with pytest.raises(GraphRewriteError, match="not uniquely disambiguated"):
        rewrite_graph(mentions, relations, resolutions, canonical)
