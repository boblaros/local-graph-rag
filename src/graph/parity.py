"""Auditable report of Native-vs-public-materialization semantic differences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rewrite import RewrittenGraph


@dataclass(frozen=True)
class ParityReport:
    status: str
    exact_chunk_identity: bool
    extraction_reused_without_builder_call: bool
    custom_kg_chunk_path_used: bool
    chunk_read_api: str
    description_placeholders: int
    description_truncations: int
    aliases_audit_only: int
    known_differences: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exact_chunk_identity": self.exact_chunk_identity,
            "extraction_reused_without_builder_call": self.extraction_reused_without_builder_call,
            "custom_kg_chunk_path_used": self.custom_kg_chunk_path_used,
            "chunk_read_api": self.chunk_read_api,
            "description_placeholders": self.description_placeholders,
            "description_truncations": self.description_truncations,
            "aliases_audit_only": self.aliases_audit_only,
            "known_differences": list(self.known_differences),
        }


def build_parity_report(
    graph: RewrittenGraph,
    *,
    chunk_read_api: str,
    description_placeholder_count: int,
    description_truncation_count: int,
) -> ParityReport:
    alias_count = sum(len(node.get("aliases", [])) for node in graph.nodes)
    differences = (
        "LightRAG ainsert_custom_kg hashes chunk content and cannot preserve native document-scoped chunk ids; it was not used.",
        "LightRAG exposes no facade method for arbitrary exact chunk reads; validation uses the reported storage interface when necessary.",
        "Public acreate_entity/acreate_relation require non-empty descriptions; missing descriptions use a deterministic tracked placeholder.",
        "Aliases and stable canonical ids are retained in audit artifacts only because LightRAG's public create schema drops extra metadata.",
        "The public create facade does not populate LightRAG's entity_chunks/relation_chunks auxiliary KV indexes, so deletion/editing parity is not claimed.",
        "Advanced descriptions are deterministically aggregated offline; Native LightRAG may use builder-role LLM summaries during graph merge.",
        "Descriptions longer than the frozen storage bound are deterministically truncated before LightRAG storage and embedding; full descriptions remain in ER audit artifacts.",
        "LightRAG represents an entity's display name as its graph node id and relationships as one undirected edge per endpoint pair.",
    )
    return ParityReport(
        status="known_differences_documented",
        exact_chunk_identity=True,
        extraction_reused_without_builder_call=True,
        custom_kg_chunk_path_used=False,
        chunk_read_api=chunk_read_api,
        description_placeholders=description_placeholder_count,
        description_truncations=description_truncation_count,
        aliases_audit_only=alias_count,
        known_differences=differences,
    )


__all__ = ["ParityReport", "build_parity_report"]
