"""Export Native LightRAG chunks and graph records to JSONL.

The exporter reads through LightRAG's storage interfaces rather than depending
on storage filenames. Existing records are reused only when their content
matches.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from src.config.native_support.hashes import (
    canonical_json,
    sha256_text,
    stable_id,
    to_jsonable,
)
from src.config.native_support.io_utils import (
    append_jsonl,
    read_jsonl,
    touch_append_only,
)
from src.config.native_support.schemas import (
    SCHEMA_VERSION,
    ChunkArtifact,
    GraphEdgeArtifact,
    GraphNodeArtifact,
)


PAGE_SIZE = 200
KV_BATCH_SIZE = 200


def _json_safe(value: Any) -> Any:
    """Preserve upstream attributes while making uncommon scalar types safe."""

    try:
        return to_jsonable(value)
    except (TypeError, ValueError):
        if is_dataclass(value) and not isinstance(value, type):
            return _json_safe(asdict(value))
        if hasattr(value, "item"):
            try:
                return _json_safe(value.item())
            except (TypeError, ValueError):
                pass
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_json_safe(item) for item in value]
        return str(value)


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip() if raw is not None else ""
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _split_lightrag_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return _unique_strings(value)
    try:
        from lightrag.constants import GRAPH_FIELD_SEP
    except ImportError:  # Importability for exporter unit tests without LightRAG.
        GRAPH_FIELD_SEP = "<SEP>"
    return _unique_strings(str(value).split(GRAPH_FIELD_SEP))


def _batches(values: Sequence[str], size: int = KV_BATCH_SIZE) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _load_existing(path: Path, key: str, run_id: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path, tolerate_truncated_tail=False):
        assert isinstance(record, dict)
        if record.get("run_id") != run_id:
            raise RuntimeError(f"raw artifact belongs to another run: {path}")
        item_id = str(record.get(key, ""))
        if not item_id:
            raise RuntimeError(f"raw artifact lacks {key}: {path}")
        if item_id in result:
            raise RuntimeError(f"duplicate {key}={item_id!r} in {path}")
        result[item_id] = record
    return result


def _immutable_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"created_at", "timestamp"}
    }


def _append_if_new(
    path: Path,
    key: str,
    record: Any,
    existing: dict[str, dict[str, Any]],
) -> bool:
    payload = record.model_dump(mode="json", exclude_none=False)
    item_id = str(payload[key])
    prior = existing.get(item_id)
    if prior is not None:
        if canonical_json(_immutable_payload(prior)) != canonical_json(
            _immutable_payload(payload)
        ):
            raise RuntimeError(
                f"immutable export conflict for {key}={item_id!r} in {path}; "
                "use a fresh run_id/workspace"
            )
        return False
    append_jsonl(path, record)
    existing[item_id] = payload
    return True


def _load_document_catalog(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    catalog: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(source):
        assert isinstance(record, dict)
        document_id = str(record.get("document_id", "")).strip()
        if not document_id:
            raise ValueError(f"input document lacks document_id: {source}")
        if document_id in catalog:
            raise ValueError(f"duplicate document_id={document_id!r}: {source}")
        # Deliberately exclude text: exporter metadata must never be folded into
        # the text LightRAG chunked.
        catalog[document_id] = {
            key: _json_safe(value) for key, value in record.items() if key != "text"
        }
    return catalog


async def _enumerate_status_documents(rag: Any) -> list[tuple[str, Any]]:
    page = 1
    rows: list[tuple[str, Any]] = []
    while True:
        current, total = await rag.doc_status.get_docs_paginated(
            page=page,
            page_size=PAGE_SIZE,
            sort_field="id",
            sort_direction="asc",
        )
        rows.extend(current)
        if not current or len(rows) >= int(total):
            break
        page += 1
    return rows


async def _collect_actual_chunks(
    rag: Any,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    statuses = await _enumerate_status_documents(rag)
    chunk_to_document: dict[str, str] = {}
    status_provenance: dict[str, dict[str, Any]] = {}
    ordered_chunk_ids: list[str] = []

    for document_id_raw, status in statuses:
        document_id = str(document_id_raw)
        chunks = getattr(status, "chunks_list", None) or []
        status_mapping = _json_safe(asdict(status) if is_dataclass(status) else status)
        for chunk_id_raw in chunks:
            chunk_id = str(chunk_id_raw)
            prior = chunk_to_document.get(chunk_id)
            if prior is not None and prior != document_id:
                raise RuntimeError(
                    f"LightRAG chunk {chunk_id!r} belongs to both {prior!r} "
                    f"and {document_id!r}"
                )
            if prior is None:
                ordered_chunk_ids.append(chunk_id)
            chunk_to_document[chunk_id] = document_id
            status_provenance[chunk_id] = {
                "document_status": status_mapping,
                "status_document_id": document_id,
            }

    chunks_by_id: dict[str, dict[str, Any]] = {}
    for batch in _batches(ordered_chunk_ids):
        values = await rag.text_chunks.get_by_ids(batch)
        if len(values) != len(batch):
            raise RuntimeError(
                "text_chunks.get_by_ids did not preserve batch cardinality"
            )
        for requested_id, value in zip(batch, values, strict=True):
            if value is None:
                raise RuntimeError(
                    f"doc_status references missing text chunk {requested_id!r}"
                )
            record = dict(value)
            storage_id = str(record.get("_id") or requested_id)
            if storage_id != requested_id:
                raise RuntimeError(
                    f"text chunk ID mismatch: requested {requested_id!r}, got {storage_id!r}"
                )
            record["_id"] = requested_id
            chunks_by_id[requested_id] = record

    records = [chunks_by_id[chunk_id] for chunk_id in ordered_chunk_ids]
    return records, chunk_to_document, status_provenance


def _count_chunk_tokens(rag: Any, chunk: Mapping[str, Any], text: str) -> int:
    for field in ("tokens", "token_count"):
        value = chunk.get(field)
        if isinstance(value, int) and value >= 0:
            return value
    tokenizer = getattr(rag, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("chunk has no token count and LightRAG has no tokenizer")
    return len(tokenizer.encode(text))


def _chunk_order(chunk: Mapping[str, Any]) -> int:
    for field in ("chunk_order_index", "chunk_order", "order"):
        value = chunk.get(field)
        if isinstance(value, int) and value >= 0:
            return value
    raise RuntimeError(f"chunk lacks a non-negative order field: {chunk.get('_id')!r}")


async def _full_chunk_provenance(
    storage: Any,
    key: str,
    fallback: Sequence[str],
) -> tuple[list[str], dict[str, Any] | None]:
    stored: Any = None
    if storage is not None:
        stored = await storage.get_by_id(key)
    if isinstance(stored, Mapping):
        chunk_ids = _unique_strings(stored.get("chunk_ids", []))
        if chunk_ids:
            return chunk_ids, dict(stored)
    return _unique_strings(fallback), dict(stored) if isinstance(
        stored, Mapping
    ) else None


def _documents_for_chunks(
    chunk_ids: Sequence[str], chunk_to_document: Mapping[str, str]
) -> list[str]:
    return _unique_strings(
        chunk_to_document[chunk_id]
        for chunk_id in chunk_ids
        if chunk_id in chunk_to_document
    )


async def export_workspace(
    rag: Any,
    *,
    run_id: str,
    run_dir: str | Path,
    documents_path: str | Path | None = None,
) -> dict[str, int]:
    """Append missing chunk/node/edge records and return export counts.

    The caller must initialize LightRAG storages first.  No LLM or embedding
    function is invoked by this routine.
    """

    directory = Path(run_dir).resolve()
    artifacts = directory / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    chunk_path = artifacts / "chunks.jsonl"
    node_path = artifacts / "graph_nodes.jsonl"
    edge_path = artifacts / "graph_edges.jsonl"
    for raw_path in (chunk_path, node_path, edge_path):
        touch_append_only(raw_path)
    existing_chunks = _load_existing(chunk_path, "chunk_id", run_id)
    existing_nodes = _load_existing(node_path, "node_id", run_id)
    existing_edges = _load_existing(edge_path, "edge_id", run_id)
    document_catalog = _load_document_catalog(documents_path)

    (
        chunk_values,
        status_chunk_documents,
        status_provenance,
    ) = await _collect_actual_chunks(rag)
    chunk_to_document: dict[str, str] = dict(status_chunk_documents)
    chunks_by_id: dict[str, dict[str, Any]] = {}
    appended_chunks = 0

    for chunk_value in chunk_values:
        chunk = dict(chunk_value)
        chunk_id = str(chunk.get("_id") or "")
        if not chunk_id:
            raise RuntimeError("LightRAG text chunk lacks storage _id")
        document_id = str(
            chunk.get("full_doc_id") or status_chunk_documents.get(chunk_id) or ""
        )
        if not document_id:
            raise RuntimeError(f"LightRAG chunk {chunk_id!r} lacks document provenance")
        if (
            chunk_id in status_chunk_documents
            and status_chunk_documents[chunk_id] != document_id
        ):
            raise RuntimeError(
                f"conflicting document provenance for chunk {chunk_id!r}"
            )
        text = chunk.get("content")
        if not isinstance(text, str):
            raise RuntimeError(f"LightRAG chunk {chunk_id!r} lacks source text")
        chunk_to_document[chunk_id] = document_id
        chunks_by_id[chunk_id] = chunk
        technical = {
            "lightrag_storage_record": _json_safe(chunk),
            **status_provenance.get(chunk_id, {}),
        }
        if document_id in document_catalog:
            technical["pilot_document"] = document_catalog[document_id]
        artifact = ChunkArtifact(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            text_sha256=sha256_text(text),
            token_count=_count_chunk_tokens(rag, chunk, text),
            chunk_order=_chunk_order(chunk),
            technical_provenance=technical,
        )
        appended_chunks += int(
            _append_if_new(chunk_path, "chunk_id", artifact, existing_chunks)
        )

    raw_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
    raw_edges = await rag.chunk_entity_relation_graph.get_all_edges()
    appended_nodes = 0
    appended_edges = 0

    for raw_node in sorted(
        raw_nodes,
        key=lambda item: str(
            item.get("id") or item.get("entity_id") or item.get("name") or ""
        ),
    ):
        node = dict(raw_node)
        node_id_value = node.get("id") or node.get("entity_id") or node.get("name")
        if node_id_value is None or not str(node_id_value):
            raise RuntimeError("LightRAG graph node lacks an identifier")
        node_id = str(node_id_value)
        capped_chunk_ids = _split_lightrag_field(node.get("source_id"))
        full_chunk_ids, tracking = await _full_chunk_provenance(
            getattr(rag, "entity_chunks", None), node_id, capped_chunk_ids
        )
        document_ids = _documents_for_chunks(full_chunk_ids, chunk_to_document)
        artifact = GraphNodeArtifact(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            node_id=node_id,
            attributes=_json_safe(node),
            source_chunk_ids=full_chunk_ids,
            document_ids=document_ids,
            technical_provenance={
                "capped_graph_source_chunk_ids": capped_chunk_ids,
                "full_chunk_tracking": _json_safe(tracking),
                "source_file_paths": _split_lightrag_field(node.get("file_path")),
                "unresolved_source_chunk_ids": [
                    chunk_id
                    for chunk_id in full_chunk_ids
                    if chunk_id not in chunk_to_document
                ],
            },
        )
        appended_nodes += int(
            _append_if_new(node_path, "node_id", artifact, existing_nodes)
        )

    def edge_sort_key(item: Mapping[str, Any]) -> tuple[str, str]:
        return (str(item.get("source", "")), str(item.get("target", "")))

    try:
        from lightrag.utils import make_relation_chunk_key
    except ImportError:

        def make_relation_chunk_key(source: str, target: str) -> str:  # type: ignore
            return "<SEP>".join(sorted((source, target)))

    for raw_edge in sorted(raw_edges, key=edge_sort_key):
        edge = dict(raw_edge)
        if edge.get("source") is None or edge.get("target") is None:
            raise RuntimeError("LightRAG graph edge lacks source/target")
        source = str(edge["source"])
        target = str(edge["target"])
        normalized_source, normalized_target = sorted((source, target))
        edge_id = stable_id("edge_", normalized_source, normalized_target)
        relation_key = make_relation_chunk_key(source, target)
        capped_chunk_ids = _split_lightrag_field(edge.get("source_id"))
        full_chunk_ids, tracking = await _full_chunk_provenance(
            getattr(rag, "relation_chunks", None), relation_key, capped_chunk_ids
        )
        document_ids = _documents_for_chunks(full_chunk_ids, chunk_to_document)
        artifact = GraphEdgeArtifact(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            edge_id=edge_id,
            source=source,
            target=target,
            attributes=_json_safe(edge),
            source_chunk_ids=full_chunk_ids,
            document_ids=document_ids,
            technical_provenance={
                "relation_chunk_storage_key": relation_key,
                "capped_graph_source_chunk_ids": capped_chunk_ids,
                "full_chunk_tracking": _json_safe(tracking),
                "source_file_paths": _split_lightrag_field(edge.get("file_path")),
                "unresolved_source_chunk_ids": [
                    chunk_id
                    for chunk_id in full_chunk_ids
                    if chunk_id not in chunk_to_document
                ],
            },
        )
        appended_edges += int(
            _append_if_new(edge_path, "edge_id", artifact, existing_edges)
        )

    return {
        "chunks_total": len(chunk_values),
        "chunks_appended": appended_chunks,
        "nodes_total": len(raw_nodes),
        "nodes_appended": appended_nodes,
        "edges_total": len(raw_edges),
        "edges_appended": appended_edges,
    }


__all__ = ["export_workspace"]
