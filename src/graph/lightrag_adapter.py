"""Read, write, and validate experiment workspaces through LightRAG.

Chunks are stored with extraction disabled, and nodes and edges are written
with LightRAG's create methods. Full graph reads and exact chunk reads use the
documented read-only storage interfaces described in ``LIGHTRAG_PARITY.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class LightRAGAdapterError(RuntimeError):
    """Raised when a LightRAG workspace operation fails validation."""


def _value(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _required_text(record: Any, names: tuple[str, ...], label: str) -> str:
    value = _value(record, *names)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise LightRAGAdapterError(f"{label} is required ({'/'.join(names)})")
    return text


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise LightRAGAdapterError(
        f"expected mapping-like value, got {type(value).__name__}"
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _status_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().casefold()


@dataclass(frozen=True, order=True)
class ChunkSnapshot:
    document_id: str
    chunk_order: int
    chunk_id: str
    content: str = field(compare=False)
    token_count: int | None = field(default=None, compare=False)

    @classmethod
    def from_record(cls, record: Any) -> "ChunkSnapshot":
        document_id = _required_text(
            record, ("document_id", "full_doc_id"), "chunk document_id"
        )
        chunk_id = _required_text(record, ("chunk_id", "_id", "id"), "chunk_id")
        content = _value(record, "content", "text")
        if not isinstance(content, str):
            raise LightRAGAdapterError(f"chunk {chunk_id!r} has no string content/text")
        order_raw = _value(record, "chunk_order", "chunk_order_index", "order")
        if not isinstance(order_raw, int) or order_raw < 0:
            raise LightRAGAdapterError(
                f"chunk {chunk_id!r} has no non-negative integer order"
            )
        token_raw = _value(record, "token_count", "tokens")
        token_count = (
            token_raw if isinstance(token_raw, int) and token_raw >= 0 else None
        )
        return cls(document_id, order_raw, chunk_id, content, token_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_order_index": self.chunk_order,
            "tokens": self.token_count,
        }


@dataclass
class StagedChunkResult:
    track_id: str
    expected_chunks: list[ChunkSnapshot]
    stored_chunks: list[dict[str, Any]]
    document_statuses: dict[str, dict[str, Any]]
    chunk_read_api: str


@dataclass
class WorkspaceSnapshot:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    graph_is_truncated: bool
    chunk_read_api: str


class SnapshotChunker:
    """Replay immutable chunks through LightRAG's public skip-KG pipeline.

    The legacy chunker callback does not receive ``document_id``.  Therefore the
    immutable document body hash must identify exactly one document.  Duplicate
    corpus bodies fail closed rather than silently receiving another document's
    chunk ids.
    """

    def __init__(self, documents: Sequence[Any], chunks: Sequence[Any]):
        self._document_by_hash: dict[str, tuple[str, str]] = {}
        self._chunks_by_document: dict[str, list[ChunkSnapshot]] = defaultdict(list)
        self._used_documents: set[str] = set()

        document_ids: set[str] = set()
        for document in documents:
            document_id = _required_text(document, ("document_id", "id"), "document_id")
            text = _value(document, "text", "content")
            if not isinstance(text, str):
                raise LightRAGAdapterError(
                    f"document {document_id!r} has no string text/content"
                )
            if document_id in document_ids:
                raise LightRAGAdapterError(f"duplicate document_id {document_id!r}")
            document_ids.add(document_id)
            digest = _sha256_text(text)
            previous = self._document_by_hash.get(digest)
            if previous is not None:
                raise LightRAGAdapterError(
                    "snapshot chunker cannot disambiguate identical document bodies: "
                    f"{previous[0]!r} and {document_id!r}"
                )
            self._document_by_hash[digest] = (document_id, text)

        chunk_ids: set[str] = set()
        orders_by_document: dict[str, set[int]] = defaultdict(set)
        for record in chunks:
            chunk = ChunkSnapshot.from_record(record)
            if chunk.document_id not in document_ids:
                raise LightRAGAdapterError(
                    f"chunk {chunk.chunk_id!r} belongs to unknown document {chunk.document_id!r}"
                )
            if chunk.chunk_id in chunk_ids:
                raise LightRAGAdapterError(
                    f"duplicate snapshot chunk_id {chunk.chunk_id!r}"
                )
            if chunk.chunk_order in orders_by_document[chunk.document_id]:
                raise LightRAGAdapterError(
                    f"duplicate chunk order {chunk.chunk_order} in {chunk.document_id!r}"
                )
            # build_chunks_dict_from_chunking_result preserves an explicit id
            # only when it is already document-prefixed.
            if not chunk.chunk_id.startswith(f"{chunk.document_id}-"):
                raise LightRAGAdapterError(
                    f"snapshot chunk id {chunk.chunk_id!r} is not prefixed by "
                    f"document id {chunk.document_id!r}; LightRAG would rewrite it"
                )
            chunk_ids.add(chunk.chunk_id)
            orders_by_document[chunk.document_id].add(chunk.chunk_order)
            self._chunks_by_document[chunk.document_id].append(chunk)

        missing = sorted(document_ids - set(self._chunks_by_document))
        if missing:
            raise LightRAGAdapterError(
                f"{len(missing)} document(s) have no staged chunks; first={missing[0]!r}"
            )
        for document_id in self._chunks_by_document:
            self._chunks_by_document[document_id].sort()

    @property
    def expected_chunks(self) -> list[ChunkSnapshot]:
        return [
            chunk
            for document_id in sorted(self._chunks_by_document)
            for chunk in self._chunks_by_document[document_id]
        ]

    def __call__(
        self,
        tokenizer: Any,
        content: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        chunk_overlap_token_size: int = 100,
        chunk_token_size: int = 1200,
    ) -> list[dict[str, Any]]:
        del split_by_character, split_by_character_only
        del chunk_overlap_token_size, chunk_token_size
        digest = _sha256_text(content)
        document = self._document_by_hash.get(digest)
        if document is None or document[1] != content:
            raise LightRAGAdapterError(
                "LightRAG supplied document content not present in immutable snapshot"
            )
        document_id = document[0]
        if document_id in self._used_documents:
            raise LightRAGAdapterError(
                f"snapshot chunker invoked more than once for document {document_id!r}"
            )
        self._used_documents.add(document_id)
        result: list[dict[str, Any]] = []
        for chunk in self._chunks_by_document[document_id]:
            token_count = chunk.token_count
            if token_count is None:
                encoder = getattr(tokenizer, "encode", None)
                if not callable(encoder):
                    raise LightRAGAdapterError(
                        f"chunk {chunk.chunk_id!r} has no token count and tokenizer cannot encode"
                    )
                token_count = len(encoder(chunk.content))
            result.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "tokens": token_count,
                    "chunk_order_index": chunk.chunk_order,
                }
            )
        return result

    def assert_complete(self) -> None:
        expected = {document_id for document_id, _ in self._document_by_hash.values()}
        missing = sorted(expected - self._used_documents)
        if missing:
            raise LightRAGAdapterError(
                f"LightRAG did not invoke snapshot chunker for {len(missing)} document(s); "
                f"first={missing[0]!r}"
            )


class PublicLightRAGAdapter:
    """LightRAG workspace operations and exact read-only chunk access."""

    def __init__(self, rag: Any):
        self.rag = rag
        self.chunk_read_api = "unresolved"

    async def initialize(self) -> None:
        await self.rag.initialize_storages()

    async def finalize(self) -> None:
        await self.rag.finalize_storages()

    async def assert_clean_workspace(self) -> None:
        labels = list(await self.rag.get_graph_labels())
        if labels:
            raise LightRAGAdapterError(
                f"advanced workspace is not clean: {len(labels)} graph node(s) already exist"
            )
        counts = await self.rag.get_processing_status()
        if isinstance(counts, Mapping):
            total = sum(
                int(value)
                for value in counts.values()
                if isinstance(value, (int, float))
            )
            if total:
                raise LightRAGAdapterError(
                    f"advanced workspace is not clean: {total} document status record(s) exist"
                )

    async def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        facade = getattr(self.rag, "aget_chunks_by_ids", None)
        if callable(facade):
            values = await facade(list(chunk_ids))
            self.chunk_read_api = "public_facade:aget_chunks_by_ids"
        else:
            storage = getattr(self.rag, "text_chunks", None)
            getter = getattr(storage, "get_by_ids", None)
            if not callable(getter):
                raise LightRAGAdapterError(
                    "this LightRAG checkout has no public facade for exact chunk reads "
                    "and text_chunks.get_by_ids is unavailable"
                )
            values = await getter(list(chunk_ids))
            self.chunk_read_api = "storage_interface:text_chunks.get_by_ids"
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise LightRAGAdapterError("chunk read API did not return a sequence")
        if len(values) != len(chunk_ids):
            raise LightRAGAdapterError(
                "chunk read API did not preserve requested cardinality: "
                f"expected {len(chunk_ids)}, found {len(values)}"
            )
        result: list[dict[str, Any]] = []
        for requested_id, value in zip(chunk_ids, values, strict=True):
            if value is None:
                raise LightRAGAdapterError(f"stored chunk {requested_id!r} is missing")
            record = _as_mapping(value)
            stored_id = str(record.get("_id") or record.get("chunk_id") or requested_id)
            if stored_id != requested_id:
                raise LightRAGAdapterError(
                    f"chunk read id mismatch: requested {requested_id!r}, found {stored_id!r}"
                )
            record["_id"] = requested_id
            result.append(record)
        return result

    async def stage_snapshot_chunks(
        self,
        documents: Sequence[Any],
        chunks: Sequence[Any],
        *,
        track_id: str,
    ) -> StagedChunkResult:
        if not track_id.strip():
            raise LightRAGAdapterError("track_id must be non-empty")
        snapshot_chunker = SnapshotChunker(documents, chunks)
        document_ids = [
            _required_text(document, ("document_id", "id"), "document_id")
            for document in documents
        ]
        texts = []
        file_paths = []
        for document, document_id in zip(documents, document_ids, strict=True):
            text = _value(document, "text", "content")
            if not isinstance(text, str):
                raise LightRAGAdapterError(
                    f"document {document_id!r} has no string text/content"
                )
            texts.append(text)
            file_paths.append(
                str(_value(document, "file_path", default=document_id) or document_id)
            )

        original_chunker = getattr(self.rag, "chunking_func", None)
        self.rag.chunking_func = snapshot_chunker
        try:
            returned_track_id = await self.rag.apipeline_enqueue_documents(
                texts,
                ids=document_ids,
                file_paths=file_paths,
                track_id=track_id,
                process_options="!",
            )
            if returned_track_id != track_id:
                raise LightRAGAdapterError(
                    f"LightRAG changed track id {track_id!r} to {returned_track_id!r}"
                )
            await self.rag.apipeline_process_enqueue_documents()
        finally:
            self.rag.chunking_func = original_chunker
        snapshot_chunker.assert_complete()

        raw_statuses = await self.rag.aget_docs_by_track_id(track_id)
        if not isinstance(raw_statuses, Mapping):
            raise LightRAGAdapterError("aget_docs_by_track_id did not return a mapping")
        statuses: dict[str, dict[str, Any]] = {}
        expected_ids = set(document_ids)
        for document_id, raw_status in raw_statuses.items():
            if str(document_id) in expected_ids:
                statuses[str(document_id)] = _as_mapping(raw_status)
        missing_status = sorted(expected_ids - set(statuses))
        if missing_status:
            raise LightRAGAdapterError(
                f"missing processed status for document {missing_status[0]!r}"
            )
        for document_id, status in statuses.items():
            if _status_text(status.get("status")) != "processed":
                raise LightRAGAdapterError(
                    f"document {document_id!r} did not reach PROCESSED: "
                    f"{status.get('status')!r}"
                )

        expected_chunks = snapshot_chunker.expected_chunks
        stored_chunks = await self.get_chunks_by_ids(
            [chunk.chunk_id for chunk in expected_chunks]
        )
        return StagedChunkResult(
            track_id=track_id,
            expected_chunks=expected_chunks,
            stored_chunks=stored_chunks,
            document_statuses=statuses,
            chunk_read_api=self.chunk_read_api,
        )

    async def create_entity(self, name: str, data: Mapping[str, Any]) -> dict[str, Any]:
        result = await self.rag.acreate_entity(name, dict(data))
        return _as_mapping(result)

    async def create_relation(
        self, source: str, target: str, data: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await self.rag.acreate_relation(source, target, dict(data))
        return _as_mapping(result)

    async def export_workspace(
        self,
        *,
        expected_node_count: int,
        chunk_ids: Sequence[str],
    ) -> WorkspaceSnapshot:
        del expected_node_count
        graph_storage = self.rag.chunk_entity_relation_graph
        nodes = [
            _as_mapping(value) for value in await graph_storage.get_all_nodes()
        ]
        edges = [
            _as_mapping(value) for value in await graph_storage.get_all_edges()
        ]
        chunks = await self.get_chunks_by_ids(chunk_ids)
        return WorkspaceSnapshot(
            nodes=nodes,
            edges=edges,
            chunks=chunks,
            graph_is_truncated=False,
            chunk_read_api=self.chunk_read_api,
        )

    async def smoke_retrieval(
        self,
        cases: Sequence[Mapping[str, Any]],
        *,
        query_param_factory: Callable[[str], Any] | None = None,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        if query_param_factory is None:
            from lightrag import QueryParam

            def default_query_param_factory(selected_mode: str) -> Any:
                return QueryParam(
                    mode=selected_mode, only_need_context=True, stream=False
                )

            query_param_factory = default_query_param_factory
        results: list[dict[str, Any]] = []
        for case in cases:
            query = _required_text(case, ("query", "question"), "smoke query")
            expected_status = str(case.get("expected_status", "success")).casefold()
            raw = await self.rag.aquery_data(query, query_param_factory(mode))
            response = _as_mapping(raw)
            actual_status = str(response.get("status", "failure")).casefold()
            passed = actual_status == expected_status
            results.append(
                {
                    "query": query,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "passed": passed,
                    "result": response,
                }
            )
        return results


__all__ = [
    "ChunkSnapshot",
    "LightRAGAdapterError",
    "PublicLightRAGAdapter",
    "SnapshotChunker",
    "StagedChunkResult",
    "WorkspaceSnapshot",
]
