"""Build and verify an immutable corpus extraction snapshot."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import (
    NormalizedChunkResult,
    NormalizedEntityMention,
    NormalizedRelation,
    StagingManifest,
)
from .normalization import normalize_extraction_calls


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable staging artifact conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _document_ids(path: Path) -> list[str]:
    result: list[str] = []
    for row in read_jsonl(path):
        document_id = str(row.get("document_id") or "").strip()
        if not document_id:
            raise ValueError(f"document row lacks document_id: {path}")
        result.append(document_id)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate document IDs: {path}")
    return result


def _validate_raw_lineage(
    chunks: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> None:
    chunk_catalog = {
        str(chunk.get("chunk_id") or ""): str(chunk.get("document_id") or "")
        for chunk in chunks
    }
    if "" in chunk_catalog:
        raise ValueError("chunk export contains a row without chunk_id")
    if len(chunk_catalog) != len(chunks):
        raise ValueError("chunk IDs are not unique")
    call_ids: set[str] = set()
    for call in calls:
        call_id = str(call.get("call_id") or "")
        chunk_id = str(call.get("chunk_id") or "")
        document_id = str(call.get("document_id") or "")
        if not call_id or not chunk_id:
            raise ValueError("raw extraction call lacks call_id or chunk_id")
        if call_id in call_ids:
            # Physical retries intentionally share a logical call_id only when
            # attempt_number differs. Exact duplicate attempt rows are invalid.
            key = (call_id, int(call.get("attempt_number") or 1))
            duplicates = [
                item
                for item in calls
                if str(item.get("call_id") or "") == key[0]
                and int(item.get("attempt_number") or 1) == key[1]
            ]
            if len(duplicates) > 1:
                raise ValueError(f"duplicate extraction physical attempt: {key}")
        call_ids.add(call_id)
        if chunk_id not in chunk_catalog:
            raise ValueError(f"raw extraction call references unknown chunk {chunk_id}")
        if document_id and chunk_catalog[chunk_id] != document_id:
            raise ValueError(f"raw extraction call document mismatch for {chunk_id}")


def _dump_rows(
    values: Iterable[Any], *, base_run_id: str
) -> tuple[list[dict[str, Any]], bytes, str]:
    rows = [
        {
            "base_run_id": base_run_id,
            **value.model_dump(mode="json", exclude_none=False),
        }
        for value in values
    ]
    content = _jsonl_bytes(rows)
    return rows, content, sha256_bytes(content)


def stage_native_artifacts(
    native_run_dir: str | Path,
    *,
    documents_path: str | Path,
    questions_path: str | Path,
    corpus_manifest_path: str | Path,
    base_run_id: str,
    builder_model: str,
    builder_model_digest: str,
    extraction_prompt_version: str,
    seed: int,
    generation_parameters: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> StagingManifest:
    """Normalize a completed native run without making any model call.

    The native workspace and its append-only raw artifacts are read-only. The
    derived snapshot is written under ``artifacts/extraction`` unless an
    explicit separate destination is supplied.
    """

    run_dir = Path(native_run_dir).resolve()
    artifacts_dir = run_dir / "artifacts"
    chunks_path = artifacts_dir / "chunks.jsonl"
    calls_path = artifacts_dir / "extraction_calls.jsonl"
    for path in (chunks_path, calls_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    documents = Path(documents_path).resolve()
    questions = Path(questions_path).resolve()
    corpus_manifest = Path(corpus_manifest_path).resolve()
    for path in (documents, questions, corpus_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_payload = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError("corpus manifest must be a JSON object")
    subset_id = str(manifest_payload.get("subset_id") or "").strip()
    if not subset_id:
        raise ValueError("corpus manifest lacks subset_id")
    chunks = read_jsonl(chunks_path)
    calls = read_jsonl(calls_path)
    if not chunks:
        raise ValueError("native workspace export contains no chunks")
    _validate_raw_lineage(chunks, calls)

    expected_document_ids = _document_ids(documents)
    expected_set = set(expected_document_ids)
    chunk_documents = {str(row.get("document_id") or "") for row in chunks}
    if chunk_documents != expected_set:
        missing = sorted(expected_set - chunk_documents)
        unexpected = sorted(chunk_documents - expected_set)
        raise ValueError(
            "staged chunk corpus differs from immutable documents: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    normalized = normalize_extraction_calls(chunks, calls)
    if len(normalized) != len(chunks):
        raise RuntimeError("normalizer did not emit exactly one result per chunk")
    entities = [item for chunk in normalized for item in chunk.entities]
    relations = [item for chunk in normalized for item in chunk.relations]
    mention_ids = [item.mention_id for item in entities]
    relation_ids = [item.relation_id for item in relations]
    if len(mention_ids) != len(set(mention_ids)):
        raise RuntimeError("mention IDs are not globally unique")
    if len(relation_ids) != len(set(relation_ids)):
        raise RuntimeError("relation IDs are not globally unique")

    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else artifacts_dir / "extraction"
    )
    chunk_rows, chunk_content, chunk_hash = _dump_rows(
        normalized, base_run_id=base_run_id
    )
    _, entity_content, entity_hash = _dump_rows(entities, base_run_id=base_run_id)
    _, relation_content, relation_hash = _dump_rows(relations, base_run_id=base_run_id)
    _write_immutable(target / "normalized_chunks.jsonl", chunk_content)
    _write_immutable(target / "normalized_entities.jsonl", entity_content)
    _write_immutable(target / "normalized_relations.jsonl", relation_content)

    failed_parse_chunks = [
        item.chunk_id
        for item in normalized
        if any(
            record.status in {"failed", "no_response"} for record in item.parse_records
        )
    ]
    staged_by_document: dict[str, int] = defaultdict(int)
    for item in normalized:
        staged_by_document[item.document_id] += 1
    completed_documents = sum(
        1
        for document_id in expected_document_ids
        if staged_by_document[document_id] > 0
    )
    input_hashes = {
        "corpus_manifest": sha256_file(corpus_manifest),
        "documents": sha256_file(documents),
        "questions": sha256_file(questions),
        "chunks": sha256_file(chunks_path),
        "raw_extraction_calls": sha256_file(calls_path),
    }
    identity = {
        "base_run_id": base_run_id,
        "subset_id": subset_id,
        "input_hashes": input_hashes,
        "normalized_chunks_sha256": chunk_hash,
        "normalized_entities_sha256": entity_hash,
        "normalized_relations_sha256": relation_hash,
        "builder_model": builder_model,
        "builder_model_digest": builder_model_digest,
        "extraction_prompt_version": extraction_prompt_version,
        "seed": seed,
        "generation_parameters": dict(generation_parameters),
    }
    staged_manifest = StagingManifest(
        base_run_id=base_run_id,
        subset_id=subset_id,
        corpus_manifest_sha256=input_hashes["corpus_manifest"],
        documents_sha256=input_hashes["documents"],
        questions_sha256=input_hashes["questions"],
        chunks_sha256=input_hashes["chunks"],
        raw_extraction_calls_sha256=input_hashes["raw_extraction_calls"],
        normalized_chunks_sha256=chunk_hash,
        normalized_entities_sha256=entity_hash,
        normalized_relations_sha256=relation_hash,
        base_extraction_sha256=sha256_json(identity),
        expected_documents=len(expected_document_ids),
        completed_documents=completed_documents,
        expected_chunks=len(chunks),
        staged_chunks=len(chunk_rows),
        failed_parse_chunks=failed_parse_chunks,
        builder_model=builder_model,
        builder_model_digest=builder_model_digest,
        extraction_prompt_version=extraction_prompt_version,
        seed=seed,
        generation_parameters=dict(generation_parameters),
        input_hashes=input_hashes,
    )
    manifest_content = (
        json.dumps(
            staged_manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_immutable(target / "staging_manifest.json", manifest_content)
    return staged_manifest


def _rows_for_base(path: Path, base_run_id: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        if row.pop("base_run_id", None) != base_run_id:
            raise RuntimeError(f"staging row belongs to another base run: {path}")
    return rows


def load_staged_snapshot(
    staging_dir: str | Path,
) -> tuple[
    StagingManifest,
    list[NormalizedChunkResult],
    list[NormalizedEntityMention],
    list[NormalizedRelation],
]:
    """Load a snapshot only after verifying every immutable artifact hash."""

    directory = Path(staging_dir).resolve()
    manifest = StagingManifest.model_validate_json(
        (directory / "staging_manifest.json").read_text(encoding="utf-8")
    )
    paths = {
        "chunks": directory / "normalized_chunks.jsonl",
        "entities": directory / "normalized_entities.jsonl",
        "relations": directory / "normalized_relations.jsonl",
    }
    expected = {
        "chunks": manifest.normalized_chunks_sha256,
        "entities": manifest.normalized_entities_sha256,
        "relations": manifest.normalized_relations_sha256,
    }
    for name, path in paths.items():
        if sha256_file(path) != expected[name]:
            raise RuntimeError(f"staging artifact hash mismatch: {path}")
    chunks = [
        NormalizedChunkResult.model_validate(row)
        for row in _rows_for_base(paths["chunks"], manifest.base_run_id)
    ]
    entities = [
        NormalizedEntityMention.model_validate(row)
        for row in _rows_for_base(paths["entities"], manifest.base_run_id)
    ]
    relations = [
        NormalizedRelation.model_validate(row)
        for row in _rows_for_base(paths["relations"], manifest.base_run_id)
    ]
    if len(chunks) != manifest.staged_chunks:
        raise RuntimeError("staging manifest chunk count mismatch")
    return manifest, chunks, entities, relations


__all__ = [
    "canonical_json",
    "load_staged_snapshot",
    "read_jsonl",
    "sha256_file",
    "sha256_json",
    "stage_native_artifacts",
]
