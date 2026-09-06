"""Run resumable Native LightRAG indexing for one fixed configuration.

Only each document's ``text`` field is sent to LightRAG. Stable
``document_id`` values are used as LightRAG IDs and citation file paths; other
metadata does not enter the chunk text. Per-document state is stored in SQLite,
while extraction calls and event logs are append-only JSONL.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config.native_runtime import (
    FrozenRun,
    build_run_config,
    lightrag_source_identity,
    load_run_config,
    make_run_id,
    prepare_run,
    validate_frozen_inputs,
    validate_lightrag_runtime,
)
from src.config.native_support.hashes import sha256_text
from src.config.native_support.io_utils import read_jsonl
from src.extraction.native_capture import (
    build_lightrag,
    get_extraction_adapter,
    resolve_ollama_model_identity,
)
from src.graph.native_export import export_workspace


INDEXING_STAGE = "indexing"
EXPORT_STAGE = "workspace_export"
EXPORT_ITEM = "chunks_and_graph"


class BatchIndexingError(RuntimeError):
    """Raised after LightRAG persisted one or more failed document statuses."""


def _quantization_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _validate_frozen_runtime_identity(frozen_run: FrozenRun) -> None:
    config = frozen_run.lock.config
    missing = [
        field
        for field in (
            "builder_model_digest",
            "keyword_model_digest",
            "answer_model_digest",
            "embedding_model_digest",
        )
        if not getattr(config, field)
    ]
    if getattr(config, "query_model", None) and not getattr(
        config, "query_model_digest", None
    ):
        missing.append("query_model_digest")
    if missing or config.quantization.casefold() in {"auto", "resolve", "unknown"}:
        details = ", ".join(missing) if missing else "quantization"
        raise RuntimeError(
            "frozen run lacks preflight-resolved model identity fields "
            f"({details}); create it with run_indexing.py --config"
        )

    validate_lightrag_runtime(frozen_run)


async def _preflight_config(args: argparse.Namespace) -> FrozenRun:
    """Resolve immutable Ollama identities before config.lock.json is created."""

    source_config = load_run_config(args.config)
    settings = dict(source_config.lightrag or {})
    base_host = settings.get("ollama_host")
    role_specs = {
        "builder": (
            source_config.builder_model,
            settings.get("llm_host", base_host),
        ),
        "embedding": (
            source_config.embedding_model,
            settings.get("embedding_host", base_host),
        ),
        "keyword": (
            source_config.keyword_model or source_config.answer_model,
            source_config.keyword.get("host")
            or source_config.keyword.get("ollama_host")
            or settings.get("llm_host", base_host),
        ),
        "answer": (
            source_config.answer_model,
            source_config.answer.get("host")
            or source_config.answer.get("ollama_host")
            or base_host,
        ),
    }
    query_model = getattr(source_config, "query_model", None)
    query_settings = dict(getattr(source_config, "query", None) or {})
    if query_model:
        role_specs["query"] = (
            query_model,
            query_settings.get("host")
            or query_settings.get("ollama_host")
            or settings.get("llm_host", base_host),
        )
    if not role_specs["embedding"][0]:
        raise ValueError("embedding_model is required")

    cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    identities: dict[str, dict[str, Any]] = {}
    for role, (model_name_raw, host_raw) in role_specs.items():
        model_name = str(model_name_raw)
        host = str(host_raw) if host_raw is not None else None
        key = (model_name, host)
        if key not in cache:
            cache[key] = await resolve_ollama_model_identity(model_name, host=host)
        identity = cache[key]
        if not identity or not identity.get("digest"):
            raise RuntimeError(
                f"Ollama model {model_name!r} for role {role!r} is missing or has no digest"
            )
        identities[role] = dict(identity)

    builder_quantization = identities["builder"].get("quantization")
    configured_quantization = source_config.quantization.strip()
    if configured_quantization.casefold() in {"auto", "resolve", "unknown"}:
        if not builder_quantization:
            raise RuntimeError("Ollama did not report builder model quantization")
        effective_quantization = str(builder_quantization)
    else:
        effective_quantization = configured_quantization
        if builder_quantization and _quantization_key(
            effective_quantization
        ) != _quantization_key(str(builder_quantization)):
            raise RuntimeError(
                "configured quantization does not match Ollama: "
                f"configured {effective_quantization}, found {builder_quantization}"
            )

    payload = source_config.model_dump(mode="json", exclude_none=False)
    for role, field_name in (
        ("builder", "builder_model_digest"),
        ("embedding", "embedding_model_digest"),
        ("keyword", "keyword_model_digest"),
        ("answer", "answer_model_digest"),
    ):
        configured = payload.get(field_name)
        resolved = identities[role]["digest"]
        if configured and configured != resolved:
            raise RuntimeError(
                f"configured {field_name} does not match Ollama: "
                f"expected {configured}, found {resolved}"
            )
        payload[field_name] = resolved
    if "query" in identities:
        configured = payload.get("query_model_digest")
        resolved = identities["query"]["digest"]
        if configured and configured != resolved:
            raise RuntimeError(
                "configured query_model_digest does not match Ollama: "
                f"expected {configured}, found {resolved}"
            )
        payload["query_model_digest"] = resolved
    payload["quantization"] = effective_quantization
    source_identity = lightrag_source_identity()
    if not source_identity.get("git_commit"):
        raise RuntimeError("unable to resolve the local LightRAG checkout commit")
    if not source_identity.get("python_tree_sha256"):
        raise RuntimeError("unable to fingerprint the local LightRAG source tree")
    metadata = dict(payload.get("metadata") or {})
    metadata["ollama_model_identities"] = identities
    metadata["expected_lightrag_git_commit"] = source_identity["git_commit"]
    metadata["expected_lightrag_git_describe"] = source_identity.get("git_describe")
    metadata["expected_lightrag_source_sha256"] = source_identity["python_tree_sha256"]
    payload["metadata"] = metadata
    effective = build_run_config(
        payload,
        input_base_dir=source_config._input_base_dir,
        validate_inputs=True,
    )
    ollama_executable = shutil.which("ollama")
    ollama_version: str | None = None
    if ollama_executable:
        try:
            ollama_version = subprocess.run(
                [ollama_executable, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            ollama_version = None
    run_id = make_run_id(effective)
    return prepare_run(
        effective,
        runs_root=args.runs_root,
        input_base_dir=source_config._input_base_dir,
        environment_extra={
            "ollama_cli_version": ollama_version,
            "ollama_model_identities": identities,
            "workspace_dir": str(Path(args.runs_root).resolve() / run_id / "workspace"),
            "lightrag_workspace": f"pilot_{sha256_text(run_id)[:20]}",
            "lightrag_source_checkout": source_identity,
        },
    )


def _load_documents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    text_hashes: dict[str, str] = {}
    for line_number, raw in enumerate(read_jsonl(path), start=1):
        assert isinstance(raw, dict)
        document_id = str(raw.get("document_id", "")).strip()
        text = raw.get("text")
        expected_hash = str(raw.get("text_sha256", "")).strip().lower()
        if not document_id:
            raise ValueError(f"{path}:{line_number}: missing document_id")
        if document_id in ids:
            raise ValueError(
                f"{path}:{line_number}: duplicate document_id {document_id!r}"
            )
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number}: document text must be non-empty")
        actual_hash = sha256_text(text)
        if expected_hash != actual_hash:
            raise ValueError(
                f"{path}:{line_number}: text_sha256 mismatch for {document_id!r}"
            )
        prior_id = text_hashes.get(actual_hash)
        if prior_id is not None:
            raise ValueError(
                "input contains duplicate document text, which LightRAG would "
                f"content-deduplicate: {prior_id!r} and {document_id!r}"
            )
        ids.add(document_id)
        text_hashes[actual_hash] = document_id
        records.append(dict(raw))
    if not records:
        raise ValueError(f"document file is empty: {path}")
    return records


def _status_value(record: Mapping[str, Any] | None) -> str | None:
    if not record:
        return None
    value = record.get("status")
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().lower() if value is not None else None


async def _lightrag_document_status(
    rag: Any, document_id: str
) -> dict[str, Any] | None:
    value = await rag.doc_status.get_by_id(document_id)
    return dict(value) if isinstance(value, Mapping) else None


async def _enumerate_lightrag_document_ids(rag: Any) -> set[str]:
    page = 1
    rows: list[tuple[str, Any]] = []
    while True:
        current, total = await rag.doc_status.get_docs_paginated(
            page=page,
            page_size=200,
            sort_field="id",
            sort_direction="asc",
        )
        rows.extend(current)
        if not current or len(rows) >= int(total):
            break
        page += 1
    return {str(document_id) for document_id, _ in rows}


async def _reconcile_status(
    rag: Any,
    status_store: Any,
    documents: Sequence[Mapping[str, Any]],
) -> None:
    """Reconcile a committed LightRAG document after an interrupted run."""

    for document in documents:
        document_id = str(document["document_id"])
        light_status = await _lightrag_document_status(rag, document_id)
        light_value = _status_value(light_status)
        harness_completed = status_store.is_completed(INDEXING_STAGE, document_id)
        if light_value == "processed" and not harness_completed:
            status_store.mark_completed(INDEXING_STAGE, document_id)
        elif harness_completed and light_value != "processed":
            raise RuntimeError(
                f"SQLite marks {document_id!r} completed but LightRAG status is "
                f"{light_value!r}; refusing to reuse an inconsistent workspace"
            )


async def _finish_claimed_batch(
    rag: Any,
    status_store: Any,
    claimed: Sequence[Mapping[str, Any]],
    *,
    call_error: BaseException | None = None,
) -> list[str]:
    failures: list[str] = []
    for document in claimed:
        document_id = str(document["document_id"])
        state = await _lightrag_document_status(rag, document_id)
        value = _status_value(state)
        if value == "processed":
            status_store.mark_completed(INDEXING_STAGE, document_id)
            continue
        error_message = str((state or {}).get("error_msg") or "").strip()
        if not error_message and call_error is not None:
            error_message = str(call_error)
        if not error_message:
            error_message = f"LightRAG finished ainsert with status={value!r}"
        status_store.mark_failed(
            INDEXING_STAGE,
            document_id,
            error_message,
            error_type=(
                type(call_error).__name__
                if call_error is not None
                else "LightRAGStatusError"
            ),
        )
        failures.append(document_id)
    return failures


def _batch_size(frozen_run: FrozenRun) -> int:
    value = (frozen_run.lock.config.lightrag or {}).get("index_batch_size", 4)
    size = int(value)
    if size <= 0:
        raise ValueError("frozen lightrag.index_batch_size must be positive")
    return size


async def run_indexing(
    frozen_run: FrozenRun,
    *,
    export_after_indexing: bool = True,
    reclaim_running: bool = False,
    document_limit: int | None = None,
) -> dict[str, Any]:
    """Index all fixed input documents and optionally export the workspace."""

    stage_started = time.perf_counter()
    _validate_frozen_runtime_identity(frozen_run)
    documents_path, _ = validate_frozen_inputs(frozen_run)
    documents = _load_documents(documents_path)
    if document_limit is not None:
        if document_limit < 1:
            raise ValueError("document_limit must be positive")
        documents = documents[:document_limit]
    document_ids = [str(document["document_id"]) for document in documents]
    logger = frozen_run.event_logger(default_stage=INDEXING_STAGE)
    max_extract_input_tokens = int(
        (frozen_run.lock.config.lightrag or {}).get("max_extract_input_tokens", 0)
    )
    if max_extract_input_tokens != 8192:
        raise ValueError("frozen lightrag.max_extract_input_tokens must be 8192")
    status_store = frozen_run.open_status()
    status_store.initialize_items(INDEXING_STAGE, document_ids)
    if reclaim_running:
        reclaimed = status_store.reset_running(stage=INDEXING_STAGE)
        if reclaimed:
            logger.warning(
                "indexing.running_reclaimed",
                "stale running indexing statuses returned to pending",
                payload={"count": reclaimed},
            )

    rag: Any | None = None
    adapter: Any | None = None
    initialized = False
    indexed_this_invocation = 0
    previous_max_extract_input_tokens = os.environ.get("MAX_EXTRACT_INPUT_TOKENS")
    os.environ["MAX_EXTRACT_INPUT_TOKENS"] = str(max_extract_input_tokens)
    try:
        rag = build_lightrag(frozen_run, extraction_capture=True)
        adapter = get_extraction_adapter(rag)
        logger.info(
            "indexing.started",
            "Native LightRAG indexing stage started",
            payload={
                "documents": len(documents),
                "documents_path": str(documents_path),
                "batch_size": _batch_size(frozen_run),
                "workspace": str(frozen_run.paths.workspace_dir),
                "max_extract_input_tokens": max_extract_input_tokens,
            },
        )
        await rag.initialize_storages()
        initialized = True

        workspace_document_ids = await _enumerate_lightrag_document_ids(rag)
        unexpected = sorted(workspace_document_ids.difference(document_ids))
        if unexpected:
            raise RuntimeError(
                "run workspace contains documents outside the fixed input: "
                + ", ".join(unexpected[:10])
            )

        await _reconcile_status(rag, status_store, documents)

        lightrag_settings = frozen_run.lock.config.lightrag or {}
        host = lightrag_settings.get("llm_host", lightrag_settings.get("ollama_host"))
        builder_identity = await resolve_ollama_model_identity(
            frozen_run.lock.config.builder_model,
            host=str(host) if host is not None else None,
        )
        configured_digest = frozen_run.lock.config.builder_model_digest
        if not builder_identity or builder_identity.get("digest") != configured_digest:
            raise RuntimeError(
                "frozen builder_model_digest does not match the local Ollama model"
            )
        actual_quantization = builder_identity.get("quantization")
        if actual_quantization and _quantization_key(
            str(actual_quantization)
        ) != _quantization_key(frozen_run.lock.config.quantization):
            raise RuntimeError(
                "frozen quantization does not match the local Ollama builder model"
            )
        model_digest = configured_digest
        adapter.model_digest = model_digest
        embedding_host = lightrag_settings.get(
            "embedding_host", lightrag_settings.get("ollama_host")
        )
        embedding_identity = await resolve_ollama_model_identity(
            str(frozen_run.lock.config.embedding_model),
            host=str(embedding_host) if embedding_host is not None else None,
        )
        if (
            not embedding_identity
            or embedding_identity.get("digest")
            != frozen_run.lock.config.embedding_model_digest
        ):
            raise RuntimeError(
                "frozen embedding_model_digest does not match the local Ollama model"
            )
        logger.info(
            "indexing.model_resolved",
            "builder model identity resolved",
            payload={
                "model": frozen_run.lock.config.builder_model,
                "digest": model_digest,
                "quantization": frozen_run.lock.config.quantization,
            },
        )

        batch_size = _batch_size(frozen_run)
        async with adapter:
            for start in range(0, len(documents), batch_size):
                batch = documents[start : start + batch_size]
                claimed: list[dict[str, Any]] = []
                for document in batch:
                    document_id = str(document["document_id"])
                    if status_store.is_completed(INDEXING_STAGE, document_id):
                        continue
                    if status_store.claim(
                        INDEXING_STAGE,
                        document_id,
                        retry_failed=True,
                        reclaim_running=False,
                    ):
                        claimed.append(document)
                if not claimed:
                    continue

                claimed_ids = [str(document["document_id"]) for document in claimed]
                batch_ordinal = f"{start // batch_size:04d}"
                log_item_id = claimed_ids[0] if len(claimed_ids) == 1 else batch_ordinal
                logger.info(
                    "indexing.batch_started",
                    "indexing batch started",
                    item_id=log_item_id,
                    payload={
                        "batch_ordinal": batch_ordinal,
                        "document_ids": claimed_ids,
                    },
                )
                call_error: Exception | None = None
                try:
                    # Exact source text only. Stable document IDs are also used
                    # as LightRAG citation paths; no metadata is prepended.
                    await rag.ainsert(
                        [str(document["text"]) for document in claimed],
                        ids=claimed_ids,
                        file_paths=claimed_ids,
                        track_id=(
                            f"experiment-{frozen_run.lock.run_id[:48]}-"
                            f"{start // batch_size:04d}"
                        ),
                    )
                except Exception as error:
                    call_error = error

                failures = await _finish_claimed_batch(
                    rag,
                    status_store,
                    claimed,
                    call_error=call_error,
                )
                succeeded = len(claimed) - len(failures)
                indexed_this_invocation += succeeded
                logger.log(
                    "indexing.batch_completed"
                    if not failures
                    else "indexing.batch_failed",
                    "indexing batch completed"
                    if not failures
                    else "indexing batch has failures",
                    level="INFO" if not failures else "ERROR",
                    item_id=log_item_id,
                    payload={
                        "batch_ordinal": batch_ordinal,
                        "document_ids": claimed_ids,
                        "succeeded": succeeded,
                        "failed_document_ids": failures,
                        "traceback": (
                            "".join(
                                traceback.format_exception(
                                    type(call_error),
                                    call_error,
                                    call_error.__traceback__,
                                )
                            )
                            if call_error is not None
                            else None
                        ),
                    },
                    error_type=(type(call_error).__name__ if call_error else None),
                    error_message=(str(call_error) if call_error else None),
                )
                if failures:
                    # LightRAG automatically retries all failed workspace docs
                    # on the next processing pass. Stop here so those retries
                    # remain aligned with the SQLite retry attempt.
                    if call_error is not None:
                        raise call_error
                    raise BatchIndexingError(
                        "LightRAG failed documents: " + ", ".join(failures)
                    )

        incomplete = [
            document_id
            for document_id in document_ids
            if not status_store.is_completed(INDEXING_STAGE, document_id)
        ]
        if incomplete:
            running = [
                record.item_id
                for record in status_store.list(stage=INDEXING_STAGE, status="running")
            ]
            hint = (
                " (use --reclaim-running after confirming no process is active)"
                if running
                else ""
            )
            raise RuntimeError(
                f"indexing is incomplete for {len(incomplete)} document(s){hint}"
            )

        export_result: dict[str, int] | None = None
        if export_after_indexing:
            status_store.initialize_items(EXPORT_STAGE, [EXPORT_ITEM])
            if not status_store.is_completed(EXPORT_STAGE, EXPORT_ITEM):
                if not status_store.claim(
                    EXPORT_STAGE,
                    EXPORT_ITEM,
                    retry_failed=True,
                    reclaim_running=reclaim_running,
                ):
                    raise RuntimeError("workspace export is already marked running")
                try:
                    export_result = await export_workspace(
                        rag,
                        run_id=frozen_run.lock.run_id,
                        run_dir=frozen_run.paths.run_dir,
                        documents_path=documents_path,
                    )
                except Exception as error:
                    status_store.mark_failed(EXPORT_STAGE, EXPORT_ITEM, error)
                    raise
                else:
                    status_store.mark_completed(EXPORT_STAGE, EXPORT_ITEM)
            else:
                export_result = {
                    "chunks_total": len(read_jsonl(frozen_run.paths.chunks_jsonl)),
                    "chunks_appended": 0,
                    "nodes_total": len(read_jsonl(frozen_run.paths.graph_nodes_jsonl)),
                    "nodes_appended": 0,
                    "edges_total": len(read_jsonl(frozen_run.paths.graph_edges_jsonl)),
                    "edges_appended": 0,
                }

        result = {
            "run_id": frozen_run.lock.run_id,
            "documents_total": len(documents),
            "documents_indexed_this_invocation": indexed_this_invocation,
            "status_counts": status_store.counts(stage=INDEXING_STAGE),
            "export": export_result,
        }
        logger.info(
            "indexing.completed",
            "Native LightRAG indexing stage completed",
            elapsed_ms=(time.perf_counter() - stage_started) * 1000.0,
            payload=result,
        )
        return result
    except Exception as error:
        logger.exception(
            "indexing.failed",
            error,
            message="Native LightRAG indexing stage failed",
            elapsed_ms=(time.perf_counter() - stage_started) * 1000.0,
        )
        raise
    finally:
        try:
            if initialized and rag is not None:
                await rag.finalize_storages()
        finally:
            status_store.close()
            if previous_max_extract_input_tokens is None:
                os.environ.pop("MAX_EXTRACT_INPUT_TOKENS", None)
            else:
                os.environ["MAX_EXTRACT_INPUT_TOKENS"] = (
                    previous_max_extract_input_tokens
                )


__all__ = ["BatchIndexingError", "run_indexing"]
