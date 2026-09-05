"""Run structured Native LightRAG retrieval and fixed-model answer generation.

Retrieval and answering are deliberately separate stages.  Retrieval opens the
already-indexed per-run LightRAG workspace and appends the complete structured
result to ``artifacts/retrieval.jsonl``.  Answering reads only those immutable
records and calls the configured answer model directly; it never asks
LightRAG to retrieve again.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import re
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config.native_runtime import (
    FrozenRun,
    load_frozen_run,
    resolved_input_paths,
    validate_frozen_inputs,
    validate_lightrag_runtime,
)
from src.config.native_support.hashes import (
    canonical_json,
    sha256_text,
    stable_id,
    to_jsonable,
)
from src.config.native_support.io_utils import append_jsonl, read_jsonl
from src.config.native_support.schemas import AnswerArtifact, RetrievalArtifact
from src.config.ollama_identity import resolve_ollama_model_identity
from src.answering.pilot_normalization import (
    INSUFFICIENT_INFORMATION,
    normalize_answer,
)
from src.graph.native_lightrag import build_lightrag


ANSWER_PROMPT_VERSION = "native-lightrag-pilot-answer-v1"
DEFAULT_ANSWER_SYSTEM_PROMPT = (
    "Answer the question using only the supplied Native LightRAG context. "
    "Do not use outside knowledge. If the context does not contain enough "
    f"evidence, output exactly {INSUFFICIENT_INFORMATION}. Give only the "
    "concise answer needed to answer the question."
)
_SOURCE_SEPARATOR_RE = re.compile(r"(?:<SEP>|[|;,])")


def _quantization_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _validate_checkout_and_lock(frozen_run: FrozenRun) -> None:
    config = frozen_run.lock.config
    missing = [
        name
        for name in (
            "builder_model_digest",
            "keyword_model_digest",
            "embedding_model_digest",
            "answer_model_digest",
        )
        if not getattr(config, name)
    ]
    if missing or config.quantization.casefold() in {"auto", "resolve", "unknown"}:
        raise RuntimeError(
            "run was frozen without resolved model identities; recreate it via "
            "run_indexing.py --config"
        )
    validate_lightrag_runtime(frozen_run)
    validate_frozen_inputs(frozen_run)


async def _verify_retrieval_model_identities(frozen_run: FrozenRun) -> None:
    config = frozen_run.lock.config
    settings = dict(config.lightrag or {})
    base_host = settings.get("ollama_host")
    builder_identity = await resolve_ollama_model_identity(
        config.builder_model,
        host=(
            str(settings.get("llm_host", base_host))
            if settings.get("llm_host", base_host) is not None
            else None
        ),
    )
    embedding_identity = await resolve_ollama_model_identity(
        str(config.embedding_model),
        host=(
            str(settings.get("embedding_host", base_host))
            if settings.get("embedding_host", base_host) is not None
            else None
        ),
    )
    keyword_model = str(config.keyword_model or config.answer_model)
    keyword_settings = dict(config.keyword or {})
    keyword_host = (
        keyword_settings.get("host")
        or keyword_settings.get("ollama_host")
        or settings.get("llm_host", base_host)
    )
    keyword_identity = await resolve_ollama_model_identity(
        keyword_model,
        host=str(keyword_host) if keyword_host is not None else None,
    )
    query_model = getattr(config, "query_model", None)
    query_identity = None
    if query_model:
        query_settings = dict(getattr(config, "query", None) or {})
        query_host = (
            query_settings.get("host")
            or query_settings.get("ollama_host")
            or settings.get("llm_host", base_host)
        )
        query_identity = await resolve_ollama_model_identity(
            str(query_model),
            host=str(query_host) if query_host is not None else None,
        )
    if (
        not builder_identity
        or builder_identity.get("digest") != config.builder_model_digest
    ):
        raise RuntimeError("builder model digest differs from config.lock.json")
    actual_quantization = builder_identity.get("quantization")
    if actual_quantization and _quantization_key(
        str(actual_quantization)
    ) != _quantization_key(config.quantization):
        raise RuntimeError("builder model quantization differs from config.lock.json")
    if (
        not embedding_identity
        or embedding_identity.get("digest") != config.embedding_model_digest
    ):
        raise RuntimeError("embedding model digest differs from config.lock.json")
    if (
        not keyword_identity
        or keyword_identity.get("digest") != config.keyword_model_digest
    ):
        raise RuntimeError("keyword model digest differs from config.lock.json")
    if query_model and (
        not query_identity
        or query_identity.get("digest") != getattr(config, "query_model_digest", None)
    ):
        raise RuntimeError("query model digest differs from config.lock.json")


def _latest_by_question(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        if question_id is not None:
            latest[str(question_id)] = dict(record)
    return latest


def _validate_retrieval_resume_artifact(
    frozen_run: FrozenRun,
    question: Mapping[str, Any],
    record: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    if record is None:
        return False, "completed status has no retrieval artifact"
    try:
        artifact = RetrievalArtifact.model_validate(record)
    except Exception as error:
        return False, f"retrieval artifact schema validation failed: {error}"
    config = frozen_run.lock.config
    expected_result_id = stable_id(
        "ret_",
        frozen_run.lock.run_id,
        str(question["question_id"]),
        str(question["question"]),
        config.retrieval_mode,
        config.retrieval,
    )
    checks = {
        "run_id": artifact.run_id == frozen_run.lock.run_id,
        "question_id": artifact.question_id == str(question["question_id"]),
        "retrieval_result_id": artifact.retrieval_result_id == expected_result_id,
        "retrieval_mode": artifact.retrieval_mode == "hybrid",
        "keyword_model": artifact.keyword_model == config.keyword_model,
        "keyword_model_digest": artifact.keyword_model_digest
        == config.keyword_model_digest,
        "embedding_model": artifact.embedding_model == config.embedding_model,
        "embedding_model_digest": artifact.embedding_model_digest
        == config.embedding_model_digest,
        "no_error": artifact.error_type is None,
        "upstream_success": artifact.raw_result.get("status") == "success",
        "context_sha256": artifact.context_sha256
        == sha256_text(artifact.context or ""),
    }
    failures = [name for name, value in checks.items() if not value]
    return (
        not failures,
        ""
        if not failures
        else "invalid retrieval artifact fields: " + ", ".join(failures),
    )


def _validate_answer_resume_artifact(
    frozen_run: FrozenRun,
    question: Mapping[str, Any],
    retrieval: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    if record is None:
        return False, "completed status has no answer artifact"
    try:
        artifact = AnswerArtifact.model_validate(record)
    except Exception as error:
        return False, f"answer artifact schema validation failed: {error}"
    provider = (artifact.model_extra or {}).get("provider_metadata") or {}
    expected_retrieval_id = (retrieval or {}).get("retrieval_result_id")
    retrieval_context = str((retrieval or {}).get("context") or "")
    expected_context_hash = sha256_text(retrieval_context)
    checks = {
        "run_id": artifact.run_id == frozen_run.lock.run_id,
        "question_id": artifact.question_id == str(question["question_id"]),
        "retrieval_result_id": bool(expected_retrieval_id)
        and artifact.retrieval_result_id == str(expected_retrieval_id),
        "model_name": artifact.model_name == frozen_run.lock.config.answer_model,
        "model_digest": artifact.model_digest
        == frozen_run.lock.config.answer_model_digest,
        "provider_model": isinstance(provider, Mapping)
        and provider.get("model") == frozen_run.lock.config.answer_model,
        "retrieval_context_sha256": artifact.retrieval_context_sha256
        == expected_context_hash,
        "retrieval_artifact_context_sha256": bool(retrieval)
        and retrieval.get("context_sha256") == expected_context_hash,
        "no_error": artifact.error_type is None,
        "nonempty_answer": bool((artifact.raw_answer or "").strip()),
    }
    failures = [name for name, value in checks.items() if not value]
    return (
        not failures,
        ""
        if not failures
        else "invalid answer artifact fields: " + ", ".join(failures),
    )


def _load_questions(
    frozen_run: FrozenRun,
    override: str | Path | None,
) -> tuple[Path, list[dict[str, Any]]]:
    _, frozen_path = validate_frozen_inputs(frozen_run)
    if override is not None:
        path = Path(override).expanduser().resolve()
        if path != frozen_path:
            raise ValueError(
                "--questions cannot replace the immutable questions frozen for "
                f"this run: expected {frozen_path}, found {path}"
            )
    else:
        path = frozen_path
    if not path.is_file():
        raise FileNotFoundError(
            f"questions JSONL cannot be resolved from the frozen config: {path}"
        )
    return path, [dict(row) for row in read_jsonl(path)]


def _select_questions(
    questions: Sequence[dict[str, Any]],
    question_ids: Sequence[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    requested = set(question_ids)
    selected = [
        question
        for question in questions
        if not requested or str(question.get("question_id")) in requested
    ]
    if requested:
        found = {str(question.get("question_id")) for question in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"unknown question IDs: {', '.join(missing)}")
    return selected if limit is None else selected[:limit]


def _safe_json(value: Any) -> Any:
    """Make upstream results JSON-safe while preserving every accessible field."""

    try:
        return to_jsonable(value)
    except (TypeError, ValueError):
        if hasattr(value, "model_dump"):
            return _safe_json(value.model_dump(mode="json", exclude_none=False))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return _safe_json(dataclasses.asdict(value))
        if isinstance(value, Mapping):
            return {str(key): _safe_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_safe_json(item) for item in value]
        return repr(value)


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        converted = _safe_json(item)
        if isinstance(converted, dict):
            result.append(converted)
        else:
            result.append({"value": converted})
    return result


def _split_source_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [
        piece.strip() for piece in _SOURCE_SEPARATOR_RE.split(text) if piece.strip()
    ]


def _chunk_document_maps(
    frozen_run: FrozenRun,
) -> tuple[dict[str, str], dict[str, str]]:
    chunk_to_document: dict[str, str] = {}
    if frozen_run.paths.chunks_jsonl.exists():
        for chunk in read_jsonl(frozen_run.paths.chunks_jsonl):
            chunk_id = str(chunk.get("chunk_id") or "")
            document_id = str(chunk.get("document_id") or "")
            if chunk_id and document_id:
                chunk_to_document[chunk_id] = document_id

    document_to_url: dict[str, str] = {}
    try:
        documents_path, _ = resolved_input_paths(frozen_run.lock.config)
    except (FileNotFoundError, ValueError):
        documents_path = Path()
    if documents_path.is_file():
        for document in read_jsonl(documents_path):
            document_id = str(document.get("document_id") or "")
            url = str(document.get("url") or "")
            if document_id and url:
                document_to_url[document_id] = url
    return chunk_to_document, document_to_url


def _recover_documents_and_urls(
    data: Mapping[str, Any],
    chunk_to_document: Mapping[str, str],
    document_to_url: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    document_ids: list[str] = []
    urls: list[str] = []

    def add_document(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        document_id = chunk_to_document.get(text, text)
        if document_id in document_to_url and document_id not in document_ids:
            document_ids.append(document_id)
            url = document_to_url[document_id]
            if url and url not in urls:
                urls.append(url)

    def inspect_item(item: Any) -> None:
        if not isinstance(item, Mapping):
            return
        for key in ("chunk_id", "document_id", "doc_id"):
            if item.get(key) is not None:
                for identifier in _split_source_ids(item[key]):
                    add_document(identifier)
        for key in ("source_id", "source_chunk_ids"):
            if item.get(key) is not None:
                for identifier in _split_source_ids(item[key]):
                    add_document(identifier)
        for key in ("url", "source_url", "document_url", "file_path"):
            raw = str(item.get(key) or "").strip()
            if not raw:
                continue
            if raw.startswith(("http://", "https://")):
                if raw not in urls:
                    urls.append(raw)
                for document_id, url in document_to_url.items():
                    if url == raw and document_id not in document_ids:
                        document_ids.append(document_id)
            else:
                for identifier in _split_source_ids(raw):
                    add_document(identifier)

    # Context chunks/references define the evidence order used for @k metrics.
    for collection in (
        data.get("chunks", []),
        data.get("references", []),
        data.get("entities", []),
        data.get("relationships", []),
    ):
        if isinstance(collection, list):
            for item in collection:
                inspect_item(item)
    return document_ids, urls


def _context_token_count(rag: Any, context: str) -> int | None:
    tokenizer = getattr(rag, "tokenizer", None)
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        return None
    try:
        return len(encoder(context))
    except Exception:
        return None


def _query_param(config: Mapping[str, Any], retrieval_mode: str) -> Any:
    from lightrag import QueryParam

    field_names = {field.name for field in dataclasses.fields(QueryParam)}
    harness_only = {"timeout_seconds"}
    unknown = sorted(set(config) - field_names - harness_only)
    if unknown:
        raise ValueError(
            "unknown config.retrieval keys for this LightRAG checkout: "
            + ", ".join(unknown)
        )
    kwargs = {key: value for key, value in config.items() if key in field_names}
    kwargs.update(
        {
            "mode": retrieval_mode,
            "only_need_context": True,
            "only_need_prompt": False,
            "stream": False,
            "include_references": True,
        }
    )
    return QueryParam(**kwargs)


async def _retrieve_one(
    rag: Any,
    frozen_run: FrozenRun,
    question: Mapping[str, Any],
    *,
    chunk_to_document: Mapping[str, str],
    document_to_url: Mapping[str, str],
) -> RetrievalArtifact:
    config = frozen_run.lock.config
    question_id = str(question["question_id"])
    question_text = str(question["question"])
    result_id = stable_id(
        "ret_",
        frozen_run.lock.run_id,
        question_id,
        question_text,
        config.retrieval_mode,
        config.retrieval,
    )
    started = time.perf_counter()
    raw_result: dict[str, Any] = {}
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    try:
        timeout = config.retrieval.get("timeout_seconds")
        if timeout is None:
            upstream = await rag.aquery_llm(
                question_text,
                param=_query_param(config.retrieval, config.retrieval_mode),
            )
        else:
            async with asyncio.timeout(float(timeout)):
                upstream = await rag.aquery_llm(
                    question_text,
                    param=_query_param(config.retrieval, config.retrieval_mode),
                )
        converted = _safe_json(upstream)
        raw_result = converted if isinstance(converted, dict) else {"value": converted}
        if raw_result.get("status") != "success":
            error_type = "LightRAGRetrievalError"
            error_message = str(raw_result.get("message") or "retrieval failed")
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        error_traceback = traceback.format_exc()

    data = raw_result.get("data")
    data = data if isinstance(data, Mapping) else {}
    metadata = raw_result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    keywords = metadata.get("keywords")
    keywords = keywords if isinstance(keywords, Mapping) else {}
    llm_response = raw_result.get("llm_response")
    llm_response = llm_response if isinstance(llm_response, Mapping) else {}
    context_value = llm_response.get("content")
    context = context_value if isinstance(context_value, str) else ""
    document_ids, urls = _recover_documents_and_urls(
        data, chunk_to_document, document_to_url
    )
    raw_result.setdefault("pilot_model_roles", {})
    if isinstance(raw_result["pilot_model_roles"], dict):
        raw_result["pilot_model_roles"].update(
            {
                "keyword": {
                    "model": config.keyword_model or config.answer_model,
                    "digest": config.keyword_model_digest,
                },
                "embedding": {
                    "model": config.embedding_model,
                    "digest": config.embedding_model_digest,
                },
                "answer": {
                    "model": config.answer_model,
                    "digest": config.answer_model_digest,
                    "called_during_retrieval": False,
                },
            }
        )

    return RetrievalArtifact(
        run_id=frozen_run.lock.run_id,
        retrieval_result_id=result_id,
        question_id=question_id,
        question_type=str(question.get("question_type") or "unknown"),
        retrieval_mode=config.retrieval_mode,
        high_level_keywords=[str(value) for value in keywords.get("high_level", [])],
        low_level_keywords=[str(value) for value in keywords.get("low_level", [])],
        retrieved_entities=_as_dicts(data.get("entities")),
        retrieved_relationships=_as_dicts(data.get("relationships")),
        retrieved_chunks=_as_dicts(data.get("chunks")),
        references=_as_dicts(data.get("references")),
        retrieved_document_ids=document_ids,
        retrieved_urls=urls,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        context_token_count=_context_token_count(rag, context),
        context=context,
        context_sha256=sha256_text(context),
        raw_result=raw_result,
        error_type=error_type,
        error_message=error_message,
        question=question_text,
        keyword_model=config.keyword_model or config.answer_model,
        keyword_model_digest=config.keyword_model_digest,
        embedding_model=config.embedding_model,
        embedding_model_digest=config.embedding_model_digest,
        error_traceback=error_traceback,
    )


async def run_retrieval(
    frozen_run: FrozenRun,
    questions: Sequence[dict[str, Any]],
    *,
    retry_failed: bool = True,
    reclaim_running: bool = False,
) -> dict[str, int]:
    """Append structured retrieval records, resuming via status.sqlite."""

    logger = frozen_run.event_logger(default_stage="retrieval")
    _validate_checkout_and_lock(frozen_run)
    await _verify_retrieval_model_identities(frozen_run)
    existing_retrieval = (
        read_jsonl(frozen_run.paths.retrieval_jsonl)
        if frozen_run.paths.retrieval_jsonl.exists()
        else []
    )
    retrieval_by_question = _latest_by_question(existing_retrieval)
    chunk_to_document, document_to_url = _chunk_document_maps(frozen_run)
    rag = build_lightrag(frozen_run, extraction_capture=False)
    await rag.initialize_storages()
    try:
        with frozen_run.open_status() as status:
            question_ids = [str(question["question_id"]) for question in questions]
            status.initialize_items("retrieval", question_ids)
            for question in questions:
                question_id = str(question["question_id"])
                if status.is_completed("retrieval", question_id):
                    valid, reason = _validate_retrieval_resume_artifact(
                        frozen_run,
                        question,
                        retrieval_by_question.get(question_id),
                    )
                    if valid:
                        continue
                    status.mark_failed(
                        "retrieval",
                        question_id,
                        reason,
                        error_type="ArtifactIntegrityError",
                    )
                    logger.error(
                        "question.resume_artifact_invalid",
                        reason,
                        item_id=question_id,
                        error_type="ArtifactIntegrityError",
                        error_message=reason,
                    )
                if not status.claim(
                    "retrieval",
                    question_id,
                    retry_failed=retry_failed,
                    reclaim_running=reclaim_running,
                ):
                    continue
                started = time.perf_counter()
                logger.info(
                    "question.started",
                    "structured retrieval started",
                    item_id=question_id,
                )
                try:
                    record = await _retrieve_one(
                        rag,
                        frozen_run,
                        question,
                        chunk_to_document=chunk_to_document,
                        document_to_url=document_to_url,
                    )
                    append_jsonl(frozen_run.paths.retrieval_jsonl, record)
                    retrieval_by_question[question_id] = record.model_dump(
                        mode="json", exclude_none=False
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if record.error_type:
                        status.mark_failed(
                            "retrieval",
                            question_id,
                            record.error_message or "retrieval failed",
                            error_type=record.error_type,
                        )
                        logger.error(
                            "question.failed",
                            "structured retrieval failed",
                            item_id=question_id,
                            elapsed_ms=elapsed_ms,
                            error_type=record.error_type,
                            error_message=record.error_message,
                            payload={
                                "traceback": (record.model_extra or {}).get(
                                    "error_traceback"
                                )
                            },
                        )
                    else:
                        status.mark_completed("retrieval", question_id)
                        logger.info(
                            "question.completed",
                            "structured retrieval completed",
                            item_id=question_id,
                            elapsed_ms=elapsed_ms,
                            payload={
                                "retrieval_result_id": record.retrieval_result_id,
                                "entities": len(record.retrieved_entities),
                                "relationships": len(record.retrieved_relationships),
                                "chunks": len(record.retrieved_chunks),
                            },
                        )
                except Exception as exc:
                    status.mark_failed("retrieval", question_id, exc)
                    logger.exception(
                        "question.failed",
                        exc,
                        message="structured retrieval artifact write failed",
                        item_id=question_id,
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    )
        with frozen_run.open_status() as status:
            return status.counts(stage="retrieval")
    finally:
        await rag.finalize_storages()


def _response_field(response: Any, name: str, default: Any = None) -> Any:
    if isinstance(response, Mapping):
        return response.get(name, default)
    return getattr(response, name, default)


def _answer_content(response: Any) -> str:
    message = _response_field(response, "message", {})
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


async def _ollama_model_digest(client: Any, model_name: str) -> str | None:
    try:
        response = await client.list()
    except Exception:
        return None
    models = _response_field(response, "models", []) or []
    requested = model_name if ":" in model_name else f"{model_name}:latest"
    for model in models:
        name = str(_response_field(model, "model", _response_field(model, "name", "")))
        if name in {model_name, requested}:
            digest = _response_field(model, "digest")
            return str(digest) if digest else None
    return None


def _answer_messages(
    question: str, context: str, system_prompt: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nNative LightRAG context:\n{context}",
        },
    ]


async def _answer_one(
    client: Any,
    frozen_run: FrozenRun,
    question: Mapping[str, Any],
    retrieval: Mapping[str, Any] | None,
    *,
    actual_model_digest: str | None,
) -> AnswerArtifact:
    config = frozen_run.lock.config
    answer_config = dict(config.answer)
    system_prompt = str(
        answer_config.pop("system_prompt", DEFAULT_ANSWER_SYSTEM_PROMPT)
    )
    options = dict(answer_config.pop("options", {}))
    options.setdefault("temperature", 0.0)
    if "seed" in options and int(options["seed"]) != int(config.seed):
        raise ValueError("answer.options.seed must equal the frozen run seed")
    options["seed"] = int(config.seed)
    think = answer_config.pop("think", False)
    keep_alive = answer_config.pop("keep_alive", None)
    # Client construction consumes these; they must not leak into chat().
    answer_config.pop("host", None)
    answer_config.pop("ollama_host", None)
    answer_config.pop("timeout_seconds", None)
    if answer_config:
        raise ValueError(
            "unknown config.answer keys: " + ", ".join(sorted(answer_config))
        )

    question_id = str(question["question_id"])
    context = str((retrieval or {}).get("context") or "")
    context_hash = sha256_text(context)
    retrieval_result_id = str(
        (retrieval or {}).get("retrieval_result_id")
        or stable_id("ret_missing_", frozen_run.lock.run_id, question_id)
    )
    messages = _answer_messages(str(question["question"]), context, system_prompt)
    prompt_hash = sha256_text(
        canonical_json(
            {
                "prompt_version": ANSWER_PROMPT_VERSION,
                "model": config.answer_model,
                "messages": messages,
                "options": options,
                "think": think,
            }
        )
    )

    raw_answer: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    provider_metadata: dict[str, Any] = {}
    response_model_name: str | None = None
    error_traceback: str | None = None
    provider_call_attempted = False
    started = time.perf_counter()
    if retrieval is None:
        error_type = "MissingRetrievalArtifact"
        error_message = "No saved retrieval artifact exists for this question"
    elif retrieval.get("error_type"):
        error_type = "RetrievalArtifactError"
        error_message = str(
            retrieval.get("error_message") or retrieval.get("error_type")
        )
    else:
        try:
            kwargs: dict[str, Any] = {
                "model": config.answer_model,
                "messages": messages,
                "options": options,
                "stream": False,
                "think": think,
            }
            if keep_alive is not None:
                kwargs["keep_alive"] = keep_alive
            provider_call_attempted = True
            response = await client.chat(**kwargs)
            provider_metadata = {
                key: _safe_json(_response_field(response, key))
                for key in (
                    "model",
                    "created_at",
                    "done",
                    "done_reason",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                )
                if _response_field(response, key) is not None
            }
            response_model_name = str(_response_field(response, "model") or "")
            if response_model_name != config.answer_model:
                raise RuntimeError(
                    "Ollama answer response model differs from frozen config: "
                    f"expected {config.answer_model!r}, found {response_model_name!r}"
                )
            raw_answer = _answer_content(response)
            input_value = _response_field(response, "prompt_eval_count")
            output_value = _response_field(response, "eval_count")
            input_tokens = int(input_value) if input_value is not None else None
            output_tokens = int(output_value) if output_value is not None else None
            if not raw_answer.strip():
                error_type = "EmptyAnswer"
                error_message = "Ollama returned an empty answer"
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            error_traceback = traceback.format_exc()

    return AnswerArtifact(
        run_id=frozen_run.lock.run_id,
        question_id=question_id,
        question_type=str(question.get("question_type") or "unknown"),
        gold_answer=str(question.get("gold_answer") or ""),
        answerable=bool(question.get("answerable", True)),
        gold_urls=[str(url) for url in question.get("gold_urls", [])],
        retrieval_result_id=retrieval_result_id,
        model_name=response_model_name or config.answer_model,
        model_digest=(
            actual_model_digest or config.answer_model_digest
            if response_model_name in (None, "", config.answer_model)
            else None
        ),
        raw_answer=raw_answer,
        normalized_answer=normalize_answer(raw_answer)
        if raw_answer is not None
        else None,
        prompt_sha256=prompt_hash,
        retrieval_context_sha256=context_hash,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        ),
        prompt_version=ANSWER_PROMPT_VERSION,
        provider_metadata=provider_metadata,
        error_traceback=error_traceback,
        failure_phase=(
            "model_call" if provider_call_attempted and error_type is not None else None
        ),
        requested_model=config.answer_model if provider_call_attempted else None,
        actual_response_model=response_model_name,
        error_type=error_type,
        error_message=error_message,
    )


async def run_answers(
    frozen_run: FrozenRun,
    questions: Sequence[dict[str, Any]],
    *,
    retry_failed: bool = True,
    reclaim_running: bool = False,
) -> dict[str, int]:
    """Generate answers from the latest saved retrieval records only."""

    try:
        from ollama import AsyncClient
    except ImportError as exc:  # pragma: no cover - dependency setup error
        raise RuntimeError("answer generation requires the ollama package") from exc

    config = frozen_run.lock.config
    _validate_checkout_and_lock(frozen_run)
    host = str(
        config.answer.get("host")
        or config.answer.get("ollama_host")
        or config.lightrag.get("ollama_host")
        or "http://localhost:11434"
    )
    timeout = float(config.answer.get("timeout_seconds", 300.0))
    client = AsyncClient(host=host, timeout=timeout)
    actual_digest = await _ollama_model_digest(client, config.answer_model)
    if not actual_digest:
        close = getattr(getattr(client, "_client", None), "aclose", None)
        if callable(close):
            await close()
        raise RuntimeError("configured answer model is unavailable from Ollama")
    if actual_digest != config.answer_model_digest:
        close = getattr(getattr(client, "_client", None), "aclose", None)
        if callable(close):
            await close()
        raise RuntimeError(
            "answer model digest differs from config.lock.json: "
            f"expected {config.answer_model_digest}, found {actual_digest}"
        )

    retrieval_records = (
        read_jsonl(frozen_run.paths.retrieval_jsonl)
        if frozen_run.paths.retrieval_jsonl.exists()
        else []
    )
    retrieval_by_question = _latest_by_question(retrieval_records)
    answer_records = (
        read_jsonl(frozen_run.paths.answers_jsonl)
        if frozen_run.paths.answers_jsonl.exists()
        else []
    )
    answers_by_question = _latest_by_question(answer_records)
    logger = frozen_run.event_logger(default_stage="answers")
    with frozen_run.open_status() as status:
        question_ids = [str(question["question_id"]) for question in questions]
        status.initialize_items("answers", question_ids)
        for question in questions:
            question_id = str(question["question_id"])
            if status.is_completed("answers", question_id):
                valid, reason = _validate_answer_resume_artifact(
                    frozen_run,
                    question,
                    retrieval_by_question.get(question_id),
                    answers_by_question.get(question_id),
                )
                if valid:
                    continue
                status.mark_failed(
                    "answers",
                    question_id,
                    reason,
                    error_type="ArtifactIntegrityError",
                )
                logger.error(
                    "question.resume_artifact_invalid",
                    reason,
                    item_id=question_id,
                    error_type="ArtifactIntegrityError",
                    error_message=reason,
                )
            if not status.claim(
                "answers",
                question_id,
                retry_failed=retry_failed,
                reclaim_running=reclaim_running,
            ):
                continue
            started = time.perf_counter()
            logger.info(
                "question.started",
                "answer generation started",
                item_id=question_id,
                payload={
                    "role": "answer",
                    "actual_model": config.answer_model,
                    "actual_model_digest": actual_digest,
                    "retrieval_reused": True,
                },
            )
            try:
                record = await _answer_one(
                    client,
                    frozen_run,
                    question,
                    retrieval_by_question.get(question_id),
                    actual_model_digest=actual_digest,
                )
                append_jsonl(frozen_run.paths.answers_jsonl, record)
                answers_by_question[question_id] = record.model_dump(
                    mode="json", exclude_none=False
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if record.error_type:
                    record_extra = record.model_extra or {}
                    error_payload: dict[str, Any] = {
                        "traceback": record_extra.get("error_traceback")
                    }
                    if record_extra.get("failure_phase") == "model_call":
                        error_payload.update(
                            {
                                "role": "answer",
                                "failure_phase": "model_call",
                                "requested_model": record_extra.get("requested_model"),
                                "actual_model": record_extra.get(
                                    "actual_response_model"
                                ),
                                "requested_model_digest": actual_digest,
                            }
                        )
                    status.mark_failed(
                        "answers",
                        question_id,
                        record.error_message or "answer generation failed",
                        error_type=record.error_type,
                    )
                    logger.error(
                        "question.failed",
                        "answer generation failed",
                        item_id=question_id,
                        elapsed_ms=elapsed_ms,
                        error_type=record.error_type,
                        error_message=record.error_message,
                        payload=error_payload,
                    )
                else:
                    status.mark_completed("answers", question_id)
                    logger.info(
                        "question.completed",
                        "answer generation completed",
                        item_id=question_id,
                        elapsed_ms=elapsed_ms,
                        payload={
                            "retrieval_result_id": record.retrieval_result_id,
                            "role": "answer",
                            "actual_model": record.model_name,
                            "actual_model_digest": record.model_digest,
                            "provider_reported_model": (record.model_extra or {})
                            .get("provider_metadata", {})
                            .get("model"),
                        },
                    )
            except Exception as exc:
                status.mark_failed("answers", question_id, exc)
                logger.exception(
                    "question.failed",
                    exc,
                    message="answer artifact write failed",
                    item_id=question_id,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
        counts = status.counts(stage="answers")
    close = getattr(getattr(client, "_client", None), "aclose", None)
    if callable(close):
        await close()
    return counts


async def _async_main(args: argparse.Namespace) -> int:
    frozen_run = load_frozen_run(args.run_dir)
    _, questions = _load_questions(frozen_run, args.questions)
    selected = _select_questions(questions, args.question_id, args.limit)
    summaries: dict[str, dict[str, int]] = {}
    if args.stage in {"retrieval", "all"}:
        summaries["retrieval"] = await run_retrieval(
            frozen_run,
            selected,
            retry_failed=not args.no_retry_failed,
            reclaim_running=args.reclaim_running,
        )
    if args.stage in {"answers", "all"}:
        summaries["answers"] = await run_answers(
            frozen_run,
            selected,
            retry_failed=not args.no_retry_failed,
            reclaim_running=args.reclaim_running,
        )
    print(canonical_json({"run_id": frozen_run.lock.run_id, "status": summaries}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("retrieval", "answers", "all"),
        help="Run retrieval, answer generation from saved retrieval, or both.",
    )
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-retry-failed", action="store_true")
    parser.add_argument(
        "--reclaim-running",
        action="store_true",
        help="Explicitly reclaim status rows left running after an interrupted process.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    return asyncio.run(_async_main(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_PROMPT_VERSION",
    "DEFAULT_ANSWER_SYSTEM_PROMPT",
    "build_parser",
    "main",
    "run_answers",
    "run_retrieval",
]
