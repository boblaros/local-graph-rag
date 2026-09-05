"""Create Native LightRAG instances and record raw extraction responses.

The pinned LightRAG version has no public extraction callback with chunk and
document identifiers, so this module uses the two internal access points
documented in ``LIGHTRAG_PARITY.md``. It records each physical Ollama response
and returns the same bytes to LightRAG's parser without changing the response.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
NATIVE_EMBEDDING_TRANSIENT_MAX_ATTEMPTS = 6
NATIVE_CAPTURE_VERSION = "native-json-passive-capture-v1"


def _is_transient_embedding_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    message = str(error).casefold()
    return "eof" in message or "connection reset" in message


async def _embedding_with_transient_retry(
    operation: Callable[[], Any],
) -> tuple[Any, int]:
    """Retry transport-only embedding failures; return result and attempts."""

    for attempt in range(1, NATIVE_EMBEDDING_TRANSIENT_MAX_ATTEMPTS + 1):
        try:
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            return result, attempt
        except Exception as error:
            if (
                attempt == NATIVE_EMBEDDING_TRANSIENT_MAX_ATTEMPTS
                or not _is_transient_embedding_error(error)
            ):
                raise
            await asyncio.sleep(0.25 * attempt)
    raise AssertionError("embedding retry loop exhausted without returning")


def utc_now_iso() -> str:
    """Return an RFC 3339/ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"JSONL record must be a mapping, got {type(value).__name__}")


class _FallbackAppendOnlyJsonlWriter:
    """Small fallback used only when the shared I/O module is unavailable.

    The normal pipeline adapts ``io_utils.append_jsonl`` below. Keeping this
    fallback local makes the adapter importable in isolated unit tests while
    preserving append-only behavior.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Any) -> None:
        payload = canonical_json(_as_record(record))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def make_append_only_writer(path: str | Path) -> Any:
    """Construct the shared append-only writer, with a conservative fallback."""

    # The harness exposes append_jsonl as a function rather than a stateful
    # writer. Adapt it here so extraction workers use the same file lock and
    # fsync behavior as every other raw artifact.
    try:
        from src.config.native_support.io_utils import append_jsonl
    except ImportError:
        return _FallbackAppendOnlyJsonlWriter(path)

    class _SharedWriter:
        def append(self, record: Any) -> None:
            append_jsonl(path, record)

    return _SharedWriter()


def append_with_writer(writer: Any, record: Mapping[str, Any]) -> None:
    """Append a record across the small set of supported writer interfaces."""

    for method_name in ("append", "write", "write_record"):
        method = getattr(writer, method_name, None)
        if callable(method):
            method(dict(record))
            return
    raise TypeError(
        "append-only writer must expose append(), write(), or write_record()"
    )


class AsyncArtifactSink:
    """Serialize concurrent extraction records into one append-only JSONL file."""

    def __init__(self, path: str | Path, writer: Any | None = None):
        self.path = Path(path)
        self.writer = writer or make_append_only_writer(self.path)
        self._lock = asyncio.Lock()

    async def emit(self, record: Mapping[str, Any]) -> None:
        # Validate before an append-only write makes a malformed raw record
        # immutable.
        from src.config.native_support.schemas import ExtractionCallArtifact

        validated = ExtractionCallArtifact.model_validate(dict(record))
        async with self._lock:
            append_with_writer(self.writer, validated)


@dataclass
class ExtractionTrace:
    run_id: str
    call_id: str
    document_id: str | None
    chunk_id: str
    chunk_text_hash: str | None
    prompt_hash: str
    gleaning_round: int
    call_kind: str
    input_tokens_estimate: int | None = None
    physical_attempts: int = 0
    started_at: str = field(default_factory=utc_now_iso)

    def next_attempt(self) -> int:
        self.physical_attempts += 1
        return self.physical_attempts


def _strip_code_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    if first_newline < 0:
        return stripped
    return stripped[first_newline + 1 : -3].strip()


def _remove_think_tags(value: str) -> str:
    try:
        from lightrag.utils import remove_think_tags

        return remove_think_tags(value)
    except Exception:
        return value


def _stock_json_contract_is_valid(payload: Any) -> bool:
    """Validate the documented stock LightRAG JSON record shape.

    This is an audit metric only.  The response is always returned unchanged to
    LightRAG, whose own parser remains authoritative for the Native graph.
    """

    if not isinstance(payload, dict):
        return False
    entities = payload.get("entities")
    relationships = payload.get("relationships")
    if not isinstance(entities, list) or not isinstance(relationships, list):
        return False
    entity_fields = {"name", "type", "description"}
    relation_fields = {"source", "target", "keywords", "description"}
    return all(
        isinstance(item, dict)
        and entity_fields.issubset(item)
        and all(isinstance(item[field], str) for field in entity_fields)
        and all(item[field].strip() for field in entity_fields)
        for item in entities
    ) and all(
        isinstance(item, dict)
        and relation_fields.issubset(item)
        and all(
            isinstance(item[field], str)
            for field in relation_fields - {"keywords"}
        )
        and all(item[field].strip() for field in relation_fields - {"keywords"})
        and isinstance(item["keywords"], str)
        for item in relationships
    )


def inspect_extraction_response(raw_response: str | None) -> dict[str, Any]:
    """Measure strict JSON validity separately from LightRAG-tolerant parsing."""

    result = {
        "json_valid": False,
        "schema_valid": False,
        "parse_success": False,
        "entity_count": 0,
        "relationship_count": 0,
    }
    if raw_response is None or not isinstance(raw_response, str):
        return result

    strict_payload: Any = None
    try:
        strict_payload = json.loads(raw_response)
        result["json_valid"] = True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    parsed = strict_payload
    if parsed is None:
        cleaned = _strip_code_fence(_remove_think_tags(raw_response))
        try:
            import json_repair

            parsed = json_repair.loads(cleaned)
        except Exception:
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        entities = parsed.get("entities", [])
        relationships = parsed.get("relationships", [])
        if isinstance(entities, list) and isinstance(relationships, list):
            result["parse_success"] = True
            result["schema_valid"] = _stock_json_contract_is_valid(parsed)
            result["entity_count"] = sum(isinstance(item, dict) for item in entities)
            result["relationship_count"] = sum(
                isinstance(item, dict) for item in relationships
            )
    return result


def _safe_token_count(tokenizer: Any, texts: list[str]) -> int | None:
    if tokenizer is None:
        return None
    try:
        return sum(len(tokenizer.encode(text or "")) for text in texts)
    except Exception:
        return None


def _error_fields(error: BaseException | None) -> dict[str, Any]:
    if error is None:
        return {"error_type": None, "error_message": None}
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _base_call_record(
    trace: ExtractionTrace,
    *,
    attempt_number: int,
    model_name: str | None,
    model_digest: str | None,
    raw_response: str | None,
    latency_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
    error: BaseException | None,
    cache_hit: bool = False,
    provider_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation = inspect_extraction_response(raw_response)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "run_id": trace.run_id,
        "call_id": trace.call_id,
        "document_id": trace.document_id or f"unresolved:{trace.chunk_id}",
        "chunk_id": trace.chunk_id,
        "chunk_sha256": trace.chunk_text_hash,
        "model_name": model_name or "unknown",
        "model_digest": model_digest,
        "prompt_sha256": trace.prompt_hash,
        "raw_response": raw_response,
        "response_sha256": (
            sha256_text(raw_response) if raw_response is not None else None
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": round(latency_ms, 3),
        "attempt_number": max(1, attempt_number),
        "gleaning_round": trace.gleaning_round,
        **validation,
        **_error_fields(error),
        "technical_provenance": {
            "call_kind": trace.call_kind,
            "cache_hit": cache_hit,
            "logical_call_started_at": trace.started_at,
            "physical_attempt_number": attempt_number,
            "document_resolution_succeeded": trace.document_id is not None,
            "provider": dict(provider_metadata or {}),
        },
    }


def _response_value(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, Mapping):
        return response.get(key, default)
    return getattr(response, key, default)


def _response_message_content(response: Any) -> str:
    message = _response_value(response, "message", {})
    if isinstance(message, Mapping):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


async def resolve_ollama_model_digest(
    model_name: str,
    *,
    host: str | None = None,
    api_key: str | None = None,
) -> str | None:
    """Resolve the immutable local Ollama digest without invoking a model."""

    identity = await resolve_ollama_model_identity(
        model_name,
        host=host,
        api_key=api_key,
    )
    return str(identity["digest"]) if identity and identity.get("digest") else None


async def resolve_ollama_model_identity(
    model_name: str,
    *,
    host: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Resolve digest and quantization through Ollama's read-only list API."""

    import ollama

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    client = ollama.AsyncClient(host=host, headers=headers)
    try:
        response = await client.list()
        models = _response_value(response, "models", []) or []
        aliases = {model_name, model_name.removesuffix(":latest")}
        for model in models:
            name = _response_value(model, "model") or _response_value(model, "name")
            if not name:
                continue
            candidate_aliases = {str(name), str(name).removesuffix(":latest")}
            if aliases & candidate_aliases:
                digest = _response_value(model, "digest")
                list_details = _response_value(model, "details", {}) or {}
                details = list_details
                model_info: Mapping[str, Any] = {}
                capabilities: list[str] = []
                # Ollama's list endpoint can report ``quantization_level`` as
                # ``unknown`` for imported GGUF models even though ``show``
                # exposes the exact Q4_K_M value.  Keep the immutable digest
                # from list, but enrich mutable descriptive fields from show.
                try:
                    shown = await client.show(str(name))
                except Exception:
                    shown = None
                if shown is not None:
                    shown_details = _response_value(shown, "details", {}) or {}
                    if shown_details:
                        details = shown_details
                    shown_info = (
                        _response_value(shown, "modelinfo")
                        or _response_value(shown, "model_info")
                        or {}
                    )
                    if isinstance(shown_info, Mapping):
                        model_info = shown_info
                    raw_capabilities = _response_value(shown, "capabilities", []) or []
                    capabilities = [str(value) for value in raw_capabilities]

                quantization = _response_value(details, "quantization_level")
                if str(quantization or "").strip().casefold() == "unknown":
                    quantization = _response_value(list_details, "quantization_level")

                architecture = str(
                    model_info.get("general.architecture")
                    or _response_value(details, "family")
                    or ""
                )
                return {
                    "requested_name": model_name,
                    "resolved_name": str(name),
                    "digest": str(digest) if digest else None,
                    "quantization": str(quantization) if quantization else None,
                    "parameter_size": (
                        str(_response_value(details, "parameter_size"))
                        if _response_value(details, "parameter_size")
                        else None
                    ),
                    "family": (
                        str(_response_value(details, "family"))
                        if _response_value(details, "family")
                        else None
                    ),
                    "format": (
                        str(_response_value(details, "format"))
                        if _response_value(details, "format")
                        else None
                    ),
                    "context_length": (
                        int(model_info[f"{architecture}.context_length"])
                        if architecture
                        and model_info.get(f"{architecture}.context_length") is not None
                        else None
                    ),
                    "embedding_length": (
                        int(model_info[f"{architecture}.embedding_length"])
                        if architecture
                        and model_info.get(f"{architecture}.embedding_length")
                        is not None
                        else None
                    ),
                    "capabilities": capabilities,
                }
        return None
    finally:
        await client._client.aclose()


def _unwrap_partial(func: Callable[..., Any]) -> Callable[..., Any]:
    while isinstance(func, partial):
        func = func.func
    return func


def _looks_like_ollama_func(func: Callable[..., Any]) -> bool:
    unwrapped = _unwrap_partial(func)
    return getattr(unwrapped, "__module__", "") == "lightrag.llm.ollama"


def make_instrumented_generic_complete(
    base_func: Callable[..., Any],
    *,
    sink: AsyncArtifactSink,
    model_name: str | None,
    model_digest: str | None,
) -> Callable[..., Any]:
    """Instrument a non-Ollama provider as one visible physical attempt."""

    async def complete(*args: Any, **kwargs: Any) -> Any:
        # The cache wrapper transports this private object explicitly through
        # LightRAG's role queue. Never infer it from worker-task context:
        # workers are long lived, so inherited context may belong to an older
        # extraction when the same model later handles a merge summary.
        trace = kwargs.pop("_pilot_extraction_trace", None)
        if not isinstance(trace, ExtractionTrace):
            trace = None
        if trace is None:
            return await base_func(*args, **kwargs)

        attempt_number = trace.next_attempt()
        started = time.perf_counter()
        try:
            response = await base_func(*args, **kwargs)
        except Exception as error:
            await sink.emit(
                _base_call_record(
                    trace,
                    attempt_number=attempt_number,
                    model_name=model_name,
                    model_digest=model_digest,
                    raw_response=None,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    input_tokens=trace.input_tokens_estimate,
                    output_tokens=None,
                    error=error,
                )
            )
            raise

        raw_response = response if isinstance(response, str) else str(response)
        await sink.emit(
            _base_call_record(
                trace,
                attempt_number=attempt_number,
                model_name=model_name,
                model_digest=model_digest,
                raw_response=raw_response,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=trace.input_tokens_estimate,
                output_tokens=None,
                error=None,
                provider_metadata={
                    "native_capture": {
                        "version": NATIVE_CAPTURE_VERSION,
                        "response_mutated": False,
                    }
                },
            )
        )
        return response

    return complete


def make_instrumented_ollama_complete(
    *,
    sink: AsyncArtifactSink,
    configured_model_name: str | None = None,
    model_digest: str | None = None,
) -> Callable[..., Any]:
    """Create an Ollama completion function with physical-attempt auditing.

    The physical attempt is wrapped with the *same* ``AsyncRetrying`` object
    used by the local LightRAG Ollama provider.  Consequently stop/wait/retry
    exception behavior remains version-identical while the full ChatResponse is
    still available for token and duration capture.
    """

    import ollama
    import lightrag.llm.ollama as lightrag_ollama

    async def physical_attempt(
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        enable_cot: bool = False,
        image_inputs: list[Any] | None = None,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        del enable_cot  # The native provider intentionally ignores this option.
        kwargs = dict(kwargs)
        trace = kwargs.pop("_pilot_extraction_trace", None)
        if not isinstance(trace, ExtractionTrace):
            trace = None
        stream = bool(kwargs.get("stream"))
        kwargs.pop("max_tokens", None)

        if kwargs.get("response_format") is None:
            if kwargs.pop("entity_extraction", False):
                warnings.warn(
                    "entity_extraction=True is deprecated; use response_format",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs["response_format"] = {"type": "json_object"}
            elif kwargs.pop("keyword_extraction", False):
                warnings.warn(
                    "keyword_extraction=True is deprecated; use response_format",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs.pop("entity_extraction", None)
            kwargs.pop("keyword_extraction", None)

        # Preserve LightRAG's official response format unchanged.  Its Ollama
        # adapter maps {"type": "json_object"} to native ``format="json"``.
        lightrag_ollama._normalize_ollama_response_format(kwargs)
        host = kwargs.pop("host", None)
        timeout = kwargs.pop("timeout", None)
        if timeout == 0:
            timeout = None
        kwargs.pop("hashing_kv", None)
        api_key = kwargs.pop("api_key", None) or os.getenv("OLLAMA_API_KEY")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"LightRAG/{lightrag_ollama.__api_version__}",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        host = lightrag_ollama._coerce_host_for_cloud_model(host, model)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_inputs:
            from lightrag.llm._vision_utils import normalize_image_inputs

            user_message["images"] = [
                image.base64_str for image in normalize_image_inputs(image_inputs)
            ]
        messages.append(user_message)

        attempt_number = trace.next_attempt() if trace is not None else 0
        started = time.perf_counter()
        client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
        error_model_name = model
        error_model_digest = model_digest
        error_raw_response: str | None = None
        error_input_tokens = trace.input_tokens_estimate if trace is not None else None
        error_output_tokens: int | None = None
        error_provider_metadata: dict[str, Any] = {"provider": "ollama"}
        try:
            response = await client.chat(model=model, messages=messages, **kwargs)
            if stream:
                # Extraction calls are non-streaming.  Preserve provider behavior
                # for any other extract-role consumer without buffering its stream.
                async def inner() -> AsyncIterator[str]:
                    try:
                        async for chunk in response:
                            yield _response_message_content(chunk)
                    finally:
                        await client._client.aclose()

                return inner()

            raw_response = _response_message_content(response)
            response_model = str(_response_value(response, "model") or "")
            if trace is None and response_model != model:
                raise RuntimeError(
                    "Ollama extraction response model differs from frozen request: "
                    f"expected {model!r}, found {response_model!r}"
                )
            if trace is not None:
                prompt_tokens = _response_value(response, "prompt_eval_count")
                output_tokens = _response_value(response, "eval_count")
                provider_metadata = {
                    "provider": "ollama",
                    "response_model": _response_value(response, "model"),
                    "done_reason": _response_value(response, "done_reason"),
                    "total_duration_ns": _response_value(response, "total_duration"),
                    "load_duration_ns": _response_value(response, "load_duration"),
                    "prompt_eval_duration_ns": _response_value(
                        response, "prompt_eval_duration"
                    ),
                    "eval_duration_ns": _response_value(response, "eval_duration"),
                    "native_capture": {
                        "version": NATIVE_CAPTURE_VERSION,
                        "response_mutated": False,
                    },
                }
                error_model_name = response_model or "unknown"
                error_raw_response = raw_response
                error_input_tokens = (
                    int(prompt_tokens) if prompt_tokens is not None else None
                )
                error_output_tokens = (
                    int(output_tokens) if output_tokens is not None else None
                )
                error_provider_metadata = provider_metadata
                if response_model != model:
                    error_model_digest = None
                    raise RuntimeError(
                        "Ollama extraction response model differs from frozen request: "
                        f"expected {model!r}, found {response_model!r}"
                    )
                await sink.emit(
                    _base_call_record(
                        trace,
                        attempt_number=attempt_number,
                        model_name=str(_response_value(response, "model") or model),
                        model_digest=model_digest,
                        raw_response=raw_response,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        input_tokens=(
                            int(prompt_tokens) if prompt_tokens is not None else None
                        ),
                        output_tokens=(
                            int(output_tokens) if output_tokens is not None else None
                        ),
                        error=None,
                        provider_metadata=provider_metadata,
                    )
                )
            return raw_response
        except Exception as error:
            if trace is not None:
                await sink.emit(
                    _base_call_record(
                        trace,
                        attempt_number=attempt_number,
                        model_name=error_model_name,
                        model_digest=error_model_digest,
                        raw_response=error_raw_response,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        input_tokens=error_input_tokens,
                        output_tokens=error_output_tokens,
                        error=error,
                        provider_metadata=error_provider_metadata,
                    )
                )
            raise
        finally:
            if not stream:
                await client._client.aclose()

    # Reuse the exact stop/wait/retry/reraise policy from this checkout.
    retrying = lightrag_ollama._ollama_model_if_cache.retry.copy()
    retrying_attempt = retrying.wraps(physical_attempt)

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        enable_cot: bool = False,
        keyword_extraction: bool = False,
        entity_extraction: bool = False,
        **kwargs: Any,
    ) -> Any:
        if keyword_extraction:
            kwargs.setdefault("keyword_extraction", True)
        if entity_extraction:
            kwargs.setdefault("entity_extraction", True)

        hashing_kv = kwargs.get("hashing_kv")
        model_name = configured_model_name
        if model_name is None and hashing_kv is not None:
            model_name = hashing_kv.global_config.get("llm_model_name")
        if not model_name:
            model_name = kwargs.pop("model", None)
        if not model_name:
            raise ValueError("Unable to resolve Ollama extract model name")

        return await retrying_attempt(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            enable_cot=enable_cot,
            **kwargs,
        )

    return complete


class ExtractionAuditAdapter(AbstractAsyncContextManager["ExtractionAuditAdapter"]):
    """Record raw extraction calls during one indexing run."""

    def __init__(
        self,
        rag: Any,
        *,
        run_id: str,
        artifact_path: str | Path,
        model_name: str | None = None,
        model_digest: str | None = None,
        provider: str | None = None,
        writer: Any | None = None,
    ) -> None:
        self.rag = rag
        self.run_id = run_id
        self.model_name = model_name
        self.model_digest = model_digest
        self.provider = (provider or "").strip().lower()
        self.sink = AsyncArtifactSink(artifact_path, writer=writer)
        self._operate_module: Any = None
        self._original_cache_wrapper: Callable[..., Any] | None = None
        self._original_extract_func: Callable[..., Any] | None = None
        self._installed = False

    async def _resolve_chunk(self, chunk_id: str) -> tuple[str | None, str | None]:
        try:
            chunk = await self.rag.text_chunks.get_by_id(chunk_id)
        except Exception:
            return None, None
        if not isinstance(chunk, Mapping):
            return None, None
        document_id = chunk.get("full_doc_id")
        content = chunk.get("content")
        return (
            str(document_id) if document_id else None,
            sha256_text(str(content)) if content is not None else None,
        )

    def _build_trace(
        self,
        *,
        chunk_id: str,
        document_id: str | None,
        chunk_text_hash: str | None,
        user_prompt: str,
        system_prompt: str | None,
        history_messages: list[dict[str, Any]] | None,
        response_format: Any,
    ) -> ExtractionTrace:
        history = history_messages or []
        prompt_payload = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "history_messages": history,
            "response_format": response_format,
        }
        tokenizer = getattr(self.rag, "tokenizer", None)
        token_texts = [system_prompt or ""]
        token_texts.extend(str(message.get("content", "")) for message in history)
        token_texts.append(user_prompt)
        return ExtractionTrace(
            run_id=self.run_id,
            call_id=f"ext_{uuid.uuid4().hex}",
            document_id=document_id,
            chunk_id=chunk_id,
            chunk_text_hash=chunk_text_hash,
            prompt_hash=sha256_text(canonical_json(prompt_payload)),
            gleaning_round=1 if history else 0,
            call_kind="gleaning" if history else "initial",
            input_tokens_estimate=_safe_token_count(tokenizer, token_texts),
        )

    def _make_cache_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(original)

        async def audited(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = signature.bind_partial(*args, **kwargs)
                arguments = bound.arguments
            except TypeError:
                return await original(*args, **kwargs)

            cache_type = str(arguments.get("cache_type", "extract"))
            chunk_id = arguments.get("chunk_id")
            if cache_type != "extract" or not chunk_id:
                return await original(*args, **kwargs)

            user_prompt = str(arguments.get("user_prompt", ""))
            system_prompt = arguments.get("system_prompt")
            history_messages = arguments.get("history_messages")
            response_format = arguments.get("response_format")
            document_id, chunk_text_hash = await self._resolve_chunk(str(chunk_id))
            trace = self._build_trace(
                chunk_id=str(chunk_id),
                document_id=document_id,
                chunk_text_hash=chunk_text_hash,
                user_prompt=user_prompt,
                system_prompt=(
                    str(system_prompt) if system_prompt is not None else None
                ),
                history_messages=history_messages,
                response_format=response_format,
            )
            started = time.perf_counter()
            try:
                role_func = arguments.get("use_llm_func")
                if not callable(role_func):
                    return await original(*args, **kwargs)

                async def traced_role_func(*role_args: Any, **role_kwargs: Any) -> Any:
                    role_kwargs["_pilot_extraction_trace"] = trace
                    return await role_func(*role_args, **role_kwargs)

                # Passing the trace as a private queued kwarg is necessary:
                # priority_limit_async_func_call uses long-lived worker Tasks.
                # An explicit per-call value cannot leak into later summary
                # calls handled by the same worker.
                bound.arguments["use_llm_func"] = traced_role_func
                result = await original(*bound.args, **bound.kwargs)
                if trace.physical_attempts == 0:
                    raw_response = (
                        result[0]
                        if isinstance(result, tuple) and result
                        else str(result)
                    )
                    await self.sink.emit(
                        _base_call_record(
                            trace,
                            attempt_number=0,
                            model_name=self.model_name,
                            model_digest=self.model_digest,
                            raw_response=raw_response,
                            latency_ms=(time.perf_counter() - started) * 1000,
                            input_tokens=trace.input_tokens_estimate,
                            output_tokens=_safe_token_count(
                                getattr(self.rag, "tokenizer", None), [raw_response]
                            ),
                            error=None,
                            cache_hit=True,
                            provider_metadata={
                                "source": "lightrag_llm_cache",
                                "native_capture": {
                                    "version": NATIVE_CAPTURE_VERSION,
                                    "response_mutated": False,
                                },
                            },
                        )
                    )
                return result
            except Exception as error:
                if trace.physical_attempts == 0:
                    await self.sink.emit(
                        _base_call_record(
                            trace,
                            attempt_number=0,
                            model_name=self.model_name,
                            model_digest=self.model_digest,
                            raw_response=None,
                            latency_ms=(time.perf_counter() - started) * 1000,
                            input_tokens=trace.input_tokens_estimate,
                            output_tokens=None,
                            error=error,
                            provider_metadata={"source": "lightrag_call_wrapper"},
                        )
                    )
                raise

        return audited

    async def install(self) -> None:
        if self._installed:
            return
        import lightrag.operate as operate

        self._operate_module = operate
        self._original_cache_wrapper = operate.use_llm_func_with_cache
        operate.use_llm_func_with_cache = self._make_cache_wrapper(
            self._original_cache_wrapper
        )

        state = self.rag._role_llm_states["extract"]
        self._original_extract_func = state.raw_func
        use_ollama = self.provider == "ollama" or _looks_like_ollama_func(
            self._original_extract_func
        )
        if use_ollama:
            instrumented = make_instrumented_ollama_complete(
                sink=self.sink,
                configured_model_name=self.model_name,
                model_digest=self.model_digest,
            )
        else:
            instrumented = make_instrumented_generic_complete(
                self._original_extract_func,
                sink=self.sink,
                model_name=self.model_name,
                model_digest=self.model_digest,
            )

        try:
            updater = getattr(self.rag, "aupdate_llm_role_config", None)
            if callable(updater):
                await updater("extract", model_func=instrumented)
            else:
                self.rag.update_llm_role_config("extract", model_func=instrumented)
        except Exception:
            operate.use_llm_func_with_cache = self._original_cache_wrapper
            raise
        self._installed = True

    async def restore(self) -> None:
        if not self._installed:
            return
        assert self._original_cache_wrapper is not None
        assert self._original_extract_func is not None
        self._operate_module.use_llm_func_with_cache = self._original_cache_wrapper
        updater = getattr(self.rag, "aupdate_llm_role_config", None)
        if callable(updater):
            await updater("extract", model_func=self._original_extract_func)
        else:
            self.rag.update_llm_role_config(
                "extract", model_func=self._original_extract_func
            )
        waiter = getattr(self.rag, "wait_for_retired_llm_queues", None)
        if callable(waiter):
            await waiter()
        self._installed = False

    async def __aenter__(self) -> "ExtractionAuditAdapter":
        await self.install()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.restore()


_PILOT_EXTRACTION_ADAPTER_ATTRIBUTE = "_pilot_extraction_audit_adapter"
_LIGHTRAG_HARNESS_KEYS = {
    "provider",
    "embedding_provider",
    "ollama_host",
    "llm_host",
    "embedding_host",
    "embedding_dim",
    "embedding_max_token_size",
    "max_extract_input_tokens",
    "llm_options",
    "llm_timeout",
    "llm_model_kwargs",
    "embedding_kwargs",
    "constructor_kwargs",
    "index_batch_size",
    "addon_params",
}
_PROTECTED_LIGHTRAG_FIELDS = {
    "working_dir",
    "workspace",
    "llm_model_func",
    "llm_model_name",
    "embedding_func",
    "role_llm_configs",
    "addon_params",
}


def _model_bound_ollama_complete(
    model_name: str,
    *,
    role: str,
    model_digest: str | None,
    event_logger: Any,
) -> Callable[..., Any]:
    """Bind an Ollama model to one LightRAG role and audit logical calls.

    The local LightRAG 1.5.2 base Ollama callable resolves its model from the
    global base config, so merely supplying role-specific kwargs is
    insufficient.  This small adapter mirrors LightRAG's own API-server role
    adapter and delegates to its retrying Ollama transport without patching the
    installed package.
    """

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        enable_cot: bool = False,
        **kwargs: Any,
    ) -> Any:
        from lightrag.llm.ollama import _ollama_model_if_cache

        started = time.perf_counter()
        prompt_hash = sha256_text(prompt)
        try:
            result = await _ollama_model_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                enable_cot=enable_cot,
                **kwargs,
            )
        except Exception as error:
            event_logger.exception(
                "llm_role.call_failed",
                error,
                message=f"{role} role Ollama call failed",
                payload={
                    "role": role,
                    "model": model_name,
                    "model_digest": model_digest,
                    "prompt_sha256": prompt_hash,
                },
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            raise
        event_logger.info(
            "llm_role.called",
            f"{role} role called configured Ollama model",
            payload={
                "role": role,
                "model": model_name,
                "model_digest": model_digest,
                "prompt_sha256": prompt_hash,
            },
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return result

    return complete


def _normalize_runtime_role_overrides(
    role_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    overrides = {
        str(role): dict(payload) for role, payload in dict(role_overrides or {}).items()
    }
    unsupported_roles = sorted(set(overrides) - {"keyword", "query"})
    if unsupported_roles:
        raise ValueError(
            "runtime role overrides are restricted to retrieval roles: "
            f"{unsupported_roles}"
        )
    for role, payload in overrides.items():
        unknown = sorted(set(payload) - {"model", "model_digest", "settings"})
        if unknown:
            raise ValueError(f"unsupported {role} role override fields: {unknown}")
        if not str(payload.get("model") or "").strip():
            raise ValueError(f"{role} role override requires a non-empty model")
        if not str(payload.get("model_digest") or "").strip():
            raise ValueError(f"{role} role override requires a model digest")
        if not isinstance(payload.get("settings"), Mapping):
            raise ValueError(f"{role} role override requires explicit settings")
    return overrides


def build_lightrag(
    frozen_run: Any,
    extraction_capture: bool = False,
    role_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Any:
    """Create a Native LightRAG instance from a fixed run configuration.

    The function does not contact Ollama or initialize storage. Callers must
    initialize and finalize storage. With ``extraction_capture=True``, it also
    attaches the raw-response recorder used during indexing.
    """

    from src.config.native_runtime import validate_lightrag_runtime

    validate_lightrag_runtime(frozen_run)

    from lightrag import LightRAG, RoleLLMConfig
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    config = frozen_run.lock.config
    overrides = _normalize_runtime_role_overrides(role_overrides)
    settings = dict(config.lightrag or {})
    provider = str(settings.get("provider", "ollama")).strip().lower()
    embedding_provider = (
        str(settings.get("embedding_provider", provider)).strip().lower()
    )
    if provider != "ollama" or embedding_provider != "ollama":
        raise NotImplementedError(
            "the Native LightRAG implementation supports only "
            "provider=ollama and embedding_provider=ollama"
        )

    embedding_model = str(config.embedding_model or "").strip()
    if not embedding_model:
        raise ValueError("frozen config must set embedding_model")
    if "embedding_dim" not in settings:
        raise ValueError("frozen config lightrag.embedding_dim is required")
    embedding_dim = int(settings["embedding_dim"])
    if embedding_dim <= 0:
        raise ValueError("lightrag.embedding_dim must be positive")
    embedding_max_tokens = int(settings.get("embedding_max_token_size", 8192))
    if embedding_max_tokens <= 0:
        raise ValueError("lightrag.embedding_max_token_size must be positive")

    ollama_host = settings.get("ollama_host")
    llm_host = settings.get("llm_host", ollama_host)
    embedding_host = settings.get("embedding_host", ollama_host)
    llm_model_kwargs = dict(settings.get("llm_model_kwargs") or {})
    if llm_host is not None:
        llm_model_kwargs.setdefault("host", str(llm_host))
    if settings.get("llm_timeout") is not None:
        llm_model_kwargs.setdefault("timeout", int(settings["llm_timeout"]))
    llm_options = dict(settings.get("llm_options") or {})
    configured_seed = llm_options.get("seed")
    if configured_seed is not None and int(configured_seed) != int(config.seed):
        raise ValueError("lightrag.llm_options.seed must equal the frozen run seed")
    llm_options["seed"] = int(config.seed)
    llm_options.setdefault("temperature", 0.0)
    llm_model_kwargs.setdefault("options", llm_options)

    keyword_override = overrides.get("keyword")
    keyword_model = str(
        (keyword_override or {}).get("model")
        or config.keyword_model
        or config.answer_model
    ).strip()
    if not keyword_model:
        raise ValueError("frozen config must set keyword_model")
    keyword_settings = dict(
        (keyword_override or {}).get("settings") or config.keyword or {}
    )
    keyword_host = keyword_settings.get("host", keyword_settings.get("ollama_host"))
    if keyword_host is None:
        keyword_host = llm_host
    keyword_timeout = int(
        keyword_settings.get("timeout_seconds", settings.get("llm_timeout", 240))
    )
    keyword_options = dict(keyword_settings.get("options") or {})
    keyword_options.setdefault("seed", int(config.seed))
    keyword_options.setdefault("temperature", 0.0)

    answer_settings = dict(config.answer or {})
    answer_model = str(config.answer_model).strip()
    answer_host = answer_settings.get("host", answer_settings.get("ollama_host"))
    if answer_host is None:
        answer_host = llm_host
    answer_options = dict(answer_settings.get("options") or {})
    answer_options.setdefault("seed", int(config.seed))
    answer_options.setdefault("temperature", 0.0)

    # Older saved configurations omit a separate query role and use the answer
    # role instead. Current experiment configurations provide it explicitly.
    query_override = overrides.get("query")
    query_model = str(
        (query_override or {}).get("model")
        or getattr(config, "query_model", None)
        or answer_model
    ).strip()
    query_model_digest = (
        (query_override or {}).get("model_digest")
        or getattr(config, "query_model_digest", None)
        or config.answer_model_digest
    )
    query_settings = dict(
        (query_override or {}).get("settings")
        or getattr(config, "query", None)
        or answer_settings
    )
    query_host = query_settings.get("host", query_settings.get("ollama_host"))
    if query_host is None:
        query_host = llm_host
    query_timeout = int(
        query_settings.get("timeout_seconds", settings.get("llm_timeout", 240))
    )
    query_options = dict(query_settings.get("options") or {})
    query_options.setdefault("seed", int(config.seed))
    query_options.setdefault("temperature", 0.0)

    role_max_async = int(settings.get("llm_model_max_async", 1))
    if role_max_async != 1:
        raise ValueError("experiment requires lightrag.llm_model_max_async=1")
    event_logger = frozen_run.event_logger(default_stage="llm")

    def role_config(
        role: str,
        model_name: str,
        model_digest: str | None,
        *,
        host: Any,
        timeout: int,
        options: Mapping[str, Any],
        think: bool | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "timeout": timeout,
            "options": dict(options),
        }
        if host is not None:
            kwargs["host"] = str(host)
        if think is not None:
            kwargs["think"] = bool(think)
        return RoleLLMConfig(
            func=_model_bound_ollama_complete(
                model_name,
                role=role,
                model_digest=model_digest,
                event_logger=event_logger,
            ),
            kwargs=kwargs,
            max_async=1,
            timeout=timeout,
            metadata={
                "binding": "ollama",
                "model": model_name,
                "model_digest": model_digest,
                "host": str(host) if host is not None else None,
                "role": role,
            },
        )

    role_llm_configs = {
        "extract": role_config(
            "extract",
            config.builder_model,
            config.builder_model_digest,
            host=llm_host,
            timeout=int(settings.get("llm_timeout", 240)),
            options=llm_options,
            think=(
                bool(llm_model_kwargs["think"]) if "think" in llm_model_kwargs else None
            ),
        ),
        "keyword": role_config(
            "keyword",
            keyword_model,
            (keyword_override or {}).get("model_digest") or config.keyword_model_digest,
            host=keyword_host,
            timeout=keyword_timeout,
            options=keyword_options,
            think=(
                bool(keyword_settings["think"]) if "think" in keyword_settings else None
            ),
        ),
        # Retrieval requests context only, so query is normally not called.
        # Binding it to the fixed answer model prevents an accidental fallback
        # to the builder if a future API path does invoke query generation.
        "query": role_config(
            "query",
            query_model,
            query_model_digest,
            host=query_host,
            timeout=query_timeout,
            options=query_options,
            think=(
                bool(query_settings["think"]) if "think" in query_settings else None
            ),
        ),
        "vlm": role_config(
            "vlm",
            config.builder_model,
            config.builder_model_digest,
            host=llm_host,
            timeout=int(settings.get("llm_timeout", 240)),
            options=llm_options,
            think=(
                bool(llm_model_kwargs["think"]) if "think" in llm_model_kwargs else None
            ),
        ),
    }

    embedding_kwargs = dict(settings.get("embedding_kwargs") or {})
    if embedding_host is not None:
        embedding_kwargs.setdefault("host", str(embedding_host))

    async def audited_embedding_func(texts: list[str], **call_kwargs: Any) -> Any:
        requested_model = str(call_kwargs.pop("embed_model", embedding_model))
        if requested_model != embedding_model:
            raise RuntimeError(
                "embedding role attempted to override the frozen model: "
                f"{requested_model!r} != {embedding_model!r}"
            )
        effective_kwargs = {**embedding_kwargs, **call_kwargs}
        started = time.perf_counter()
        try:
            result, attempts = await _embedding_with_transient_retry(
                lambda: ollama_embed.func(
                    texts,
                    embed_model=embedding_model,
                    **effective_kwargs,
                )
            )
        except Exception as error:
            event_logger.exception(
                "embedding_role.call_failed",
                error,
                message="embedding role Ollama call failed",
                payload={
                    "role": "embedding",
                    "model": embedding_model,
                    "model_digest": config.embedding_model_digest,
                    "batch_size": len(texts),
                },
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            raise
        shape = getattr(result, "shape", None)
        returned_dimension = int(shape[-1]) if shape and len(shape) > 1 else None
        event_logger.info(
            "embedding_role.called",
            "embedding role called frozen Ollama model",
            payload={
                "role": "embedding",
                "model": embedding_model,
                "model_digest": config.embedding_model_digest,
                "batch_size": len(texts),
                "returned_dimension": returned_dimension,
                "attempts": attempts,
            },
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return result

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=embedding_max_tokens,
        model_name=embedding_model,
        supports_asymmetric=True,
        func=audited_embedding_func,
    )

    valid_constructor_fields = {item.name for item in fields(LightRAG) if item.init}
    constructor_kwargs = dict(settings.get("constructor_kwargs") or {})
    for key, value in settings.items():
        if key in _LIGHTRAG_HARNESS_KEYS:
            continue
        if key in valid_constructor_fields:
            if key in constructor_kwargs and constructor_kwargs[key] != value:
                raise ValueError(
                    f"conflicting lightrag.{key} and constructor_kwargs.{key}"
                )
            constructor_kwargs[key] = value
        else:
            raise ValueError(f"unknown frozen config lightrag key: {key!r}")
    protected = sorted(_PROTECTED_LIGHTRAG_FIELDS.intersection(constructor_kwargs))
    if protected:
        raise ValueError(
            "run-isolation/model fields cannot be overridden in "
            f"lightrag.constructor_kwargs: {', '.join(protected)}"
        )

    # LightRAG shares in-memory storage between instances with the same name.
    # rather than by working_dir. Use a run-derived namespace as well as the
    # run-specific directory, so two models opened in one Python process cannot
    # share storage state. Physical files remain under run/workspace/.
    workspace_name = f"pilot_{sha256_text(frozen_run.lock.run_id)[:20]}"
    addon_params = dict(settings.get("addon_params") or {})
    rag = LightRAG(
        working_dir=str(frozen_run.paths.workspace_dir.resolve()),
        workspace=workspace_name,
        llm_model_func=ollama_model_complete,
        llm_model_name=config.builder_model,
        llm_model_kwargs=llm_model_kwargs,
        role_llm_configs=role_llm_configs,
        embedding_func=embedding_func,
        addon_params=addon_params,
        **constructor_kwargs,
    )
    if Path(rag.working_dir).resolve() != frozen_run.paths.workspace_dir.resolve():
        raise RuntimeError("LightRAG working_dir escaped the frozen run workspace")
    if rag.workspace != workspace_name:
        raise RuntimeError("LightRAG workspace namespace is not run-isolated")

    resolved_chunker = dict(rag.addon_params.get("chunker") or {})
    resolved_fixed = dict(resolved_chunker.get("fixed_token") or {})
    expected_fixed = {
        "chunk_token_size": 1200,
        "chunk_overlap_token_size": 100,
        "split_by_character": None,
        "split_by_character_only": False,
    }
    if resolved_chunker.get("chunk_token_size") != 1200 or any(
        resolved_fixed.get(key) != value for key, value in expected_fixed.items()
    ):
        raise RuntimeError(
            "LightRAG did not resolve the frozen fixed-token chunker to 1200/100"
        )

    role_snapshot = rag.get_llm_role_config()
    expected_role_models = {
        "extract": config.builder_model,
        "keyword": keyword_model,
        "query": query_model,
        "vlm": config.builder_model,
    }
    for role, expected_model in expected_role_models.items():
        actual = role_snapshot.get(role, {})
        if actual.get("model") != expected_model:
            raise RuntimeError(
                f"LightRAG role {role!r} resolved model {actual.get('model')!r}; "
                f"expected {expected_model!r}"
            )
        if int(actual.get("max_async", 0)) != 1:
            raise RuntimeError(f"LightRAG role {role!r} concurrency is not 1")
    event_logger.info(
        "llm_roles.configured",
        "role-specific LightRAG models configured and verified",
        payload={
            "roles": role_snapshot,
            "embedding": {
                "model": embedding_model,
                "model_digest": config.embedding_model_digest,
                "dimension": embedding_dim,
            },
            "fixed_token_chunker": {
                "chunk_token_size": resolved_chunker.get("chunk_token_size"),
                **{key: resolved_fixed.get(key) for key in expected_fixed},
            },
            "max_extract_input_tokens": settings.get("max_extract_input_tokens"),
        },
    )

    if extraction_capture:
        adapter = ExtractionAuditAdapter(
            rag,
            run_id=frozen_run.lock.run_id,
            artifact_path=frozen_run.paths.extraction_calls_jsonl,
            model_name=config.builder_model,
            model_digest=config.builder_model_digest,
            provider=provider,
        )
        setattr(rag, _PILOT_EXTRACTION_ADAPTER_ATTRIBUTE, adapter)
    return rag


def get_extraction_adapter(rag: Any) -> ExtractionAuditAdapter:
    """Return the capture adapter attached by ``build_lightrag(..., True)``."""

    adapter = getattr(rag, _PILOT_EXTRACTION_ADAPTER_ATTRIBUTE, None)
    if not isinstance(adapter, ExtractionAuditAdapter):
        raise RuntimeError("LightRAG was not built with extraction_capture=True")
    return adapter


def extract_role_runtime_metadata(rag: Any) -> dict[str, Any]:
    """Return the sanitized runtime identity of LightRAG's extract role."""

    getter = getattr(rag, "get_llm_role_config", None)
    if callable(getter):
        try:
            value = getter("extract")
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            pass
    identity = getattr(rag, "_build_global_config", lambda: {})().get(
        "llm_cache_identities", {}
    )
    value = identity.get("extract", {}) if isinstance(identity, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "ExtractionAuditAdapter",
    "build_lightrag",
    "extract_role_runtime_metadata",
    "get_extraction_adapter",
    "inspect_extraction_response",
    "make_instrumented_ollama_complete",
    "resolve_ollama_model_digest",
    "resolve_ollama_model_identity",
]
