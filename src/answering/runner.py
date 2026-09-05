"""Generate answers without performing retrieval a second time."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ANSWER_SCHEMA_VERSION = "3.0.0"
AnswerOutcome = Literal["success", "failure", "skipped_upstream_failure"]
ANSWER_PROMPT_VERSION = "experiment-grounded-answer-v1"
INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
SYSTEM_PROMPT = (
    "Answer using only the supplied retrieval context. Do not use outside "
    "knowledge. If the context is insufficient, output exactly "
    f"{INSUFFICIENT_INFORMATION}. Return only the concise answer."
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *values: Any) -> str:
    return prefix + _sha256(_canonical(values))[:24]


class AnswerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ANSWER_SCHEMA_VERSION
    question_id: str
    question_type: str
    answerable: bool
    gold_answer: str
    base_run_id: str
    variant_run_id: str
    base_extraction_sha256: str
    builder_model: str
    graph_regime: str
    retrieval_result_id: str
    answer_result_id: str
    answer_model: str
    answer_model_digest: str
    prompt_version: str = ANSWER_PROMPT_VERSION
    prompt_sha256: str
    retrieval_context_sha256: str
    answer_outcome: AnswerOutcome = "success"
    raw_answer: str
    normalized_answer: str
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> "AnswerRecord":
        if self.answer_outcome == "success":
            if not self.raw_answer.strip() or self.error_type:
                raise ValueError("successful answer must be non-empty and error-free")
        elif (
            self.raw_answer
            or self.normalized_answer
            or not self.error_type
            or not self.error_message
        ):
            raise ValueError("failed/skipped answer requires an explicit error")
        if self.answer_outcome == "failure" and self.latency_ms is None:
            raise ValueError("attempted answer failure requires latency")
        if (
            self.answer_outcome == "skipped_upstream_failure"
            and self.latency_ms is not None
        ):
            raise ValueError("skipped upstream failure must not report model latency")
        return self


def _read_existing(path: Path) -> dict[str, AnswerRecord]:
    if not path.exists():
        return {}
    result: dict[str, AnswerRecord] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = AnswerRecord.model_validate_json(line)
        if record.answer_result_id in result:
            raise RuntimeError(f"duplicate answer_result_id at {path}:{line_number}")
        result[record.answer_result_id] = record
    return result


def _append(path: Path, record: AnswerRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical(record.model_dump(mode="json")) + "\n")
        handle.flush()


def _answer_text(response: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(response, str):
        return response, {}
    if isinstance(response, Mapping):
        message = response.get("message")
        if isinstance(message, Mapping):
            content = message.get("content", "")
        else:
            content = response.get("response", response.get("content", ""))
        return str(content or ""), dict(response)
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    metadata = (
        response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
    )
    return str(content or ""), dict(metadata)


async def _default_ollama_call(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    options: Mapping[str, Any],
    host: str | None,
    think: bool,
    timeout_seconds: float | None,
) -> Any:
    from ollama import AsyncClient

    client_kwargs = {"timeout": timeout_seconds} if timeout_seconds is not None else {}
    client = (
        AsyncClient(host=host, **client_kwargs)
        if host
        else AsyncClient(**client_kwargs)
    )
    return await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        options=dict(options),
        think=think,
    )


async def run_answering(
    retrieval_records: Sequence[Mapping[str, Any] | Any],
    questions: Sequence[Mapping[str, Any]],
    *,
    artifact_path: str | Path,
    answer_model: str,
    answer_model_digest: str,
    generation_options: Mapping[str, Any],
    think: bool = False,
    ollama_host: str | None = None,
    timeout_seconds: float | None = None,
    answer_call: Callable[..., Awaitable[Any] | Any] | None = None,
) -> list[AnswerRecord]:
    """Answer from saved contexts; this function has no LightRAG dependency."""

    if not answer_model.strip() or not answer_model_digest.strip():
        raise ValueError("answer role must have an explicit tag and digest")
    question_catalog = {str(row.get("question_id")): dict(row) for row in questions}
    if len(question_catalog) != len(questions):
        raise ValueError("question IDs must be unique")
    path = Path(artifact_path)
    existing = _read_existing(path)
    results: list[AnswerRecord] = []

    for raw_retrieval in retrieval_records:
        retrieval = (
            raw_retrieval.model_dump(mode="json")
            if hasattr(raw_retrieval, "model_dump")
            else dict(raw_retrieval)
        )
        question_id = str(retrieval.get("question_id") or "")
        question = question_catalog.get(question_id)
        if question is None:
            raise ValueError(f"retrieval references unknown question {question_id}")
        context = str(retrieval.get("context") or "")
        context_hash = _sha256(context)
        if retrieval.get("context_sha256") != context_hash:
            raise RuntimeError(f"retrieval context hash mismatch for {question_id}")
        user_prompt = (
            f"Question:\n{str(question.get('question') or '').strip()}\n\n"
            f"Retrieval context:\n{context}"
        )
        prompt_hash = _sha256(
            _canonical(
                {
                    "version": ANSWER_PROMPT_VERSION,
                    "system": SYSTEM_PROMPT,
                    "user": user_prompt,
                    "options": dict(generation_options),
                    "think": think,
                }
            )
        )
        answer_id = _stable_id(
            "ans_",
            retrieval.get("variant_run_id"),
            retrieval.get("retrieval_result_id"),
            answer_model,
            answer_model_digest,
            prompt_hash,
        )
        prior = existing.get(answer_id)
        if prior is not None:
            if (
                prior.retrieval_result_id != retrieval.get("retrieval_result_id")
                or prior.retrieval_context_sha256 != context_hash
            ):
                raise RuntimeError("completed answer artifact has incompatible input")
            results.append(prior)
            continue

        retrieval_failed = retrieval.get("retrieval_outcome") == "failure"
        metadata: dict[str, Any] = {}
        error: Exception | None = None
        elapsed: float | None = None
        answer = ""
        if retrieval_failed:
            error = RuntimeError(
                "answering skipped because the upstream retrieval failed"
            )
            answer_outcome: AnswerOutcome = "skipped_upstream_failure"
        else:
            started = time.perf_counter()
            caller = answer_call or _default_ollama_call
            try:
                response = caller(
                    model=answer_model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    options=dict(generation_options),
                    host=ollama_host,
                    think=think,
                    timeout_seconds=timeout_seconds,
                )
                if inspect.isawaitable(response):
                    response = await response
                answer, metadata = _answer_text(response)
                answer = answer.strip()
                if not answer:
                    raise RuntimeError(
                        f"answer model returned empty output for {question_id}"
                    )
                answer_outcome = "success"
            except Exception as caught:
                error = caught
                answer = ""
                metadata = {}
                answer_outcome = "failure"
            elapsed = (time.perf_counter() - started) * 1000.0
        record = AnswerRecord(
            question_id=question_id,
            question_type=str(question.get("question_type") or "unknown"),
            answerable=bool(question.get("answerable")),
            gold_answer=str(
                question.get("answer") or question.get("gold_answer") or ""
            ),
            base_run_id=str(retrieval.get("base_run_id") or ""),
            variant_run_id=str(retrieval.get("variant_run_id") or ""),
            base_extraction_sha256=str(retrieval.get("base_extraction_sha256") or ""),
            builder_model=str(retrieval.get("builder_model") or ""),
            graph_regime=str(retrieval.get("graph_regime") or ""),
            retrieval_result_id=str(retrieval.get("retrieval_result_id") or ""),
            answer_result_id=answer_id,
            answer_model=answer_model,
            answer_model_digest=answer_model_digest,
            prompt_sha256=prompt_hash,
            retrieval_context_sha256=context_hash,
            answer_outcome=answer_outcome,
            raw_answer=answer,
            normalized_answer=" ".join(answer.casefold().split()),
            latency_ms=elapsed,
            input_tokens=(
                int(metadata["prompt_eval_count"])
                if metadata.get("prompt_eval_count") is not None
                else None
            ),
            output_tokens=(
                int(metadata["eval_count"])
                if metadata.get("eval_count") is not None
                else None
            ),
            provider_metadata=metadata,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )
        _append(path, record)
        existing[answer_id] = record
        results.append(record)
    return results


__all__ = [
    "ANSWER_PROMPT_VERSION",
    "ANSWER_SCHEMA_VERSION",
    "AnswerOutcome",
    "AnswerRecord",
    "INSUFFICIENT_INFORMATION",
    "run_answering",
]
