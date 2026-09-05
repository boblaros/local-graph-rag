"""Keep only the Ollama models required by the current experiment stage.

This prevents sequential stages from overlapping in memory without changing
model or context settings.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class OllamaLifecycleError(RuntimeError):
    """Raised when the required Ollama models cannot be loaded exclusively."""


@dataclass(frozen=True)
class ModelRequirement:
    """One model that may remain loaded for a stage."""

    role: str
    tag: str
    digest: str | None


def _deduplicate_requirements(
    requirements: Sequence[ModelRequirement],
) -> tuple[ModelRequirement, ...]:
    unique: list[ModelRequirement] = []
    seen: set[tuple[str, str | None]] = set()
    for requirement in requirements:
        key = (requirement.tag, requirement.digest)
        if key not in seen:
            unique.append(requirement)
            seen.add(key)
    return tuple(unique)


def required_models_for_stage(config: Any, stage: str) -> tuple[ModelRequirement, ...]:
    """Return the only model identities allowed to remain loaded for ``stage``."""

    if stage == "index":
        requirements = (
            ModelRequirement(
                role="extract",
                tag=config.builder_model,
                digest=config.builder_model_digest,
            ),
            ModelRequirement(
                role="embedding",
                tag=str(config.embedding_model or ""),
                digest=config.embedding_model_digest,
            ),
        )
    elif stage == "retrieve":
        requirements = (
            ModelRequirement(
                role="keyword",
                tag=str(config.keyword_model or config.answer_model),
                digest=config.keyword_model_digest or config.answer_model_digest,
            ),
            ModelRequirement(
                role="embedding",
                tag=str(config.embedding_model or ""),
                digest=config.embedding_model_digest,
            ),
        )
    elif stage == "answer":
        requirements = (
            ModelRequirement(
                role="answer",
                tag=config.answer_model,
                digest=config.answer_model_digest,
            ),
        )
    elif stage in {"preflight", "validate", "evaluate", "cleanup"}:
        requirements = ()
    else:
        raise ValueError(f"unsupported lifecycle stage {stage!r}")
    return _deduplicate_requirements(
        tuple(requirement for requirement in requirements if requirement.tag)
    )


def _record_value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _loaded_model_rows(response: Any) -> list[dict[str, Any]]:
    models = _record_value(response, "models") or []
    rows: list[dict[str, Any]] = []
    for model in models:
        name = _record_value(model, "model") or _record_value(model, "name")
        rows.append(
            {
                "name": str(name or ""),
                "digest": (
                    str(digest) if (digest := _record_value(model, "digest")) else None
                ),
                "size_bytes": _optional_int(_record_value(model, "size")),
                "size_vram_bytes": _optional_int(_record_value(model, "size_vram")),
                "context_length": _optional_int(_record_value(model, "context_length")),
            }
        )
    return rows


def _is_required(
    loaded: Mapping[str, Any], requirements: Sequence[ModelRequirement]
) -> bool:
    loaded_name = str(loaded.get("name") or "")
    loaded_digest = str(loaded.get("digest") or "")
    return any(
        loaded_name == requirement.tag
        or (
            bool(loaded_digest)
            and bool(requirement.digest)
            and loaded_digest == requirement.digest
        )
        for requirement in requirements
    )


async def _close_client(client: Any) -> None:
    close = getattr(client, "aclose", None)
    if close is None:
        close = getattr(getattr(client, "_client", None), "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def reconcile_ollama_models(
    config: Any,
    *,
    stage: str,
    phase: str,
    logger: Any | None = None,
    timeout_seconds: float = 15.0,
    poll_interval_seconds: float = 0.25,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Unload every resident runner that is not needed by ``stage``.

    The operation is intentionally exclusive: any unrelated resident Ollama
    runner is also unloaded. The experiment owns the local Ollama runtime while
    it is active so that memory pressure remains consistent.
    """

    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")

    requirements = required_models_for_stage(config, stage)
    host = str(
        config.lightrag.get("ollama_host")
        or config.lightrag.get("llm_host")
        or "http://localhost:11434"
    )
    if client_factory is None:
        import ollama

        client_factory = ollama.AsyncClient
    client = client_factory(host=host, timeout=max(5.0, timeout_seconds))
    unloaded: list[str] = []
    try:
        loaded_before = _loaded_model_rows(await client.ps())
        unwanted_before = [
            row for row in loaded_before if not _is_required(row, requirements)
        ]
        unnamed = [row for row in unwanted_before if not row["name"]]
        if unnamed:
            raise OllamaLifecycleError(
                "Ollama reported a resident runner without a model name; "
                "refusing to continue because it cannot be unloaded safely"
            )

        for row in unwanted_before:
            model_name = str(row["name"])
            await client.generate(
                model=model_name,
                prompt="",
                stream=False,
                keep_alive=0,
            )
            unloaded.append(model_name)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            loaded_after = _loaded_model_rows(await client.ps())
            unwanted_after = [
                row for row in loaded_after if not _is_required(row, requirements)
            ]
            if not unwanted_after:
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                names = [str(row.get("name") or "<unnamed>") for row in unwanted_after]
                raise OllamaLifecycleError(
                    "Ollama did not unload non-required model(s) within "
                    f"{timeout_seconds:g}s: {', '.join(names)}"
                )
            await asyncio.sleep(min(poll_interval_seconds, remaining))

        report = {
            "stage": stage,
            "phase": phase,
            "required_models": [
                {
                    "role": requirement.role,
                    "tag": requirement.tag,
                    "digest": requirement.digest,
                }
                for requirement in requirements
            ],
            "loaded_before": loaded_before,
            "unloaded_models": unloaded,
            "loaded_after": loaded_after,
        }
        if logger is not None:
            required_names = [requirement.tag for requirement in requirements]
            resident_after = [
                str(row.get("name") or "<unnamed>") for row in loaded_after
            ]
            logger.info(
                "model_lifecycle.reconciled",
                f"Ollama runners reconciled for {stage} ({phase}); "
                f"required={required_names or ['none']} "
                f"unloaded={unloaded or ['none']} "
                f"resident_after={resident_after or ['none']}",
                payload=report,
            )
        return report
    finally:
        await _close_client(client)


__all__ = [
    "ModelRequirement",
    "OllamaLifecycleError",
    "reconcile_ollama_models",
    "required_models_for_stage",
]
