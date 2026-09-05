"""Read-only Ollama identity and LightRAG prompt-bundle resolution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .loader import LoadedExperimentConfig
from .models import ExperimentConfig


def _value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _ollama_tag_key(value: str) -> str:
    """Normalize only Ollama's optional ``:latest``/case presentation.

    This is identity matching, not model selection: the actually reported
    model name is still frozen verbatim in ``resolved_name``.
    """

    return value.strip().removesuffix(":latest").casefold()


def resolve_extraction_prompt_sha256(
    *, use_json: bool, addon_params: Mapping[str, Any] | None = None
) -> str:
    """Fingerprint the exact installed LightRAG extraction prompt bundle.

    Chunk text is deliberately excluded: this identifies prompt templates and
    the resolved profile, while each physical call separately hashes its full
    effective prompt in ``extraction_calls.jsonl``.
    """

    from lightrag.prompt import PROMPTS, resolve_entity_extraction_prompt_profile

    mode = "json" if use_json else "text"
    keys = [
        "entity_extraction_section_context",
        f"entity_extraction_{mode}_system_prompt"
        if use_json
        else "entity_extraction_system_prompt",
        f"entity_extraction_{mode}_user_prompt"
        if use_json
        else "entity_extraction_user_prompt",
        (
            "entity_continue_extraction_json_user_prompt"
            if use_json
            else "entity_continue_extraction_user_prompt"
        ),
        "DEFAULT_TUPLE_DELIMITER",
        "DEFAULT_COMPLETION_DELIMITER",
    ]
    payload = {
        "mode": mode,
        "templates": {key: PROMPTS[key] for key in keys},
        "profile": resolve_entity_extraction_prompt_profile(addon_params, use_json),
        # Freeze the official LightRAG provider contract as well as its prompt
        # bundle.  JSON extraction uses the stock generic JSON-object mode; no
        # experiment-specific schema is injected.
        "response_format": {"type": "json_object"} if use_json else None,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


async def resolve_ollama_inventory(
    requested_tags: list[str], *, host: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Resolve only explicitly requested local models; never select a model."""

    import ollama

    requested = {tag.strip() for tag in requested_tags if tag.strip()}
    aliases: dict[str, str] = {}
    for tag in requested:
        key = _ollama_tag_key(tag)
        previous = aliases.get(key)
        if previous is not None and previous != tag:
            raise ValueError(
                "requested Ollama tags are ambiguous after canonical matching: "
                f"{previous!r}, {tag!r}"
            )
        aliases[key] = tag
    identities: dict[str, dict[str, Any]] = {}
    inventory: dict[str, str] = {}
    async with ollama.AsyncClient(host=host) as client:
        response = await client.list()
        models = _value(response, "models", default=[]) or []
        for model in models:
            name = str(_value(model, "model", "name", default="") or "")
            digest = str(_value(model, "digest", default="") or "").removeprefix(
                "sha256:"
            )
            if name and digest:
                inventory[name] = digest
                inventory[name.removesuffix(":latest")] = digest
            requested_tag = aliases.get(_ollama_tag_key(name))
            if requested_tag is None or not digest:
                continue
            details = _value(model, "details", default={}) or {}
            model_info: Mapping[str, Any] = {}
            capabilities: list[str] = []
            try:
                shown = await client.show(name)
            except Exception:
                shown = None
            if shown is not None:
                details = _value(shown, "details", default=details) or details
                raw_info = _value(shown, "modelinfo", "model_info", default={}) or {}
                if isinstance(raw_info, Mapping):
                    model_info = raw_info
                capabilities = [
                    str(item)
                    for item in (_value(shown, "capabilities", default=[]) or [])
                ]
            architecture = str(
                model_info.get("general.architecture")
                or _value(details, "family", default="")
                or ""
            )
            identities[requested_tag] = {
                "requested_tag": requested_tag,
                "resolved_name": name,
                "digest": digest,
                "quantization": _value(details, "quantization_level"),
                "context_length": (
                    int(model_info[f"{architecture}.context_length"])
                    if architecture
                    and model_info.get(f"{architecture}.context_length") is not None
                    else None
                ),
                "embedding_length": (
                    int(model_info[f"{architecture}.embedding_length"])
                    if architecture
                    and model_info.get(f"{architecture}.embedding_length") is not None
                    else None
                ),
                "capabilities": capabilities,
            }
    return identities, inventory


async def resolve_experiment_config(
    loaded: LoadedExperimentConfig,
) -> tuple[ExperimentConfig, dict[str, str]]:
    """Return a resolved copy after read-only Ollama list/show calls.

    Missing role tags remain missing. This preserves the fail-closed contract,
    especially for an unselected ER judge.
    """

    config = loaded.config
    role_models = [config.roles.query, config.roles.answer, config.roles.embedding]
    if config.roles.er_judge is not None:
        role_models.append(config.roles.er_judge)
    if config.roles.rr_verifier is not None:
        role_models.append(config.roles.rr_verifier)
    requested_tags = [
        str(model.requested_tag)
        for model in [*config.builders, *role_models]
        if model.requested_tag
    ]
    identities, inventory = await resolve_ollama_inventory(
        requested_tags, host=config.runtime.ollama_host
    )
    payload = config.model_dump(mode="python", exclude_none=False)

    def apply_identity(target: dict[str, Any]) -> None:
        requested_tag = target.get("requested_tag")
        if not requested_tag:
            return
        identity = identities.get(str(requested_tag))
        if identity is None:
            return
        # Resolution fills templates but never rewrites an already frozen
        # identity in memory.  Keeping the configured values lets preflight
        # compare them with the current inventory and report tag/digest drift
        # instead of silently blessing a different model.
        if target.get("resolved_name") is None:
            target["resolved_name"] = identity["resolved_name"]
        if target.get("digest") is None:
            target["digest"] = identity["digest"]

    for builder in payload["builders"]:
        apply_identity(builder)
        identity = identities.get(str(builder.get("requested_tag") or ""))
        reported_quantization = (
            str((identity or {}).get("quantization") or "").replace("-", "_").casefold()
        )
        expected_quantization = str(builder.get("quantization") or "").casefold()
        if reported_quantization not in {"", "unknown"} and (
            reported_quantization != expected_quantization
        ):
            raise ValueError(
                "resolved builder quantization mismatch for "
                f"{builder.get('key')}: expected {expected_quantization}, "
                f"found {reported_quantization}"
            )
    for role_name in ("query", "answer", "embedding", "er_judge", "rr_verifier"):
        role = payload["roles"].get(role_name)
        if isinstance(role, dict):
            apply_identity(role)
    embedding_payload = payload["roles"]["embedding"]
    embedding_identity = identities.get(
        str(embedding_payload.get("requested_tag") or "")
    )
    if embedding_identity is not None:
        if embedding_payload.get("dimension") is None:
            embedding_payload["dimension"] = embedding_identity.get("embedding_length")
        if embedding_payload.get("max_tokens") is None:
            embedding_payload["max_tokens"] = embedding_identity.get("context_length")
    if payload["extraction"].get("prompt_sha256") is None:
        payload["extraction"]["prompt_sha256"] = resolve_extraction_prompt_sha256(
            use_json=bool(config.extraction.json_extraction),
            addon_params=(
                {"entity_types_guidance": config.extraction.entity_types_guidance}
                if config.extraction.entity_types_guidance
                else None
            ),
        )
    return ExperimentConfig.model_validate(payload), inventory


def write_resolved_config(path: str | Path, config: ExperimentConfig) -> Path:
    """Write a new resolved YAML; never overwrite different frozen content."""

    destination = Path(path)
    content = yaml.safe_dump(
        config.model_dump(mode="json", exclude_none=False),
        allow_unicode=True,
        sort_keys=False,
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"refusing to overwrite a different resolved config: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


__all__ = [
    "resolve_experiment_config",
    "resolve_extraction_prompt_sha256",
    "resolve_ollama_inventory",
    "write_resolved_config",
]
