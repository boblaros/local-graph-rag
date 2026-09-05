"""Connect the experiment configuration to Native LightRAG indexing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import LoadedExperimentConfig
from src.extraction.staging import StagingManifest, stage_native_artifacts


def _model_name(model: Any) -> str:
    value = getattr(model, "resolved_name", None) or getattr(
        model, "requested_tag", None
    )
    if not value:
        raise ValueError("model identity has neither resolved_name nor requested_tag")
    return str(value)


def _required_digest(model: Any, role: str) -> str:
    value = getattr(model, "digest", None)
    if not value:
        raise ValueError(f"{role} model digest is unresolved")
    return str(value)


def _generation(role: Any, *, fallback_output_tokens: int) -> dict[str, Any]:
    generation = getattr(role, "generation", None)
    if generation is None:
        raise ValueError("query and answer roles require explicit generation settings")
    return {
        "temperature": float(generation.temperature),
        "seed": int(generation.seed),
        "num_ctx": int(generation.context_window),
        "num_predict": int(generation.output_tokens or fallback_output_tokens),
    }


def build_legacy_run_payload(
    loaded: LoadedExperimentConfig,
    *,
    builder_key: str,
    graph_regime: str,
    base_run_id: str,
    variant_run_id: str,
    er_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate an experiment condition into the LightRAG run schema."""

    config = loaded.config
    builder = config.builders_by_key.get(builder_key)
    if builder is None:
        raise ValueError(f"unknown builder key {builder_key!r}")
    config.assert_ready(
        builder_key=builder_key,
        require_er=graph_regime in {"advanced_lightrag_er", "advanced_lightrag_er_rr"},
        require_rr=graph_regime == "advanced_lightrag_er_rr",
    )
    query = config.roles.query
    answer = config.roles.answer
    embedding = config.roles.embedding
    extraction = config.extraction
    chunking = extraction.chunking
    runtime = config.runtime
    host = str(getattr(runtime, "ollama_host", "http://localhost:11434"))
    query_options = _generation(query, fallback_output_tokens=256)
    answer_options = _generation(answer, fallback_output_tokens=64)
    lightrag_overrides = runtime.lightrag.legacy_payload()
    retrieval_overrides = runtime.retrieval.legacy_payload()

    lightrag = {
        "provider": "ollama",
        "embedding_provider": "ollama",
        "ollama_host": host,
        "llm_host": host,
        "embedding_host": host,
        "llm_timeout": int(runtime.model_timeout_seconds),
        "llm_options": {
            "temperature": extraction.temperature,
            "seed": extraction.seed,
            "num_ctx": extraction.context_window,
            "num_predict": extraction.output_tokens,
        },
        # Ollama's thinking switch is a top-level chat parameter, not an item
        # inside ``options``.  Keep it frozen separately from the token knobs.
        "llm_model_kwargs": {"think": bool(extraction.think)},
        "embedding_dim": embedding.dimension,
        "embedding_max_token_size": embedding.max_tokens,
        "max_extract_input_tokens": runtime.lightrag.max_extract_input_tokens,
        "index_batch_size": runtime.lightrag.index_batch_size,
        "chunk_token_size": chunking.chunk_token_size,
        "chunk_overlap_token_size": chunking.chunk_overlap_token_size,
        "addon_params": {
            "chunker": {
                "chunk_token_size": chunking.chunk_token_size,
                "fixed_token": {
                    "chunk_token_size": chunking.chunk_token_size,
                    "chunk_overlap_token_size": chunking.chunk_overlap_token_size,
                    "split_by_character": chunking.split_by_character,
                    "split_by_character_only": chunking.split_by_character_only,
                },
            }
        },
        "entity_extraction_use_json": extraction.json_extraction,
        "entity_extract_max_gleaning": extraction.max_gleaning,
        "enable_llm_cache": runtime.lightrag.enable_llm_cache,
        "enable_llm_cache_for_entity_extract": (
            runtime.lightrag.enable_llm_cache_for_entity_extract
        ),
        "llm_model_max_async": extraction.llm_concurrency,
        "embedding_func_max_async": runtime.lightrag.embedding_func_max_async,
        "max_parallel_insert": extraction.document_insertion_concurrency,
        "kg_chunk_pick_method": runtime.lightrag.kg_linked_chunk_selection,
        **lightrag_overrides,
    }
    if extraction.entity_types_guidance:
        lightrag["addon_params"]["entity_types_guidance"] = (
            extraction.entity_types_guidance
        )
    retrieval = {
        "top_k": 20,
        "chunk_top_k": 20,
        "max_entity_tokens": 2048,
        "max_relation_tokens": 3072,
        "max_total_tokens": 7000,
        "enable_rerank": False,
        "timeout_seconds": int(runtime.retrieval.timeout_seconds),
        **retrieval_overrides,
    }
    metadata = {
        "experiment": config.experiment_id,
        "builder_key": builder_key,
        "subset_id": config.corpus.subset_id,
        "corpus_manifest_path": str(loaded.manifest_path),
        "corpus_manifest_sha256": config.corpus.manifest_sha256,
        "base_run_id": base_run_id,
        "variant_run_id": variant_run_id,
        "extraction_prompt_version": extraction.prompt_version,
        "extraction_prompt_sha256": extraction.prompt_sha256,
        "er_lineage": dict(er_lineage or {}),
    }
    return {
        "schema_version": "1.0.0",
        "builder_model": _model_name(builder),
        "builder_model_digest": _required_digest(builder, f"builder {builder_key}"),
        "quantization": builder.quantization,
        "graph_regime": graph_regime,
        "retrieval_mode": "hybrid",
        "seed": extraction.seed,
        # Keyword extraction and future query generation share the explicit
        # query role; neither falls back to the builder or answer role.
        "keyword_model": _model_name(query),
        "keyword_model_digest": _required_digest(query, "query"),
        "query_model": _model_name(query),
        "query_model_digest": _required_digest(query, "query"),
        "answer_model": _model_name(answer),
        "answer_model_digest": _required_digest(answer, "answer"),
        "embedding_model": _model_name(embedding),
        "embedding_model_digest": _required_digest(embedding, "embedding"),
        "documents_path": str(loaded.documents_path),
        "questions_path": str(loaded.questions_path),
        "lightrag": lightrag,
        "keyword": {
            "ollama_host": host,
            "timeout_seconds": int(runtime.model_timeout_seconds),
            "think": bool(query.generation.think),
            "options": query_options,
        },
        "query": {
            "ollama_host": host,
            "timeout_seconds": int(runtime.model_timeout_seconds),
            "think": bool(query.generation.think),
            "options": query_options,
        },
        "retrieval": retrieval,
        "answer": {
            "ollama_host": host,
            "timeout_seconds": int(runtime.model_timeout_seconds),
            "think": bool(answer.generation.think),
            "options": answer_options,
        },
        "metadata": metadata,
    }


def _write_config_compatible(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"generated immutable config conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


async def prepare_lightrag_run(
    loaded: LoadedExperimentConfig,
    *,
    builder_key: str,
    graph_regime: str,
    base_run_id: str,
    variant_run_id: str,
    er_lineage: Mapping[str, Any] | None = None,
    physical_attempt: int | None = None,
) -> Any:
    """Create the fixed LightRAG workspace configuration for one condition."""

    payload = build_legacy_run_payload(
        loaded,
        builder_key=builder_key,
        graph_regime=graph_regime,
        base_run_id=base_run_id,
        variant_run_id=variant_run_id,
        er_lineage=er_lineage,
    )
    if physical_attempt is not None:
        if graph_regime not in {"advanced_lightrag_er", "advanced_lightrag_er_rr"} or physical_attempt < 1:
            raise ValueError(
                "physical_attempt is a positive ER-materialization attempt only"
            )
        metadata = dict(payload.get("metadata") or {})
        metadata["physical_materialization_attempt"] = physical_attempt
        payload["metadata"] = metadata
    scope_id = base_run_id if graph_regime == "native_lightrag" else variant_run_id
    suffix = f".attempt-{physical_attempt:04d}" if physical_attempt is not None else ""
    generated = loaded.runs_root / "_generated_configs" / f"{scope_id}{suffix}.json"
    _write_config_compatible(generated, payload)
    runs_root = (
        loaded.runs_root
        / ("base" if graph_regime == "native_lightrag" else "variants")
        / scope_id
    )
    if physical_attempt is not None:
        runs_root = runs_root / "materialization_attempts" / f"{physical_attempt:04d}"
    from src.orchestration.native_indexing import _preflight_config

    return await _preflight_config(
        argparse.Namespace(config=str(generated), runs_root=str(runs_root))
    )


async def build_native_and_stage(
    loaded: LoadedExperimentConfig,
    *,
    builder_key: str,
    base_run_id: str,
    variant_run_id: str,
    reclaim_running: bool = False,
    document_limit: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any], StagingManifest]:
    """Make the sole builder extraction call path, then freeze staging."""

    frozen = await prepare_lightrag_run(
        loaded,
        builder_key=builder_key,
        graph_regime="native_lightrag",
        base_run_id=base_run_id,
        variant_run_id=variant_run_id,
    )
    from src.orchestration.native_indexing import run_indexing

    result = await run_indexing(
        frozen,
        export_after_indexing=True,
        reclaim_running=reclaim_running,
        document_limit=document_limit,
    )
    builder = loaded.config.builders_by_key[builder_key]
    extraction = loaded.config.extraction
    staged = stage_native_artifacts(
        frozen.paths.run_dir,
        documents_path=loaded.documents_path,
        questions_path=loaded.questions_path,
        corpus_manifest_path=loaded.manifest_path,
        base_run_id=base_run_id,
        builder_model=_model_name(builder),
        builder_model_digest=_required_digest(builder, f"builder {builder_key}"),
        extraction_prompt_version=extraction.prompt_version,
        seed=extraction.seed,
        generation_parameters={
            "temperature": extraction.temperature,
            "context_window": extraction.context_window,
            "output_tokens": extraction.output_tokens,
            "think": extraction.think,
            "max_gleaning": extraction.max_gleaning,
            "json_extraction": extraction.json_extraction,
        },
        output_dir=output_dir,
    )
    return frozen, result, staged


def workspace_tree_sha256(path: str | Path) -> str:
    """Read-only proof that the native workspace was not modified by ER."""

    root = Path(path).resolve()
    digest = hashlib.sha256()
    files = (
        item
        for item in root.rglob("*")
        if item.is_file()
        # Finder metadata is not part of the LightRAG workspace and may appear
        # merely because a directory was opened in macOS Finder.
        and item.name != ".DS_Store"
        and not item.name.startswith("._")
    )
    for file_path in sorted(files):
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "build_legacy_run_payload",
    "build_native_and_stage",
    "prepare_lightrag_run",
    "workspace_tree_sha256",
]
