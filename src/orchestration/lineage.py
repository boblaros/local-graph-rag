"""Deterministic run identities and immutable, hash-verified manifests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import ExperimentConfig


LINEAGE_SCHEMA_VERSION = "2.0.0"
BASE_ID_VERSION = "base-run-id-v2"
VARIANT_ID_VERSION = "variant-run-id-v2"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json", exclude_none=False))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Fingerprint an entire workspace tree without reading implementation JSON."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    entries: list[dict[str, Any]] = []
    for child in sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    ):
        relative = child.relative_to(root).as_posix()
        if child.is_symlink():
            entries.append(
                {"path": relative, "type": "symlink", "target": os.readlink(child)}
            )
        elif child.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
        elif child.is_dir():
            entries.append({"path": relative, "type": "directory"})
    return sha256_json({"tree_hash_version": "workspace-tree-v1", "entries": entries})


def full_config_sha256(config: ExperimentConfig) -> str:
    return sha256_json(config.model_dump(mode="json", exclude_none=False))


def _extraction_runtime_payload(config: ExperimentConfig) -> dict[str, Any]:
    """Settings outside ``ExtractionConfig`` that change staged extraction."""

    return {
        "max_extract_input_tokens": config.runtime.lightrag.max_extract_input_tokens,
        "tiktoken_model_name": config.runtime.lightrag.tiktoken_model_name,
    }


def _resolved_embedding_payload(config: ExperimentConfig) -> dict[str, Any]:
    embedding = config.roles.embedding
    unresolved = embedding.unresolved_fields("roles.embedding")
    if unresolved:
        raise ValueError("embedding identity is unresolved: " + ", ".join(unresolved))
    return embedding.model_dump(mode="json", exclude_none=False)


def graph_identity_payload(config: ExperimentConfig) -> dict[str, Any]:
    """Return graph-build dimensions shared by Native and ER workspaces.

    Query/answer generation and retrieval parameters are deliberately excluded:
    they consume an already materialized workspace and therefore need their own
    downstream result identities, not a different graph variant.
    """

    lightrag = config.runtime.lightrag
    embedding_cache: dict[str, Any] = {
        "enabled": lightrag.embedding_cache.enabled,
    }
    if lightrag.embedding_cache.enabled:
        embedding_cache.update(
            similarity_threshold=lightrag.embedding_cache.similarity_threshold,
            use_llm_check=lightrag.embedding_cache.use_llm_check,
        )
    return {
        "materialization_version": "lightrag-public-graph-v1",
        "embedding": _resolved_embedding_payload(config),
        "materialization": {
            "embedding_provider": lightrag.embedding_provider,
            "storage": lightrag.storage.model_dump(mode="json", exclude_none=False),
            "kg_linked_chunk_selection": lightrag.kg_linked_chunk_selection,
            # Similarity-based embedding reuse can alter persisted vectors, so it
            # is scientific graph lineage rather than an operational cache flag.
            "embedding_cache": embedding_cache,
        },
    }


def _resolved_builder(config: ExperimentConfig, builder_key: str) -> Any:
    builder = config.builders_by_key.get(builder_key)
    if builder is None:
        raise KeyError(f"unknown builder: {builder_key}")
    unresolved = builder.unresolved_fields(f"builders.{builder_key}")
    if unresolved:
        raise ValueError("builder identity is unresolved: " + ", ".join(unresolved))
    if config.extraction.prompt_sha256 is None:
        raise ValueError("extraction.prompt_sha256 must be frozen before base identity")
    return builder


def base_identity_payload(config: ExperimentConfig, builder_key: str) -> dict[str, Any]:
    """Return exactly the scientific dimensions shared by both graph branches."""

    builder = _resolved_builder(config, builder_key)
    return {
        "identity_version": BASE_ID_VERSION,
        "corpus": {
            "subset_id": config.corpus.subset_id,
            "manifest_sha256": config.corpus.manifest_sha256,
            "documents_sha256": config.corpus.documents_sha256,
            "questions_sha256": config.corpus.questions_sha256,
            "expected_documents": config.corpus.expected_documents,
        },
        "chunking": config.extraction.chunking.model_dump(
            mode="json", exclude_none=False
        ),
        "extraction": {
            key: value
            for key, value in config.extraction.model_dump(
                mode="json", exclude_none=False
            ).items()
            if key != "chunking"
        },
        "extraction_runtime": _extraction_runtime_payload(config),
        "builder": {
            "key": builder.key,
            "requested_tag": builder.requested_tag,
            "resolved_name": builder.resolved_name,
            "digest": builder.digest,
            "quantization": builder.quantization,
        },
    }


def compute_base_run_id(config: ExperimentConfig, builder_key: str) -> str:
    return f"base_{sha256_json(base_identity_payload(config, builder_key))[:24]}"


def base_config_sha256(config: ExperimentConfig, builder_key: str) -> str:
    """Full hash used by the base freeze and extraction-stage resume check."""

    return sha256_json(base_identity_payload(config, builder_key))


def variant_identity_payload(
    config: ExperimentConfig,
    *,
    base_run_id: str,
    graph_regime: Literal[
        "native_lightrag", "advanced_lightrag_er", "advanced_lightrag_er_rr"
    ],
) -> dict[str, Any]:
    if graph_regime not in config.graph_regimes:
        raise ValueError(f"unsupported graph regime: {graph_regime}")
    payload: dict[str, Any] = {
        "identity_version": VARIANT_ID_VERSION,
        "base_run_id": base_run_id,
        "graph_regime": graph_regime,
        "graph": graph_identity_payload(config),
    }
    if graph_regime in {"advanced_lightrag_er", "advanced_lightrag_er_rr"}:
        judge = config.roles.er_judge
        if judge is None:
            raise ValueError("roles.er_judge is required for the ER variant")
        unresolved = judge.unresolved_fields("roles.er_judge")
        if unresolved:
            raise ValueError(
                "ER judge identity is unresolved: " + ", ".join(unresolved)
            )
        payload["entity_resolution"] = config.entity_resolution.model_dump(
            mode="json", exclude_none=False
        )
        payload["graph_materialization"] = config.graph_materialization.model_dump(
            mode="json", exclude_none=False
        )
        payload["judge"] = judge.model_dump(mode="json", exclude_none=False)
    if graph_regime == "advanced_lightrag_er_rr":
        verifier = config.roles.rr_verifier
        if verifier is None:
            raise ValueError("roles.rr_verifier is required for the ER+RR variant")
        unresolved = verifier.unresolved_fields("roles.rr_verifier")
        if unresolved:
            raise ValueError(
                "RR verifier identity is unresolved: " + ", ".join(unresolved)
            )
        payload["relation_recovery"] = config.relation_recovery.model_dump(
            mode="json", exclude_none=False
        )
        payload["rr_verifier"] = verifier.model_dump(mode="json", exclude_none=False)
    return payload


def variant_config_sha256(
    config: ExperimentConfig,
    *,
    base_run_id: str,
    graph_regime: Literal[
        "native_lightrag", "advanced_lightrag_er", "advanced_lightrag_er_rr"
    ],
) -> str:
    """Full hash used to resume graph construction/materialization stages."""

    return sha256_json(
        variant_identity_payload(
            config, base_run_id=base_run_id, graph_regime=graph_regime
        )
    )


def compute_variant_run_id(
    config: ExperimentConfig,
    *,
    base_run_id: str,
    graph_regime: Literal[
        "native_lightrag", "advanced_lightrag_er", "advanced_lightrag_er_rr"
    ],
) -> str:
    digest = sha256_json(
        variant_identity_payload(
            config, base_run_id=base_run_id, graph_regime=graph_regime
        )
    )
    regime = {
        "native_lightrag": "native",
        "advanced_lightrag_er": "er",
        "advanced_lightrag_er_rr": "rr",
    }[graph_regime]
    return f"variant_{regime}_{digest[:24]}"


class ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactRef(ImmutableModel):
    logical_name: str
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    schema_version: str
    record_count: int | None = Field(default=None, ge=0)

    @field_validator("sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("artifact sha256 must be lowercase hexadecimal")
        return value

    @field_validator("logical_name", "path", "schema_version")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(
                "artifact name, path, and schema version must be non-empty"
            )
        return value


def _jsonl_record_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def fingerprint_artifact(
    path: str | Path,
    *,
    logical_name: str,
    schema_version: str,
    root: str | Path | None = None,
    record_count: int | None = None,
) -> ArtifactRef:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if root is None:
        stored_path = str(artifact)
    else:
        root_path = Path(root).resolve()
        try:
            stored_path = artifact.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"artifact is outside its declared root: {artifact}"
            ) from exc
    if record_count is None and artifact.suffix == ".jsonl":
        record_count = _jsonl_record_count(artifact)
    return ArtifactRef(
        logical_name=logical_name,
        path=stored_path,
        sha256=sha256_file(artifact),
        size_bytes=artifact.stat().st_size,
        schema_version=schema_version,
        record_count=record_count,
    )


def resolve_artifact_path(ref: ArtifactRef, *, root: str | Path | None = None) -> Path:
    path = Path(ref.path)
    if path.is_absolute():
        return path.resolve()
    if root is None:
        raise ValueError(f"relative artifact path requires a root: {ref.path}")
    root_path = Path(root).resolve()
    resolved = (root_path / path).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"unsafe artifact path escapes root: {ref.path}") from exc
    return resolved


def verify_artifact(
    ref: ArtifactRef, *, root: str | Path | None = None
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        path = resolve_artifact_path(ref, root=root)
    except ValueError as exc:
        return False, [str(exc)]
    if not path.is_file():
        return False, [f"missing artifact: {path}"]
    if path.stat().st_size != ref.size_bytes:
        reasons.append(f"size mismatch for {ref.logical_name}")
    if sha256_file(path) != ref.sha256:
        reasons.append(f"sha256 mismatch for {ref.logical_name}")
    if ref.record_count is not None and path.suffix == ".jsonl":
        if _jsonl_record_count(path) != ref.record_count:
            reasons.append(f"record count mismatch for {ref.logical_name}")
    return not reasons, reasons


def artifact_maps_compatible(
    expected: Mapping[str, ArtifactRef], actual: Mapping[str, ArtifactRef]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            reasons.append("missing artifact refs: " + ", ".join(missing))
        if extra:
            reasons.append("unexpected artifact refs: " + ", ".join(extra))
    for key in sorted(set(expected) & set(actual)):
        if expected[key] != actual[key]:
            reasons.append(f"artifact ref mismatch: {key}")
    return not reasons, reasons


def compute_artifact_set_sha256(
    base_run_id: str, artifacts: Mapping[str, ArtifactRef]
) -> str:
    # Paths are operational locators, not extraction content. Excluding them
    # keeps the base extraction hash portable across clean reproductions while
    # StageStatusStore still compares and verifies the complete ArtifactRef.
    return sha256_json(
        {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "base_run_id": base_run_id,
            "artifacts": {
                key: {
                    "logical_name": artifacts[key].logical_name,
                    "sha256": artifacts[key].sha256,
                    "size_bytes": artifacts[key].size_bytes,
                    "schema_version": artifacts[key].schema_version,
                    "record_count": artifacts[key].record_count,
                }
                for key in sorted(artifacts)
            },
        }
    )


class BaseRunManifest(ImmutableModel):
    schema_version: Literal["2.0.0"] = LINEAGE_SCHEMA_VERSION
    manifest_type: Literal["base_run"] = "base_run"
    identity_version: Literal["base-run-id-v1", "base-run-id-v2"] = BASE_ID_VERSION
    base_run_id: str
    builder_key: str
    builder_requested_tag: str
    builder_resolved_name: str
    builder_digest: str
    subset_id: str
    corpus_manifest_sha256: str
    documents_sha256: str
    questions_sha256: str
    chunking_config_sha256: str
    extraction_config_sha256: str
    extraction_version: str
    extraction_prompt_version: str
    extraction_prompt_sha256: str
    extraction_seed: int
    extraction_generation_parameters: dict[str, Any]
    base_config_sha256: str | None = None
    extraction_runtime_config_sha256: str | None = None
    # Retained only so already-frozen v1 manifests remain readable. New v2 base
    # manifests deliberately do not bind reusable extraction to downstream roles.
    full_config_sha256: str | None = None
    base_extraction_sha256: str | None = None
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    frozen_at: str | None = None

    @model_validator(mode="after")
    def _artifact_hash(self) -> "BaseRunManifest":
        if self.identity_version == BASE_ID_VERSION and (
            self.base_config_sha256 is None
            or self.extraction_runtime_config_sha256 is None
        ):
            raise ValueError("v2 base manifest requires complete base config lineage")
        if (
            self.identity_version == BASE_ID_VERSION
            and self.base_config_sha256 is not None
            and self.base_run_id != f"base_{self.base_config_sha256[:24]}"
        ):
            raise ValueError("v2 base manifest ID does not match base config hash")
        if self.artifacts:
            expected = compute_artifact_set_sha256(self.base_run_id, self.artifacts)
            if self.base_extraction_sha256 != expected:
                raise ValueError("base_extraction_sha256 does not match artifacts")
        elif self.base_extraction_sha256 is not None:
            raise ValueError("base extraction hash requires artifact refs")
        return self


class VariantRunManifest(ImmutableModel):
    schema_version: Literal["2.0.0"] = LINEAGE_SCHEMA_VERSION
    manifest_type: Literal["variant_run"] = "variant_run"
    identity_version: Literal["variant-run-id-v1", "variant-run-id-v2"] = (
        VARIANT_ID_VERSION
    )
    variant_run_id: str
    base_run_id: str
    base_extraction_sha256: str
    corpus_manifest_sha256: str
    graph_regime: Literal[
        "native_lightrag", "advanced_lightrag_er", "advanced_lightrag_er_rr"
    ]
    variant_config_sha256: str | None = None
    graph_config_sha256: str | None = None
    embedding_requested_tag: str | None = None
    embedding_resolved_name: str | None = None
    embedding_digest: str | None = None
    embedding_dimension: int | None = Field(default=None, gt=0)
    embedding_max_tokens: int | None = Field(default=None, gt=0)
    embedding_config_sha256: str | None = None
    er_version: str | None = None
    er_config_sha256: str | None = None
    judge_tag: str | None = None
    judge_digest: str | None = None
    judge_prompt_version: str | None = None
    judge_config_sha256: str | None = None
    merge_plan_sha256: str | None = None
    rr_version: str | None = None
    rr_config_sha256: str | None = None
    rr_verifier_tag: str | None = None
    rr_verifier_digest: str | None = None
    rr_prompt_version: str | None = None
    rr_verifier_config_sha256: str | None = None
    rr_plan_sha256: str | None = None
    rr_results_sha256: str | None = None
    # Retained only for backwards-compatible reads of v1 manifests. A graph
    # manifest must not change merely because query/answer/metadata changed.
    full_config_sha256: str | None = None
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    frozen_at: str | None = None

    @model_validator(mode="after")
    def _regime_lineage(self) -> "VariantRunManifest":
        graph_fields = (
            self.variant_config_sha256,
            self.graph_config_sha256,
            self.embedding_requested_tag,
            self.embedding_resolved_name,
            self.embedding_digest,
            self.embedding_dimension,
            self.embedding_max_tokens,
            self.embedding_config_sha256,
        )
        if self.identity_version == VARIANT_ID_VERSION and any(
            value is None for value in graph_fields
        ):
            raise ValueError("v2 variant manifest requires complete graph lineage")
        if (
            self.identity_version == VARIANT_ID_VERSION
            and self.variant_config_sha256 is not None
        ):
            regime = {
                "native_lightrag": "native",
                "advanced_lightrag_er": "er",
                "advanced_lightrag_er_rr": "rr",
            }[self.graph_regime]
            expected_id = f"variant_{regime}_{self.variant_config_sha256[:24]}"
            if self.variant_run_id != expected_id:
                raise ValueError(
                    "v2 variant manifest ID does not match variant config hash"
                )
        er_fields = (
            self.er_version,
            self.er_config_sha256,
            self.judge_tag,
            self.judge_digest,
            self.judge_prompt_version,
            self.judge_config_sha256,
            self.merge_plan_sha256,
        )
        if self.graph_regime in {
            "advanced_lightrag_er",
            "advanced_lightrag_er_rr",
        } and any(value is None for value in er_fields):
            raise ValueError("post-ER variant manifest requires complete ER lineage")
        if self.graph_regime == "native_lightrag" and any(
            value is not None for value in er_fields
        ):
            raise ValueError("native variant must not carry ER lineage")
        rr_fields = (
            self.rr_version,
            self.rr_config_sha256,
            self.rr_verifier_tag,
            self.rr_verifier_digest,
            self.rr_prompt_version,
            self.rr_verifier_config_sha256,
            self.rr_plan_sha256,
            self.rr_results_sha256,
        )
        if self.graph_regime == "advanced_lightrag_er_rr" and any(
            value is None for value in rr_fields
        ):
            raise ValueError("ER+RR variant manifest requires complete RR lineage")
        if self.graph_regime != "advanced_lightrag_er_rr" and any(
            value is not None for value in rr_fields
        ):
            raise ValueError("only ER+RR variant may carry RR lineage")
        return self


def build_base_manifest(
    config: ExperimentConfig,
    builder_key: str,
    *,
    artifacts: Mapping[str, ArtifactRef] | None = None,
) -> BaseRunManifest:
    builder = _resolved_builder(config, builder_key)
    base_run_id = compute_base_run_id(config, builder_key)
    artifact_map = dict(artifacts or {})
    extraction_hash = (
        compute_artifact_set_sha256(base_run_id, artifact_map) if artifact_map else None
    )
    return BaseRunManifest(
        base_run_id=base_run_id,
        builder_key=builder.key,
        builder_requested_tag=builder.requested_tag,
        builder_resolved_name=builder.resolved_name,
        builder_digest=builder.digest,
        subset_id=config.corpus.subset_id,
        corpus_manifest_sha256=config.corpus.manifest_sha256,
        documents_sha256=config.corpus.documents_sha256,
        questions_sha256=config.corpus.questions_sha256,
        chunking_config_sha256=sha256_json(config.extraction.chunking),
        extraction_config_sha256=sha256_json(config.extraction),
        extraction_version=config.extraction.version,
        extraction_prompt_version=config.extraction.prompt_version,
        extraction_prompt_sha256=config.extraction.prompt_sha256,
        extraction_seed=config.extraction.seed,
        extraction_generation_parameters={
            "temperature": config.extraction.temperature,
            "context_window": config.extraction.context_window,
            "output_tokens": config.extraction.output_tokens,
            "max_gleaning": config.extraction.max_gleaning,
            "json_extraction": config.extraction.json_extraction,
            "llm_concurrency": config.extraction.llm_concurrency,
            "document_insertion_concurrency": (
                config.extraction.document_insertion_concurrency
            ),
        },
        base_config_sha256=base_config_sha256(config, builder_key),
        extraction_runtime_config_sha256=sha256_json(
            _extraction_runtime_payload(config)
        ),
        base_extraction_sha256=extraction_hash,
        artifacts=artifact_map,
    )


def build_variant_manifest(
    config: ExperimentConfig,
    *,
    base_manifest: BaseRunManifest,
    graph_regime: Literal[
        "native_lightrag", "advanced_lightrag_er", "advanced_lightrag_er_rr"
    ],
    merge_plan: ArtifactRef | None = None,
    rr_plan: ArtifactRef | None = None,
    rr_results: ArtifactRef | None = None,
    artifacts: Mapping[str, ArtifactRef] | None = None,
) -> VariantRunManifest:
    if base_manifest.base_extraction_sha256 is None:
        raise ValueError("variant requires a completed base extraction manifest")
    variant_run_id = compute_variant_run_id(
        config,
        base_run_id=base_manifest.base_run_id,
        graph_regime=graph_regime,
    )
    graph_payload = graph_identity_payload(config)
    embedding = config.roles.embedding
    assert embedding.resolved_name is not None
    assert embedding.digest is not None
    assert embedding.dimension is not None
    assert embedding.max_tokens is not None
    kwargs: dict[str, Any] = {}
    if graph_regime in {"advanced_lightrag_er", "advanced_lightrag_er_rr"}:
        if merge_plan is None:
            raise ValueError("ER variant requires an immutable merge plan")
        judge = config.roles.er_judge
        assert judge is not None and judge.digest is not None
        kwargs = {
            "er_version": config.entity_resolution.version,
            "er_config_sha256": sha256_json(config.entity_resolution),
            "judge_tag": judge.resolved_name,
            "judge_digest": judge.digest,
            "judge_prompt_version": judge.prompt_version,
            "judge_config_sha256": sha256_json(judge.identity_payload()),
            "merge_plan_sha256": merge_plan.sha256,
        }
    if graph_regime == "advanced_lightrag_er_rr":
        if rr_plan is None or rr_results is None:
            raise ValueError("ER+RR variant requires immutable RR plan and results")
        verifier = config.roles.rr_verifier
        assert verifier is not None and verifier.digest is not None
        kwargs.update(
            rr_version=config.relation_recovery.version,
            rr_config_sha256=sha256_json(config.relation_recovery),
            rr_verifier_tag=verifier.resolved_name,
            rr_verifier_digest=verifier.digest,
            rr_prompt_version=verifier.prompt_version,
            rr_verifier_config_sha256=sha256_json(verifier.identity_payload()),
            rr_plan_sha256=rr_plan.sha256,
            rr_results_sha256=rr_results.sha256,
        )
    return VariantRunManifest(
        variant_run_id=variant_run_id,
        base_run_id=base_manifest.base_run_id,
        base_extraction_sha256=base_manifest.base_extraction_sha256,
        corpus_manifest_sha256=base_manifest.corpus_manifest_sha256,
        graph_regime=graph_regime,
        variant_config_sha256=variant_config_sha256(
            config,
            base_run_id=base_manifest.base_run_id,
            graph_regime=graph_regime,
        ),
        graph_config_sha256=sha256_json(graph_payload),
        embedding_requested_tag=embedding.requested_tag,
        embedding_resolved_name=embedding.resolved_name,
        embedding_digest=embedding.digest,
        embedding_dimension=embedding.dimension,
        embedding_max_tokens=embedding.max_tokens,
        embedding_config_sha256=sha256_json(graph_payload["embedding"]),
        artifacts=dict(artifacts or {}),
        **kwargs,
    )


class FreezeResult(ImmutableModel):
    path: str
    created: bool
    sha256: str


class FrozenBaseConfig(ImmutableModel):
    """Immutable extraction-only config stored under a reusable base run."""

    schema_version: Literal["2.0.0"] = LINEAGE_SCHEMA_VERSION
    manifest_type: Literal["frozen_base_config"] = "frozen_base_config"
    base_run_id: str
    base_config_sha256: str
    base_config: dict[str, Any]
    frozen_at: str

    @model_validator(mode="after")
    def _base_hash(self) -> "FrozenBaseConfig":
        expected_hash = sha256_json(self.base_config)
        if expected_hash != self.base_config_sha256:
            raise ValueError("frozen base config SHA-256 mismatch")
        if self.base_run_id != f"base_{expected_hash[:24]}":
            raise ValueError("frozen base config run ID mismatch")
        return self


class FrozenExperimentConfig(ImmutableModel):
    schema_version: Literal["2.0.0"] = LINEAGE_SCHEMA_VERSION
    manifest_type: Literal["frozen_experiment_config"] = "frozen_experiment_config"
    config_sha256: str
    config: dict[str, Any]
    frozen_at: str

    @model_validator(mode="after")
    def _config_hash(self) -> "FrozenExperimentConfig":
        validated = ExperimentConfig.model_validate(self.config)
        if full_config_sha256(validated) != self.config_sha256:
            raise ValueError("frozen experiment config SHA-256 mismatch")
        return self


def _without_volatile_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("frozen_at", None)
    return result


def freeze_manifest(
    path: str | Path,
    manifest: BaseRunManifest | VariantRunManifest,
) -> FreezeResult:
    """Create an immutable manifest, or validate an identical existing one."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    proposed = manifest.model_dump(mode="json", exclude_none=False)
    if destination.exists():
        with destination.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if _without_volatile_fields(existing) != _without_volatile_fields(proposed):
            raise RuntimeError(f"immutable manifest conflict: {destination}")
        return FreezeResult(
            path=str(destination.resolve()),
            created=False,
            sha256=sha256_file(destination),
        )
    proposed["frozen_at"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    encoded = (
        json.dumps(
            proposed,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        # Another process won the immutable create race. Re-enter the same
        # compatibility check instead of treating an identical freeze as a
        # failure.
        return freeze_manifest(destination, manifest)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return FreezeResult(
        path=str(destination.resolve()), created=True, sha256=sha256_file(destination)
    )


def freeze_experiment_config(
    path: str | Path,
    config: ExperimentConfig,
) -> FreezeResult:
    """Persist the fully resolved configuration once and verify on resume."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_payload = config.model_dump(mode="json", exclude_none=False)
    config_hash = full_config_sha256(config)
    if destination.exists():
        frozen = load_frozen_experiment_config(destination)
        if frozen.config_sha256 != config_hash or frozen.config != config_payload:
            raise RuntimeError(f"immutable frozen config conflict: {destination}")
        return FreezeResult(
            path=str(destination.resolve()),
            created=False,
            sha256=sha256_file(destination),
        )
    payload = FrozenExperimentConfig(
        config_sha256=config_hash,
        config=config_payload,
        frozen_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    encoded = (
        json.dumps(
            payload.model_dump(mode="json", exclude_none=False),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return freeze_experiment_config(destination, config)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return FreezeResult(
        path=str(destination.resolve()), created=True, sha256=sha256_file(destination)
    )


def freeze_base_config(
    path: str | Path,
    config: ExperimentConfig,
    builder_key: str,
) -> FreezeResult:
    """Freeze only dimensions that determine the reusable extraction base."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    base_config = base_identity_payload(config, builder_key)
    base_hash = sha256_json(base_config)
    base_run_id = compute_base_run_id(config, builder_key)
    if destination.exists():
        frozen = load_frozen_base_config(destination)
        if (
            frozen.base_run_id != base_run_id
            or frozen.base_config_sha256 != base_hash
            or frozen.base_config != base_config
        ):
            raise RuntimeError(f"immutable frozen base config conflict: {destination}")
        return FreezeResult(
            path=str(destination.resolve()),
            created=False,
            sha256=sha256_file(destination),
        )
    payload = FrozenBaseConfig(
        base_run_id=base_run_id,
        base_config_sha256=base_hash,
        base_config=base_config,
        frozen_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    encoded = (
        json.dumps(
            payload.model_dump(mode="json", exclude_none=False),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return freeze_base_config(destination, config, builder_key)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return FreezeResult(
        path=str(destination.resolve()), created=True, sha256=sha256_file(destination)
    )


def load_frozen_base_config(path: str | Path) -> FrozenBaseConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return FrozenBaseConfig.model_validate(json.load(handle))


def load_frozen_experiment_config(path: str | Path) -> FrozenExperimentConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return FrozenExperimentConfig.model_validate(json.load(handle))


__all__ = [
    "ArtifactRef",
    "BASE_ID_VERSION",
    "BaseRunManifest",
    "FreezeResult",
    "FrozenBaseConfig",
    "FrozenExperimentConfig",
    "LINEAGE_SCHEMA_VERSION",
    "VARIANT_ID_VERSION",
    "VariantRunManifest",
    "artifact_maps_compatible",
    "base_config_sha256",
    "base_identity_payload",
    "build_base_manifest",
    "build_variant_manifest",
    "canonical_json",
    "compute_artifact_set_sha256",
    "compute_base_run_id",
    "compute_variant_run_id",
    "fingerprint_artifact",
    "freeze_base_config",
    "freeze_manifest",
    "freeze_experiment_config",
    "full_config_sha256",
    "graph_identity_payload",
    "load_frozen_base_config",
    "resolve_artifact_path",
    "load_frozen_experiment_config",
    "sha256_bytes",
    "sha256_file",
    "sha256_directory",
    "sha256_json",
    "sha256_text",
    "variant_config_sha256",
    "variant_identity_payload",
    "verify_artifact",
]
