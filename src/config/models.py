"""Strict, versioned configuration for the generalized experiment harness."""

from __future__ import annotations

from typing import Any, ClassVar, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


CONFIG_SCHEMA_VERSION = "4.0.0"
GRAPH_REGIMES = (
    "native_lightrag",
    "advanced_lightrag_er",
    "advanced_lightrag_er_rr",
)
SUPPORTED_ER_CANDIDATE_METHODS = (
    "exact_name",
    "containment",
    "fuzzy_name",
    "acronym",
    "embedding_neighbour",
    "reciprocal_embedding_neighbour",
)

EXPECTED_BUILDERS: tuple[tuple[str, str], ...] = (
    (
        "qwen35_0_8b",
        "hf.co/lmstudio-community/Qwen3.5-0.8B-GGUF:Q4_K_M",
    ),
    (
        "qwen35_2b",
        "hf.co/lmstudio-community/Qwen3.5-2B-GGUF:Q4_K_M",
    ),
    (
        "qwen35_4b",
        "hf.co/lmstudio-community/Qwen3.5-4B-GGUF:Q4_K_M",
    ),
    (
        "qwen35_9b",
        "hf.co/lmstudio-community/Qwen3.5-9B-GGUF:Q4_K_M",
    ),
    (
        "qwen3_0_6b",
        "hf.co/lmstudio-community/Qwen3-0.6B-GGUF:Q4_K_M",
    ),
    (
        "qwen3_1_7b",
        "hf.co/lmstudio-community/Qwen3-1.7B-GGUF:Q4_K_M",
    ),
    (
        "qwen3_4b",
        "hf.co/lmstudio-community/Qwen3-4B-GGUF:Q4_K_M",
    ),
    (
        "qwen3_8b",
        "hf.co/lmstudio-community/Qwen3-8B-GGUF:Q4_K_M",
    ),
    (
        "gemma3_270m",
        "hf.co/lmstudio-community/gemma-3-270m-it-GGUF:Q4_K_M",
    ),
    (
        "gemma3_1b",
        "hf.co/lmstudio-community/gemma-3-1b-it-GGUF:Q4_K_M",
    ),
    (
        "gemma3_4b",
        "hf.co/lmstudio-community/gemma-3-4b-it-GGUF:Q4_K_M",
    ),
    (
        "gemma3_12b",
        "hf.co/lmstudio-community/gemma-3-12b-it-GGUF:Q4_K_M",
    ),
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _validate_sha256(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return normalized


class ModelIdentity(StrictModel):
    """Requested and actually resolved Ollama identity.

    ``requested_tag`` may be absent in a template. A model-consuming stage is
    forbidden until all three identity fields are populated and preflight has
    matched them against Ollama.
    """

    requested_tag: str | None = None
    resolved_name: str | None = None
    digest: str | None = None
    provider: Literal["ollama"] = "ollama"

    @field_validator("requested_tag", "resolved_name")
    @classmethod
    def _nonempty_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("model names must be null or non-empty")
        return value

    @field_validator("digest")
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        return _validate_sha256(value, field_name="model digest")

    def unresolved_fields(self, prefix: str) -> list[str]:
        return [
            f"{prefix}.{name}"
            for name in ("requested_tag", "resolved_name", "digest")
            if getattr(self, name) is None
        ]


class GenerationSettings(StrictModel):
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int = 42
    context_window: int = Field(default=8192, gt=0)
    output_tokens: int = Field(gt=0)
    think: bool = False


class RoleModel(ModelIdentity):
    generation: GenerationSettings


class EmbeddingModel(ModelIdentity):
    dimension: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)

    def unresolved_fields(self, prefix: str) -> list[str]:
        unresolved = super().unresolved_fields(prefix)
        for name in ("dimension", "max_tokens"):
            if getattr(self, name) is None:
                unresolved.append(f"{prefix}.{name}")
        return unresolved


class ERJudge(ModelIdentity):
    prompt_version: str
    temperature: float = Field(ge=0.0)
    seed: int
    context_window: int = Field(gt=0)
    output_tokens: int = Field(gt=0)

    @field_validator("prompt_version")
    @classmethod
    def _prompt_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ER judge prompt_version must be non-empty")
        return value

    def identity_payload(self) -> dict[str, Any]:
        """Return the exact payload consumed by the ER judge/cache adapter."""

        unresolved = self.unresolved_fields("roles.er_judge")
        if unresolved:
            raise ValueError(
                "ER judge identity is unresolved: " + ", ".join(unresolved)
            )
        return {
            # The public Ollama call receives the actually resolved name. The
            # requested alias remains separately frozen in this config.
            "model_tag": self.resolved_name,
            "model_digest": self.digest,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "seed": self.seed,
            "generation_parameters": {
                "num_ctx": self.context_window,
                "num_predict": self.output_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }


class RRVerifier(ModelIdentity):
    """Frozen identity of the relation-recovery verifier.

    This is deliberately a separate role even when it resolves to the same
    physical model as answer/ER judge: its prompt and generation settings are
    independent scientific dimensions.
    """

    prompt_version: str
    temperature: float = Field(ge=0.0)
    seed: int
    context_window: int = Field(gt=0)
    output_tokens: int = Field(gt=0)

    @field_validator("prompt_version")
    @classmethod
    def _prompt_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("RR verifier prompt_version must be non-empty")
        return value

    def identity_payload(self) -> dict[str, Any]:
        unresolved = self.unresolved_fields("roles.rr_verifier")
        if unresolved:
            raise ValueError(
                "RR verifier identity is unresolved: " + ", ".join(unresolved)
            )
        return {
            "model_tag": self.resolved_name,
            "model_digest": self.digest,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "seed": self.seed,
            "generation_parameters": {
                "num_ctx": self.context_window,
                "num_predict": self.output_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }


class ModelRoles(StrictModel):
    """Roles are intentionally independent; no role inherits the builder."""

    query: RoleModel
    answer: RoleModel
    embedding: EmbeddingModel
    er_judge: ERJudge | None = None
    rr_verifier: RRVerifier | None = None


class BuilderModel(ModelIdentity):
    key: str
    display_name: str
    family: Literal["qwen3.5", "qwen3", "gemma3"]
    quantization: Literal["Q4_K_M"] = "Q4_K_M"


class CorpusConfig(StrictModel):
    subset_id: str
    manifest_path: str
    manifest_sha256: str
    documents_path: str
    documents_sha256: str
    questions_path: str
    questions_sha256: str
    expected_documents: int = Field(gt=0)
    expected_questions: int = Field(gt=0)
    expected_gold_documents: int = Field(ge=0)
    expected_hard_negatives: int = Field(ge=0)

    @field_validator("manifest_sha256", "documents_sha256", "questions_sha256")
    @classmethod
    def _hashes(cls, value: str) -> str:
        result = _validate_sha256(value, field_name="corpus hash")
        assert result is not None
        return result

    @field_validator("subset_id", "manifest_path", "documents_path", "questions_path")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("corpus identifiers and paths must be non-empty")
        return value


class ChunkingConfig(StrictModel):
    version: str = "lightrag-fixed-token-v1"
    chunk_token_size: int = Field(default=1200, gt=0)
    chunk_overlap_token_size: int = Field(default=100, ge=0)
    split_by_character: str | None = None
    split_by_character_only: bool = False

    @model_validator(mode="after")
    def _overlap_is_smaller(self) -> "ChunkingConfig":
        if self.chunk_overlap_token_size >= self.chunk_token_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class ExtractionConfig(StrictModel):
    version: str = "native-lightrag-extraction-v1"
    prompt_version: str
    prompt_sha256: str | None = None
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int = 42
    context_window: int = Field(default=8192, gt=0)
    output_tokens: int = Field(default=4096, gt=0)
    think: bool = False
    entity_types_guidance: str | None = None
    max_gleaning: int = Field(default=1, ge=0)
    json_extraction: bool = True
    llm_concurrency: int = Field(default=1, gt=0)
    document_insertion_concurrency: int = Field(default=1, gt=0)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    @field_validator("prompt_version", "version")
    @classmethod
    def _versions(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("extraction versions must be non-empty")
        return value

    @field_validator("prompt_sha256")
    @classmethod
    def _prompt_hash(cls, value: str | None) -> str | None:
        return _validate_sha256(value, field_name="extraction prompt hash")

    @field_validator("entity_types_guidance")
    @classmethod
    def _entity_types_guidance(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("entity_types_guidance must be null or non-empty")
        return value


class ERDecisionPolicyConfig(StrictModel):
    """Explicit identity of the fixed ER pair-decision policy."""

    version: Literal["expert-fixed-v1"] = "expert-fixed-v1"
    reject_below_score: Literal[0.5] = 0.5
    auto_merge_at_score: Literal[0.76] = 0.76
    judge_candidate_method: Literal["reciprocal_embedding_neighbour"] = (
        "reciprocal_embedding_neighbour"
    )
    judge_mutual_top_k: Literal[1] = 1
    max_judge_calls_per_run: int = Field(default=650, ge=0)


class EntityResolutionConfig(StrictModel):
    allowed_signal_weights: ClassVar[frozenset[str]] = frozenset(
        {
            "name_exact",
            "lexical",
            "acronym",
            "containment",
            "description",
            "embedding",
            "type",
            "neighbourhood",
            "relation_context",
            "provenance",
            "frequency",
            "source_diversity",
        }
    )
    version: Literal["corpus-native-merge-only-er-v8"] = (
        "corpus-native-merge-only-er-v8"
    )
    config_version: Literal["8.0.0"] = "8.0.0"
    decision_policy: ERDecisionPolicyConfig
    fuzzy_candidate_threshold: float = Field(ge=0.0, le=1.0)
    containment_min_chars: int = Field(gt=0)
    max_block_size: int = Field(gt=1)
    embedding_neighbor_k: int = Field(gt=0)
    embedding_lsh_tables: int = Field(gt=0)
    embedding_lsh_bits: int = Field(gt=0)
    embedding_lsh_max_bucket: int = Field(gt=0)
    candidate_methods: tuple[str, ...]
    signal_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "name_exact": 0.24,
            "lexical": 0.15,
            "acronym": 0.09,
            "containment": 0.08,
            "description": 0.12,
            "embedding": 0.12,
            "type": 0.08,
            "neighbourhood": 0.05,
            "relation_context": 0.04,
            "provenance": 0.01,
            "frequency": 0.01,
            "source_diversity": 0.01,
        }
    )

    @model_validator(mode="after")
    def _validate_er_policy(self) -> "EntityResolutionConfig":
        if any(weight < 0 for weight in self.signal_weights.values()):
            raise ValueError("ER signal weights must be non-negative")
        if not any(weight > 0 for weight in self.signal_weights.values()):
            raise ValueError("at least one ER signal weight must be positive")
        if set(self.signal_weights) != self.allowed_signal_weights:
            missing = sorted(self.allowed_signal_weights - set(self.signal_weights))
            extra = sorted(set(self.signal_weights) - self.allowed_signal_weights)
            raise ValueError(
                f"ER signal weights must use the exact scorer keys; "
                f"missing={missing}, extra={extra}"
            )
        if len(self.candidate_methods) != len(set(self.candidate_methods)):
            raise ValueError("ER candidate methods must be unique")
        if self.candidate_methods != SUPPORTED_ER_CANDIDATE_METHODS:
            raise ValueError(
                "ER candidate methods must exactly match the implemented generator: "
                f"{SUPPORTED_ER_CANDIDATE_METHODS!r}"
            )
        if self.decision_policy.judge_candidate_method not in self.candidate_methods:
            raise ValueError(
                "ER judge candidate method must be an implemented candidate route"
            )
        return self

    def pipeline_payload(self) -> dict[str, Any]:
        """Translate exactly to ``entity_resolution.models.ERConfig`` fields."""

        from src.entity_resolution.models import ERDecisionPolicy

        return {
            "decision_policy": ERDecisionPolicy(
                **self.decision_policy.model_dump(mode="python")
            ),
            "fuzzy_candidate_threshold": self.fuzzy_candidate_threshold,
            "containment_min_chars": self.containment_min_chars,
            "max_block_size": self.max_block_size,
            "er_version": self.version,
            "scoring_weights": dict(self.signal_weights),
        }

    def embedding_candidate_payload(self) -> dict[str, int]:
        """Return the frozen ANN/LSH pre-candidate parameters."""

        return {
            "embedding_neighbor_k": self.embedding_neighbor_k,
            "embedding_lsh_tables": self.embedding_lsh_tables,
            "embedding_lsh_bits": self.embedding_lsh_bits,
            "embedding_lsh_max_bucket": self.embedding_lsh_max_bucket,
        }


class GraphMaterializationConfig(StrictModel):
    """Storage policy for materialized derived graphs."""

    missing_description_policy: Literal["preserve_with_audited_placeholders"] | None = (
        None
    )


class RelationRecoveryConfig(StrictModel):
    """Minimal evidence-bound relation recovery after frozen ER."""

    version: Literal["evidence-bound-relation-recovery-v1"] = (
        "evidence-bound-relation-recovery-v1"
    )
    config_version: Literal["1.0.0"] = "1.0.0"
    candidate_policy_version: Literal["same-chunk-missing-canonical-pair-v1"] = (
        "same-chunk-missing-canonical-pair-v1"
    )
    require_exact_evidence_quote: Literal[True] = True
    include_existing_local_relations: Literal[True] = True
    max_calls_per_chunk: Literal[1] = 1
    retry_count: Literal[0] = 0


class LightRAGStorageConfig(StrictModel):
    """Explicit public LightRAG storage implementations for every workspace."""

    kv_storage: str
    vector_storage: str
    graph_storage: str
    doc_status_storage: str

    @field_validator(
        "kv_storage", "vector_storage", "graph_storage", "doc_status_storage"
    )
    @classmethod
    def _storage_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("LightRAG storage backend names must be non-empty")
        return value


class EmbeddingCacheConfig(StrictModel):
    enabled: bool
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    use_llm_check: bool


class LightRAGRuntimeConfig(StrictModel):
    """Non-scientific LightRAG settings which must never come from defaults."""

    provider: Literal["ollama"]
    embedding_provider: Literal["ollama"]
    llm_timeout_seconds: float = Field(gt=0)
    max_extract_input_tokens: int = Field(gt=0)
    index_batch_size: int = Field(gt=0)
    storage: LightRAGStorageConfig
    tiktoken_model_name: str
    embedding_func_max_async: int = Field(gt=0)
    kg_linked_chunk_selection: Literal["WEIGHT", "VECTOR"]
    enable_llm_cache: bool
    enable_llm_cache_for_entity_extract: bool
    embedding_cache: EmbeddingCacheConfig

    @field_validator("tiktoken_model_name")
    @classmethod
    def _tokenizer_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tiktoken_model_name must be non-empty")
        return value

    def legacy_payload(self) -> dict[str, Any]:
        """Flatten the explicit runtime fields to LightRAG constructor names."""

        return {
            "provider": self.provider,
            "embedding_provider": self.embedding_provider,
            "llm_timeout": self.llm_timeout_seconds,
            "max_extract_input_tokens": self.max_extract_input_tokens,
            "index_batch_size": self.index_batch_size,
            **self.storage.model_dump(mode="json"),
            "tiktoken_model_name": self.tiktoken_model_name,
            "embedding_func_max_async": self.embedding_func_max_async,
            "kg_chunk_pick_method": self.kg_linked_chunk_selection,
            "enable_llm_cache": self.enable_llm_cache,
            "enable_llm_cache_for_entity_extract": self.enable_llm_cache_for_entity_extract,
            "embedding_cache_config": self.embedding_cache.model_dump(mode="json"),
        }


class RetrievalConfig(StrictModel):
    mode: Literal["hybrid"]
    top_k: int = Field(gt=0)
    chunk_top_k: int = Field(gt=0)
    max_entity_tokens: int = Field(gt=0)
    max_relation_tokens: int = Field(gt=0)
    max_total_tokens: int = Field(gt=0)
    enable_rerank: bool
    timeout_seconds: float = Field(gt=0)

    def legacy_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("mode")
        return payload


class ERJudgeBudgetOverride(StrictModel):
    """Attempt-level resource ceiling which must not alter ER pair semantics."""

    max_judge_calls_per_run: int = Field(gt=0)
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ER judge budget override reason must be non-empty")
        return value


class RuntimeConfig(StrictModel):
    """Execution settings shared by all conditions and frozen with each run."""

    runs_root: str
    minimum_free_disk_gib: float = Field(ge=0.0)
    artifact_schema_version: str
    ollama_host: str
    model_timeout_seconds: float = Field(gt=0)
    lightrag: LightRAGRuntimeConfig
    retrieval: RetrievalConfig
    er_judge_budget_overrides: dict[str, ERJudgeBudgetOverride] = Field(
        default_factory=dict
    )

    @field_validator("runs_root", "artifact_schema_version")
    @classmethod
    def _runtime_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("runtime paths and versions must be non-empty")
        return value

    @field_validator("ollama_host")
    @classmethod
    def _ollama_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("runtime.ollama_host must be an explicit HTTP(S) URL")
        return value


class ExperimentConfig(StrictModel):
    schema_version: Literal["4.0.0"] = CONFIG_SCHEMA_VERSION
    experiment_id: str
    corpus: CorpusConfig
    builders: tuple[BuilderModel, ...]
    roles: ModelRoles
    extraction: ExtractionConfig
    entity_resolution: EntityResolutionConfig
    relation_recovery: RelationRecoveryConfig
    graph_materialization: GraphMaterializationConfig
    graph_regimes: tuple[
        Literal[
            "native_lightrag",
            "advanced_lightrag_er",
            "advanced_lightrag_er_rr",
        ],
        ...,
    ] = GRAPH_REGIMES
    runtime: RuntimeConfig
    metadata: dict[str, Any] = Field(default_factory=dict)

    expected_builders: ClassVar[tuple[tuple[str, str], ...]] = EXPECTED_BUILDERS

    @model_validator(mode="after")
    def _scientific_dimensions(self) -> "ExperimentConfig":
        actual = tuple(
            (builder.key, builder.requested_tag) for builder in self.builders
        )
        if actual != self.expected_builders:
            raise ValueError(
                "builders must contain the exact 12 ordered experiment tags; "
                f"got {actual!r}"
            )
        if tuple(self.graph_regimes) != GRAPH_REGIMES:
            raise ValueError(f"graph_regimes must be exactly {GRAPH_REGIMES!r}")
        if self.corpus.expected_documents != (
            self.corpus.expected_gold_documents + self.corpus.expected_hard_negatives
        ):
            raise ValueError("expected corpus role counts do not sum to documents")
        unknown_budget_overrides = sorted(
            set(self.runtime.er_judge_budget_overrides) - set(self.builders_by_key)
        )
        if unknown_budget_overrides:
            raise ValueError(
                "ER judge budget overrides reference unknown builders: "
                f"{unknown_budget_overrides}"
            )
        planned_budget = self.entity_resolution.decision_policy.max_judge_calls_per_run
        invalid_budget_overrides = sorted(
            key
            for key, override in self.runtime.er_judge_budget_overrides.items()
            if override.max_judge_calls_per_run < planned_budget
        )
        if invalid_budget_overrides:
            raise ValueError(
                "operational ER judge budget overrides cannot reduce the frozen "
                f"planned budget: {invalid_budget_overrides}"
            )
        return self

    @property
    def builders_by_key(self) -> dict[str, BuilderModel]:
        return {builder.key: builder for builder in self.builders}

    def effective_er_judge_budget(
        self, builder_key: str
    ) -> tuple[int, ERJudgeBudgetOverride | None]:
        """Return the frozen default or an explicitly audited operational override."""

        if builder_key not in self.builders_by_key:
            raise KeyError(f"unknown builder: {builder_key}")
        override = self.runtime.er_judge_budget_overrides.get(builder_key)
        if override is not None:
            return override.max_judge_calls_per_run, override
        return self.entity_resolution.decision_policy.max_judge_calls_per_run, None

    def unresolved_requirements(
        self,
        *,
        builder_key: str | None = None,
        require_all_builders: bool = False,
        require_er: bool = True,
        require_rr: bool = False,
    ) -> list[str]:
        """List configuration fields that must be resolved before execution."""

        unresolved: list[str] = []
        if builder_key is not None and builder_key not in self.builders_by_key:
            return [f"builders.{builder_key}"]
        selected = (
            self.builders
            if require_all_builders or builder_key is None
            else (self.builders_by_key[builder_key],)
        )
        for builder in selected:
            unresolved.extend(builder.unresolved_fields(f"builders.{builder.key}"))
        unresolved.extend(self.roles.query.unresolved_fields("roles.query"))
        unresolved.extend(self.roles.answer.unresolved_fields("roles.answer"))
        unresolved.extend(self.roles.embedding.unresolved_fields("roles.embedding"))
        if self.extraction.prompt_sha256 is None:
            unresolved.append("extraction.prompt_sha256")
        if require_er:
            if self.graph_materialization.missing_description_policy is None:
                unresolved.append("graph_materialization.missing_description_policy")
            if self.roles.er_judge is None:
                unresolved.append("roles.er_judge")
            else:
                unresolved.extend(
                    self.roles.er_judge.unresolved_fields("roles.er_judge")
                )
        if require_rr:
            if not require_er:
                unresolved.append("relation_recovery.requires_entity_resolution")
            if self.roles.rr_verifier is None:
                unresolved.append("roles.rr_verifier")
            else:
                unresolved.extend(
                    self.roles.rr_verifier.unresolved_fields("roles.rr_verifier")
                )
        return sorted(set(unresolved))

    def assert_ready(
        self,
        *,
        builder_key: str | None = None,
        require_all_builders: bool = False,
        require_er: bool = True,
        require_rr: bool = False,
    ) -> None:
        unresolved = self.unresolved_requirements(
            builder_key=builder_key,
            require_all_builders=require_all_builders,
            require_er=require_er,
            require_rr=require_rr,
        )
        if unresolved:
            raise ValueError(
                "experiment configuration is not executable; unresolved: "
                + ", ".join(unresolved)
            )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "EXPECTED_BUILDERS",
    "GRAPH_REGIMES",
    "SUPPORTED_ER_CANDIDATE_METHODS",
    "BuilderModel",
    "ChunkingConfig",
    "CorpusConfig",
    "EmbeddingModel",
    "EmbeddingCacheConfig",
    "ERDecisionPolicyConfig",
    "ERJudgeBudgetOverride",
    "EntityResolutionConfig",
    "ERJudge",
    "ExperimentConfig",
    "ExtractionConfig",
    "GenerationSettings",
    "GraphMaterializationConfig",
    "LightRAGRuntimeConfig",
    "LightRAGStorageConfig",
    "ModelIdentity",
    "ModelRoles",
    "RoleModel",
    "RetrievalConfig",
    "RelationRecoveryConfig",
    "RRVerifier",
    "RuntimeConfig",
]
