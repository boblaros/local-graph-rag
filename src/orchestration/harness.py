"""Run the Native, ER, ER+RR, retrieval, answering, and evaluation stages.

Only :meth:`ExperimentHarness.native_build` performs builder extraction. Every
later graph stage uses the normalized records saved by that method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config import LoadedExperimentConfig
from src.entity_resolution import (
    SCHEMA_VERSION as ER_ARTIFACT_SCHEMA_VERSION,
    CorpusSnapshot,
    ERConfig,
    ERLineage,
    EntityResolutionPipeline,
    JudgeCache,
    JudgeBudgetExceededError,
    OllamaEmbeddingIdentity,
    OllamaERJudge,
    augment_mentions_with_ollama_embeddings,
    build_native_entity_embedding_records,
    build_lsh_cosine_neighbor_evidence,
    content_hash,
)
from src.extraction import load_staged_snapshot
from src.graph import materialize_finalize_reopen, rewrite_graph
from src.relation_recovery import (
    OllamaRRVerifier,
    aggregate_rr_graph,
    build_rr_plan,
    graph_edge_pairs,
    validate_verifier_response,
)

from .artifacts import (
    jsonable,
    read_enveloped_jsonl,
    read_json,
    read_jsonl,
    write_immutable_json,
    write_immutable_jsonl,
)
from .lineage import (
    ArtifactRef,
    BaseRunManifest,
    VariantRunManifest,
    base_config_sha256,
    build_base_manifest,
    build_variant_manifest,
    compute_base_run_id,
    compute_variant_run_id,
    fingerprint_artifact,
    freeze_base_config,
    freeze_manifest,
    full_config_sha256,
    resolve_artifact_path,
    sha256_json,
    variant_config_sha256,
    verify_artifact,
)
from .native import build_native_and_stage, prepare_lightrag_run, workspace_tree_sha256
from .paths import ConditionPaths
from .preflight import validate_preflight
from .quality_gates import (
    QualityGateReport,
    validate_er_quality_gate,
    validate_extraction_quality_gate,
    validate_final_workspace_quality_gate,
    validate_rr_quality_gate,
)
from .status import StageExpectation, StageStatus, StageStatusStore


GraphRegime = Literal[
    "native_lightrag", "advanced_lightrag_er", "advanced_lightrag_er_rr"
]
EVALUATION_SCHEMA_VERSION = "5.0.0"


@dataclass(frozen=True)
class StageOutcome:
    stage: str
    item_id: str
    reused: bool
    payload: dict[str, Any]


def _required(value: str | None, label: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{label} must be resolved and non-empty")
    return value


def _validate_native_entity_group_coverage(
    mentions: list[Any], native_nodes: list[dict[str, Any]]
) -> None:
    """Require staged entity names to reproduce the Native graph node IDs."""

    staged_names = {str(item.original_name) for item in mentions}
    node_ids = [str(row.get("node_id") or "").strip() for row in native_nodes]
    native_names = set(node_ids)
    if (
        not staged_names
        or not native_names
        or "" in native_names
        or len(node_ids) != len(native_names)
        or staged_names != native_names
    ):
        missing = sorted(native_names - staged_names)
        extra = sorted(staged_names - native_names)
        raise RuntimeError(
            "merge-only ER cannot reproduce exact Native entity groups: "
            f"missing_from_staging={missing[:3]}, absent_from_native={extra[:3]}"
        )


def _load_questions(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ids = [str(row.get("question_id") or "") for row in rows]
    if not ids or "" in ids or len(ids) != len(set(ids)):
        raise ValueError(f"question catalog has missing/duplicate IDs: {path}")
    return rows


def _load_documents(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ids = [str(row.get("document_id") or "") for row in rows]
    if not ids or "" in ids or len(ids) != len(set(ids)):
        raise ValueError(f"document catalog has missing/duplicate IDs: {path}")
    return rows


def _smoke_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answerable = next((row for row in questions if row.get("answerable") is True), None)
    unanswerable = next(
        (row for row in questions if row.get("answerable") is False), None
    )
    if answerable is None or unanswerable is None:
        raise ValueError("smoke requires one answerable and one unanswerable question")
    return [answerable, unanswerable]


def _query_parameters(config: Any) -> dict[str, Any]:
    return {
        "top_k": int(config.top_k),
        "chunk_top_k": int(config.chunk_top_k),
        "max_entity_tokens": int(config.max_entity_tokens),
        "max_relation_tokens": int(config.max_relation_tokens),
        "max_total_tokens": int(config.max_total_tokens),
        "enable_rerank": bool(config.enable_rerank),
    }


def _read_base_manifest(path: Path) -> BaseRunManifest:
    return BaseRunManifest.model_validate(read_json(path))


def _er_runtime_config(config: Any) -> ERConfig:
    return ERConfig(**config.pipeline_payload())


def _condition_file(
    config: Any,
    paths: ConditionPaths,
    graph_regime: GraphRegime,
    *,
    directory: str,
    stem: str,
    suffix: str,
) -> Path:
    """Keep downstream outputs from different frozen role configs disjoint."""

    condition_hash = full_config_sha256(config)[:16]
    return (
        paths.variant_dir(graph_regime) / directory / f"{stem}.{condition_hash}{suffix}"
    )


def _qa_artifact_path(
    status: StageStatusStore,
    config: Any,
    paths: ConditionPaths,
    graph_regime: GraphRegime,
    *,
    stage: Literal["retrieval", "answering"],
) -> Path:
    """Resolve a completed QA artifact from lifecycle truth.

    QA filenames include the full config hash that was current when the stage
    ran. Evaluation may legitimately happen after an analysis-only config
    identity change, so reconstructing the filename from today's config can
    miss an immutable, completed artifact. Prefer the completed lifecycle ref
    and verify it fail-closed; retain the historical filename fallback only for
    callers that have not created lifecycle records.
    """

    variant_run_id = {
        "native_lightrag": paths.native_variant_run_id,
        "advanced_lightrag_er": paths.er_variant_run_id,
        "advanced_lightrag_er_rr": paths.rr_variant_run_id,
    }[graph_regime]
    output_key = "retrieval" if stage == "retrieval" else "answers"
    stem = "retrieval" if stage == "retrieval" else "answers"
    logical_name = "retrieval_results" if stage == "retrieval" else "answer_results"
    fallback = _condition_file(
        config,
        paths,
        graph_regime,
        directory="artifacts",
        stem=stem,
        suffix=".jsonl",
    )
    record = status.get(stage, variant_run_id)
    if record is None:
        return fallback
    if record.status != StageStatus.COMPLETED:
        raise RuntimeError(
            f"{stage} lifecycle is not completed for {variant_run_id}: "
            f"{record.status.value}"
        )
    if record.item_id != variant_run_id or record.lineage_id != variant_run_id:
        raise RuntimeError(f"{stage} lifecycle lineage mismatch for {variant_run_id}")
    ref = record.output_artifacts.get(output_key)
    if ref is None:
        raise RuntimeError(
            f"completed {stage} lifecycle has no {output_key} artifact: "
            f"{variant_run_id}"
        )
    expected_questions = int(config.corpus.expected_questions)
    if (
        ref.logical_name != logical_name
        or ref.schema_version != "3.0.0"
        or ref.record_count != expected_questions
    ):
        raise RuntimeError(
            f"completed {stage} lifecycle artifact contract mismatch for "
            f"{variant_run_id}"
        )
    valid, reasons = verify_artifact(ref, root=status.artifact_root)
    if not valid:
        raise RuntimeError(
            f"completed {stage} lifecycle artifact verification failed for "
            f"{variant_run_id}: {'; '.join(reasons)}"
        )
    artifact = resolve_artifact_path(ref, root=status.artifact_root)
    expected_parent = (paths.variant_dir(graph_regime) / "artifacts").resolve()
    if artifact.parent != expected_parent:
        raise RuntimeError(
            f"completed {stage} lifecycle artifact is outside its variant: "
            f"{variant_run_id}"
        )
    return artifact


def _evaluation_identity_sha256(config: Any) -> str:
    from src.evaluation import (
        CASCADE_COMPARISON_PLAN,
        DEFAULT_BOOTSTRAP_SAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        DEFAULT_CONFIDENCE_LEVEL,
        DEFAULT_PAIRED_METRIC_FIELDS,
        DEFAULT_RETRIEVAL_CUTOFFS,
        SOURCE_QUESTION_TYPE_WEIGHTS,
    )

    return sha256_json(
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "full_config_sha256": full_config_sha256(config),
            "comparison_plan": CASCADE_COMPARISON_PLAN,
            "retrieval_cutoffs": DEFAULT_RETRIEVAL_CUTOFFS,
            "paired_metric_fields": DEFAULT_PAIRED_METRIC_FIELDS,
            "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
            "bootstrap_confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "bootstrap_stratification": "question_type",
            "extraction_bootstrap_cluster_unit": "document_id",
            "source_question_type_weights": SOURCE_QUESTION_TYPE_WEIGHTS,
        }
    )


def _primary_analysis_identity_sha256(config: Any) -> str:
    from src.evaluation import (
        CASCADE_COMPARISON_PLAN,
        PRIMARY_ANALYSIS_SCHEMA_VERSION,
    )

    return sha256_json(
        {
            "schema_version": PRIMARY_ANALYSIS_SCHEMA_VERSION,
            "evaluation_identity_sha256": _evaluation_identity_sha256(config),
            "comparison_plan": CASCADE_COMPARISON_PLAN,
            "expected_builder_keys": [builder.key for builder in config.builders],
            "expected_question_count": config.corpus.expected_questions,
        }
    )


def _exploratory_identity_sha256(config: Any) -> str:
    from src.evaluation import (
        CASCADE_COMPARISON_PLAN,
        DEFAULT_BOOTSTRAP_SAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        DEFAULT_CONFIDENCE_LEVEL,
        DEFAULT_PAIRED_METRIC_FIELDS,
        EXPLORATORY_BINARY_METRICS,
        EXPLORATORY_SCHEMA_VERSION,
        builder_metadata_from_config,
    )

    return sha256_json(
        {
            "schema_version": EXPLORATORY_SCHEMA_VERSION,
            "evaluation_identity_sha256": _evaluation_identity_sha256(config),
            "primary_analysis_identity_sha256": _primary_analysis_identity_sha256(
                config
            ),
            "comparison_plan": CASCADE_COMPARISON_PLAN,
            "graph_regimes": [
                "native_lightrag",
                "advanced_lightrag_er",
                "advanced_lightrag_er_rr",
            ],
            "metric_fields": DEFAULT_PAIRED_METRIC_FIELDS,
            "binary_metric_fields": sorted(EXPLORATORY_BINARY_METRICS),
            "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
            "bootstrap_confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "bootstrap_stratification": "question_type",
            "mcnemar": "exact_two_sided",
            "holm_scope": "separate_by_estimand_graph_regime_metric",
            "builder_metadata": builder_metadata_from_config(config.builders),
        }
    )


def _metric_file(
    config: Any,
    paths: ConditionPaths,
    graph_regime: GraphRegime,
    *,
    stem: str,
    suffix: str,
) -> Path:
    condition_hash = _evaluation_identity_sha256(config)[:16]
    return (
        paths.variant_dir(graph_regime) / "metrics" / f"{stem}.{condition_hash}{suffix}"
    )


def _query_generation_parameters(config: Any) -> dict[str, Any]:
    generation = config.roles.query.generation
    return {
        "temperature": float(generation.temperature),
        "seed": int(generation.seed),
        "num_ctx": int(generation.context_window),
        "num_predict": int(generation.output_tokens),
        "think": bool(generation.think),
    }


def _published_er_materialization(
    paths: ConditionPaths,
    *,
    merge_plan_sha256: str,
) -> tuple[dict[str, Any], dict[str, ArtifactRef]] | None:
    """Recover the publish-complete window before the mutable status commit.

    Materialization artifacts and the workspace are immutable; the status file
    is intentionally the last mutable commit.  A process may therefore die
    after publishing a valid variant but before ``mark_completed``.  Verify the
    entire publication rather than attempting a second physical build.
    """

    locator_path = paths.locator("advanced_lightrag_er")
    manifest_path = paths.er_dir / "variant_manifest.json"
    gate_path = paths.er_dir / "gates" / "final_workspace.json"
    if not all(path.is_file() for path in (locator_path, manifest_path, gate_path)):
        return None

    locator = read_json(locator_path)
    manifest = VariantRunManifest.model_validate(read_json(manifest_path))
    gate = QualityGateReport.model_validate(read_json(gate_path))
    if not gate.passed:
        raise RuntimeError("published ER workspace has a failed final quality gate")
    if (
        manifest.graph_regime != "advanced_lightrag_er"
        or manifest.base_run_id != paths.base_run_id
        or manifest.variant_run_id != paths.er_variant_run_id
        or locator.get("base_run_id") != paths.base_run_id
        or locator.get("variant_run_id") != paths.er_variant_run_id
        or locator.get("graph_regime") != "advanced_lightrag_er"
        or locator.get("merge_plan_sha256") != merge_plan_sha256
        or manifest.merge_plan_sha256 != merge_plan_sha256
    ):
        raise RuntimeError("published ER materialization has incompatible lineage")
    if locator.get("native_workspace_sha256_before") != locator.get(
        "native_workspace_sha256_after"
    ):
        raise RuntimeError("published ER materialization changed the native workspace")

    artifact_refs = {
        name: ArtifactRef.model_validate(ref)
        for name, ref in manifest.artifacts.items()
    }
    if not artifact_refs:
        raise RuntimeError("published ER manifest has no graph artifact refs")
    invalid_artifacts: list[str] = []
    for name, ref in artifact_refs.items():
        valid, reasons = verify_artifact(ref)
        if not valid:
            invalid_artifacts.extend(f"{name}: {reason}" for reason in reasons)
    if invalid_artifacts:
        raise RuntimeError(
            "published ER graph artifact verification failed: " + invalid_artifacts[0]
        )

    workspace_dir = Path(str(locator.get("workspace_dir") or ""))
    if not workspace_dir.is_dir() or workspace_tree_sha256(
        workspace_dir
    ) != locator.get("workspace_sha256"):
        raise RuntimeError("published ER workspace hash mismatch")
    native_locator = read_json(paths.locator("native_lightrag"))
    native_workspace = Path(str(native_locator.get("workspace_dir") or ""))
    expected_native_hash = locator.get("native_workspace_sha256_after")
    if (
        not native_workspace.is_dir()
        or workspace_tree_sha256(native_workspace) != expected_native_hash
    ):
        raise RuntimeError("native workspace changed after ER publication")

    outputs = {
        "variant_manifest": fingerprint_artifact(
            manifest_path,
            logical_name="advanced_variant_manifest",
            schema_version="2.0.0",
        ),
        "workspace_locator": fingerprint_artifact(
            locator_path,
            logical_name="advanced_workspace_locator",
            schema_version="2.0.0",
        ),
        "final_workspace_gate": fingerprint_artifact(
            gate_path,
            logical_name="graph/final_workspace_gate",
            schema_version="2.0.0",
        ),
    }
    return locator, outputs


def _judge_budget_gate_path(paths: ConditionPaths, *, attempt_count: int) -> Path:
    """Keep a failed budget gate and give every retry its own immutable audit."""

    legacy = paths.er_dir / "gates" / "judge_budget.json"
    if attempt_count <= 1 or not legacy.exists():
        return legacy
    return paths.er_dir / "gates" / f"judge_budget.attempt-{attempt_count:04d}.json"


class ExperimentHarness:
    """One condition-at-a-time lifecycle with hash-aware stage reuse."""

    def __init__(self, loaded: LoadedExperimentConfig):
        self.loaded = loaded
        self.config = loaded.config
        self.runs_root = loaded.runs_root
        self.status = StageStatusStore(self.runs_root / "status.json")

    def condition_paths(self, builder_key: str) -> ConditionPaths:
        base = compute_base_run_id(self.config, builder_key)
        native = compute_variant_run_id(
            self.config, base_run_id=base, graph_regime="native_lightrag"
        )
        advanced = compute_variant_run_id(
            self.config, base_run_id=base, graph_regime="advanced_lightrag_er"
        )
        rr = compute_variant_run_id(
            self.config, base_run_id=base, graph_regime="advanced_lightrag_er_rr"
        )
        return ConditionPaths(self.runs_root, base, native, advanced, rr)

    def _expectation(
        self,
        lineage_id: str,
        *,
        inputs: dict[str, Any] | None = None,
        config_sha256: str | None = None,
    ) -> StageExpectation:
        return StageExpectation(
            lineage_id=lineage_id,
            config_sha256=config_sha256 or full_config_sha256(self.config),
            input_artifacts=dict(inputs or {}),
        )

    def _freeze_config(self, paths: ConditionPaths, builder_key: str) -> Path:
        destination = paths.base_dir / "config.frozen.json"
        freeze_base_config(destination, self.config, builder_key)
        return destination

    async def preflight(self, builder_key: str) -> QualityGateReport:
        """Resolve a read-only local inventory and validate all frozen inputs."""

        from src.config import (
            resolve_extraction_prompt_sha256,
            resolve_ollama_inventory,
        )

        requested = [
            str(model.requested_tag)
            for model in (
                *self.config.builders,
                self.config.roles.query,
                self.config.roles.answer,
                self.config.roles.embedding,
                *(
                    (self.config.roles.er_judge,)
                    if self.config.roles.er_judge is not None
                    else ()
                ),
                *(
                    (self.config.roles.rr_verifier,)
                    if self.config.roles.rr_verifier is not None
                    else ()
                ),
            )
            if model.requested_tag
        ]
        _, inventory = await resolve_ollama_inventory(
            requested, host=self.config.runtime.ollama_host
        )
        report = validate_preflight(
            self.loaded,
            model_inventory=inventory,
            builder_key=builder_key,
            require_er=True,
            require_rr=True,
            effective_prompt_sha256=resolve_extraction_prompt_sha256(
                use_json=self.config.extraction.json_extraction,
                addon_params=(
                    {
                        "entity_types_guidance": (
                            self.config.extraction.entity_types_guidance
                        )
                    }
                    if self.config.extraction.entity_types_guidance
                    else None
                ),
            ),
            fail_closed=False,
        )
        paths = self.condition_paths(builder_key)
        report_path = (
            paths.base_dir
            / "gates"
            / (
                f"preflight-{full_config_sha256(self.config)[:16]}-"
                f"{sha256_json(report)[:16]}.json"
            )
        )
        write_immutable_json(report_path, report)
        return report.require_passed()

    async def _verify_current_models(self, *models: Any) -> None:
        """Fail before a stage if a frozen Ollama tag/digest has drifted."""

        from src.config import resolve_ollama_inventory

        requested = [
            _required(model.requested_tag, "requested model tag") for model in models
        ]
        identities, _ = await resolve_ollama_inventory(
            requested, host=self.config.runtime.ollama_host
        )
        errors: list[str] = []
        for model, requested_tag in zip(models, requested, strict=True):
            discovered = identities.get(requested_tag)
            if discovered is None:
                errors.append(f"unavailable: {requested_tag}")
                continue
            expected_name = _required(model.resolved_name, "resolved model name")
            expected_digest = _required(model.digest, "resolved model digest")
            if discovered.get("resolved_name") != expected_name:
                errors.append(
                    f"resolved name drift for {requested_tag}: "
                    f"{discovered.get('resolved_name')!r} != {expected_name!r}"
                )
            if discovered.get("digest") != expected_digest:
                errors.append(f"digest drift for {requested_tag}")
        if errors:
            raise RuntimeError(
                "frozen model identity check failed: " + "; ".join(errors)
            )

    def _extraction_gate(
        self,
        paths: ConditionPaths,
        *,
        native_run_dir: Path,
        fail_closed: bool,
    ) -> QualityGateReport:
        manifest, chunks, entities, relations = load_staged_snapshot(paths.staging_dir)
        documents = _load_documents(self.loaded.documents_path)
        calls = read_jsonl(native_run_dir / "artifacts" / "extraction_calls.jsonl")
        from src.extraction.native_capture import NATIVE_CAPTURE_VERSION

        return validate_extraction_quality_gate(
            expected_document_ids=[str(row["document_id"]) for row in documents],
            manifest=manifest,
            chunks=chunks,
            extraction_calls=calls,
            entities=entities,
            relations=relations,
            native_capture_version=NATIVE_CAPTURE_VERSION,
            fail_closed=fail_closed,
        )

    async def _validate_native_workspace(
        self,
        frozen: Any,
        *,
        paths: ConditionPaths,
        base_extraction_sha256: str,
    ) -> tuple[QualityGateReport, dict[str, Any]]:
        from lightrag import QueryParam
        from src.extraction.native_capture import build_lightrag
        from src.graph import PublicLightRAGAdapter

        _, chunks, _, _ = load_staged_snapshot(paths.staging_dir)
        graph_nodes = read_jsonl(frozen.paths.graph_nodes_jsonl)
        graph_edges = read_jsonl(frozen.paths.graph_edges_jsonl)
        rag = build_lightrag(frozen, extraction_capture=False)
        adapter = PublicLightRAGAdapter(rag)
        initialized = False
        try:
            await adapter.initialize()
            initialized = True
            snapshot = await adapter.export_workspace(
                expected_node_count=len(graph_nodes),
                chunk_ids=[chunk.chunk_id for chunk in chunks],
            )
            smoke_rows = _smoke_questions(_load_questions(self.loaded.questions_path))
            smoke = await adapter.smoke_retrieval(
                [
                    {
                        "question": row["question"],
                        # Retrieval over a fixed full corpus should execute for
                        # both dataset labels; answerability is evaluated later.
                        "expected_status": "success",
                    }
                    for row in smoke_rows
                ],
                query_param_factory=lambda mode: QueryParam(
                    mode=mode,
                    only_need_context=True,
                    stream=False,
                    **_query_parameters(self.config.runtime.retrieval),
                ),
            )
        finally:
            if initialized:
                await adapter.finalize()
        gate = validate_final_workspace_quality_gate(
            graph_regime="native_lightrag",
            base_extraction_sha256=base_extraction_sha256,
            expected_base_extraction_sha256=base_extraction_sha256,
            # get_knowledge_graph is capped by LightRAG's max_graph_nodes
            # setting. The uncapped JSONL exports were produced from the same
            # persisted workspace through get_all_nodes/get_all_edges, so use
            # them for complete structural validation.
            nodes=graph_nodes,
            edges=graph_edges,
            chunks=snapshot.chunks,
            expected_chunk_ids=[chunk.chunk_id for chunk in chunks],
            induced_self_loops=[],
            workspace_clean=True,
            persisted=True,
            finalized=True,
            reopened=True,
            export_succeeded=True,
            smoke_retrieval_answerable=bool(smoke[0]["passed"]),
            smoke_retrieval_unanswerable=bool(smoke[1]["passed"]),
            fail_closed=False,
        )
        return gate, {
            "workspace": jsonable(snapshot),
            "full_graph": {
                "nodes": len(graph_nodes),
                "edges": len(graph_edges),
                "nodes_path": str(frozen.paths.graph_nodes_jsonl.resolve()),
                "edges_path": str(frozen.paths.graph_edges_jsonl.resolve()),
            },
            "smoke": smoke,
        }

    async def native_build(
        self, builder_key: str, *, reclaim_running: bool = False
    ) -> StageOutcome:
        """Perform the sole native extraction and freeze its reusable snapshot."""

        # Freeze only extraction identity here. Graph/downstream identities are
        # independently frozen by their own manifests and status expectations.
        self.config.assert_ready(builder_key=builder_key, require_er=False)
        paths = self.condition_paths(builder_key)
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._freeze_config(paths, builder_key)
        expectation = self._expectation(
            paths.base_run_id,
            config_sha256=base_config_sha256(self.config, builder_key),
        )
        claim = self.status.claim(
            "native_extraction_build",
            paths.base_run_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        if claim.reused:
            native_locator = paths.locator("native_lightrag")
            if not native_locator.is_file():
                raise RuntimeError(
                    "the immutable base extraction is already complete, but the "
                    "requested Native graph identity has changed. Pinned LightRAG "
                    "cannot rebuild its Native graph from staged extraction without "
                    "another builder call; preserve the frozen embedding/graph "
                    "settings or start an explicitly new base extraction identity"
                )
            return StageOutcome(
                "native_extraction_build",
                paths.base_run_id,
                True,
                read_json(native_locator),
            )
        if not claim.claimed:
            raise RuntimeError(
                "native extraction stage is already active or not claimable"
            )
        try:
            preflight_report = await self.preflight(builder_key)
            preflight_report_path = (
                paths.base_dir
                / "gates"
                / (
                    f"preflight-{full_config_sha256(self.config)[:16]}-"
                    f"{sha256_json(preflight_report)[:16]}.json"
                )
            )
            frozen, indexing_result, _ = await build_native_and_stage(
                self.loaded,
                builder_key=builder_key,
                base_run_id=paths.base_run_id,
                variant_run_id=paths.native_variant_run_id,
                reclaim_running=reclaim_running,
                output_dir=paths.staging_dir,
            )
            extraction_gate = self._extraction_gate(
                paths, native_run_dir=frozen.paths.run_dir, fail_closed=False
            )
            extraction_gate_path = paths.base_dir / "gates" / "extraction.json"
            write_immutable_json(extraction_gate_path, extraction_gate)
            if not extraction_gate.passed:
                self.status.mark_blocked_by_quality_gate(
                    "native_extraction_build",
                    paths.base_run_id,
                    "; ".join(extraction_gate.errors),
                )
                raise RuntimeError(
                    "native extraction blocked by extraction quality gate"
                )

            artifact_paths = {
                "raw_extraction_calls": frozen.paths.extraction_calls_jsonl,
                "native_chunks": frozen.paths.chunks_jsonl,
                "normalized_chunks": paths.staging_dir / "normalized_chunks.jsonl",
                "normalized_entities": paths.staging_dir / "normalized_entities.jsonl",
                "normalized_relations": paths.staging_dir
                / "normalized_relations.jsonl",
                "staging_manifest": paths.staging_dir / "staging_manifest.json",
                "extraction_gate": extraction_gate_path,
            }
            refs = {
                name: fingerprint_artifact(
                    path,
                    logical_name=name,
                    schema_version=self.config.runtime.artifact_schema_version,
                )
                for name, path in artifact_paths.items()
            }
            base_manifest = build_base_manifest(
                self.config, builder_key, artifacts=refs
            )
            freeze_manifest(paths.base_manifest, base_manifest)
            assert base_manifest.base_extraction_sha256 is not None

            native_gate, native_audit = await self._validate_native_workspace(
                frozen,
                paths=paths,
                base_extraction_sha256=base_manifest.base_extraction_sha256,
            )
            native_gate_path = paths.native_dir / "gates" / "final_workspace.json"
            native_audit_path = paths.native_dir / "artifacts" / "workspace_audit.json"
            write_immutable_json(native_gate_path, native_gate)
            write_immutable_json(native_audit_path, native_audit)
            if not native_gate.passed:
                self.status.mark_blocked_by_quality_gate(
                    "native_extraction_build",
                    paths.base_run_id,
                    "; ".join(native_gate.errors),
                )
                raise RuntimeError("native workspace blocked by final workspace gate")

            locator = {
                "schema_version": "2.0.0",
                "builder_key": builder_key,
                "base_run_id": paths.base_run_id,
                "variant_run_id": paths.native_variant_run_id,
                "graph_regime": "native_lightrag",
                "base_extraction_sha256": base_manifest.base_extraction_sha256,
                "legacy_run_id": frozen.lock.run_id,
                "legacy_run_dir": str(frozen.paths.run_dir.resolve()),
                "workspace_dir": str(frozen.paths.workspace_dir.resolve()),
                "workspace_sha256": workspace_tree_sha256(frozen.paths.workspace_dir),
                "config_frozen_path": str(config_path.resolve()),
            }
            locator_path = paths.locator("native_lightrag")
            write_immutable_json(locator_path, locator)
            variant_artifacts = {
                "native_workspace_gate": fingerprint_artifact(
                    native_gate_path,
                    logical_name="native_workspace_gate",
                    schema_version="2.0.0",
                ),
                "workspace_audit": fingerprint_artifact(
                    native_audit_path,
                    logical_name="workspace_audit",
                    schema_version="2.0.0",
                ),
            }
            native_manifest = build_variant_manifest(
                self.config,
                base_manifest=base_manifest,
                graph_regime="native_lightrag",
                artifacts=variant_artifacts,
            )
            variant_manifest_path = paths.native_dir / "variant_manifest.json"
            freeze_manifest(variant_manifest_path, native_manifest)
            outputs = {
                "base_manifest": fingerprint_artifact(
                    paths.base_manifest,
                    logical_name="base_manifest",
                    schema_version="2.0.0",
                ),
                "native_variant_manifest": fingerprint_artifact(
                    variant_manifest_path,
                    logical_name="native_variant_manifest",
                    schema_version="2.0.0",
                ),
                "native_locator": fingerprint_artifact(
                    locator_path,
                    logical_name="native_locator",
                    schema_version="2.0.0",
                ),
                "preflight_gate": fingerprint_artifact(
                    preflight_report_path,
                    logical_name="preflight_gate",
                    schema_version="2.0.0",
                ),
            }
            self.status.mark_completed(
                "native_extraction_build", paths.base_run_id, outputs
            )
            return StageOutcome(
                "native_extraction_build",
                paths.base_run_id,
                False,
                {**locator, "indexing": indexing_result},
            )
        except Exception as error:
            record = self.status.get("native_extraction_build", paths.base_run_id)
            if record is not None and record.status.value == "running":
                self.status.mark_failed(
                    "native_extraction_build", paths.base_run_id, error
                )
            raise

    def _load_er_artifacts(self, paths: ConditionPaths) -> dict[str, Any]:
        result: dict[str, Any] = {}
        header: dict[str, Any] | None = None
        for name in (
            "entity_profiles",
            "candidate_pairs",
            "pair_scores",
            "pair_decisions",
            "cannot_links",
            "clusters",
            "canonical_entities",
            "mention_to_canonical",
            "aliases",
        ):
            current_header, records = read_enveloped_jsonl(
                paths.er_artifacts / f"{name}.jsonl"
            )
            if header is None:
                header = current_header
            elif current_header.get("lineage") != header.get("lineage"):
                raise RuntimeError("ER artifacts have inconsistent lineage")
            result[name] = records
        for name in ("merge_plan", "summary"):
            envelope = read_json(paths.er_artifacts / f"{name}.json")
            if envelope.get("lineage") != (header or {}).get("lineage"):
                raise RuntimeError("ER JSON artifact lineage mismatch")
            payload = envelope.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"invalid ER {name} payload")
            result[name] = payload
        result["lineage"] = (header or {}).get("lineage")
        return result

    def _judge(self) -> tuple[OllamaERJudge, JudgeCache]:
        import ollama

        judge = self.config.roles.er_judge
        if judge is None:
            raise ValueError("roles.er_judge must be selected")
        tag = _required(judge.resolved_name, "ER judge resolved name")
        digest = _required(judge.digest, "ER judge digest")
        client = ollama.Client(
            host=self.config.runtime.ollama_host,
            timeout=self.config.runtime.model_timeout_seconds,
        )
        adapter = OllamaERJudge(
            client=client,
            model_tag=tag,
            model_digest=digest,
            prompt_version=judge.prompt_version,
            temperature=judge.temperature,
            seed=judge.seed,
            options={
                "num_ctx": judge.context_window,
                "num_predict": judge.output_tokens,
            },
        )
        return adapter, JudgeCache(self.runs_root / "judge_cache")

    def _rr_verifier(self) -> OllamaRRVerifier:
        import ollama

        verifier = self.config.roles.rr_verifier
        if verifier is None:
            raise ValueError("roles.rr_verifier must be selected")
        return OllamaRRVerifier(
            client=ollama.Client(
                host=self.config.runtime.ollama_host,
                timeout=self.config.runtime.model_timeout_seconds,
            ),
            model_tag=_required(verifier.resolved_name, "RR verifier resolved name"),
            model_digest=_required(verifier.digest, "RR verifier digest"),
            prompt_version=verifier.prompt_version,
            temperature=verifier.temperature,
            seed=verifier.seed,
            options={
                "num_ctx": verifier.context_window,
                "num_predict": verifier.output_tokens,
            },
        )

    async def er_plan(
        self, builder_key: str, *, reclaim_running: bool = False
    ) -> StageOutcome:
        """Run corpus-level ER only after the complete extraction gate passes."""

        self.config.assert_ready(builder_key=builder_key, require_er=True)
        paths = self.condition_paths(builder_key)
        base_manifest = _read_base_manifest(paths.base_manifest)
        locator = read_json(paths.locator("native_lightrag"))
        inputs = {
            "base_manifest": fingerprint_artifact(
                paths.base_manifest,
                logical_name="base_manifest",
                schema_version="2.0.0",
            )
        }
        expectation = self._expectation(
            paths.er_variant_run_id,
            config_sha256=variant_config_sha256(
                self.config,
                base_run_id=paths.base_run_id,
                graph_regime="advanced_lightrag_er",
            ),
            inputs=inputs,
        )
        claim = self.status.claim(
            "er_plan",
            paths.er_variant_run_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        if claim.reused:
            return StageOutcome(
                "er_plan",
                paths.er_variant_run_id,
                True,
                read_json(paths.er_artifacts / "summary.json"),
            )
        if not claim.claimed:
            raise RuntimeError("ER planning stage is already active or not claimable")
        effective_judge_budget, budget_override = self.config.effective_er_judge_budget(
            builder_key
        )
        judge_budget_path = _judge_budget_gate_path(
            paths, attempt_count=claim.record.attempt_count
        )
        budget_override_payload = (
            budget_override.model_dump(mode="json")
            if budget_override is not None
            else None
        )
        try:
            await self._verify_current_models(
                self.config.roles.embedding,
                self.config.roles.er_judge,
            )
            native_run_dir = Path(str(locator["legacy_run_dir"]))
            extraction_gate = self._extraction_gate(
                paths, native_run_dir=native_run_dir, fail_closed=False
            )
            if not extraction_gate.passed:
                self.status.mark_blocked_by_quality_gate(
                    "er_plan",
                    paths.er_variant_run_id,
                    "; ".join(extraction_gate.errors),
                )
                raise RuntimeError("ER is blocked until the complete corpus is staged")
            staging, chunks, mentions, relations = load_staged_snapshot(
                paths.staging_dir
            )
            if not staging.corpus_complete:
                raise RuntimeError("ER cannot run on an incomplete staging manifest")
            native_nodes = read_jsonl(
                native_run_dir / "artifacts" / "graph_nodes.jsonl"
            )
            _validate_native_entity_group_coverage(mentions, native_nodes)
            snapshot = CorpusSnapshot(
                expected_document_ids=tuple(
                    str(row["document_id"])
                    for row in _load_documents(self.loaded.documents_path)
                ),
                completed_document_ids=tuple(
                    sorted({item.document_id for item in chunks})
                ),
                expected_chunk_ids=tuple(item.chunk_id for item in chunks),
                completed_chunk_ids=tuple(item.chunk_id for item in chunks),
            )
            runtime_config = _er_runtime_config(self.config.entity_resolution)
            embedding_role = self.config.roles.embedding
            embedding_identity = OllamaEmbeddingIdentity(
                resolved_tag=_required(
                    embedding_role.resolved_name, "embedding resolved name"
                ),
                resolved_digest=_required(embedding_role.digest, "embedding digest"),
                dimension=int(embedding_role.dimension or 0),
            )
            import ollama

            started = time.perf_counter()
            async with ollama.AsyncClient(
                host=self.config.runtime.ollama_host,
                timeout=self.config.runtime.model_timeout_seconds,
            ) as embedding_client:
                native_entity_records = build_native_entity_embedding_records(mentions)
                augmented_native_entities = (
                    await augment_mentions_with_ollama_embeddings(
                        native_entity_records,
                        client=embedding_client,
                        identity=embedding_identity,
                        cache_dir=self.runs_root / "embedding_cache",
                        batch_size=self.config.runtime.lightrag.index_batch_size,
                    )
                )
            # Embed lineage is derived metadata, not a mutation of immutable
            # staging. Keep cache-hit state out of scientific artifacts.
            for item in augmented_native_entities:
                provenance = dict(item.get("provenance") or {})
                provenance["er_embedding"] = {
                    key: item[key]
                    for key in (
                        "embedding_cache_key",
                        "embedding_dimension",
                        "embedding_model_digest",
                        "embedding_model_tag",
                        "embedding_profile_text_sha256",
                        "embedding_profile_text_version",
                    )
                }
                item["provenance"] = provenance
            configured_candidate_options = (
                self.config.entity_resolution.embedding_candidate_payload()
            )
            candidate_options = {
                "k": configured_candidate_options["embedding_neighbor_k"],
                "tables": configured_candidate_options["embedding_lsh_tables"],
                "bits": configured_candidate_options["embedding_lsh_bits"],
                "max_bucket": configured_candidate_options["embedding_lsh_max_bucket"],
            }
            embedding_neighbors = build_lsh_cosine_neighbor_evidence(
                augmented_native_entities,
                seed=self.config.extraction.seed,
                **candidate_options,
            )
            judge, cache = self._judge()
            result = EntityResolutionPipeline(runtime_config).run(
                snapshot=snapshot,
                mentions=mentions,
                relations=relations,
                lineage=ERLineage(
                    base_run_id=paths.base_run_id,
                    base_extraction_hash=_required(
                        base_manifest.base_extraction_sha256,
                        "base extraction SHA-256",
                    ),
                    corpus_manifest_hash=self.config.corpus.manifest_sha256,
                    input_hashes={
                        **staging.input_hashes,
                        "normalized_chunks": staging.normalized_chunks_sha256,
                        "normalized_entities": staging.normalized_entities_sha256,
                        "normalized_relations": staging.normalized_relations_sha256,
                        "embedding_model_digest": embedding_identity.resolved_digest,
                        "embedding_candidate_config": content_hash(
                            {
                                **candidate_options,
                                "seed": self.config.extraction.seed,
                            }
                        ),
                    },
                    er_version=runtime_config.er_version,
                ),
                artifact_dir=paths.er_artifacts,
                judge=judge,
                judge_cache=cache,
                embedding_neighbors=embedding_neighbors,
                embedded_native_entities=augmented_native_entities,
                operational_max_judge_calls_per_run=effective_judge_budget,
            )
            write_immutable_json(
                judge_budget_path,
                {
                    "schema_version": "1.0.0",
                    "builder_key": builder_key,
                    "base_run_id": paths.base_run_id,
                    "variant_run_id": paths.er_variant_run_id,
                    "physical_er_plan_attempt": claim.record.attempt_count,
                    "operational_override": budget_override_payload,
                    **dict(result.summary["judge_budget"]),
                },
            )
            er_runtime_seconds = time.perf_counter() - started
            judge_decisions = [
                decision for decision in result.decisions if decision.source == "judge"
            ]
            current_judge_calls = sum(
                decision.judge_cache_hit is False for decision in judge_decisions
            )
            current_judge_cache_hits = sum(
                decision.judge_cache_hit is True for decision in judge_decisions
            )

            def judge_tokens(field: str, *, cache_hits: bool | None) -> int:
                return sum(
                    int(decision.judge_metadata.get(field) or 0)
                    for decision in judge_decisions
                    if cache_hits is None or decision.judge_cache_hit is cache_hits
                )

            runtime_report = {
                "schema_version": "1.0.0",
                "base_run_id": paths.base_run_id,
                "variant_run_id": paths.er_variant_run_id,
                "er_runtime_seconds": er_runtime_seconds,
                # Actual calls and tokens charged to this completed attempt.
                "judge_calls": current_judge_calls,
                "judge_cache_hits": current_judge_cache_hits,
                "judge_prompt_tokens": judge_tokens(
                    "prompt_eval_count", cache_hits=False
                ),
                "judge_output_tokens": judge_tokens("eval_count", cache_hits=False),
                "judge_tokens": judge_tokens("prompt_eval_count", cache_hits=False)
                + judge_tokens("eval_count", cache_hits=False),
                # Cached results retain the original provider metadata.  These
                # totals make a resumed plan auditable without pretending that
                # cache hits incurred a second model call or cost.
                "judge_decisions": len(judge_decisions),
                "judge_cached_original_prompt_tokens": judge_tokens(
                    "prompt_eval_count", cache_hits=None
                ),
                "judge_cached_original_output_tokens": judge_tokens(
                    "eval_count", cache_hits=None
                ),
                "judge_cost": 0.0,
                "judge_cost_basis": "local_ollama_no_api_charge",
                "judge_budget": {
                    **dict(result.summary["judge_budget"]),
                    "operational_override": budget_override_payload,
                },
            }
            graph = rewrite_graph(
                mentions,
                relations,
                result.mention_to_canonical,
                result.canonical_entities,
            )
            er_gate = validate_er_quality_gate(
                extraction_report=extraction_gate,
                config=self.config,
                base_extraction_sha256=_required(
                    base_manifest.base_extraction_sha256,
                    "base extraction SHA-256",
                ),
                profiles=result.profiles,
                candidate_pairs=result.candidates,
                pair_scores=result.scores,
                pair_decisions=result.decisions,
                cannot_links=result.cannot_links,
                clusters=result.clusters,
                canonical_entities=result.canonical_entities,
                mention_to_canonical=result.mention_to_canonical,
                lineage=result.lineage,
                judge_identity=judge.identity,
                rewritten_relations=graph.edges,
                expected_er_config_hash=runtime_config.config_hash,
                operational_max_judge_calls_per_run=effective_judge_budget,
                fail_closed=False,
            )
            gate_path = paths.er_dir / "gates" / "er.json"
            write_immutable_json(gate_path, er_gate)
            if not er_gate.passed:
                self.status.mark_blocked_by_quality_gate(
                    "er_plan", paths.er_variant_run_id, "; ".join(er_gate.errors)
                )
                raise RuntimeError("ER plan blocked by ER quality gate")
            runtime_path = paths.er_artifacts / "runtime.json"
            if runtime_path.exists():
                existing_runtime = read_json(runtime_path)
                stable_runtime_fields = (
                    "schema_version",
                    "base_run_id",
                    "variant_run_id",
                )
                if any(
                    existing_runtime.get(field) != runtime_report.get(field)
                    for field in stable_runtime_fields
                ):
                    raise RuntimeError(
                        "existing ER runtime report has incompatible lineage"
                    )
                # A runtime report is written only after a passed ER gate.  If
                # an interruption happened immediately afterwards, preserve
                # that first valid timing instead of making resume conflict on
                # inherently variable elapsed time/cache-hit counters.
                runtime_report = existing_runtime
            else:
                write_immutable_json(runtime_path, runtime_report)
            outputs = {
                filename: fingerprint_artifact(
                    paths.er_artifacts / filename,
                    logical_name=f"er/{filename}",
                    schema_version=ER_ARTIFACT_SCHEMA_VERSION,
                )
                for filename in result.artifact_hashes
            }
            outputs["er_gate"] = fingerprint_artifact(
                gate_path, logical_name="er_gate", schema_version="2.0.0"
            )
            outputs["judge_budget_gate"] = fingerprint_artifact(
                judge_budget_path,
                logical_name="er/judge_budget_gate",
                schema_version="1.0.0",
            )
            outputs["er_runtime"] = fingerprint_artifact(
                runtime_path,
                logical_name="er/runtime.json",
                schema_version="1.0.0",
            )
            self.status.mark_completed("er_plan", paths.er_variant_run_id, outputs)
            return StageOutcome(
                "er_plan",
                paths.er_variant_run_id,
                False,
                {"summary": dict(result.summary), "gate": jsonable(er_gate)},
            )
        except JudgeBudgetExceededError as error:
            write_immutable_json(
                judge_budget_path,
                {
                    "schema_version": "1.0.0",
                    "builder_key": builder_key,
                    "base_run_id": paths.base_run_id,
                    "variant_run_id": paths.er_variant_run_id,
                    "physical_er_plan_attempt": claim.record.attempt_count,
                    "operational_override": budget_override_payload,
                    **error.report,
                },
            )
            self.status.mark_blocked_by_quality_gate(
                "er_plan", paths.er_variant_run_id, str(error)
            )
            raise
        except Exception as error:
            record = self.status.get("er_plan", paths.er_variant_run_id)
            if record is not None and record.status.value == "running":
                self.status.mark_failed("er_plan", paths.er_variant_run_id, error)
            raise

    def er_quality_gate(self, builder_key: str) -> QualityGateReport:
        """Revalidate saved ER artifacts without invoking the judge."""

        paths = self.condition_paths(builder_key)
        base_manifest = _read_base_manifest(paths.base_manifest)
        locator = read_json(paths.locator("native_lightrag"))
        extraction_gate = self._extraction_gate(
            paths,
            native_run_dir=Path(str(locator["legacy_run_dir"])),
            fail_closed=False,
        )
        er = self._load_er_artifacts(paths)
        _, _, mentions, relations = load_staged_snapshot(paths.staging_dir)
        graph = rewrite_graph(
            mentions,
            relations,
            er["mention_to_canonical"],
            er["canonical_entities"],
        )
        judge, _ = self._judge()
        runtime_config = _er_runtime_config(self.config.entity_resolution)
        effective_judge_budget, _ = self.config.effective_er_judge_budget(builder_key)
        return validate_er_quality_gate(
            extraction_report=extraction_gate,
            config=self.config,
            base_extraction_sha256=_required(
                base_manifest.base_extraction_sha256, "base extraction SHA-256"
            ),
            profiles=er["entity_profiles"],
            candidate_pairs=er["candidate_pairs"],
            pair_scores=er["pair_scores"],
            pair_decisions=er["pair_decisions"],
            cannot_links=er["cannot_links"],
            clusters=er["clusters"],
            canonical_entities=er["canonical_entities"],
            mention_to_canonical=er["mention_to_canonical"],
            lineage=er["lineage"],
            judge_identity=judge.identity,
            rewritten_relations=graph.edges,
            expected_er_config_hash=runtime_config.config_hash,
            operational_max_judge_calls_per_run=effective_judge_budget,
            fail_closed=True,
        )

    def _load_rr_plan(
        self, paths: ConditionPaths
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return (
            read_jsonl(paths.rr_artifacts / "chunk_plan.jsonl"),
            read_json(paths.rr_artifacts / "plan_summary.json"),
        )

    def _er_rewritten_graph(self, paths: ConditionPaths) -> tuple[Any, list[Any]]:
        _, chunks, mentions, relations = load_staged_snapshot(paths.staging_dir)
        er = self._load_er_artifacts(paths)
        return (
            rewrite_graph(
                mentions,
                relations,
                er["mention_to_canonical"],
                er["canonical_entities"],
            ),
            chunks,
        )

    async def rr_plan(
        self, builder_key: str, *, reclaim_running: bool = False
    ) -> StageOutcome:
        """Freeze all same-chunk missing ER pairs without any model call."""

        self.config.assert_ready(
            builder_key=builder_key, require_er=True, require_rr=True
        )
        paths = self.condition_paths(builder_key)
        self.er_quality_gate(builder_key)
        merge_ref = fingerprint_artifact(
            paths.er_artifacts / "merge_plan.json",
            logical_name="er/merge_plan.json",
            schema_version="1.0.0",
        )
        inputs = {
            "merge_plan": merge_ref,
            "normalized_chunks": fingerprint_artifact(
                paths.staging_dir / "normalized_chunks.jsonl",
                logical_name="extraction/normalized_chunks.jsonl",
                schema_version="3.0.0",
            ),
            "mention_to_canonical": fingerprint_artifact(
                paths.er_artifacts / "mention_to_canonical.jsonl",
                logical_name="er/mention_to_canonical.jsonl",
                schema_version="1.0.0",
            ),
            "canonical_entities": fingerprint_artifact(
                paths.er_artifacts / "canonical_entities.jsonl",
                logical_name="er/canonical_entities.jsonl",
                schema_version="1.0.0",
            ),
        }
        expectation = self._expectation(
            paths.rr_variant_run_id,
            config_sha256=variant_config_sha256(
                self.config,
                base_run_id=paths.base_run_id,
                graph_regime="advanced_lightrag_er_rr",
            ),
            inputs=inputs,
        )
        claim = self.status.claim(
            "rr_plan",
            paths.rr_variant_run_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        if claim.reused:
            return StageOutcome(
                "rr_plan",
                paths.rr_variant_run_id,
                True,
                read_json(paths.rr_artifacts / "plan_summary.json"),
            )
        if not claim.claimed:
            raise RuntimeError("RR planning stage is already active or not claimable")
        try:
            graph, chunks = self._er_rewritten_graph(paths)
            rows, summary = build_rr_plan(
                chunks=chunks,
                nodes=graph.nodes,
                edges=graph.edges,
                mention_to_canonical=graph.mention_to_canonical,
                candidate_policy_version=(
                    self.config.relation_recovery.candidate_policy_version
                ),
            )
            verifier = self.config.roles.rr_verifier
            assert verifier is not None
            prompt_budget = verifier.context_window - verifier.output_tokens
            maximum_estimated_prompt = int(
                summary["prompt_estimated_tokens_chars_div_4"]["maximum"]
            )
            if maximum_estimated_prompt > prompt_budget:
                raise RuntimeError(
                    "RR prompt estimate exceeds the frozen context budget: "
                    f"maximum={maximum_estimated_prompt}, budget={prompt_budget}"
                )
            summary = {
                **summary,
                "builder_key": builder_key,
                "base_run_id": paths.base_run_id,
                "er_variant_run_id": paths.er_variant_run_id,
                "rr_variant_run_id": paths.rr_variant_run_id,
                "merge_plan_sha256": merge_ref.sha256,
                "rr_config_sha256": sha256_json(self.config.relation_recovery),
                "rr_verifier_config_sha256": sha256_json(verifier.identity_payload()),
                "context_budget": {
                    "context_window": verifier.context_window,
                    "reserved_output_tokens": verifier.output_tokens,
                    "maximum_estimated_prompt_tokens_chars_div_4": (
                        maximum_estimated_prompt
                    ),
                    "estimated_prompt_budget_tokens": prompt_budget,
                    "passed": True,
                },
                "input_graph_sha256": sha256_json(
                    {
                        "nodes": graph.nodes,
                        "edges": graph.edges,
                        "mention_to_canonical": graph.mention_to_canonical,
                    }
                ),
            }
            plan_path = paths.rr_artifacts / "chunk_plan.jsonl"
            summary_path = paths.rr_artifacts / "plan_summary.json"
            write_immutable_jsonl(plan_path, rows)
            write_immutable_json(summary_path, summary)
            outputs = {
                "chunk_plan": fingerprint_artifact(
                    plan_path,
                    logical_name="rr/chunk_plan.jsonl",
                    schema_version="1.0.0",
                ),
                "plan_summary": fingerprint_artifact(
                    summary_path,
                    logical_name="rr/plan_summary.json",
                    schema_version="1.0.0",
                ),
            }
            self.status.mark_completed("rr_plan", paths.rr_variant_run_id, outputs)
            return StageOutcome("rr_plan", paths.rr_variant_run_id, False, summary)
        except Exception as error:
            record = self.status.get("rr_plan", paths.rr_variant_run_id)
            if record is not None and record.status.value == "running":
                self.status.mark_failed("rr_plan", paths.rr_variant_run_id, error)
            raise

    async def rr_verify(
        self, builder_key: str, *, reclaim_running: bool = False
    ) -> StageOutcome:
        """Verify each eligible chunk once, caching raw responses immutably."""

        self.config.assert_ready(
            builder_key=builder_key, require_er=True, require_rr=True
        )
        paths = self.condition_paths(builder_key)
        plan_rows, plan_summary = self._load_rr_plan(paths)
        plan_ref = fingerprint_artifact(
            paths.rr_artifacts / "chunk_plan.jsonl",
            logical_name="rr/chunk_plan.jsonl",
            schema_version="1.0.0",
        )
        expectation = self._expectation(
            paths.rr_variant_run_id,
            config_sha256=variant_config_sha256(
                self.config,
                base_run_id=paths.base_run_id,
                graph_regime="advanced_lightrag_er_rr",
            ),
            inputs={"chunk_plan": plan_ref},
        )
        claim = self.status.claim(
            "rr_verification",
            paths.rr_variant_run_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        summary_path = paths.rr_artifacts / "verification_summary.json"
        if claim.reused:
            return StageOutcome(
                "rr_verification",
                paths.rr_variant_run_id,
                True,
                read_json(summary_path),
            )
        if not claim.claimed:
            raise RuntimeError("RR verification is already active or not claimable")
        try:
            verifier_role = self.config.roles.rr_verifier
            assert verifier_role is not None
            await self._verify_current_models(verifier_role)
            verifier = self._rr_verifier()
            graph, chunks = self._er_rewritten_graph(paths)
            expected_rows, expected_summary = build_rr_plan(
                chunks=chunks,
                nodes=graph.nodes,
                edges=graph.edges,
                mention_to_canonical=graph.mention_to_canonical,
                candidate_policy_version=(
                    self.config.relation_recovery.candidate_policy_version
                ),
            )
            if plan_rows != expected_rows or plan_summary.get(
                "rr_plan_sha256"
            ) != expected_summary.get("rr_plan_sha256"):
                raise RuntimeError("saved RR plan no longer matches frozen ER inputs")

            chunk_by_id = {str(chunk.chunk_id): chunk for chunk in chunks}
            frozen_pairs = graph_edge_pairs(graph.edges)
            cache_dir = paths.rr_artifacts / "chunk_results"
            marker_dir = paths.rr_artifacts / "call_markers"
            failure_dir = paths.rr_artifacts / "call_failures"
            results: list[dict[str, Any]] = []
            new_calls = 0
            cache_hits = 0
            new_request_shas: set[str] = set()
            started = time.perf_counter()
            for plan_row in plan_rows:
                if not plan_row.get("eligible"):
                    continue
                chunk_id = str(plan_row["chunk_id"])
                chunk = chunk_by_id[chunk_id]
                request_sha = verifier.request_sha256(
                    chunk_text=chunk.text, plan_row=plan_row
                )
                cache_key = sha256_json([chunk_id, request_sha])[:32]
                result_path = cache_dir / f"{cache_key}.json"
                marker_path = marker_dir / f"{cache_key}.json"
                if result_path.is_file():
                    cached = read_json(result_path)
                    if (
                        cached.get("chunk_id") != chunk_id
                        or cached.get("request_sha256") != request_sha
                        or cached.get("candidate_set_sha256")
                        != plan_row.get("candidate_set_sha256")
                    ):
                        raise RuntimeError(f"RR cache identity mismatch for {chunk_id}")
                    recomputed = validate_verifier_response(
                        raw_content=str(cached.get("raw_content") or ""),
                        chunk_text=chunk.text,
                        plan_row=plan_row,
                        frozen_er_pairs=frozen_pairs,
                        verifier_identity=verifier.identity,
                        request_sha256=request_sha,
                        provider_metadata=cached.get("provider_metadata") or {},
                    )
                    if cached.get("verification") != recomputed:
                        raise RuntimeError(f"RR cached validation drift for {chunk_id}")
                    results.append(cached)
                    cache_hits += 1
                    continue
                if marker_path.exists():
                    raise RuntimeError(
                        "RR refuses a second physical call after an interrupted/failed "
                        f"attempt for chunk {chunk_id}; inspect {marker_path}"
                    )
                write_immutable_json(
                    marker_path,
                    {
                        "schema_version": "1.0.0",
                        "chunk_id": chunk_id,
                        "request_sha256": request_sha,
                        "candidate_set_sha256": plan_row["candidate_set_sha256"],
                        "physical_call_limit": 1,
                    },
                )
                try:
                    raw = verifier.verify(chunk_text=chunk.text, plan_row=plan_row)
                except Exception as error:
                    write_immutable_json(
                        failure_dir / f"{cache_key}.json",
                        {
                            "schema_version": "1.0.0",
                            "chunk_id": chunk_id,
                            "request_sha256": request_sha,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "retry_permitted": False,
                        },
                    )
                    raise
                if raw["request_sha256"] != request_sha:
                    raise RuntimeError(f"RR request identity drift for {chunk_id}")
                verification = validate_verifier_response(
                    raw_content=raw["raw_content"],
                    chunk_text=chunk.text,
                    plan_row=plan_row,
                    frozen_er_pairs=frozen_pairs,
                    verifier_identity=verifier.identity,
                    request_sha256=request_sha,
                    provider_metadata=raw["provider_metadata"],
                )
                cached = {
                    "schema_version": "1.0.0",
                    "record_type": "rr_chunk_result",
                    "document_id": plan_row["document_id"],
                    "chunk_id": chunk_id,
                    "chunk_sha256": plan_row["chunk_sha256"],
                    "candidate_set_sha256": plan_row["candidate_set_sha256"],
                    "request_sha256": request_sha,
                    "raw_content": raw["raw_content"],
                    "provider_metadata": raw["provider_metadata"],
                    "verification": verification,
                }
                write_immutable_json(result_path, cached)
                results.append(cached)
                new_calls += 1
                new_request_shas.add(request_sha)

            results.sort(key=lambda row: str(row["chunk_id"]))
            accepted = [
                relation
                for result in results
                for relation in result["verification"]["accepted_relations"]
            ]
            result_rows_path = paths.rr_artifacts / "verification_results.jsonl"
            accepted_path = paths.rr_artifacts / "accepted_relations.jsonl"
            write_immutable_jsonl(result_rows_path, results)
            write_immutable_jsonl(accepted_path, accepted)
            gate = validate_rr_quality_gate(
                config=self.config,
                chunks=chunks,
                er_nodes=graph.nodes,
                er_edges=graph.edges,
                mention_to_canonical=graph.mention_to_canonical,
                plan_rows=plan_rows,
                plan_summary=plan_summary,
                chunk_results=results,
                fail_closed=False,
            )
            gate_path = paths.rr_dir / "gates" / "rr.json"
            write_immutable_json(gate_path, gate)
            if not gate.passed:
                self.status.mark_blocked_by_quality_gate(
                    "rr_verification",
                    paths.rr_variant_run_id,
                    "; ".join(gate.errors),
                )
                raise RuntimeError("RR verification blocked by quality gate")
            summary = {
                "schema_version": "1.0.0",
                "builder_key": builder_key,
                "base_run_id": paths.base_run_id,
                "er_variant_run_id": paths.er_variant_run_id,
                "rr_variant_run_id": paths.rr_variant_run_id,
                "rr_plan_sha256": plan_summary["rr_plan_sha256"],
                "eligible_chunks": len(results),
                "new_physical_calls": new_calls,
                "cached_completed_chunks": cache_hits,
                "valid_chunks": sum(
                    row["verification"]["status"] == "valid" for row in results
                ),
                "invalid_chunks_rejected": sum(
                    row["verification"]["status"] == "invalid" for row in results
                ),
                "accepted_candidate_instances": len(accepted),
                "recovered_pairs": len(
                    {
                        tuple(sorted((row["entity_a_id"], row["entity_b_id"])))
                        for row in accepted
                    }
                ),
                "runtime_seconds": time.perf_counter() - started,
                "prompt_tokens_this_attempt": sum(
                    int(row["provider_metadata"].get("prompt_eval_count") or 0)
                    for row in results
                    if row["request_sha256"] in new_request_shas
                ),
                "output_tokens_this_attempt": sum(
                    int(row["provider_metadata"].get("eval_count") or 0)
                    for row in results
                    if row["request_sha256"] in new_request_shas
                ),
                "cached_original_prompt_tokens_all_completed": sum(
                    int(row["provider_metadata"].get("prompt_eval_count") or 0)
                    for row in results
                ),
                "cached_original_output_tokens_all_completed": sum(
                    int(row["provider_metadata"].get("eval_count") or 0)
                    for row in results
                ),
            }
            write_immutable_json(summary_path, summary)
            outputs = {
                "verification_results": fingerprint_artifact(
                    result_rows_path,
                    logical_name="rr/verification_results.jsonl",
                    schema_version="1.0.0",
                ),
                "accepted_relations": fingerprint_artifact(
                    accepted_path,
                    logical_name="rr/accepted_relations.jsonl",
                    schema_version="1.0.0",
                ),
                "verification_summary": fingerprint_artifact(
                    summary_path,
                    logical_name="rr/verification_summary.json",
                    schema_version="1.0.0",
                ),
                "rr_gate": fingerprint_artifact(
                    gate_path,
                    logical_name="rr/quality_gate.json",
                    schema_version="2.0.0",
                ),
            }
            self.status.mark_completed(
                "rr_verification", paths.rr_variant_run_id, outputs
            )
            return StageOutcome(
                "rr_verification", paths.rr_variant_run_id, False, summary
            )
        except Exception as error:
            record = self.status.get("rr_verification", paths.rr_variant_run_id)
            if record is not None and record.status.value == "running":
                self.status.mark_failed(
                    "rr_verification", paths.rr_variant_run_id, error
                )
            raise

    def rr_quality_gate(self, builder_key: str) -> QualityGateReport:
        """Revalidate frozen plan/results without calling the verifier."""

        paths = self.condition_paths(builder_key)
        plan_rows, plan_summary = self._load_rr_plan(paths)
        graph, chunks = self._er_rewritten_graph(paths)
        results = read_jsonl(paths.rr_artifacts / "verification_results.jsonl")
        rr_nodes_path = paths.rr_graph_artifacts / "rewritten_nodes.jsonl"
        rr_edges_path = paths.rr_graph_artifacts / "rewritten_edges.jsonl"
        return validate_rr_quality_gate(
            config=self.config,
            chunks=chunks,
            er_nodes=graph.nodes,
            er_edges=graph.edges,
            mention_to_canonical=graph.mention_to_canonical,
            plan_rows=plan_rows,
            plan_summary=plan_summary,
            chunk_results=results,
            rr_nodes=read_jsonl(rr_nodes_path) if rr_nodes_path.is_file() else None,
            rr_edges=read_jsonl(rr_edges_path) if rr_edges_path.is_file() else None,
            fail_closed=True,
        )

    async def er_materialize(
        self, builder_key: str, *, reclaim_running: bool = False
    ) -> StageOutcome:
        """Build the ER graph in a fresh physical workspace from staging only."""

        self.config.assert_ready(builder_key=builder_key, require_er=True)
        paths = self.condition_paths(builder_key)
        self.er_quality_gate(builder_key)
        base_manifest = _read_base_manifest(paths.base_manifest)
        base_hash = _required(
            base_manifest.base_extraction_sha256, "base extraction SHA-256"
        )
        merge_ref = fingerprint_artifact(
            paths.er_artifacts / "merge_plan.json",
            logical_name="er/merge_plan.json",
            schema_version="1.0.0",
        )
        expectation = self._expectation(
            paths.er_variant_run_id,
            config_sha256=variant_config_sha256(
                self.config,
                base_run_id=paths.base_run_id,
                graph_regime="advanced_lightrag_er",
            ),
            inputs={
                "base_manifest": fingerprint_artifact(
                    paths.base_manifest,
                    logical_name="base_manifest",
                    schema_version="2.0.0",
                ),
                "merge_plan": merge_ref,
            },
        )
        claim = self.status.claim(
            "er_materialization",
            paths.er_variant_run_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        if claim.reused:
            return StageOutcome(
                "er_materialization",
                paths.er_variant_run_id,
                True,
                read_json(paths.locator("advanced_lightrag_er")),
            )
        if not claim.claimed:
            raise RuntimeError("ER materialization is already active or not claimable")
        try:
            published = _published_er_materialization(
                paths,
                merge_plan_sha256=merge_ref.sha256,
            )
            if published is not None:
                locator, outputs = published
                self.status.mark_completed(
                    "er_materialization", paths.er_variant_run_id, outputs
                )
                return StageOutcome(
                    "er_materialization",
                    paths.er_variant_run_id,
                    True,
                    {**locator, "recovered_after_crash": True},
                )
            await self._verify_current_models(
                self.config.builders_by_key[builder_key],
                self.config.roles.query,
                self.config.roles.answer,
                self.config.roles.embedding,
                self.config.roles.er_judge,
            )
            _, chunks, mentions, relations = load_staged_snapshot(paths.staging_dir)
            er = self._load_er_artifacts(paths)
            graph = rewrite_graph(
                mentions,
                relations,
                er["mention_to_canonical"],
                er["canonical_entities"],
            )
            paths.graph_artifacts.mkdir(parents=True, exist_ok=True)
            rewrite_paths = {
                "rewritten_nodes": paths.graph_artifacts / "rewritten_nodes.jsonl",
                "rewritten_edges": paths.graph_artifacts / "rewritten_edges.jsonl",
                "rewritten_mapping": paths.graph_artifacts
                / "mention_to_canonical.jsonl",
                "induced_self_loops": paths.graph_artifacts
                / "induced_self_loops.jsonl",
                "preexisting_self_loops": paths.graph_artifacts
                / "preexisting_self_loops.jsonl",
            }
            write_immutable_jsonl(rewrite_paths["rewritten_nodes"], graph.nodes)
            write_immutable_jsonl(rewrite_paths["rewritten_edges"], graph.edges)
            write_immutable_jsonl(
                rewrite_paths["rewritten_mapping"], graph.mention_to_canonical
            )
            write_immutable_jsonl(
                rewrite_paths["induced_self_loops"], graph.induced_self_loops
            )
            write_immutable_jsonl(
                rewrite_paths["preexisting_self_loops"], graph.preexisting_self_loops
            )
            rewrite_summary_path = paths.graph_artifacts / "rewrite_summary.json"
            write_immutable_json(rewrite_summary_path, graph.summary)

            native_locator = read_json(paths.locator("native_lightrag"))
            native_workspace = Path(str(native_locator["workspace_dir"]))
            native_before = workspace_tree_sha256(native_workspace)
            frozen = await prepare_lightrag_run(
                self.loaded,
                builder_key=builder_key,
                graph_regime="advanced_lightrag_er",
                base_run_id=paths.base_run_id,
                variant_run_id=paths.er_variant_run_id,
                er_lineage=dict(er["lineage"] or {}),
                physical_attempt=claim.record.attempt_count,
            )
            from lightrag import QueryParam
            from src.extraction.native_capture import build_lightrag

            def rag_factory() -> Any:
                return build_lightrag(frozen, extraction_capture=False)

            questions = _smoke_questions(_load_questions(self.loaded.questions_path))
            outcome = await materialize_finalize_reopen(
                rag_factory,
                documents=_load_documents(self.loaded.documents_path),
                chunks=chunks,
                graph=graph,
                track_id=(
                    f"er-{paths.er_variant_run_id[-20:]}-"
                    f"{claim.record.attempt_count:04d}"
                ),
                smoke_cases=[
                    {
                        "question": row["question"],
                        "expected_status": "success",
                    }
                    for row in questions
                ],
                query_param_factory=lambda mode: QueryParam(
                    mode=mode,
                    only_need_context=True,
                    stream=False,
                    **_query_parameters(self.config.runtime.retrieval),
                ),
            )
            native_after = workspace_tree_sha256(native_workspace)
            snapshot = outcome.reopened_snapshot
            workspace_paths = {
                "nodes": paths.graph_artifacts / "nodes.jsonl",
                "edges": paths.graph_artifacts / "edges.jsonl",
                "chunks": paths.graph_artifacts / "chunks.jsonl",
            }
            lineage = {
                "base_run_id": paths.base_run_id,
                "variant_run_id": paths.er_variant_run_id,
                "base_extraction_sha256": base_hash,
                "graph_regime": "advanced_lightrag_er",
            }
            write_immutable_jsonl(
                workspace_paths["nodes"],
                ({**lineage, **row} for row in snapshot.nodes),
            )
            write_immutable_jsonl(
                workspace_paths["edges"],
                ({**lineage, **row} for row in snapshot.edges),
            )
            write_immutable_jsonl(
                workspace_paths["chunks"],
                ({**lineage, **row} for row in snapshot.chunks),
            )
            aliases_path = paths.graph_artifacts / "aliases_audit.jsonl"
            placeholders_path = paths.graph_artifacts / "description_placeholders.jsonl"
            parity_path = paths.graph_artifacts / "parity_report.json"
            materialization_path = paths.graph_artifacts / "materialization_report.json"
            write_immutable_jsonl(aliases_path, outcome.stage.aliases_audit)
            write_immutable_jsonl(
                placeholders_path, outcome.stage.description_placeholders
            )
            write_immutable_json(parity_path, outcome.parity_report)
            write_immutable_json(materialization_path, outcome.to_dict())
            final_gate = validate_final_workspace_quality_gate(
                graph_regime="advanced_lightrag_er",
                base_extraction_sha256=base_hash,
                expected_base_extraction_sha256=base_hash,
                nodes=snapshot.nodes,
                edges=snapshot.edges,
                chunks=snapshot.chunks,
                expected_chunk_ids=[chunk.chunk_id for chunk in chunks],
                induced_self_loops=graph.induced_self_loops,
                preexisting_self_loops=graph.preexisting_self_loops,
                workspace_clean=True,
                persisted=True,
                finalized=True,
                reopened=True,
                export_succeeded=True,
                smoke_retrieval_answerable=bool(outcome.smoke_results[0]["passed"]),
                smoke_retrieval_unanswerable=bool(outcome.smoke_results[1]["passed"]),
                native_workspace_sha256_before=native_before,
                native_workspace_sha256_after=native_after,
                fail_closed=False,
            )
            gate_path = paths.er_dir / "gates" / "final_workspace.json"
            write_immutable_json(gate_path, final_gate)
            if not final_gate.passed:
                self.status.mark_blocked_by_quality_gate(
                    "er_materialization",
                    paths.er_variant_run_id,
                    "; ".join(final_gate.errors),
                )
                raise RuntimeError("ER workspace blocked by final workspace gate")
            locator = {
                "schema_version": "2.0.0",
                **lineage,
                "builder_key": builder_key,
                "legacy_run_id": frozen.lock.run_id,
                "legacy_run_dir": str(frozen.paths.run_dir.resolve()),
                "workspace_dir": str(frozen.paths.workspace_dir.resolve()),
                "workspace_sha256": workspace_tree_sha256(frozen.paths.workspace_dir),
                "physical_materialization_attempt": claim.record.attempt_count,
                "native_workspace_sha256_before": native_before,
                "native_workspace_sha256_after": native_after,
                "merge_plan_sha256": merge_ref.sha256,
            }
            locator_path = paths.locator("advanced_lightrag_er")
            write_immutable_json(locator_path, locator)
            variant_artifacts = {
                name: fingerprint_artifact(
                    path,
                    logical_name=f"graph/{name}",
                    schema_version="2.0.0",
                )
                for name, path in {
                    **rewrite_paths,
                    **workspace_paths,
                    "rewrite_summary": rewrite_summary_path,
                    "aliases_audit": aliases_path,
                    "description_placeholders": placeholders_path,
                    "parity_report": parity_path,
                    "materialization_report": materialization_path,
                    "final_workspace_gate": gate_path,
                }.items()
            }
            advanced_manifest = build_variant_manifest(
                self.config,
                base_manifest=base_manifest,
                graph_regime="advanced_lightrag_er",
                merge_plan=merge_ref,
                artifacts=variant_artifacts,
            )
            manifest_path = paths.er_dir / "variant_manifest.json"
            freeze_manifest(manifest_path, advanced_manifest)
            outputs = {
                "variant_manifest": fingerprint_artifact(
                    manifest_path,
                    logical_name="advanced_variant_manifest",
                    schema_version="2.0.0",
                ),
                "workspace_locator": fingerprint_artifact(
                    locator_path,
                    logical_name="advanced_workspace_locator",
                    schema_version="2.0.0",
                ),
                "final_workspace_gate": variant_artifacts["final_workspace_gate"],
            }
            self.status.mark_completed(
                "er_materialization", paths.er_variant_run_id, outputs
            )
            return StageOutcome(
                "er_materialization", paths.er_variant_run_id, False, locator
            )
        except Exception as error:
            record = self.status.get("er_materialization", paths.er_variant_run_id)
            if record is not None and record.status.value == "running":
                self.status.mark_failed(
                    "er_materialization", paths.er_variant_run_id, error
                )
            raise

    async def rr_materialize(
        self, builder_key: str, *, reclaim_running: bool = False
    ) -> StageOutcome:
        """Build ER+RR in a fresh workspace while preserving Native and ER."""

        self.config.assert_ready(
            builder_key=builder_key, require_er=True, require_rr=True
        )
        paths = self.condition_paths(builder_key)
        self.rr_quality_gate(builder_key)
        base_manifest = _read_base_manifest(paths.base_manifest)
        base_hash = _required(
            base_manifest.base_extraction_sha256, "base extraction SHA-256"
        )
        merge_ref = fingerprint_artifact(
            paths.er_artifacts / "merge_plan.json",
            logical_name="er/merge_plan.json",
            schema_version="1.0.0",
        )
        plan_ref = fingerprint_artifact(
            paths.rr_artifacts / "chunk_plan.jsonl",
            logical_name="rr/chunk_plan.jsonl",
            schema_version="1.0.0",
        )
        results_ref = fingerprint_artifact(
            paths.rr_artifacts / "verification_results.jsonl",
            logical_name="rr/verification_results.jsonl",
            schema_version="1.0.0",
        )
        expectation = self._expectation(
            paths.rr_variant_run_id,
            config_sha256=variant_config_sha256(
                self.config,
                base_run_id=paths.base_run_id,
                graph_regime="advanced_lightrag_er_rr",
            ),
            inputs={
                "base_manifest": fingerprint_artifact(
                    paths.base_manifest,
                    logical_name="base_manifest",
                    schema_version="2.0.0",
                ),
                "merge_plan": merge_ref,
                "rr_plan": plan_ref,
                "rr_results": results_ref,
            },
        )
        claim = self.status.claim(
            "rr_materialization",
            paths.rr_variant_run_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        locator_path = paths.locator("advanced_lightrag_er_rr")
        if claim.reused:
            return StageOutcome(
                "rr_materialization",
                paths.rr_variant_run_id,
                True,
                read_json(locator_path),
            )
        if not claim.claimed:
            raise RuntimeError("RR materialization is already active or not claimable")
        try:
            # Recover a fully published immutable workspace if the process died
            # after publication but before the mutable status record was closed.
            manifest_path = paths.rr_dir / "variant_manifest.json"
            final_gate_path = paths.rr_dir / "gates" / "final_workspace.json"
            if (
                locator_path.is_file()
                and manifest_path.is_file()
                and final_gate_path.is_file()
            ):
                locator = read_json(locator_path)
                published_manifest = VariantRunManifest.model_validate(
                    read_json(manifest_path)
                )
                published_gate = QualityGateReport.model_validate(
                    read_json(final_gate_path)
                )
                if (
                    not published_gate.passed
                    or published_manifest.graph_regime != "advanced_lightrag_er_rr"
                    or published_manifest.base_run_id != paths.base_run_id
                    or published_manifest.variant_run_id != paths.rr_variant_run_id
                    or locator.get("base_run_id") != paths.base_run_id
                    or locator.get("variant_run_id") != paths.rr_variant_run_id
                    or locator.get("graph_regime") != "advanced_lightrag_er_rr"
                    or locator.get("rr_plan_sha256") != plan_ref.sha256
                    or locator.get("rr_results_sha256") != results_ref.sha256
                    or published_manifest.rr_plan_sha256 != plan_ref.sha256
                    or published_manifest.rr_results_sha256 != results_ref.sha256
                ):
                    raise RuntimeError("published RR workspace has different inputs")
                if locator.get("native_workspace_sha256_before") != locator.get(
                    "native_workspace_sha256_after"
                ) or locator.get("er_workspace_sha256_before") != locator.get(
                    "er_workspace_sha256_after"
                ):
                    raise RuntimeError(
                        "published RR workspace changed a parent workspace"
                    )
                published_workspace = Path(str(locator.get("workspace_dir") or ""))
                if not published_workspace.is_dir() or workspace_tree_sha256(
                    published_workspace
                ) != locator.get("workspace_sha256"):
                    raise RuntimeError("published RR workspace hash is invalid")
                artifact_errors: list[str] = []
                for name, ref in published_manifest.artifacts.items():
                    valid, reasons = verify_artifact(ref)
                    if not valid:
                        artifact_errors.extend(
                            f"{name}: {reason}" for reason in reasons
                        )
                if artifact_errors:
                    raise RuntimeError(
                        "published RR artifacts are invalid: "
                        + "; ".join(artifact_errors)
                    )
                outputs = {
                    "variant_manifest": fingerprint_artifact(
                        manifest_path,
                        logical_name="rr_variant_manifest",
                        schema_version="2.0.0",
                    ),
                    "workspace_locator": fingerprint_artifact(
                        locator_path,
                        logical_name="rr_workspace_locator",
                        schema_version="2.0.0",
                    ),
                    "final_workspace_gate": fingerprint_artifact(
                        final_gate_path,
                        logical_name="graph/final_workspace_gate",
                        schema_version="2.0.0",
                    ),
                }
                self.status.mark_completed(
                    "rr_materialization", paths.rr_variant_run_id, outputs
                )
                return StageOutcome(
                    "rr_materialization",
                    paths.rr_variant_run_id,
                    True,
                    {**locator, "recovered_after_crash": True},
                )

            await self._verify_current_models(
                self.config.builders_by_key[builder_key],
                self.config.roles.query,
                self.config.roles.answer,
                self.config.roles.embedding,
                self.config.roles.er_judge,
                self.config.roles.rr_verifier,
            )
            er_graph, chunks = self._er_rewritten_graph(paths)
            accepted = read_jsonl(paths.rr_artifacts / "accepted_relations.jsonl")
            graph = aggregate_rr_graph(er_graph, accepted)
            rr_graph_dir = paths.rr_graph_artifacts
            rewrite_paths = {
                "rewritten_nodes": rr_graph_dir / "rewritten_nodes.jsonl",
                "rewritten_edges": rr_graph_dir / "rewritten_edges.jsonl",
                "rewritten_mapping": rr_graph_dir / "mention_to_canonical.jsonl",
                "induced_self_loops": rr_graph_dir / "induced_self_loops.jsonl",
                "preexisting_self_loops": rr_graph_dir / "preexisting_self_loops.jsonl",
            }
            write_immutable_jsonl(rewrite_paths["rewritten_nodes"], graph.nodes)
            write_immutable_jsonl(rewrite_paths["rewritten_edges"], graph.edges)
            write_immutable_jsonl(
                rewrite_paths["rewritten_mapping"], graph.mention_to_canonical
            )
            write_immutable_jsonl(
                rewrite_paths["induced_self_loops"], graph.induced_self_loops
            )
            write_immutable_jsonl(
                rewrite_paths["preexisting_self_loops"], graph.preexisting_self_loops
            )
            rewrite_summary_path = rr_graph_dir / "rewrite_summary.json"
            write_immutable_json(rewrite_summary_path, graph.summary)

            native_locator = read_json(paths.locator("native_lightrag"))
            er_locator = read_json(paths.locator("advanced_lightrag_er"))
            native_workspace = Path(str(native_locator["workspace_dir"]))
            er_workspace = Path(str(er_locator["workspace_dir"]))
            native_before = workspace_tree_sha256(native_workspace)
            er_before = workspace_tree_sha256(er_workspace)
            rr_lineage = {
                "er_lineage": dict(self._load_er_artifacts(paths)["lineage"] or {}),
                "rr_plan_sha256": plan_ref.sha256,
                "rr_results_sha256": results_ref.sha256,
                "rr_config_sha256": sha256_json(self.config.relation_recovery),
            }
            frozen = await prepare_lightrag_run(
                self.loaded,
                builder_key=builder_key,
                graph_regime="advanced_lightrag_er_rr",
                base_run_id=paths.base_run_id,
                variant_run_id=paths.rr_variant_run_id,
                er_lineage=rr_lineage,
                physical_attempt=claim.record.attempt_count,
            )
            from lightrag import QueryParam
            from src.extraction.native_capture import build_lightrag

            def rag_factory() -> Any:
                return build_lightrag(frozen, extraction_capture=False)

            questions = _smoke_questions(_load_questions(self.loaded.questions_path))
            outcome = await materialize_finalize_reopen(
                rag_factory,
                documents=_load_documents(self.loaded.documents_path),
                chunks=chunks,
                graph=graph,
                track_id=(
                    f"rr-{paths.rr_variant_run_id[-20:]}-"
                    f"{claim.record.attempt_count:04d}"
                ),
                smoke_cases=[
                    {"question": row["question"], "expected_status": "success"}
                    for row in questions
                ],
                query_param_factory=lambda mode: QueryParam(
                    mode=mode,
                    only_need_context=True,
                    stream=False,
                    **_query_parameters(self.config.runtime.retrieval),
                ),
            )
            native_after = workspace_tree_sha256(native_workspace)
            er_after = workspace_tree_sha256(er_workspace)
            if native_before != native_after or er_before != er_after:
                raise RuntimeError(
                    "Native or ER workspace changed during RR publication"
                )

            snapshot = outcome.reopened_snapshot
            workspace_paths = {
                "nodes": rr_graph_dir / "nodes.jsonl",
                "edges": rr_graph_dir / "edges.jsonl",
                "chunks": rr_graph_dir / "chunks.jsonl",
            }
            lineage = {
                "base_run_id": paths.base_run_id,
                "variant_run_id": paths.rr_variant_run_id,
                "base_extraction_sha256": base_hash,
                "graph_regime": "advanced_lightrag_er_rr",
            }
            write_immutable_jsonl(
                workspace_paths["nodes"], ({**lineage, **row} for row in snapshot.nodes)
            )
            write_immutable_jsonl(
                workspace_paths["edges"], ({**lineage, **row} for row in snapshot.edges)
            )
            write_immutable_jsonl(
                workspace_paths["chunks"],
                ({**lineage, **row} for row in snapshot.chunks),
            )
            aliases_path = rr_graph_dir / "aliases_audit.jsonl"
            placeholders_path = rr_graph_dir / "description_placeholders.jsonl"
            parity_path = rr_graph_dir / "parity_report.json"
            materialization_path = rr_graph_dir / "materialization_report.json"
            write_immutable_jsonl(aliases_path, outcome.stage.aliases_audit)
            write_immutable_jsonl(
                placeholders_path, outcome.stage.description_placeholders
            )
            write_immutable_json(parity_path, outcome.parity_report)
            write_immutable_json(materialization_path, outcome.to_dict())
            final_gate = validate_final_workspace_quality_gate(
                graph_regime="advanced_lightrag_er_rr",
                base_extraction_sha256=base_hash,
                expected_base_extraction_sha256=base_hash,
                nodes=snapshot.nodes,
                edges=snapshot.edges,
                chunks=snapshot.chunks,
                expected_chunk_ids=[chunk.chunk_id for chunk in chunks],
                induced_self_loops=graph.induced_self_loops,
                preexisting_self_loops=graph.preexisting_self_loops,
                workspace_clean=True,
                persisted=True,
                finalized=True,
                reopened=True,
                export_succeeded=True,
                smoke_retrieval_answerable=bool(outcome.smoke_results[0]["passed"]),
                smoke_retrieval_unanswerable=bool(outcome.smoke_results[1]["passed"]),
                native_workspace_sha256_before=native_before,
                native_workspace_sha256_after=native_after,
                fail_closed=False,
            )
            write_immutable_json(final_gate_path, final_gate)
            materialization_rr_gate = validate_rr_quality_gate(
                config=self.config,
                chunks=chunks,
                er_nodes=er_graph.nodes,
                er_edges=er_graph.edges,
                mention_to_canonical=er_graph.mention_to_canonical,
                plan_rows=self._load_rr_plan(paths)[0],
                plan_summary=self._load_rr_plan(paths)[1],
                chunk_results=read_jsonl(
                    paths.rr_artifacts / "verification_results.jsonl"
                ),
                rr_nodes=graph.nodes,
                rr_edges=graph.edges,
                fail_closed=False,
            )
            rr_gate_path = paths.rr_dir / "gates" / "rr_materialization.json"
            write_immutable_json(rr_gate_path, materialization_rr_gate)
            if not final_gate.passed or not materialization_rr_gate.passed:
                errors = [*final_gate.errors, *materialization_rr_gate.errors]
                self.status.mark_blocked_by_quality_gate(
                    "rr_materialization", paths.rr_variant_run_id, "; ".join(errors)
                )
                raise RuntimeError("RR workspace blocked by quality gate")

            locator = {
                "schema_version": "2.0.0",
                **lineage,
                "builder_key": builder_key,
                "legacy_run_id": frozen.lock.run_id,
                "legacy_run_dir": str(frozen.paths.run_dir.resolve()),
                "workspace_dir": str(frozen.paths.workspace_dir.resolve()),
                "workspace_sha256": workspace_tree_sha256(frozen.paths.workspace_dir),
                "physical_materialization_attempt": claim.record.attempt_count,
                "native_workspace_sha256_before": native_before,
                "native_workspace_sha256_after": native_after,
                "er_workspace_sha256_before": er_before,
                "er_workspace_sha256_after": er_after,
                "merge_plan_sha256": merge_ref.sha256,
                "rr_plan_sha256": plan_ref.sha256,
                "rr_results_sha256": results_ref.sha256,
            }
            write_immutable_json(locator_path, locator)
            variant_artifacts = {
                name: fingerprint_artifact(
                    path,
                    logical_name=f"graph/{name}",
                    schema_version="2.0.0",
                )
                for name, path in {
                    **rewrite_paths,
                    **workspace_paths,
                    "rewrite_summary": rewrite_summary_path,
                    "aliases_audit": aliases_path,
                    "description_placeholders": placeholders_path,
                    "parity_report": parity_path,
                    "materialization_report": materialization_path,
                    "final_workspace_gate": final_gate_path,
                    "rr_materialization_gate": rr_gate_path,
                }.items()
            }
            manifest = build_variant_manifest(
                self.config,
                base_manifest=base_manifest,
                graph_regime="advanced_lightrag_er_rr",
                merge_plan=merge_ref,
                rr_plan=plan_ref,
                rr_results=results_ref,
                artifacts=variant_artifacts,
            )
            freeze_manifest(manifest_path, manifest)
            outputs = {
                "variant_manifest": fingerprint_artifact(
                    manifest_path,
                    logical_name="rr_variant_manifest",
                    schema_version="2.0.0",
                ),
                "workspace_locator": fingerprint_artifact(
                    locator_path,
                    logical_name="rr_workspace_locator",
                    schema_version="2.0.0",
                ),
                "final_workspace_gate": variant_artifacts["final_workspace_gate"],
            }
            self.status.mark_completed(
                "rr_materialization", paths.rr_variant_run_id, outputs
            )
            return StageOutcome(
                "rr_materialization", paths.rr_variant_run_id, False, locator
            )
        except Exception as error:
            record = self.status.get("rr_materialization", paths.rr_variant_run_id)
            if record is not None and record.status.value == "running":
                self.status.mark_failed(
                    "rr_materialization", paths.rr_variant_run_id, error
                )
            raise

    async def retrieval(
        self,
        builder_key: str,
        graph_regime: GraphRegime,
        *,
        reclaim_running: bool = False,
    ) -> StageOutcome:
        """Run context-only retrieval for all 120 questions in one variant."""

        from src.config.native_runtime import load_frozen_run
        from src.extraction.native_capture import build_lightrag
        from src.retrieval import run_retrieval

        self.config.assert_ready(
            builder_key=builder_key,
            require_er=graph_regime != "native_lightrag",
            require_rr=graph_regime == "advanced_lightrag_er_rr",
        )
        paths = self.condition_paths(builder_key)
        locator = read_json(paths.locator(graph_regime))
        base_manifest = _read_base_manifest(paths.base_manifest)
        variant_id = str(locator["variant_run_id"])
        expectation = self._expectation(
            variant_id,
            inputs={
                "workspace_locator": fingerprint_artifact(
                    paths.locator(graph_regime),
                    logical_name="workspace_locator",
                    schema_version="2.0.0",
                )
            },
        )
        claim = self.status.claim(
            "retrieval",
            variant_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        artifact_path = _condition_file(
            self.config,
            paths,
            graph_regime,
            directory="artifacts",
            stem="retrieval",
            suffix=".jsonl",
        )
        if claim.reused:
            return StageOutcome(
                "retrieval", variant_id, True, {"artifact": str(artifact_path)}
            )
        if not claim.claimed:
            raise RuntimeError("retrieval is already active or not claimable")
        rag: Any | None = None
        initialized = False
        try:
            await self._verify_current_models(
                self.config.roles.query,
                self.config.roles.embedding,
            )
            frozen = load_frozen_run(locator["legacy_run_dir"])
            query = self.config.roles.query
            query_generation = _query_generation_parameters(self.config)
            role_settings = {
                "ollama_host": self.config.runtime.ollama_host,
                "timeout_seconds": int(self.config.runtime.model_timeout_seconds),
                "options": {
                    key: value
                    for key, value in query_generation.items()
                    if key != "think"
                },
            }
            role_override = {
                "model": _required(query.resolved_name, "query model"),
                "model_digest": _required(query.digest, "query model digest"),
                "settings": role_settings,
            }
            rag = build_lightrag(
                frozen,
                extraction_capture=False,
                role_overrides={
                    "keyword": role_override,
                    "query": role_override,
                },
            )
            await rag.initialize_storages()
            initialized = True
            builder = self.config.builders_by_key[builder_key]
            records = await run_retrieval(
                rag,
                _load_questions(self.loaded.questions_path),
                artifact_path=artifact_path,
                lineage={
                    "base_run_id": paths.base_run_id,
                    "variant_run_id": variant_id,
                    "base_extraction_sha256": _required(
                        base_manifest.base_extraction_sha256,
                        "base extraction SHA-256",
                    ),
                },
                builder_model=_required(builder.resolved_name, "builder model"),
                graph_regime=graph_regime,
                query_model=_required(query.resolved_name, "query model"),
                query_model_digest=_required(query.digest, "query model digest"),
                query_generation_parameters=query_generation,
                retrieval_mode=self.config.runtime.retrieval.mode,
                retrieval_parameters=_query_parameters(self.config.runtime.retrieval),
            )
        except Exception as error:
            self.status.mark_failed("retrieval", variant_id, error)
            raise
        finally:
            if initialized and rag is not None:
                await rag.finalize_storages()
        output = fingerprint_artifact(
            artifact_path,
            logical_name="retrieval_results",
            schema_version="3.0.0",
        )
        self.status.mark_completed("retrieval", variant_id, {"retrieval": output})
        return StageOutcome(
            "retrieval",
            variant_id,
            False,
            {"artifact": str(artifact_path), "records": len(records)},
        )

    async def answering(
        self,
        builder_key: str,
        graph_regime: GraphRegime,
        *,
        reclaim_running: bool = False,
    ) -> StageOutcome:
        """Generate answers only from saved retrieval contexts."""

        from src.answering import run_answering

        self.config.assert_ready(
            builder_key=builder_key,
            require_er=graph_regime != "native_lightrag",
            require_rr=graph_regime == "advanced_lightrag_er_rr",
        )
        paths = self.condition_paths(builder_key)
        locator = read_json(paths.locator(graph_regime))
        variant_id = str(locator["variant_run_id"])
        retrieval_path = _condition_file(
            self.config,
            paths,
            graph_regime,
            directory="artifacts",
            stem="retrieval",
            suffix=".jsonl",
        )
        retrieval_ref = fingerprint_artifact(
            retrieval_path,
            logical_name="retrieval_results",
            schema_version="3.0.0",
        )
        expectation = self._expectation(variant_id, inputs={"retrieval": retrieval_ref})
        claim = self.status.claim(
            "answering",
            variant_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        artifact_path = _condition_file(
            self.config,
            paths,
            graph_regime,
            directory="artifacts",
            stem="answers",
            suffix=".jsonl",
        )
        if claim.reused:
            return StageOutcome(
                "answering", variant_id, True, {"artifact": str(artifact_path)}
            )
        if not claim.claimed:
            raise RuntimeError("answering is already active or not claimable")
        try:
            await self._verify_current_models(self.config.roles.answer)
            answer = self.config.roles.answer
            generation = answer.generation
            if generation is None:
                raise ValueError("answer generation settings are required")
            records = await run_answering(
                read_jsonl(retrieval_path),
                _load_questions(self.loaded.questions_path),
                artifact_path=artifact_path,
                answer_model=_required(answer.resolved_name, "answer model"),
                answer_model_digest=_required(answer.digest, "answer model digest"),
                generation_options={
                    "temperature": generation.temperature,
                    "seed": generation.seed,
                    "num_ctx": generation.context_window,
                    "num_predict": generation.output_tokens,
                },
                think=generation.think,
                ollama_host=self.config.runtime.ollama_host,
                timeout_seconds=self.config.runtime.model_timeout_seconds,
            )
        except Exception as error:
            self.status.mark_failed("answering", variant_id, error)
            raise
        output = fingerprint_artifact(
            artifact_path,
            logical_name="answer_results",
            schema_version="3.0.0",
        )
        self.status.mark_completed("answering", variant_id, {"answers": output})
        return StageOutcome(
            "answering",
            variant_id,
            False,
            {"artifact": str(artifact_path), "records": len(records)},
        )

    def _evaluate_once(self, builder_key: str) -> StageOutcome:
        """Compute all three within-question post-processing comparisons."""

        from src.evaluation import (
            CASCADE_COMPARISON_PLAN,
            DEFAULT_BOOTSTRAP_SAMPLES,
            DEFAULT_BOOTSTRAP_SEED,
            DEFAULT_CONFIDENCE_LEVEL,
            DEFAULT_PAIRED_METRIC_FIELDS,
            DEFAULT_RETRIEVAL_CUTOFFS,
            SOURCE_QUESTION_TYPE_WEIGHTS,
            aggregate_downstream_metrics,
            compute_downstream_metrics,
            compute_er_metrics,
            bootstrap_extraction_metrics,
            compute_extraction_efficiency,
            compute_extraction_metrics,
            compute_graph_topology,
            paired_variant_deltas,
            paired_regime_deltas,
            summarize_paired_deltas,
            summarize_regime_deltas,
        )

        paths = self.condition_paths(builder_key)
        _, chunks, mentions, _ = load_staged_snapshot(paths.staging_dir)
        er = self._load_er_artifacts(paths)
        questions = _load_questions(self.loaded.questions_path)
        variant_rows: dict[str, list[dict[str, Any]]] = {}
        variant_summaries: dict[str, dict[str, Any]] = {}
        for regime in (
            "native_lightrag",
            "advanced_lightrag_er",
            "advanced_lightrag_er_rr",
        ):
            retrievals = read_jsonl(
                _qa_artifact_path(
                    self.status,
                    self.config,
                    paths,
                    regime,
                    stage="retrieval",
                )
            )
            answers = read_jsonl(
                _qa_artifact_path(
                    self.status,
                    self.config,
                    paths,
                    regime,
                    stage="answering",
                )
            )
            rows = compute_downstream_metrics(questions, retrievals, answers)
            write_immutable_jsonl(
                _metric_file(
                    self.config,
                    paths,
                    regime,
                    stem="question_metrics",
                    suffix=".jsonl",
                ),
                rows,
            )
            variant_rows[regime] = rows
            variant_summaries[regime] = aggregate_downstream_metrics(rows)
        secondary_paired = paired_variant_deltas(
            variant_rows["native_lightrag"],
            variant_rows["advanced_lightrag_er"],
            metric_fields=DEFAULT_PAIRED_METRIC_FIELDS,
        )
        secondary_summary = summarize_paired_deltas(
            secondary_paired,
            metric_fields=DEFAULT_PAIRED_METRIC_FIELDS,
            bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
            confidence_level=DEFAULT_CONFIDENCE_LEVEL,
            seed=DEFAULT_BOOTSTRAP_SEED,
        )
        comparison_specs = {
            "incremental_ablation": (
                "advanced_lightrag_er",
                "advanced_lightrag_er_rr",
            ),
            "primary": (
                "native_lightrag",
                "advanced_lightrag_er_rr",
            ),
        }
        generic_paired: dict[str, list[dict[str, Any]]] = {}
        generic_summaries: dict[str, dict[str, Any]] = {}
        for name, (left_regime, right_regime) in comparison_specs.items():
            rows = paired_regime_deltas(
                variant_rows[left_regime],
                variant_rows[right_regime],
                left_regime=left_regime,
                right_regime=right_regime,
                metric_fields=DEFAULT_PAIRED_METRIC_FIELDS,
            )
            generic_paired[name] = rows
            generic_summaries[name] = summarize_regime_deltas(
                rows,
                left_regime=left_regime,
                right_regime=right_regime,
                metric_fields=DEFAULT_PAIRED_METRIC_FIELDS,
                bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
                confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                seed=DEFAULT_BOOTSTRAP_SEED,
            )
        native_locator = read_json(paths.locator("native_lightrag"))
        native_run_dir = Path(native_locator["legacy_run_dir"])
        native_nodes = read_jsonl(native_run_dir / "artifacts" / "graph_nodes.jsonl")
        native_edges = read_jsonl(native_run_dir / "artifacts" / "graph_edges.jsonl")
        extraction_calls = read_jsonl(
            native_run_dir / "artifacts" / "extraction_calls.jsonl"
        )
        er_nodes = read_jsonl(paths.graph_artifacts / "rewritten_nodes.jsonl")
        er_edges = read_jsonl(paths.graph_artifacts / "rewritten_edges.jsonl")
        native_topology = compute_graph_topology(native_nodes, native_edges)
        er_topology = compute_graph_topology(er_nodes, er_edges)
        rr_nodes = read_jsonl(paths.rr_graph_artifacts / "rewritten_nodes.jsonl")
        rr_edges = read_jsonl(paths.rr_graph_artifacts / "rewritten_edges.jsonl")
        rr_topology = compute_graph_topology(rr_nodes, rr_edges)
        er_metrics = compute_er_metrics(
            mentions=mentions,
            canonical_entities=er["canonical_entities"],
            candidate_pairs=er["candidate_pairs"],
            pair_decisions=er["pair_decisions"],
            mention_to_canonical=er["mention_to_canonical"],
            aliases=er["aliases"],
            merge_plan=er["merge_plan"],
            rewrite_summary=read_json(paths.graph_artifacts / "rewrite_summary.json"),
            native_graph=native_topology,
            er_graph=er_topology,
            runtime=read_json(paths.er_artifacts / "runtime.json"),
        )
        report = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "builder_key": builder_key,
            "base_run_id": paths.base_run_id,
            "evaluation_parameters": {
                "retrieval_cutoffs": list(DEFAULT_RETRIEVAL_CUTOFFS),
                "paired_bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
                "paired_bootstrap_confidence_level": DEFAULT_CONFIDENCE_LEVEL,
                "paired_bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
                "paired_bootstrap_stratification": "question_type",
                "extraction_bootstrap_cluster_unit": "document_id",
                "source_question_type_weights": SOURCE_QUESTION_TYPE_WEIGHTS,
            },
            "extraction": {
                **compute_extraction_metrics(chunks),
                "efficiency": compute_extraction_efficiency(extraction_calls),
                "document_cluster_bootstrap": bootstrap_extraction_metrics(
                    chunks,
                    extraction_calls,
                    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
                    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                    seed=DEFAULT_BOOTSTRAP_SEED,
                ),
            },
            "entity_resolution": er_metrics,
            "native_topology": native_topology,
            "er_topology": er_topology,
            "er_rr_topology": rr_topology,
            "relation_recovery": {
                "plan": read_json(paths.rr_artifacts / "plan_summary.json"),
                "verification": read_json(
                    paths.rr_artifacts / "verification_summary.json"
                ),
                "rewrite": read_json(paths.rr_graph_artifacts / "rewrite_summary.json"),
            },
            "downstream": variant_summaries,
            "paired_comparisons": {
                "primary": generic_summaries["primary"],
                "secondary": secondary_summary,
                "incremental_ablation": generic_summaries["incremental_ablation"],
            },
            "paired_question_count": len(secondary_paired),
            "comparison_plan": {
                role: {
                    **declaration,
                    "unit": "question_id",
                    "builder_key": builder_key,
                }
                for role, declaration in CASCADE_COMPARISON_PLAN.items()
            },
        }
        metrics_dir = paths.er_dir / "metrics"
        condition_hash = _evaluation_identity_sha256(self.config)[:16]
        paired_primary_path = (
            metrics_dir / f"paired_primary_er_rr_vs_native.{condition_hash}.jsonl"
        )
        paired_secondary_path = (
            metrics_dir / f"paired_secondary_er_vs_native.{condition_hash}.jsonl"
        )
        paired_incremental_path = (
            metrics_dir / f"paired_incremental_er_rr_vs_er.{condition_hash}.jsonl"
        )
        report_path = metrics_dir / f"summary.{condition_hash}.json"
        write_immutable_jsonl(paired_primary_path, generic_paired["primary"])
        write_immutable_jsonl(paired_secondary_path, secondary_paired)
        write_immutable_jsonl(
            paired_incremental_path, generic_paired["incremental_ablation"]
        )
        write_immutable_json(report_path, report)
        return StageOutcome(
            "evaluation",
            paths.base_run_id,
            False,
            {
                "summary": str(report_path),
                "paired_primary_er_rr_vs_native": str(paired_primary_path),
                "paired_secondary_er_vs_native": str(paired_secondary_path),
                "paired_incremental_er_rr_vs_er": str(paired_incremental_path),
            },
        )

    def evaluation(
        self, builder_key: str, *, reclaim_running: bool = False
    ) -> StageOutcome:
        """Resume-aware wrapper around paired metric computation."""

        self.config.assert_ready(
            builder_key=builder_key, require_er=True, require_rr=True
        )
        paths = self.condition_paths(builder_key)
        input_paths: dict[str, Path] = {
            "base_manifest": paths.base_manifest,
            "merge_plan": paths.er_artifacts / "merge_plan.json",
            "rr_results": paths.rr_artifacts / "verification_results.jsonl",
        }
        for regime in (
            "native_lightrag",
            "advanced_lightrag_er",
            "advanced_lightrag_er_rr",
        ):
            input_paths[f"{regime}_retrieval"] = _qa_artifact_path(
                self.status,
                self.config,
                paths,
                regime,
                stage="retrieval",
            )
            input_paths[f"{regime}_answers"] = _qa_artifact_path(
                self.status,
                self.config,
                paths,
                regime,
                stage="answering",
            )
        inputs = {
            name: fingerprint_artifact(
                path,
                logical_name=name,
                schema_version=(
                    "3.0.0"
                    if name.endswith(("_retrieval", "_answers"))
                    else "1.0.0"
                    if name == "rr_results"
                    else "2.0.0"
                ),
            )
            for name, path in input_paths.items()
        }
        expectation = self._expectation(
            paths.base_run_id,
            config_sha256=_evaluation_identity_sha256(self.config),
            inputs=inputs,
        )
        claim = self.status.claim(
            "evaluation",
            paths.base_run_id,
            expectation,
            reclaim_running=reclaim_running,
        )
        metrics_dir = paths.er_dir / "metrics"
        condition_hash = _evaluation_identity_sha256(self.config)[:16]
        paired_primary_path = (
            metrics_dir / f"paired_primary_er_rr_vs_native.{condition_hash}.jsonl"
        )
        paired_secondary_path = (
            metrics_dir / f"paired_secondary_er_vs_native.{condition_hash}.jsonl"
        )
        paired_incremental_path = (
            metrics_dir / f"paired_incremental_er_rr_vs_er.{condition_hash}.jsonl"
        )
        report_path = metrics_dir / f"summary.{condition_hash}.json"
        if claim.reused:
            return StageOutcome(
                "evaluation",
                paths.base_run_id,
                True,
                {
                    "summary": str(report_path),
                    "paired_primary_er_rr_vs_native": str(paired_primary_path),
                    "paired_secondary_er_vs_native": str(paired_secondary_path),
                    "paired_incremental_er_rr_vs_er": str(paired_incremental_path),
                },
            )
        if not claim.claimed:
            raise RuntimeError("evaluation is already active or not claimable")
        try:
            outcome = self._evaluate_once(builder_key)
            output_paths = {
                "summary": report_path,
                "paired_primary_er_rr_vs_native": paired_primary_path,
                "paired_secondary_er_vs_native": paired_secondary_path,
                "paired_incremental_er_rr_vs_er": paired_incremental_path,
                "native_question_metrics": _metric_file(
                    self.config,
                    paths,
                    "native_lightrag",
                    stem="question_metrics",
                    suffix=".jsonl",
                ),
                "er_question_metrics": _metric_file(
                    self.config,
                    paths,
                    "advanced_lightrag_er",
                    stem="question_metrics",
                    suffix=".jsonl",
                ),
                "er_rr_question_metrics": _metric_file(
                    self.config,
                    paths,
                    "advanced_lightrag_er_rr",
                    stem="question_metrics",
                    suffix=".jsonl",
                ),
            }
            outputs = {
                name: fingerprint_artifact(
                    path,
                    logical_name=f"evaluation/{name}",
                    schema_version=(
                        "3.0.0"
                        if name.endswith("question_metrics")
                        else EVALUATION_SCHEMA_VERSION
                    ),
                )
                for name, path in output_paths.items()
            }
            self.status.mark_completed("evaluation", paths.base_run_id, outputs)
            return outcome
        except Exception as error:
            record = self.status.get("evaluation", paths.base_run_id)
            if record is not None and record.status.value == "running":
                self.status.mark_failed("evaluation", paths.base_run_id, error)
            raise

    def primary_analysis(self) -> StageOutcome:
        """Collect primary, secondary and incremental contrasts for 12 builders."""

        from src.evaluation import collect_prespecified_cascade_comparisons

        condition_hash = _evaluation_identity_sha256(self.config)[:16]
        summaries: list[dict[str, Any]] = []
        for builder in self.config.builders:
            paths = self.condition_paths(builder.key)
            summary_path = paths.er_dir / "metrics" / f"summary.{condition_hash}.json"
            if not summary_path.exists():
                raise RuntimeError(
                    f"primary analysis requires completed evaluation: {builder.key}"
                )
            summaries.append(read_json(summary_path))
        report = collect_prespecified_cascade_comparisons(
            summaries,
            expected_builder_keys=[builder.key for builder in self.config.builders],
            expected_question_count=len(_load_questions(self.loaded.questions_path)),
        )
        report["evaluation_identity_sha256"] = _evaluation_identity_sha256(self.config)
        report["primary_analysis_identity_sha256"] = _primary_analysis_identity_sha256(
            self.config
        )
        primary_hash = _primary_analysis_identity_sha256(self.config)[:16]
        output = (
            self.loaded.runs_root
            / "analysis"
            / f"prespecified_cascade_comparisons.{primary_hash}.json"
        )
        write_immutable_json(output, report)
        return StageOutcome(
            "primary_analysis",
            _primary_analysis_identity_sha256(self.config),
            False,
            {
                "artifact": str(output),
                "primary_comparison_count": 12,
                "total_prespecified_comparison_count": 36,
            },
        )

    def exploratory_analysis(self) -> StageOutcome:
        """Build the 66-pair/family/scale/DiD global exploratory report."""

        from src.evaluation import (
            DEFAULT_BOOTSTRAP_SAMPLES,
            DEFAULT_BOOTSTRAP_SEED,
            DEFAULT_CONFIDENCE_LEVEL,
            DEFAULT_PAIRED_METRIC_FIELDS,
            DOWNSTREAM_METRICS_SCHEMA_VERSION,
            EXPLORATORY_SCHEMA_VERSION,
            PRIMARY_ANALYSIS_SCHEMA_VERSION,
            builder_metadata_from_config,
            compute_exploratory_analysis,
            summarize_builder_artifacts,
        )

        for builder in self.config.builders:
            self.config.assert_ready(
                builder_key=builder.key, require_er=True, require_rr=True
            )

        evaluation_hash = _evaluation_identity_sha256(self.config)[:16]
        primary_hash = _primary_analysis_identity_sha256(self.config)[:16]
        exploratory_hash = _exploratory_identity_sha256(self.config)[:16]
        primary_path = (
            self.loaded.runs_root
            / "analysis"
            / f"prespecified_cascade_comparisons.{primary_hash}.json"
        )
        if not primary_path.exists():
            raise RuntimeError(
                "exploratory analysis requires the completed primary-analysis artifact"
            )
        condition_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        builder_summaries: dict[str, dict[str, Any]] = {}
        input_paths: dict[str, Path] = {"primary_analysis": primary_path}
        for builder in self.config.builders:
            paths = self.condition_paths(builder.key)
            condition_rows[builder.key] = {}
            summary_path = paths.er_dir / "metrics" / f"summary.{evaluation_hash}.json"
            if not summary_path.exists():
                raise RuntimeError(
                    f"exploratory analysis requires builder summary: {builder.key}"
                )
            builder_summaries[builder.key] = read_json(summary_path)
            input_paths[f"{builder.key}/summary"] = summary_path
            for regime in (
                "native_lightrag",
                "advanced_lightrag_er",
                "advanced_lightrag_er_rr",
            ):
                metric_path = _metric_file(
                    self.config,
                    paths,
                    regime,
                    stem="question_metrics",
                    suffix=".jsonl",
                )
                if not metric_path.exists():
                    raise RuntimeError(
                        "exploratory analysis requires completed question metrics: "
                        f"{builder.key}/{regime}"
                    )
                condition_rows[builder.key][regime] = read_jsonl(metric_path)
                input_paths[f"{builder.key}/{regime}"] = metric_path
        result = compute_exploratory_analysis(
            condition_rows,
            builder_metadata=builder_metadata_from_config(self.config.builders),
            metric_fields=DEFAULT_PAIRED_METRIC_FIELDS,
            expected_question_count=self.config.corpus.expected_questions,
            bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
            confidence_level=DEFAULT_CONFIDENCE_LEVEL,
            seed=DEFAULT_BOOTSTRAP_SEED,
        )
        analysis_dir = self.loaded.runs_root / "analysis"
        pairwise_path = (
            analysis_dir / f"exploratory_model_pairs.{exploratory_hash}.jsonl"
        )
        did_path = (
            analysis_dir
            / f"exploratory_difference_in_differences.{exploratory_hash}.jsonl"
        )
        report_path = analysis_dir / f"global_experiment_report.{exploratory_hash}.json"
        write_immutable_jsonl(pairwise_path, result["pairwise_comparisons"])
        write_immutable_jsonl(did_path, result["difference_in_differences"])
        report = dict(result["global_report"])
        report["family_and_scale"]["artifact_level"] = summarize_builder_artifacts(
            builder_summaries,
            builder_metadata=builder_metadata_from_config(self.config.builders),
        )
        report["lineage"] = {
            "experiment_id": self.config.experiment_id,
            "subset_id": self.config.corpus.subset_id,
            "questions_sha256": self.config.corpus.questions_sha256,
            "full_config_sha256": full_config_sha256(self.config),
            "evaluation_identity_sha256": _evaluation_identity_sha256(self.config),
            "primary_analysis_identity_sha256": _primary_analysis_identity_sha256(
                self.config
            ),
            "exploratory_identity_sha256": _exploratory_identity_sha256(self.config),
        }
        report["input_artifacts"] = {
            name: fingerprint_artifact(
                path,
                logical_name=f"exploratory/input/{name}",
                schema_version=(
                    PRIMARY_ANALYSIS_SCHEMA_VERSION
                    if name == "primary_analysis"
                    else EVALUATION_SCHEMA_VERSION
                    if name.endswith("/summary")
                    else DOWNSTREAM_METRICS_SCHEMA_VERSION
                ),
            ).model_dump(mode="json")
            for name, path in sorted(input_paths.items())
        }
        report["output_artifacts"] = {
            "model_pairs": fingerprint_artifact(
                pairwise_path,
                logical_name="exploratory/model_pairs",
                schema_version=EXPLORATORY_SCHEMA_VERSION,
            ).model_dump(mode="json"),
            "difference_in_differences": fingerprint_artifact(
                did_path,
                logical_name="exploratory/difference_in_differences",
                schema_version=EXPLORATORY_SCHEMA_VERSION,
            ).model_dump(mode="json"),
        }
        report["primary_analysis"] = read_json(primary_path)
        write_immutable_json(report_path, report)
        return StageOutcome(
            "exploratory_analysis",
            _exploratory_identity_sha256(self.config),
            False,
            {
                "global_report": str(report_path),
                "model_pairs": str(pairwise_path),
                "difference_in_differences": str(did_path),
                "model_pair_count": 66,
            },
        )

    async def resume(
        self, builder_key: str, *, reclaim_running: bool = False
    ) -> list[StageOutcome]:
        """Advance one builder through every stage; completed hashes are reused."""

        outcomes = [
            await self.native_build(builder_key, reclaim_running=reclaim_running)
        ]
        outcomes.append(
            await self.er_plan(builder_key, reclaim_running=reclaim_running)
        )
        outcomes.append(
            await self.er_materialize(builder_key, reclaim_running=reclaim_running)
        )
        outcomes.append(
            await self.rr_plan(builder_key, reclaim_running=reclaim_running)
        )
        outcomes.append(
            await self.rr_verify(builder_key, reclaim_running=reclaim_running)
        )
        outcomes.append(
            await self.rr_materialize(builder_key, reclaim_running=reclaim_running)
        )
        for regime in (
            "native_lightrag",
            "advanced_lightrag_er",
            "advanced_lightrag_er_rr",
        ):
            outcomes.append(
                await self.retrieval(
                    builder_key, regime, reclaim_running=reclaim_running
                )
            )
            outcomes.append(
                await self.answering(
                    builder_key, regime, reclaim_running=reclaim_running
                )
            )
        outcomes.append(self.evaluation(builder_key, reclaim_running=reclaim_running))
        return outcomes


__all__ = ["ExperimentHarness", "GraphRegime", "StageOutcome"]
