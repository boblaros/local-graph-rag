from __future__ import annotations

import pytest

from src.orchestration.lineage import (
    base_config_sha256,
    build_base_manifest,
    build_variant_manifest,
    compute_artifact_set_sha256,
    compute_base_run_id,
    compute_variant_run_id,
    fingerprint_artifact,
    freeze_base_config,
    freeze_experiment_config,
    freeze_manifest,
    load_frozen_base_config,
    load_frozen_experiment_config,
    sha256_directory,
)

from ._config_helpers import digest, resolved_config


def test_base_identity_contains_only_corpus_chunking_and_extraction_dimensions() -> (
    None
):
    config = resolved_config()
    original = compute_base_run_id(config, "qwen35_0_8b")
    changed_roles = config.roles.model_copy(
        update={
            "query": config.roles.query.model_copy(
                update={"digest": digest("another-query")}
            ),
            "answer": config.roles.answer.model_copy(
                update={"digest": digest("another-answer")}
            ),
        }
    )
    changed = config.model_copy(update={"roles": changed_roles})
    assert compute_base_run_id(changed, "qwen35_0_8b") == original

    changed_downstream = config.model_copy(
        update={
            "entity_resolution": config.entity_resolution.model_copy(
                update={
                    "decision_policy": config.entity_resolution.decision_policy.model_copy(
                        update={"max_judge_calls_per_run": 649}
                    )
                }
            ),
            "metadata": {**config.metadata, "audit_note": "downstream-only"},
        }
    )
    assert compute_base_run_id(changed_downstream, "qwen35_0_8b") == original
    assert base_config_sha256(changed_downstream, "qwen35_0_8b") == (
        base_config_sha256(config, "qwen35_0_8b")
    )

    builders = list(config.builders)
    builders[0] = builders[0].model_copy(update={"digest": digest("new-builder")})
    changed_builder = config.model_copy(update={"builders": tuple(builders)})
    assert compute_base_run_id(changed_builder, "qwen35_0_8b") != original

    for field, value in (
        ("max_extract_input_tokens", 4096),
        ("tiktoken_model_name", "different-tokenizer"),
    ):
        changed_lightrag = config.runtime.lightrag.model_copy(update={field: value})
        changed_runtime = config.runtime.model_copy(
            update={"lightrag": changed_lightrag}
        )
        changed_extraction_runtime = config.model_copy(
            update={"runtime": changed_runtime}
        )
        assert (
            compute_base_run_id(changed_extraction_runtime, "qwen35_0_8b") != original
        )


def test_native_and_er_variant_identities_share_base_but_er_freezes_judge_and_plan_config() -> (
    None
):
    config = resolved_config()
    base = compute_base_run_id(config, "qwen35_0_8b")
    native = compute_variant_run_id(
        config, base_run_id=base, graph_regime="native_lightrag"
    )
    er = compute_variant_run_id(
        config, base_run_id=base, graph_regime="advanced_lightrag_er"
    )
    rr = compute_variant_run_id(
        config, base_run_id=base, graph_regime="advanced_lightrag_er_rr"
    )
    assert native.startswith("variant_native_")
    assert er.startswith("variant_er_")
    assert rr.startswith("variant_rr_")
    assert native != er
    assert rr not in {native, er}

    override = config.runtime.er_judge_budget_overrides["qwen3_8b"]
    changed_runtime = config.runtime.model_copy(
        update={
            "er_judge_budget_overrides": {
                "qwen3_8b": override.model_copy(
                    update={"max_judge_calls_per_run": 1200}
                )
            }
        }
    )
    changed_operational_budget = config.model_copy(
        update={"runtime": changed_runtime}
    )
    assert (
        compute_variant_run_id(
            changed_operational_budget,
            base_run_id=base,
            graph_regime="advanced_lightrag_er",
        )
        == er
    )
    assert (
        compute_variant_run_id(
            changed_operational_budget,
            base_run_id=base,
            graph_regime="advanced_lightrag_er_rr",
        )
        == rr
    )

    changed_er = config.model_copy(
        update={
            "entity_resolution": config.entity_resolution.model_copy(
                update={
                    "decision_policy": config.entity_resolution.decision_policy.model_copy(
                        update={"max_judge_calls_per_run": 649}
                    )
                }
            )
        }
    )
    assert (
        compute_variant_run_id(
            changed_er, base_run_id=base, graph_regime="native_lightrag"
        )
        == native
    )
    assert (
        compute_variant_run_id(
            changed_er, base_run_id=base, graph_regime="advanced_lightrag_er"
        )
        != er
    )
    assert (
        compute_variant_run_id(
            changed_er,
            base_run_id=base,
            graph_regime="advanced_lightrag_er_rr",
        )
        != rr
    )

    changed_rr = config.model_copy(
        update={
            "relation_recovery": config.relation_recovery.model_copy(
                update={"include_existing_local_relations": True}
            ),
            "roles": config.roles.model_copy(
                update={
                    "rr_verifier": config.roles.rr_verifier.model_copy(
                        update={"output_tokens": 128}
                    )
                }
            ),
        }
    )
    assert (
        compute_variant_run_id(
            changed_rr, base_run_id=base, graph_regime="native_lightrag"
        )
        == native
    )
    assert (
        compute_variant_run_id(
            changed_rr, base_run_id=base, graph_regime="advanced_lightrag_er"
        )
        == er
    )
    assert (
        compute_variant_run_id(
            changed_rr, base_run_id=base, graph_regime="advanced_lightrag_er_rr"
        )
        != rr
    )

    unresolved_policy = config.model_copy(
        update={
            "graph_materialization": config.graph_materialization.model_copy(
                update={"missing_description_policy": None}
            )
        }
    )
    assert (
        compute_variant_run_id(
            unresolved_policy, base_run_id=base, graph_regime="native_lightrag"
        )
        == native
    )
    assert (
        compute_variant_run_id(
            unresolved_policy,
            base_run_id=base,
            graph_regime="advanced_lightrag_er",
        )
        != er
    )

    changed_query = config.roles.query.model_copy(
        update={
            "generation": config.roles.query.generation.model_copy(
                update={"temperature": 0.25}
            )
        }
    )
    downstream_only = config.model_copy(
        update={
            "roles": config.roles.model_copy(update={"query": changed_query}),
            "runtime": config.runtime.model_copy(
                update={
                    "retrieval": config.runtime.retrieval.model_copy(
                        update={"top_k": 7}
                    )
                }
            ),
        }
    )
    assert (
        compute_variant_run_id(
            downstream_only, base_run_id=base, graph_regime="native_lightrag"
        )
        == native
    )
    assert (
        compute_variant_run_id(
            downstream_only,
            base_run_id=base,
            graph_regime="advanced_lightrag_er",
        )
        == er
    )

    changed_embedding = config.roles.embedding.model_copy(
        update={"digest": digest("different-embedding")}
    )
    graph_changed = config.model_copy(
        update={
            "roles": config.roles.model_copy(update={"embedding": changed_embedding})
        }
    )
    assert (
        compute_variant_run_id(
            graph_changed, base_run_id=base, graph_regime="native_lightrag"
        )
        != native
    )
    assert (
        compute_variant_run_id(
            graph_changed,
            base_run_id=base,
            graph_regime="advanced_lightrag_er",
        )
        != er
    )

    changed_storage = config.runtime.lightrag.storage.model_copy(
        update={"graph_storage": "AnotherPublicGraphStorage"}
    )
    storage_changed = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "lightrag": config.runtime.lightrag.model_copy(
                        update={"storage": changed_storage}
                    )
                }
            )
        }
    )
    assert (
        compute_variant_run_id(
            storage_changed, base_run_id=base, graph_regime="native_lightrag"
        )
        != native
    )

    operational_lightrag = config.runtime.lightrag.model_copy(
        update={
            "llm_timeout_seconds": 123,
            "index_batch_size": 3,
            "embedding_func_max_async": 4,
        }
    )
    operational_only = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"lightrag": operational_lightrag}
            )
        }
    )
    assert (
        compute_variant_run_id(
            operational_only, base_run_id=base, graph_regime="native_lightrag"
        )
        == native
    )

    chunk_selection_changed = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "lightrag": config.runtime.lightrag.model_copy(
                        update={"kg_linked_chunk_selection": "VECTOR"}
                    )
                }
            )
        }
    )
    assert (
        compute_variant_run_id(
            chunk_selection_changed,
            base_run_id=base,
            graph_regime="native_lightrag",
        )
        != native
    )

    enabled_embedding_cache = config.runtime.lightrag.embedding_cache.model_copy(
        update={"enabled": True}
    )
    cache_changed = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "lightrag": config.runtime.lightrag.model_copy(
                        update={"embedding_cache": enabled_embedding_cache}
                    )
                }
            )
        }
    )
    assert (
        compute_variant_run_id(
            cache_changed, base_run_id=base, graph_regime="native_lightrag"
        )
        != native
    )


def test_immutable_config_and_run_manifests_are_idempotent_and_conflict_safe(
    tmp_path,
) -> None:
    config = resolved_config()
    frozen_path = tmp_path / "frozen_config.json"
    first = freeze_experiment_config(frozen_path, config)
    second = freeze_experiment_config(frozen_path, config)
    assert first.created is True
    assert second.created is False
    assert first.sha256 == second.sha256
    assert load_frozen_experiment_config(frozen_path).config_sha256

    changed = config.model_copy(
        update={
            "roles": config.roles.model_copy(
                update={
                    "query": config.roles.query.model_copy(
                        update={"digest": digest("conflicting-query")}
                    )
                }
            )
        }
    )
    with pytest.raises(RuntimeError, match="immutable frozen config conflict"):
        freeze_experiment_config(frozen_path, changed)

    base_frozen_path = tmp_path / "frozen_base_config.json"
    first_base = freeze_base_config(base_frozen_path, config, "qwen35_0_8b")
    second_base = freeze_base_config(base_frozen_path, changed, "qwen35_0_8b")
    assert first_base.created is True
    assert second_base.created is False
    assert first_base.sha256 == second_base.sha256
    frozen_base = load_frozen_base_config(base_frozen_path)
    assert frozen_base.base_config_sha256 == base_config_sha256(config, "qwen35_0_8b")

    changed_extraction_runtime = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "lightrag": config.runtime.lightrag.model_copy(
                        update={"max_extract_input_tokens": 4096}
                    )
                }
            )
        }
    )
    with pytest.raises(RuntimeError, match="immutable frozen base config conflict"):
        freeze_base_config(base_frozen_path, changed_extraction_runtime, "qwen35_0_8b")

    extraction = tmp_path / "entities.jsonl"
    extraction.write_text('{"mention_id":"m1"}\n', encoding="utf-8")
    extraction_ref = fingerprint_artifact(
        extraction,
        logical_name="normalized_entities",
        schema_version="2.0.0",
        root=tmp_path,
    )
    base_manifest = build_base_manifest(
        config,
        "qwen35_0_8b",
        artifacts={"normalized_entities": extraction_ref},
    )
    assert base_manifest.extraction_prompt_sha256 == config.extraction.prompt_sha256
    assert base_manifest.extraction_seed == config.extraction.seed
    assert base_manifest.extraction_generation_parameters["max_gleaning"] == 1
    manifest_path = tmp_path / "base_manifest.json"
    assert freeze_manifest(manifest_path, base_manifest).created is True
    assert freeze_manifest(manifest_path, base_manifest).created is False

    merge_plan = tmp_path / "merge_plan.json"
    merge_plan.write_text('{"schema_version":"2.0.0"}\n', encoding="utf-8")
    merge_ref = fingerprint_artifact(
        merge_plan,
        logical_name="merge_plan",
        schema_version="2.0.0",
        root=tmp_path,
    )
    native = build_variant_manifest(
        config,
        base_manifest=base_manifest,
        graph_regime="native_lightrag",
    )
    advanced = build_variant_manifest(
        config,
        base_manifest=base_manifest,
        graph_regime="advanced_lightrag_er",
        merge_plan=merge_ref,
    )
    rr_plan = tmp_path / "rr_plan.jsonl"
    rr_results = tmp_path / "rr_results.jsonl"
    rr_plan.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    rr_results.write_text('{"chunk_id":"c1","status":"valid"}\n', encoding="utf-8")
    rr_plan_ref = fingerprint_artifact(
        rr_plan,
        logical_name="rr_plan",
        schema_version="1.0.0",
        root=tmp_path,
    )
    rr_results_ref = fingerprint_artifact(
        rr_results,
        logical_name="rr_results",
        schema_version="1.0.0",
        root=tmp_path,
    )
    rr = build_variant_manifest(
        config,
        base_manifest=base_manifest,
        graph_regime="advanced_lightrag_er_rr",
        merge_plan=merge_ref,
        rr_plan=rr_plan_ref,
        rr_results=rr_results_ref,
    )
    downstream_changed_base = build_base_manifest(
        changed,
        "qwen35_0_8b",
        artifacts={"normalized_entities": extraction_ref},
    )
    downstream_changed_native = build_variant_manifest(
        changed,
        base_manifest=downstream_changed_base,
        graph_regime="native_lightrag",
    )
    downstream_changed_advanced = build_variant_manifest(
        changed,
        base_manifest=downstream_changed_base,
        graph_regime="advanced_lightrag_er",
        merge_plan=merge_ref,
    )
    assert downstream_changed_base == base_manifest
    assert downstream_changed_native == native
    assert downstream_changed_advanced == advanced
    assert native.base_extraction_sha256 == advanced.base_extraction_sha256
    assert rr.base_extraction_sha256 == advanced.base_extraction_sha256
    assert advanced.merge_plan_sha256 == merge_ref.sha256
    assert advanced.judge_tag == config.roles.er_judge.requested_tag
    assert advanced.judge_digest == config.roles.er_judge.digest
    assert advanced.judge_config_sha256
    assert native.embedding_resolved_name == config.roles.embedding.resolved_name
    assert native.embedding_digest == config.roles.embedding.digest
    assert native.embedding_dimension == config.roles.embedding.dimension
    assert native.embedding_max_tokens == config.roles.embedding.max_tokens
    assert native.embedding_config_sha256 == advanced.embedding_config_sha256
    assert native.graph_config_sha256 == advanced.graph_config_sha256
    assert rr.graph_config_sha256 == advanced.graph_config_sha256
    assert rr.rr_plan_sha256 == rr_plan_ref.sha256
    assert rr.rr_results_sha256 == rr_results_ref.sha256
    assert rr.rr_verifier_digest == config.roles.rr_verifier.digest


def test_workspace_tree_hash_detects_any_native_mutation(tmp_path) -> None:
    workspace = tmp_path / "native"
    workspace.mkdir()
    (workspace / "graph.graphml").write_text("original", encoding="utf-8")
    before = sha256_directory(workspace)
    assert sha256_directory(workspace) == before
    (workspace / "graph.graphml").write_text("mutated", encoding="utf-8")
    assert sha256_directory(workspace) != before


def test_base_extraction_hash_is_content_addressed_not_path_addressed(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for directory in (left, right):
        (directory / "entities.jsonl").write_text(
            '{"mention_id":"m1"}\n', encoding="utf-8"
        )
    left_ref = fingerprint_artifact(
        left / "entities.jsonl",
        logical_name="normalized_entities",
        schema_version="2.0.0",
        root=left,
    )
    right_ref = fingerprint_artifact(
        right / "entities.jsonl",
        logical_name="normalized_entities",
        schema_version="2.0.0",
        root=None,
    )
    assert left_ref.path != right_ref.path
    assert compute_artifact_set_sha256("base_1", {"entities": left_ref}) == (
        compute_artifact_set_sha256("base_1", {"entities": right_ref})
    )
