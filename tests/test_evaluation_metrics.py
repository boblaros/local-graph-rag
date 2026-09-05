import pytest

from src.evaluation.er_metrics import compute_er_metrics, compute_graph_topology
from src.evaluation.extraction_metrics import (
    bootstrap_extraction_metrics,
    compute_extraction_efficiency,
    compute_extraction_metrics,
)
from src.evaluation.paired import (
    paired_regime_deltas,
    paired_variant_deltas,
    summarize_paired_deltas,
    summarize_regime_deltas,
)
from src.evaluation.primary import (
    CASCADE_COMPARISON_PLAN,
    PRIMARY_ANALYSIS_SCHEMA_VERSION,
    collect_prespecified_cascade_comparisons,
)


def test_extraction_metrics_keep_coverage_dimensions_separate() -> None:
    metrics = compute_extraction_metrics(
        [
            {
                "entity_present": True,
                "relations_present": False,
                "entities": [{"description_present": False}],
                "relations": [],
                "parse_records": [{"status": "recovered"}],
            },
            {
                "entity_present": False,
                "relations_present": False,
                "entities": [],
                "relations": [],
                "parse_records": [{"status": "strict"}],
            },
        ]
    )
    assert metrics["entity_extraction_coverage"] == 0.5
    assert metrics["relation_extraction_coverage"] == 0.0
    assert metrics["description_completeness"] == 0.0
    assert metrics["json_recovery_rate"] == 0.5


def test_extraction_efficiency_deduplicates_physical_attempt_exports() -> None:
    metrics = compute_extraction_efficiency(
        [
            {
                "call_id": "call-a",
                "attempt_number": 1,
                "latency_ms": 10,
                "input_tokens": 4,
                "output_tokens": 2,
                "gleaning_round": 0,
            },
            {
                "call_id": "call-a",
                "attempt_number": 1,
                "latency_ms": 10,
                "input_tokens": 4,
                "output_tokens": 2,
                "gleaning_round": 0,
            },
            {
                "call_id": "call-a",
                "attempt_number": 2,
                "latency_ms": 20,
                "input_tokens": 4,
                "output_tokens": 3,
                "gleaning_round": 0,
                "error_type": "RetryableError",
            },
            {
                "call_id": "call-b",
                "attempt_number": 1,
                "latency_ms": 5,
                "input_tokens": 3,
                "output_tokens": 1,
                "gleaning_round": 1,
                "technical_provenance": {"cache_hit": True},
            },
        ]
    )

    assert metrics["physical_call_count"] == 3
    assert metrics["logical_call_count"] == 2
    assert metrics["retry_count"] == 1
    assert metrics["gleaning_call_count"] == 1
    assert metrics["cache_hit_count"] == 1
    assert metrics["latency_ms_total"] == 35
    assert metrics["total_tokens"] == 17


def test_extraction_bootstrap_resamples_whole_documents_deterministically() -> None:
    chunks = [
        {
            "document_id": "d1",
            "token_count": 100,
            "entity_present": True,
            "relations_present": True,
            "entities": [{"description_present": True}],
            "relations": [{"description_present": True}],
            "parse_records": [{"status": "strict"}],
        },
        {
            "document_id": "d1",
            "token_count": 100,
            "entity_present": True,
            "relations_present": False,
            "entities": [{"description_present": False}],
            "relations": [],
            "parse_records": [{"status": "recovered"}],
        },
        {
            "document_id": "d2",
            "token_count": 100,
            "entity_present": False,
            "relations_present": False,
            "entities": [],
            "relations": [],
            "parse_records": [{"status": "strict"}],
        },
    ]
    calls = [
        {
            "document_id": "d1",
            "call_id": "c1",
            "attempt_number": 1,
            "json_valid": True,
            "schema_valid": True,
            "parse_success": True,
            "latency_ms": 10,
        },
        {
            "document_id": "d2",
            "call_id": "c2",
            "attempt_number": 1,
            "json_valid": False,
            "schema_valid": False,
            "parse_success": False,
            "latency_ms": 20,
        },
    ]

    first = bootstrap_extraction_metrics(chunks, calls, bootstrap_samples=200, seed=9)
    second = bootstrap_extraction_metrics(chunks, calls, bootstrap_samples=200, seed=9)

    assert first == second
    assert first["cluster_unit"] == "document_id"
    assert first["document_count"] == 2
    assert first["point_estimates"]["entity_extraction_coverage"] == 2 / 3
    assert (
        first["confidence_intervals"]["entity_extraction_coverage"]["replicate_count"]
        == 200
    )


def test_er_metrics_and_paired_lineage() -> None:
    er = compute_er_metrics(
        mentions=[
            {"mention_id": "a", "original_name": "Apple"},
            {"mention_id": "b", "original_name": "Apple Inc."},
        ],
        canonical_entities=[
            {
                "canonical_entity_id": "c",
                "display_name": "Apple Inc.",
                "aliases": ["Apple"],
            }
        ],
        candidate_pairs=[{"left_mention_id": "a", "right_mention_id": "b"}],
        pair_decisions=[
            {
                "left_mention_id": "a",
                "right_mention_id": "b",
                "action": "merge",
                "source": "judge",
            }
        ],
        mention_to_canonical=[
            {"mention_id": "a", "canonical_entity_id": "c"},
            {"mention_id": "b", "canonical_entity_id": "c"},
        ],
        aliases=[
            {
                "canonical_entity_id": "c",
                "display_name": "Apple Inc.",
                "aliases": ["Apple"],
            }
        ],
        merge_plan={
            "payload": {
                "decision_counts": {"merge": 1, "reject": 0, "abstain": 0},
                "cannot_links": [],
                "blocked_merges": [],
            }
        },
        rewrite_summary={
            "deduplicated_relation_count": 2,
            "induced_self_loops_removed": 1,
            "input_relation_count": 4,
            "materialized_edge_count": 1,
        },
        gold_pairs=[("a", "b")],
        gold_clusters={"a": "gold", "b": "gold"},
        native_graph={
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "isolated"}],
            "edges": [{"source": "a", "target": "b"}],
        },
        er_graph={
            "nodes": [{"id": "c"}, {"id": "isolated"}],
            "edges": [{"source": "c", "target": "isolated"}],
        },
        runtime={
            "er_runtime_seconds": 1.25,
            "judge_calls": 1,
            "judge_cache_hits": 0,
            "judge_decisions": 1,
            "judge_prompt_tokens": 12,
            "judge_output_tokens": 3,
            "judge_tokens": 15,
            "judge_cached_original_prompt_tokens": 12,
            "judge_cached_original_output_tokens": 3,
            "judge_cost": 0.0,
            "judge_cost_basis": "local_ollama_no_api_charge",
        },
    )
    assert er["pair_f1"] == 1.0
    assert er["judge_rate"] == 1.0
    assert er["node_reduction"] == 1
    assert er["edge_deduplication"] == 2
    assert er["induced_self_loops_removed"] == 1
    assert er["alias_coverage"] == 1.0
    assert er["connected_components_pre"] == 2
    assert er["isolates_pre"] == 1
    assert er["largest_component_pre"] == 2
    assert er["average_degree_pre"] == 2 / 3
    assert er["normalized_density_pre"] == 1 / 3
    assert er["judge_cache_hits"] == 0
    assert er["judge_tokens"] == 15
    assert er["judge_cost_basis"] == "local_ollama_no_api_charge"

    common = {
        "question_id": "q",
        "base_run_id": "base",
        "builder_model": "builder",
        "base_extraction_sha256": "x",
        "retrieval_result_id": "r",
        "answer_result_id": "a",
    }
    rows = paired_variant_deltas(
        [
            {
                **common,
                "graph_regime": "native_lightrag",
                "variant_run_id": "n",
                "score": 0.25,
            }
        ],
        [
            {
                **common,
                "graph_regime": "advanced_lightrag_er",
                "variant_run_id": "e",
                "score": 0.75,
            }
        ],
        metric_fields=["score"],
    )
    assert rows[0]["deltas"]["score"] == 0.5

    summary = summarize_paired_deltas(
        rows,
        metric_fields=["score"],
        bootstrap_samples=100,
        confidence_level=0.95,
        seed=42,
    )
    score = summary["metrics"]["score"]
    assert score["paired_count"] == 1
    assert score["native_mean"] == 0.25
    assert score["er_mean"] == 0.75
    assert score["mean_delta"] == 0.5
    assert score["ci_lower"] == 0.5
    assert score["ci_upper"] == 0.5

    rr_rows = paired_regime_deltas(
        [
            {
                **common,
                "graph_regime": "advanced_lightrag_er",
                "variant_run_id": "e",
                "score": 0.75,
            }
        ],
        [
            {
                **common,
                "graph_regime": "advanced_lightrag_er_rr",
                "variant_run_id": "rr",
                "score": 1.0,
            }
        ],
        left_regime="advanced_lightrag_er",
        right_regime="advanced_lightrag_er_rr",
        metric_fields=["score"],
    )
    rr_summary = summarize_regime_deltas(
        rr_rows,
        left_regime="advanced_lightrag_er",
        right_regime="advanced_lightrag_er_rr",
        metric_fields=["score"],
        bootstrap_samples=10,
    )
    assert rr_rows[0]["deltas"]["score"] == 0.25
    assert rr_summary["metrics"]["score"]["left_mean"] == 0.75
    assert rr_summary["metrics"]["score"]["right_mean"] == 1.0
    assert rr_summary["effect_direction"] == (
        "advanced_lightrag_er_rr_minus_advanced_lightrag_er"
    )


def test_graph_topology_counts_components_isolates_and_largest() -> None:
    topology = compute_graph_topology(
        ["a", "b", "c", "d"],
        [("a", "b"), ("b", "c")],
    )

    assert topology["nodes"] == 4
    assert topology["edges"] == 2
    assert topology["connected_components"] == 2
    assert topology["isolates"] == 1
    assert topology["largest_component"] == 3
    assert topology["largest_component_size"] == 3
    assert topology["largest_component_share"] == 0.75
    assert topology["average_degree"] == 1.0
    assert topology["normalized_density"] == 1 / 3
    assert topology["self_loops"] == 0
    assert topology["node_provenance_coverage"] is None


def test_paired_bootstrap_is_deterministic_and_grouped_by_question_type() -> None:
    def row(question_id: str, regime: str, score: float, question_type: str):
        return {
            "question_id": question_id,
            "question_type": question_type,
            "answerable": True,
            "base_run_id": "base",
            "builder_model": "builder",
            "base_extraction_sha256": "x",
            "graph_regime": regime,
            "variant_run_id": f"{regime}-variant",
            "retrieval_result_id": f"{regime}-retrieval-{question_id}",
            "answer_result_id": f"{regime}-answer-{question_id}",
            "score": score,
        }

    paired = paired_variant_deltas(
        [
            row("q1", "native_lightrag", 0.0, "inference"),
            row("q2", "native_lightrag", 1.0, "temporal"),
        ],
        [
            row("q1", "advanced_lightrag_er", 1.0, "inference"),
            row("q2", "advanced_lightrag_er", 0.0, "temporal"),
        ],
        metric_fields=["score"],
    )
    first = summarize_paired_deltas(
        paired, metric_fields=["score"], bootstrap_samples=500, seed=7
    )
    second = summarize_paired_deltas(
        paired, metric_fields=["score"], bootstrap_samples=500, seed=7
    )

    assert first == second
    assert first["metrics"]["score"]["mean_delta"] == 0.0
    # Stratification keeps one inference and one temporal question in every draw.
    assert first["metrics"]["score"]["ci_lower"] == 0.0
    assert first["metrics"]["score"]["ci_upper"] == 0.0
    assert first["metrics"]["score"]["native_ci_lower"] == 0.5
    assert first["metrics"]["score"]["er_ci_upper"] == 0.5
    assert first["by_question_type"]["inference"]["score"]["mean_delta"] == 1.0
    assert first["by_question_type"]["temporal"]["score"]["mean_delta"] == -1.0


def test_paired_summary_reports_ite_complete_case_and_absolute_intervals() -> None:
    def metric_row(question_id, regime, question_type, score, failed=False):
        return {
            "question_id": question_id,
            "question_type": question_type,
            "answerable": question_type != "unanswerable",
            "base_run_id": "base",
            "builder_model": "builder",
            "base_extraction_sha256": "x",
            "graph_regime": regime,
            "variant_run_id": f"{regime}-variant",
            "retrieval_result_id": f"{regime}-r-{question_id}",
            "answer_result_id": f"{regime}-a-{question_id}",
            "answer_correct": score,
            "answer_failed": failed,
        }

    definitions = [
        ("q1", "inference", 1.0, 1.0, False),
        ("q2", "comparison", 0.0, 1.0, False),
        ("q3", "temporal", 1.0, 0.0, False),
        ("q4", "unanswerable", 1.0, 0.0, True),
    ]
    paired = paired_variant_deltas(
        [
            metric_row(question, "native_lightrag", kind, native)
            for question, kind, native, _er, _failed in definitions
        ],
        [
            metric_row(
                question,
                "advanced_lightrag_er",
                kind,
                er,
                failed,
            )
            for question, kind, _native, er, failed in definitions
        ],
        metric_fields=["answer_correct"],
    )
    summary = summarize_paired_deltas(
        paired,
        metric_fields=["answer_correct"],
        bootstrap_samples=100,
        seed=3,
    )

    metric = summary["analyses"]["primary"]["micro"]["answer_correct"]
    assert summary["bootstrap"]["method"] == (
        "paired_stratified_percentile_by_question_type"
    )
    assert metric["native_ci_lower"] is not None
    assert metric["er_ci_upper"] is not None
    assert metric["delta_ci_lower"] == metric["ci_lower"]
    assert summary["analyses"]["secondary"]["question_count"] == 3
    assert summary["analyses"]["secondary"]["excluded_failure_count"] == 1


def test_graph_topology_reports_provenance_coverage_and_self_loops() -> None:
    topology = compute_graph_topology(
        [
            {"id": "a", "source_chunk_ids": ["chunk-a"]},
            {"id": "b", "source_chunk_ids": []},
        ],
        [
            {"source": "a", "target": "b", "source_chunk_ids": ["chunk-a"]},
            {"source": "a", "target": "a", "provenance": []},
        ],
    )

    assert topology["average_degree"] == 2.0
    assert topology["normalized_density"] == 1.0
    assert topology["self_loops"] == 1
    assert topology["node_provenance_coverage"] == 0.5
    assert topology["edge_provenance_coverage"] == 0.5


def test_prespecified_analysis_collects_cascade_roles_for_exactly_12_builders() -> None:
    builders = [f"builder-{index:02d}" for index in range(12)]
    summaries = [
        {
            "builder_key": builder,
            "base_run_id": f"base-{builder}",
            "paired_question_count": 120,
            "comparison_plan": {
                role: {**declaration, "unit": "question_id", "builder_key": builder}
                for role, declaration in CASCADE_COMPARISON_PLAN.items()
            },
            "paired_comparisons": {
                role: {
                    "effect_direction": declaration["effect_direction"],
                    "analyses": {"primary": {}, "secondary": {}},
                }
                for role, declaration in CASCADE_COMPARISON_PLAN.items()
            },
        }
        for builder in builders
    ]

    report = collect_prespecified_cascade_comparisons(
        summaries,
        expected_builder_keys=builders,
        expected_question_count=120,
    )

    assert report["schema_version"] == PRIMARY_ANALYSIS_SCHEMA_VERSION == "2.0.0"
    assert report["primary_comparison_count"] == 12
    assert report["total_prespecified_comparison_count"] == 36
    assert report["comparison_roles"]["primary"]["effect_direction"] == (
        "advanced_lightrag_er_rr_minus_native_lightrag"
    )
    assert report["comparison_roles"]["secondary"]["effect_direction"] == (
        "advanced_lightrag_er_minus_native_lightrag"
    )
    assert report["comparison_roles"]["incremental_ablation"][
        "effect_direction"
    ] == "advanced_lightrag_er_rr_minus_advanced_lightrag_er"
    assert [
        item["builder_key"]
        for item in report["comparison_roles"]["primary"]["builders"]
    ] == builders
    with pytest.raises(ValueError, match="builder set differs"):
        collect_prespecified_cascade_comparisons(
            summaries[:-1],
            expected_builder_keys=builders,
            expected_question_count=120,
        )
