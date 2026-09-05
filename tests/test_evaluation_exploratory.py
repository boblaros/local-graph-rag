from __future__ import annotations

import pytest

from src.evaluation.exploratory import (
    builder_metadata_from_config,
    compute_exploratory_analysis,
    exact_mcnemar,
    holm_adjust,
    summarize_builder_artifacts,
)

from ._config_helpers import resolved_config


def _conditions():
    config = resolved_config(all_builders=True)
    question_types = ("inference", "comparison", "temporal", "unanswerable")
    conditions = {}
    for builder_index, builder in enumerate(config.builders):
        conditions[builder.key] = {}
        for regime_index, regime in enumerate(
            (
                "native_lightrag",
                "advanced_lightrag_er",
                "advanced_lightrag_er_rr",
            )
        ):
            rows = []
            for question_index, question_type in enumerate(question_types):
                score = float((builder_index + question_index + regime_index) % 2)
                rows.append(
                    {
                        "question_id": f"q{question_index}",
                        "question_type": question_type,
                        "answerable": question_type != "unanswerable",
                        "graph_regime": regime,
                        "builder_model": builder.resolved_name,
                        "base_run_id": f"base-{builder.key}",
                        "variant_run_id": f"variant-{builder.key}-{regime}",
                        "base_extraction_sha256": f"extraction-{builder.key}",
                        "retrieval_result_id": f"retrieval-{builder.key}-{regime}-q{question_index}",
                        "answer_result_id": f"answer-{builder.key}-{regime}-q{question_index}",
                        "answer_correct": score,
                        "token_f1": score / 2.0
                        if question_type != "unanswerable"
                        else None,
                        "retrieval_latency_ms": float(
                            100 - builder_index + regime_index
                        ),
                        "retrieval_failed": False,
                        "answer_failed": False,
                    }
                )
            conditions[builder.key][regime] = rows
    return config, conditions


def test_exact_mcnemar_and_holm_are_deterministic() -> None:
    result = exact_mcnemar(
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    )
    assert result["left_only_success"] == 4
    assert result["right_only_success"] == 0
    assert result["p_value"] == 0.125
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(ValueError, match="binary"):
        exact_mcnemar([0.5], [0.0])


def test_exploratory_analysis_builds_66_pairs_did_family_scale_and_holm() -> None:
    config, conditions = _conditions()
    metadata = builder_metadata_from_config(config.builders)

    first = compute_exploratory_analysis(
        conditions,
        builder_metadata=metadata,
        metric_fields=["answer_correct", "token_f1", "retrieval_latency_ms"],
        expected_question_count=4,
        bootstrap_samples=20,
        seed=7,
    )
    second = compute_exploratory_analysis(
        conditions,
        builder_metadata=metadata,
        metric_fields=["answer_correct", "token_f1", "retrieval_latency_ms"],
        expected_question_count=4,
        bootstrap_samples=20,
        seed=7,
    )

    assert first == second
    report = first["global_report"]
    assert report["model_pair_count"] == 66
    assert report["schema_version"] == "2.0.0"
    assert report["within_regime_comparison_count"] == 66 * 2 * 3 * 3
    assert report["difference_in_differences_count"] == 66 * 2 * 3 * 3
    assert set(report["comparison_plan"]) == {
        "primary",
        "secondary",
        "incremental_ablation",
    }
    assert set(report["family_and_scale"]["by_family"]) == {
        "gemma3",
        "qwen3",
        "qwen3.5",
    }
    assert len(report["family_and_scale"]["builder_metrics"]) == 12
    first_builder_metrics = report["family_and_scale"]["builder_metrics"][0][
        "metrics"
    ]["answer_correct"]
    assert set(first_builder_metrics) == {
        "native",
        "er",
        "er_rr",
        "primary_effect",
        "secondary_effect",
        "incremental_ablation_effect",
        "question_counts",
    }
    assert report["mcnemar"]["families"]
    assert {
        family["test_count"] for family in report["mcnemar"]["families"].values()
    } == {66}
    pair = first["pairwise_comparisons"][0]
    assert pair["effect_direction"] == "right_minus_left"
    assert pair["left_wins"] + pair["ties"] + pair["right_wins"] == 4
    assert pair["mcnemar"]["holm_family_size"] == 66
    latency_pair = next(
        row
        for row in first["pairwise_comparisons"]
        if row["estimand"] == "intention_to_evaluate"
        and row["graph_regime"] == "native_lightrag"
        and row["metric"] == "retrieval_latency_ms"
        and row["left_builder"] == config.builders[0].key
        and row["right_builder"] == config.builders[1].key
    )
    assert latency_pair["metric_direction"] == "lower_is_better"
    assert latency_pair["right_wins"] == 4
    did = first["difference_in_differences"][0]
    assert did["contrast_role"] == "primary"
    assert did["comparison_key"] == "er_rr_minus_native"
    assert did["effect_direction"] == (
        "right_builder_contrast_minus_left_builder_contrast"
    )
    assert did["difference_ci_lower"] is not None
    assert {
        row["contrast_role"] for row in first["difference_in_differences"]
    } == {"primary", "secondary", "incremental_ablation"}


def test_exploratory_analysis_fails_closed_on_question_or_builder_mismatch() -> None:
    config, conditions = _conditions()
    metadata = builder_metadata_from_config(config.builders)
    conditions[config.builders[0].key]["native_lightrag"] = conditions[
        config.builders[0].key
    ]["native_lightrag"][:-1]

    with pytest.raises(ValueError, match="questions, expected"):
        compute_exploratory_analysis(
            conditions,
            builder_metadata=metadata,
            metric_fields=["answer_correct"],
            expected_question_count=4,
            bootstrap_samples=10,
        )

    _config, conditions = _conditions()
    conditions[config.builders[0].key]["native_lightrag"][0]["builder_model"] = (
        "wrong:model"
    )
    with pytest.raises(ValueError, match="builder model lineage mismatch"):
        compute_exploratory_analysis(
            conditions,
            builder_metadata=metadata,
            metric_fields=["answer_correct"],
            expected_question_count=4,
            bootstrap_samples=10,
        )

    _config, conditions = _conditions()
    del conditions[config.builders[0].key]["advanced_lightrag_er_rr"]
    with pytest.raises(ValueError, match="exactly three regimes"):
        compute_exploratory_analysis(
            conditions,
            builder_metadata=metadata,
            metric_fields=["answer_correct"],
            expected_question_count=4,
            bootstrap_samples=10,
        )


def test_builder_scale_is_parsed_from_frozen_display_name() -> None:
    metadata = builder_metadata_from_config(resolved_config(all_builders=True).builders)
    by_key = {row["builder_key"]: row for row in metadata}
    assert by_key["gemma3_270m"]["parameter_billions"] == 0.27
    assert by_key["qwen3_0_6b"]["scale_band"] == "sub_1b"
    assert by_key["gemma3_12b"]["scale_band"] == "8b_plus"


def test_builder_artifact_family_summary_uses_graph_as_descriptive_unit() -> None:
    metadata = builder_metadata_from_config(resolved_config(all_builders=True).builders)
    summaries = {
        row["builder_key"]: {
            "extraction": {
                "entity_extraction_coverage": 0.5,
                "efficiency": {"call_failure_rate": 0.0},
                "document_cluster_bootstrap": {"bootstrap_samples": 10_000},
            },
            "entity_resolution": {"node_reduction": 2},
            "native_topology": {"nodes": 10, "average_degree": 1.0},
            "er_topology": {"nodes": 8, "average_degree": 1.5},
            "er_rr_topology": {"nodes": 9, "average_degree": 1.7},
        }
        for row in metadata
    }

    result = summarize_builder_artifacts(
        summaries,
        builder_metadata=metadata,
    )

    assert result["by_family"]["qwen3"]["native_topology.nodes"]["model_count"] == 4
    metrics = result["builder_metrics"][0]["metrics"]
    assert metrics["primary_topology_effect.nodes"] == -1
    assert metrics["secondary_topology_effect.nodes"] == -2
    assert metrics["incremental_ablation_topology_effect.nodes"] == 1
    assert (
        "extraction.document_cluster_bootstrap.bootstrap_samples"
        not in result["builder_metrics"][0]["metrics"]
    )
    assert result["inference"].startswith("descriptive_only")
