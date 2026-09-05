from __future__ import annotations

import pytest

from src.evaluation.downstream import (
    aggregate_downstream_metrics,
    compute_downstream_metrics,
    compute_question_downstream_metrics,
)


def _lineage(question_id: str) -> dict[str, str]:
    return {
        "question_id": question_id,
        "base_run_id": "base-1",
        "variant_run_id": "variant-native",
        "base_extraction_sha256": "extraction-hash",
        "builder_model": "builder:tag",
        "graph_regime": "native_lightrag",
        "retrieval_result_id": f"ret-{question_id}",
    }


def test_answerable_question_retrieval_and_exact_answer_metrics() -> None:
    question = {
        "question_id": "q1",
        "question_type": "multi_hop",
        "answerable": True,
        "gold_answer": "Donald Trump",
        "gold_document_ids": ["doc-a", "doc-b"],
    }
    retrieval = {
        **_lineage("q1"),
        "answerable": True,
        "retrieved_document_ids": [
            "hard-1",
            "hard-2",
            "hard-3",
            "hard-4",
            "doc-b",
            "doc-a",
        ],
        "retrieved_entities": [{"id": "entity"}],
        "retrieved_relationships": [{"id": "relation"}],
        "retrieved_chunks": [{"id": "chunk"}],
        "references": [{"id": "reference"}],
        "retrieval_outcome": "context",
        "latency_ms": 12.5,
        "context_token_count": 40,
    }
    answer = {
        **_lineage("q1"),
        "answer_result_id": "ans-q1",
        "answerable": True,
        "gold_answer": "Donald Trump",
        "raw_answer": "Donald Trump",
        "normalized_answer": "donald   trump",
        "latency_ms": 20.0,
        "input_tokens": 50,
        "output_tokens": 2,
    }

    metrics = compute_question_downstream_metrics(question, retrieval, answer)

    assert metrics["retrieval_hit"] is True
    assert metrics["retrieval_recall"] == 1.0
    assert metrics["retrieval_recall_at_5"] == 0.5
    assert metrics["retrieval_recall_at_10"] == 1.0
    assert metrics["complete_chain_recall_at_5"] is False
    assert metrics["complete_chain_recall_at_10"] is True
    assert metrics["answer_exact_match"] is True
    assert metrics["token_f1"] == 1.0
    assert metrics["unanswerable_correct"] is None
    assert metrics["answer_correct"] is True
    assert metrics["base_run_id"] == "base-1"
    assert metrics["retrieval_result_id"] == "ret-q1"
    assert metrics["answer_result_id"] == "ans-q1"
    assert metrics["answer_total_tokens"] == 52
    assert metrics["retrieved_entity_count"] == 1


def test_unanswerable_correctness_uses_exact_abstention_token() -> None:
    question = {
        "question_id": "q2",
        "question_type": "unanswerable",
        "answerable": False,
        "gold_answer": "",
        "gold_document_ids": [],
    }
    retrieval = {
        **_lineage("q2"),
        "answerable": False,
        "retrieved_document_ids": [],
    }
    answer = {
        **_lineage("q2"),
        "answer_result_id": "ans-q2",
        "answerable": False,
        "raw_answer": "INSUFFICIENT_INFORMATION",
        "normalized_answer": "insufficient_information",
    }

    metrics = compute_question_downstream_metrics(question, retrieval, answer)

    assert metrics["retrieval_hit"] is None
    assert metrics["retrieval_recall"] is None
    assert metrics["answer_exact_match"] is None
    assert metrics["unanswerable_correct"] is True
    assert metrics["answer_correct"] is True
    assert metrics["hallucination"] is False


def test_downstream_aggregation_reports_null_metrics_and_question_types() -> None:
    rows = [
        {
            "question_type": "inference",
            "answerable": True,
            "retrieval_hit": True,
            "retrieval_recall": 0.5,
            "retrieval_empty": False,
            "answer_exact_match": True,
            "token_precision": 1.0,
            "token_recall": 1.0,
            "token_f1": 1.0,
            "answer_correct": True,
            "predicted_unanswerable": False,
            "retrieved_entity_count": 2,
            "retrieved_relationship_count": 1,
            "retrieved_chunk_count": 3,
            "retrieval_latency_ms": 10,
            "context_token_count": 100,
            "answer_latency_ms": 20,
            "answer_input_tokens": 120,
            "answer_output_tokens": 2,
            "answer_total_tokens": 122,
            **{
                f"{name}_{cutoff}": value
                for cutoff in (5, 10, 20)
                for name, value in (
                    ("retrieval_hit_at", True),
                    ("retrieval_recall_at", 0.5),
                    ("complete_chain_recall_at", False),
                )
            },
        },
        {
            "question_type": "unanswerable",
            "answerable": False,
            "retrieval_hit": None,
            "retrieval_recall": None,
            "retrieval_empty": True,
            "answer_exact_match": None,
            "token_precision": None,
            "token_recall": None,
            "token_f1": None,
            "answer_correct": False,
            "predicted_unanswerable": False,
            "retrieved_entity_count": 0,
            "retrieved_relationship_count": 0,
            "retrieved_chunk_count": 0,
            "retrieval_latency_ms": 5,
            "context_token_count": 0,
            "answer_latency_ms": 10,
            "answer_input_tokens": 20,
            "answer_output_tokens": 3,
            "answer_total_tokens": 23,
            **{
                f"{name}_{cutoff}": None
                for cutoff in (5, 10, 20)
                for name in (
                    "retrieval_hit_at",
                    "retrieval_recall_at",
                    "complete_chain_recall_at",
                )
            },
        },
    ]

    summary = aggregate_downstream_metrics(rows)

    assert summary["unanswerable_precision"] is None
    assert summary["unanswerable_recall"] == 0.0
    assert summary["unanswerable_f1"] is None
    assert summary["hallucination_rate"] == 1.0
    assert summary["over_abstention_rate"] == 0.0
    assert summary["retrieval_latency_ms_total"] == 15
    assert summary["answer_total_tokens"] == 145
    assert summary["by_question_type"]["inference"]["token_f1"] == 1.0
    assert summary["by_question_type"]["unanswerable"]["hallucination_rate"] == 1.0


def test_downstream_batch_requires_exact_question_sets_and_lineage() -> None:
    question = {
        "question_id": "q1",
        "answerable": True,
        "gold_answer": "answer",
        "gold_document_ids": ["doc"],
    }
    retrieval = {
        **_lineage("q1"),
        "answerable": True,
        "retrieved_document_ids": ["doc"],
    }
    answer = {
        **_lineage("q1"),
        "answer_result_id": "ans-q1",
        "answerable": True,
        "gold_answer": "answer",
        "raw_answer": "answer",
    }

    assert len(compute_downstream_metrics([question], [retrieval], [answer])) == 1
    with pytest.raises(ValueError, match="ID sets differ"):
        compute_downstream_metrics([question], [retrieval], [])

    answer_without_variant = {**answer, "variant_run_id": ""}
    with pytest.raises(ValueError, match="required lineage IDs"):
        compute_question_downstream_metrics(question, retrieval, answer_without_variant)


def test_ranked_retrieval_produces_first_rank_and_mrr() -> None:
    question = {
        "question_id": "q-rank",
        "question_type": "inference",
        "answerable": True,
        "gold_answer": "answer",
        "gold_document_ids": ["gold"],
    }
    retrieval = {
        **_lineage("q-rank"),
        "answerable": True,
        "retrieval_outcome": "context",
        "retrieved_document_ids": ["negative", "gold"],
        "ranked_retrieval_items": [
            {"rank": 1, "document_id": "negative"},
            {"rank": 2, "document_id": "gold"},
        ],
    }
    answer = {
        **_lineage("q-rank"),
        "answer_result_id": "ans-rank",
        "answerable": True,
        "gold_answer": "answer",
        "answer_outcome": "success",
        "raw_answer": "answer",
    }

    metrics = compute_question_downstream_metrics(question, retrieval, answer)

    assert metrics["first_relevant_rank"] == 2
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["reciprocal_rank_at_5"] == 0.5


def test_failures_are_zero_in_primary_and_excluded_from_complete_case() -> None:
    question = {
        "question_id": "q-failed",
        "question_type": "inference",
        "answerable": True,
        "gold_answer": "answer",
        "gold_document_ids": ["gold"],
    }
    retrieval = {
        **_lineage("q-failed"),
        "answerable": True,
        "retrieval_outcome": "failure",
        "retrieved_document_ids": [],
        "latency_ms": 7,
    }
    answer = {
        **_lineage("q-failed"),
        "answer_result_id": "ans-failed",
        "answerable": True,
        "gold_answer": "answer",
        "answer_outcome": "skipped_upstream_failure",
        "raw_answer": "",
    }

    row = compute_question_downstream_metrics(question, retrieval, answer)
    summary = aggregate_downstream_metrics([row])

    assert row["retrieval_hit"] is False
    assert row["reciprocal_rank"] == 0.0
    assert row["token_f1"] == 0.0
    assert row["answer_correct"] is False
    assert row["hallucination"] is None
    assert row["complete_case"] is False
    assert summary["analyses"]["primary"]["micro"]["answer_correct"] == 0.0
    assert summary["analyses"]["secondary"]["micro"] is None
    assert summary["analyses"]["secondary"]["excluded_failure_count"] == 1


def test_macro_source_weighted_and_latency_quantiles_are_reported() -> None:
    rows = []
    for question_type, score, latency in (
        ("inference", 1.0, 10.0),
        ("comparison", 0.0, 20.0),
        ("temporal", 0.0, 30.0),
        ("unanswerable", 1.0, 40.0),
    ):
        rows.append(
            {
                "question_type": question_type,
                "answerable": question_type != "unanswerable",
                "answer_correct": score,
                "complete_case": True,
                "retrieval_latency_ms": latency,
            }
        )

    summary = aggregate_downstream_metrics(rows)

    assert summary["analyses"]["primary"]["micro"]["answer_correct"] == 0.5
    assert summary["analyses"]["primary"]["macro"]["answer_correct"] == 0.5
    expected = (816 + 301) / 2556
    assert summary["analyses"]["primary"]["source_weighted"][
        "answer_correct"
    ] == pytest.approx(expected)
    assert summary["retrieval_latency_ms_median"] == 25.0
    assert summary["retrieval_latency_ms_p90"] == pytest.approx(37.0)
