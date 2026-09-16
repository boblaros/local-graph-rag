"""Question-level retrieval and answer metrics with strict run lineage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
from typing import Any

from .answer_normalization import (
    canonicalize_unanswerable,
    normalize_answer,
    normalized_exact_match,
    token_scores,
)


DEFAULT_UNANSWERABLE_TOKEN = "INSUFFICIENT_INFORMATION"
COMPLETE_CHAIN_CUTOFF = 5
DOWNSTREAM_METRICS_SCHEMA_VERSION = "4.0.0"
DEFAULT_PAIRED_METRIC_FIELDS = (
    "retrieval_hit",
    "retrieval_recall",
    "reciprocal_rank",
    "complete_chain_recall_at_5",
    "answer_correct",
    "token_f1",
    "hallucination",
    "over_abstention",
)
LINEAGE_FIELDS = (
    "question_id",
    "base_run_id",
    "variant_run_id",
    "base_extraction_sha256",
    "builder_model",
    "graph_regime",
    "retrieval_result_id",
    "answer_result_id",
)


def _row(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(
        f"downstream metric row must be mapping-like, got {type(value).__name__}"
    )


def _required(row: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if not str(row.get(field) or "").strip()]
    if missing:
        raise ValueError(f"{label} is missing required lineage IDs: {missing}")


def _string_set(value: Any, field: str) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"{field} must be a sequence of document IDs")
    return {str(item).strip() for item in value if str(item).strip()}


def _ordered_strings(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"{field} must be a sequence of document IDs")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _optional_number(row: Mapping[str, Any], field: str) -> int | float | None:
    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric when present")
    return value


def compute_question_downstream_metrics(
    question: Mapping[str, Any] | Any,
    retrieval_record: Mapping[str, Any] | Any,
    answer_record: Mapping[str, Any] | Any,
    *,
    unanswerable_token: str = DEFAULT_UNANSWERABLE_TOKEN,
) -> dict[str, Any]:
    """Score one retrieval/answer pair and retain every required lineage ID."""

    question_row = _row(question)
    retrieval = _row(retrieval_record)
    answer = _row(answer_record)
    _required(question_row, ("question_id",), "question")
    _required(retrieval, LINEAGE_FIELDS[:-1], "retrieval record")
    _required(answer, LINEAGE_FIELDS, "answer record")

    for field in LINEAGE_FIELDS[:-1]:
        if str(retrieval[field]) != str(answer[field]):
            raise ValueError(f"retrieval/answer lineage mismatch for {field}")
    if str(question_row["question_id"]) != str(retrieval["question_id"]):
        raise ValueError("question/retrieval question_id mismatch")
    if "answerable" not in question_row:
        raise ValueError("question must declare answerable")
    answerable = bool(question_row["answerable"])
    for label, row in (("retrieval", retrieval), ("answer", answer)):
        if "answerable" not in row or bool(row["answerable"]) != answerable:
            raise ValueError(f"{label} answerable flag does not match question")

    gold_documents = _string_set(
        question_row.get("gold_document_ids"), "gold_document_ids"
    )
    explicit_ranks = retrieval.get("ranked_retrieval_items")
    if explicit_ranks:
        if isinstance(explicit_ranks, (str, bytes)) or not isinstance(
            explicit_ranks, Iterable
        ):
            raise ValueError("ranked_retrieval_items must be a sequence")
        ranked_rows = [_row(item) for item in explicit_ranks]
        if [item.get("rank") for item in ranked_rows] != list(
            range(1, len(ranked_rows) + 1)
        ):
            raise ValueError("ranked retrieval ranks must be contiguous")
        ranked_documents = _ordered_strings(
            [item.get("document_id") for item in ranked_rows],
            "ranked_retrieval_items.document_id",
        )
        legacy_ranks = _ordered_strings(
            retrieval.get("retrieved_document_ids"), "retrieved_document_ids"
        )
        if legacy_ranks and legacy_ranks != ranked_documents:
            raise ValueError("explicit and legacy retrieval ranks disagree")
    else:
        ranked_documents = _ordered_strings(
            retrieval.get("retrieved_document_ids"), "retrieved_document_ids"
        )
    retrieved_documents = set(ranked_documents)
    retrieved_gold = gold_documents & retrieved_documents
    retrieval_outcome = str(retrieval.get("retrieval_outcome") or "context")
    retrieval_failed = retrieval_outcome == "failure"
    if retrieval_outcome not in {"context", "empty", "failure"}:
        raise ValueError(f"unknown retrieval outcome: {retrieval_outcome}")
    if retrieval_failed and ranked_documents:
        raise ValueError("failed retrieval cannot contain ranked documents")
    if answerable and gold_documents:
        retrieval_hit: bool | None = bool(retrieved_gold)
        retrieval_recall: float | None = len(retrieved_gold) / len(gold_documents)
        first_relevant_rank = next(
            (
                rank
                for rank, document_id in enumerate(ranked_documents, 1)
                if document_id in gold_documents
            ),
            None,
        )
        reciprocal_rank: float | None = (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        )
    else:
        retrieval_hit = None
        retrieval_recall = None
        first_relevant_rank = None
        reciprocal_rank = None

    raw_prediction = answer.get("raw_answer") or answer.get("normalized_answer")
    answer_outcome = str(answer.get("answer_outcome") or "success")
    answer_failed = answer_outcome != "success"
    if answer_outcome not in {"success", "failure", "skipped_upstream_failure"}:
        raise ValueError(f"unknown answer outcome: {answer_outcome}")
    if not answer_failed and not str(raw_prediction or "").strip():
        raise ValueError("answer record has no answer text")
    raw_gold = question_row.get("gold_answer") or question_row.get("answer")
    gold = normalize_answer(raw_gold)
    if answerable and not str(raw_gold or "").strip():
        raise ValueError("answerable question has no gold answer")
    recorded_gold = normalize_answer(answer.get("gold_answer"))
    if (
        answerable
        and str(answer.get("gold_answer") or "").strip()
        and recorded_gold != gold
    ):
        raise ValueError("answer record gold_answer does not match question")

    predicted = normalize_answer(raw_prediction)
    predicted_unanswerable = (
        None
        if answer_failed
        else canonicalize_unanswerable(raw_prediction) is not None
        or predicted == normalize_answer(unanswerable_token)
    )
    # Intention-to-evaluate: an operational failure is a downstream miss, while
    # latency/tokens and semantic error categories remain explicitly unavailable.
    exact_match = (
        False
        if answerable and answer_failed
        else normalized_exact_match(raw_prediction, raw_gold)
        if answerable
        else None
    )
    scores = (
        token_scores(raw_prediction, raw_gold)
        if answerable and not answer_failed
        else None
    )
    unanswerable_correct = (
        False
        if not answerable and answer_failed
        else predicted_unanswerable
        if not answerable
        else None
    )
    answer_correct = exact_match if answerable else unanswerable_correct
    result = {
        "schema_version": DOWNSTREAM_METRICS_SCHEMA_VERSION,
        **{field: answer[field] for field in LINEAGE_FIELDS},
        "question_type": str(question_row.get("question_type") or "unknown"),
        "answerable": answerable,
        "gold_document_count": len(gold_documents),
        "retrieved_document_count": len(retrieved_documents),
        "retrieved_gold_document_count": len(retrieved_gold),
        "retrieval_outcome": retrieval_outcome,
        "retrieval_failed": retrieval_failed,
        "retrieval_hit": retrieval_hit,
        "retrieval_recall": retrieval_recall,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": reciprocal_rank,
        "answer_outcome": answer_outcome,
        "answer_failed": answer_failed,
        "answer_exact_match": exact_match,
        "token_precision": (
            scores.precision if scores is not None else 0.0 if answerable else None
        ),
        "token_recall": (
            scores.recall if scores is not None else 0.0 if answerable else None
        ),
        "token_f1": scores.f1 if scores is not None else 0.0 if answerable else None,
        "predicted_unanswerable": predicted_unanswerable,
        "unanswerable_correct": unanswerable_correct,
        "answer_correct": answer_correct,
        "hallucination": (
            (not predicted_unanswerable)
            if not answerable and not answer_failed
            else None
        ),
        "over_abstention": (
            predicted_unanswerable if answerable and not answer_failed else None
        ),
        "retrieval_empty": retrieval_outcome == "empty",
        "retrieved_entity_count": len(retrieval.get("retrieved_entities") or []),
        "retrieved_relationship_count": len(
            retrieval.get("retrieved_relationships") or []
        ),
        "retrieved_chunk_count": len(retrieval.get("retrieved_chunks") or []),
        "reference_count": len(retrieval.get("references") or []),
        "retrieval_latency_ms": _optional_number(retrieval, "latency_ms"),
        "context_token_count": _optional_number(retrieval, "context_token_count"),
        "answer_latency_ms": _optional_number(answer, "latency_ms"),
        "answer_input_tokens": _optional_number(answer, "input_tokens"),
        "answer_output_tokens": _optional_number(answer, "output_tokens"),
    }
    input_tokens = result["answer_input_tokens"]
    output_tokens = result["answer_output_tokens"]
    result["answer_total_tokens"] = (
        int(input_tokens) + int(output_tokens)
        if input_tokens is not None and output_tokens is not None
        else None
    )
    result["complete_chain_recall_at_5"] = (
        gold_documents <= set(ranked_documents[:COMPLETE_CHAIN_CUTOFF])
        if answerable and gold_documents
        else None
    )
    return result


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _sum(rows: Sequence[Mapping[str, Any]], field: str) -> float | int | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    return sum(values) if values else None


def _quantile(
    rows: Sequence[Mapping[str, Any]], field: str, probability: float
) -> float | None:
    values = sorted(float(row[field]) for row in rows if row.get(field) is not None)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _metric_denominator(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    applicable = [row for row in rows if row.get(field) is not None]
    if field.startswith("retrieval") or field in {"reciprocal_rank"}:

        def failure(row: Mapping[str, Any]) -> bool:
            return bool(row.get("retrieval_failed"))

    elif field.startswith("answer") or field in {
        "token_f1",
        "unanswerable_correct",
        "hallucination",
        "over_abstention",
    }:

        def failure(row: Mapping[str, Any]) -> bool:
            return bool(row.get("answer_failed"))

    else:

        def failure(row: Mapping[str, Any]) -> bool:
            return bool(row.get("retrieval_failed") or row.get("answer_failed"))

    return {
        "expected_count": len(rows),
        "observed_count": len(applicable),
        "missing_count": len(rows) - len(applicable),
        "failure_count": sum(failure(row) for row in applicable),
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    answerable = [row for row in rows if bool(row.get("answerable"))]
    unanswerable = [row for row in rows if not bool(row.get("answerable"))]
    false_negative = len(unanswerable) - sum(
        bool(row.get("predicted_unanswerable")) for row in unanswerable
    )
    false_positive = sum(bool(row.get("predicted_unanswerable")) for row in answerable)
    result: dict[str, Any] = {
        "question_count": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "retrieval_failure_count": sum(
            bool(row.get("retrieval_failed")) for row in rows
        ),
        "answer_failure_count": sum(bool(row.get("answer_failed")) for row in rows),
        "retrieval_hit": _mean(rows, "retrieval_hit"),
        "retrieval_recall": _mean(rows, "retrieval_recall"),
        "mrr": _mean(rows, "reciprocal_rank"),
        "empty_retrieval_rate": _mean(rows, "retrieval_empty"),
        "token_f1": _mean(rows, "token_f1"),
        "answer_correct": _mean(rows, "answer_correct"),
        "hallucination_count": false_negative,
        "hallucination_rate": (
            false_negative / len(unanswerable) if unanswerable else None
        ),
        "over_abstention_count": false_positive,
        "over_abstention_rate": (
            false_positive / len(answerable) if answerable else None
        ),
        "mean_retrieved_entity_count": _mean(rows, "retrieved_entity_count"),
        "mean_retrieved_relationship_count": _mean(
            rows, "retrieved_relationship_count"
        ),
        "mean_retrieved_chunk_count": _mean(rows, "retrieved_chunk_count"),
        "retrieval_latency_ms_total": _sum(rows, "retrieval_latency_ms"),
        "retrieval_latency_ms_mean": _mean(rows, "retrieval_latency_ms"),
        "context_tokens_total": _sum(rows, "context_token_count"),
        "context_tokens_mean": _mean(rows, "context_token_count"),
        "answer_latency_ms_total": _sum(rows, "answer_latency_ms"),
        "answer_latency_ms_mean": _mean(rows, "answer_latency_ms"),
        "answer_input_tokens_total": _sum(rows, "answer_input_tokens"),
        "answer_output_tokens_total": _sum(rows, "answer_output_tokens"),
        "answer_total_tokens": _sum(rows, "answer_total_tokens"),
    }
    for prefix, field in (
        ("retrieval_latency_ms", "retrieval_latency_ms"),
        ("answer_latency_ms", "answer_latency_ms"),
    ):
        result[f"{prefix}_median"] = _quantile(rows, field, 0.50)
        result[f"{prefix}_p90"] = _quantile(rows, field, 0.90)
        result[f"{prefix}_p95"] = _quantile(rows, field, 0.95)
    result["complete_chain_recall_at_5"] = _mean(rows, "complete_chain_recall_at_5")
    denominator_fields = (
        "retrieval_hit",
        "retrieval_recall",
        "reciprocal_rank",
        "token_f1",
        "answer_correct",
        "unanswerable_correct",
        "hallucination",
        "over_abstention",
        "retrieval_latency_ms",
        "answer_latency_ms",
    )
    result["metric_denominators"] = {
        field: _metric_denominator(rows, field) for field in denominator_fields
    }
    return result


def aggregate_downstream_metrics(
    rows: Sequence[Mapping[str, Any] | Any],
) -> dict[str, Any]:
    """Micro estimates over all evaluated questions and within question types."""
    normalized = [_row(value) for value in rows]
    if not normalized:
        raise ValueError("downstream aggregation requires at least one row")
    kinds = sorted({str(row.get("question_type") or "unknown") for row in normalized})
    return {
        "schema_version": DOWNSTREAM_METRICS_SCHEMA_VERSION,
        "estimand": "intention_to_evaluate",
        "aggregation": "micro",
        **_aggregate_rows(normalized),
        "by_question_type": {
            kind: _aggregate_rows(
                [
                    row
                    for row in normalized
                    if str(row.get("question_type") or "unknown") == kind
                ]
            )
            for kind in kinds
        },
    }


def compute_downstream_metrics(
    questions: Sequence[Mapping[str, Any] | Any],
    retrieval_records: Sequence[Mapping[str, Any] | Any],
    answer_records: Sequence[Mapping[str, Any] | Any],
    *,
    unanswerable_token: str = DEFAULT_UNANSWERABLE_TOKEN,
) -> list[dict[str, Any]]:
    """Score an exact question/retrieval/answer snapshot, failing on omissions."""

    def index(
        values: Sequence[Mapping[str, Any] | Any], label: str
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for value in values:
            row = _row(value)
            question_id = str(row.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(f"{label} row has no question_id")
            if question_id in result:
                raise ValueError(f"duplicate {label} question_id: {question_id}")
            result[question_id] = row
        return result

    question_index = index(questions, "question")
    retrieval_index = index(retrieval_records, "retrieval")
    answer_index = index(answer_records, "answer")
    if not question_index:
        raise ValueError("downstream evaluation requires at least one question")
    if not (set(question_index) == set(retrieval_index) == set(answer_index)):
        raise ValueError(
            "question/retrieval/answer ID sets differ: "
            f"questions={len(question_index)}, retrieval={len(retrieval_index)}, "
            f"answers={len(answer_index)}"
        )
    return [
        compute_question_downstream_metrics(
            question_index[question_id],
            retrieval_index[question_id],
            answer_index[question_id],
            unanswerable_token=unanswerable_token,
        )
        for question_id in sorted(question_index)
    ]


__all__ = [
    "DEFAULT_UNANSWERABLE_TOKEN",
    "COMPLETE_CHAIN_CUTOFF",
    "DEFAULT_PAIRED_METRIC_FIELDS",
    "DOWNSTREAM_METRICS_SCHEMA_VERSION",
    "LINEAGE_FIELDS",
    "aggregate_downstream_metrics",
    "compute_downstream_metrics",
    "compute_question_downstream_metrics",
]
