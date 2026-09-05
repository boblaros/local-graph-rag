"""Paired cascade-regime comparisons on exactly the same questions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import random
from typing import Any, Literal

from .downstream import SOURCE_QUESTION_TYPE_WEIGHTS


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 42
PAIRED_METRICS_SCHEMA_VERSION = "3.0.0"
Aggregation = Literal["micro", "macro", "source_weighted"]


def _index(
    rows: Iterable[Mapping[str, Any]], regime: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if row.get("graph_regime") != regime:
            raise ValueError(f"expected {regime}, found {row.get('graph_regime')}")
        key = (
            str(row.get("base_run_id") or ""),
            str(row.get("builder_model") or ""),
            str(row.get("question_id") or ""),
        )
        if not all(key):
            raise ValueError(
                "paired rows require base_run_id, builder_model, question_id"
            )
        if key in result:
            raise ValueError(f"duplicate paired row: {key}")
        result[key] = row
    return result


def paired_variant_deltas(
    native_rows: Iterable[Mapping[str, Any]],
    er_rows: Iterable[Mapping[str, Any]],
    *,
    metric_fields: Sequence[str],
) -> list[dict[str, Any]]:
    native = _index(native_rows, "native_lightrag")
    advanced = _index(er_rows, "advanced_lightrag_er")
    if set(native) != set(advanced):
        missing_er = sorted(set(native) - set(advanced))
        missing_native = sorted(set(advanced) - set(native))
        raise ValueError(
            "paired comparison question sets differ: "
            f"missing_er={missing_er[:3]}, missing_native={missing_native[:3]}"
        )
    result: list[dict[str, Any]] = []
    for key in sorted(native):
        base, builder, question = key
        left = native[key]
        right = advanced[key]
        if left.get("base_extraction_sha256") != right.get("base_extraction_sha256"):
            raise ValueError(f"variants do not share extraction for {key}")
        for field in ("question_type", "answerable"):
            if left.get(field) != right.get(field):
                raise ValueError(f"variants disagree on {field} for {key}")
        deltas: dict[str, float | None] = {}
        native_values: dict[str, float | None] = {}
        er_values: dict[str, float | None] = {}
        for field in metric_fields:
            native_value = left.get(field)
            er_value = right.get(field)
            native_values[field] = (
                float(native_value) if native_value is not None else None
            )
            er_values[field] = float(er_value) if er_value is not None else None
            deltas[field] = (
                float(er_value) - float(native_value)
                if native_value is not None and er_value is not None
                else None
            )
        native_retrieval_failed = bool(left.get("retrieval_failed"))
        er_retrieval_failed = bool(right.get("retrieval_failed"))
        native_answer_failed = bool(left.get("answer_failed"))
        er_answer_failed = bool(right.get("answer_failed"))
        native_failed = native_retrieval_failed or native_answer_failed
        er_failed = er_retrieval_failed or er_answer_failed
        result.append(
            {
                "schema_version": PAIRED_METRICS_SCHEMA_VERSION,
                "question_id": question,
                "question_type": left.get("question_type"),
                "answerable": left.get("answerable"),
                "base_run_id": base,
                "builder_model": builder,
                "base_extraction_sha256": left.get("base_extraction_sha256"),
                "native_variant_run_id": left.get("variant_run_id"),
                "er_variant_run_id": right.get("variant_run_id"),
                "native_retrieval_result_id": left.get("retrieval_result_id"),
                "er_retrieval_result_id": right.get("retrieval_result_id"),
                "native_answer_result_id": left.get("answer_result_id"),
                "er_answer_result_id": right.get("answer_result_id"),
                "native_failed": native_failed,
                "er_failed": er_failed,
                "native_retrieval_failed": native_retrieval_failed,
                "er_retrieval_failed": er_retrieval_failed,
                "native_answer_failed": native_answer_failed,
                "er_answer_failed": er_answer_failed,
                "complete_case": not native_failed and not er_failed,
                "native_values": native_values,
                "er_values": er_values,
                "deltas": deltas,
            }
        )
    return result


def paired_regime_deltas(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
    *,
    left_regime: str,
    right_regime: str,
    metric_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Generic within-question comparison for cascade contrasts."""

    if not left_regime or not right_regime or left_regime == right_regime:
        raise ValueError("paired regimes must be distinct non-empty names")
    left_index = _index(left_rows, left_regime)
    right_index = _index(right_rows, right_regime)
    if set(left_index) != set(right_index):
        raise ValueError("paired comparison question sets differ")
    result: list[dict[str, Any]] = []
    for key in sorted(left_index):
        base, builder, question = key
        left = left_index[key]
        right = right_index[key]
        if left.get("base_extraction_sha256") != right.get("base_extraction_sha256"):
            raise ValueError(f"variants do not share extraction for {key}")
        for field in ("question_type", "answerable"):
            if left.get(field) != right.get(field):
                raise ValueError(f"variants disagree on {field} for {key}")
        left_values = {
            field: float(left[field]) if left.get(field) is not None else None
            for field in metric_fields
        }
        right_values = {
            field: float(right[field]) if right.get(field) is not None else None
            for field in metric_fields
        }
        deltas = {
            field: (
                right_values[field] - left_values[field]
                if right_values[field] is not None and left_values[field] is not None
                else None
            )
            for field in metric_fields
        }
        left_retrieval_failed = bool(left.get("retrieval_failed"))
        right_retrieval_failed = bool(right.get("retrieval_failed"))
        left_answer_failed = bool(left.get("answer_failed"))
        right_answer_failed = bool(right.get("answer_failed"))
        left_failed = left_retrieval_failed or left_answer_failed
        right_failed = right_retrieval_failed or right_answer_failed
        result.append(
            {
                "schema_version": PAIRED_METRICS_SCHEMA_VERSION,
                "question_id": question,
                "question_type": left.get("question_type"),
                "answerable": left.get("answerable"),
                "base_run_id": base,
                "builder_model": builder,
                "base_extraction_sha256": left.get("base_extraction_sha256"),
                "left_regime": left_regime,
                "right_regime": right_regime,
                "left_variant_run_id": left.get("variant_run_id"),
                "right_variant_run_id": right.get("variant_run_id"),
                "left_retrieval_result_id": left.get("retrieval_result_id"),
                "right_retrieval_result_id": right.get("retrieval_result_id"),
                "left_answer_result_id": left.get("answer_result_id"),
                "right_answer_result_id": right.get("answer_result_id"),
                "left_failed": left_failed,
                "right_failed": right_failed,
                "left_retrieval_failed": left_retrieval_failed,
                "right_retrieval_failed": right_retrieval_failed,
                "left_answer_failed": left_answer_failed,
                "right_answer_failed": right_answer_failed,
                "complete_case": not left_failed and not right_failed,
                "left_values": left_values,
                "right_values": right_values,
                "deltas": deltas,
            }
        )
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _source_type(question_type: str) -> str:
    normalized = question_type.casefold()
    if "comparison" in normalized:
        return "comparison"
    if "temporal" in normalized:
        return "temporal"
    if "null" in normalized or "unanswerable" in normalized:
        return "unanswerable"
    if "inference" in normalized:
        return "inference"
    raise ValueError(
        f"question type has no frozen source-weight mapping: {question_type}"
    )


def _aggregate(
    triples: Sequence[tuple[str, float, float, float]], aggregation: Aggregation
) -> tuple[float, float, float]:
    if aggregation == "micro":
        return tuple(_mean([item[index] for item in triples]) for index in (1, 2, 3))  # type: ignore[return-value]
    strata: dict[str, list[tuple[str, float, float, float]]] = defaultdict(list)
    for item in triples:
        strata[item[0]].append(item)
    stratum_values = {
        name: tuple(_mean([item[index] for item in values]) for index in (1, 2, 3))
        for name, values in strata.items()
    }
    if aggregation == "macro":
        return tuple(
            _mean([values[index] for values in stratum_values.values()])
            for index in range(3)
        )  # type: ignore[return-value]
    weights = {
        name: SOURCE_QUESTION_TYPE_WEIGHTS[_source_type(name)] for name in strata
    }
    denominator = sum(weights.values())
    return tuple(
        sum(values[index] * weights[name] for name, values in stratum_values.items())
        / denominator
        for index in range(3)
    )  # type: ignore[return-value]


def _bootstrap_plan(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int, seed: int
) -> list[list[int]]:
    """Create one shared question-index plan for every metric/estimand."""

    strata: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[str(row.get("question_type") or "unknown")].append(index)
    plan_seed = int.from_bytes(
        hashlib.sha256(f"{seed}\0paired-question-plan".encode()).digest()[:8],
        "big",
    )
    generator = random.Random(plan_seed)
    plan: list[list[int]] = []
    for _ in range(bootstrap_samples):
        sampled: list[int] = []
        for name in sorted(strata):
            indices = strata[name]
            sampled.extend(indices[generator.randrange(len(indices))] for _ in indices)
        plan.append(sampled)
    return plan


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    aggregation: Aggregation,
    bootstrap_plan: Sequence[Sequence[int]],
    confidence_level: float,
) -> dict[str, Any]:
    triples_by_index: dict[int, tuple[str, float, float, float]] = {}
    for row_index, row in enumerate(rows):
        native = row.get("native_values")
        advanced = row.get("er_values")
        deltas = row.get("deltas")
        if not all(isinstance(value, Mapping) for value in (native, advanced, deltas)):
            raise ValueError("paired row lacks native/ER/delta metric mappings")
        native_value = native.get(field)
        er_value = advanced.get(field)
        delta = deltas.get(field)
        if native_value is None or er_value is None or delta is None:
            continue
        triples_by_index[row_index] = (
            str(row.get("question_type") or "unknown"),
            float(native_value),
            float(er_value),
            float(delta),
        )
    triples = list(triples_by_index.values())
    missing_count = len(rows) - len(triples)
    if not triples:
        return {
            "paired_count": 0,
            "missing_count": missing_count,
            "failure_count": 0,
            "native_mean": None,
            "er_mean": None,
            "mean_delta": None,
            "native_ci_lower": None,
            "native_ci_upper": None,
            "er_ci_lower": None,
            "er_ci_upper": None,
            "delta_ci_lower": None,
            "delta_ci_upper": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    point = _aggregate(triples, aggregation)
    bootstrap = []
    for sampled_indices in bootstrap_plan:
        sampled = [
            triples_by_index[index]
            for index in sampled_indices
            if index in triples_by_index
        ]
        if sampled:
            bootstrap.append(_aggregate(sampled, aggregation))
    alpha = (1.0 - confidence_level) / 2.0
    if field.startswith("retrieval") or field in {
        "reciprocal_rank",
        "context_token_count",
    }:
        failure_fields = ("native_retrieval_failed", "er_retrieval_failed")
    elif field.startswith("answer") or field in {
        "token_f1",
        "unanswerable_correct",
        "hallucination",
        "over_abstention",
    }:
        failure_fields = ("native_answer_failed", "er_answer_failed")
    else:
        failure_fields = ("native_failed", "er_failed")
    intervals = [
        (
            _percentile([item[index] for item in bootstrap], alpha),
            _percentile([item[index] for item in bootstrap], 1.0 - alpha),
        )
        for index in range(3)
    ]
    return {
        "schema_version": PAIRED_METRICS_SCHEMA_VERSION,
        "paired_count": len(triples),
        "missing_count": missing_count,
        "failure_count": sum(
            bool(row.get(failure_fields[0]) or row.get(failure_fields[1]))
            for index, row in enumerate(rows)
            if index in triples_by_index
        ),
        "native_mean": point[0],
        "er_mean": point[1],
        "mean_delta": point[2],
        "native_ci_lower": intervals[0][0],
        "native_ci_upper": intervals[0][1],
        "er_ci_lower": intervals[1][0],
        "er_ci_upper": intervals[1][1],
        "delta_ci_lower": intervals[2][0],
        "delta_ci_upper": intervals[2][1],
        # Existing output names for the paired-difference interval.
        "ci_lower": intervals[2][0],
        "ci_upper": intervals[2][1],
    }


def summarize_paired_deltas(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    metric_fields: Sequence[str],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize paired effects using one stratified question resampling design."""

    if not paired_rows:
        raise ValueError("paired summary requires at least one question")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if not metric_fields or len(metric_fields) != len(set(metric_fields)):
        raise ValueError("metric_fields must be a non-empty unique sequence")

    def summarize(
        rows: Sequence[Mapping[str, Any]],
        aggregation: Aggregation,
        bootstrap_plan: Sequence[Sequence[int]],
    ) -> dict[str, Any]:
        return {
            field: _metric_summary(
                rows,
                field,
                aggregation=aggregation,
                bootstrap_plan=bootstrap_plan,
                confidence_level=confidence_level,
            )
            for field in metric_fields
        }

    def analysis(rows: Sequence[Mapping[str, Any]], estimand: str) -> dict[str, Any]:
        plan = _bootstrap_plan(rows, bootstrap_samples=bootstrap_samples, seed=seed)
        unmapped = sorted(
            {
                str(row.get("question_type") or "unknown")
                for row in rows
                if not any(
                    token in str(row.get("question_type") or "unknown").casefold()
                    for token in (
                        "inference",
                        "comparison",
                        "temporal",
                        "null",
                        "unanswerable",
                    )
                )
            }
        )
        return {
            "estimand": estimand,
            "question_count": len(rows),
            "micro": summarize(rows, "micro", plan),
            "macro": summarize(rows, "macro", plan),
            "source_weighted": (
                summarize(rows, "source_weighted", plan)
                if not unmapped
                else {
                    "available": False,
                    "unmapped_question_types": unmapped,
                }
            ),
        }

    question_types = sorted(
        {str(row.get("question_type") or "unknown") for row in paired_rows}
    )
    complete = [row for row in paired_rows if bool(row.get("complete_case", True))]
    primary = analysis(paired_rows, "intention_to_evaluate")
    secondary = (
        analysis(complete, "complete_case")
        if complete
        else {
            "estimand": "complete_case",
            "question_count": 0,
            "micro": {},
            "macro": {},
            "source_weighted": {},
        }
    )
    secondary["excluded_failure_count"] = len(paired_rows) - len(complete)
    by_question_type: dict[str, dict[str, Any]] = {}
    for question_type in question_types:
        type_rows = [
            row
            for row in paired_rows
            if str(row.get("question_type") or "unknown") == question_type
        ]
        plan = _bootstrap_plan(
            type_rows, bootstrap_samples=bootstrap_samples, seed=seed
        )
        by_question_type[question_type] = summarize(type_rows, "micro", plan)
    return {
        "schema_version": PAIRED_METRICS_SCHEMA_VERSION,
        "paired_question_count": len(paired_rows),
        "effect_direction": "advanced_lightrag_er_minus_native_lightrag",
        "bootstrap": {
            "method": "paired_stratified_percentile_by_question_type",
            "samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "seed": seed,
            "cluster_unit": "question_id",
            "shared_resample_plan_across_metrics": True,
        },
        # Backward-compatible primary micro view.
        "metrics": primary["micro"],
        "by_question_type": by_question_type,
        "analyses": {"primary": primary, "secondary": secondary},
        "source_question_type_weights": SOURCE_QUESTION_TYPE_WEIGHTS,
    }


def summarize_regime_deltas(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    left_regime: str,
    right_regime: str,
    metric_fields: Sequence[str],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize generic right-minus-left effects using the proven bootstrap."""

    compatibility_rows: list[dict[str, Any]] = []
    for raw in paired_rows:
        row = dict(raw)
        row.update(
            native_values=row.get("left_values"),
            er_values=row.get("right_values"),
            native_failed=row.get("left_failed"),
            er_failed=row.get("right_failed"),
            native_retrieval_failed=row.get("left_retrieval_failed"),
            er_retrieval_failed=row.get("right_retrieval_failed"),
            native_answer_failed=row.get("left_answer_failed"),
            er_answer_failed=row.get("right_answer_failed"),
        )
        compatibility_rows.append(row)
    summary = summarize_paired_deltas(
        compatibility_rows,
        metric_fields=metric_fields,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )

    def relabel(value: Any) -> None:
        if isinstance(value, dict):
            for old, new in (
                ("native_mean", "left_mean"),
                ("er_mean", "right_mean"),
                ("native_ci_lower", "left_ci_lower"),
                ("native_ci_upper", "left_ci_upper"),
                ("er_ci_lower", "right_ci_lower"),
                ("er_ci_upper", "right_ci_upper"),
            ):
                if old in value:
                    value[new] = value.pop(old)
            for item in value.values():
                relabel(item)
        elif isinstance(value, list):
            for item in value:
                relabel(item)

    relabel(summary)
    summary["effect_direction"] = f"{right_regime}_minus_{left_regime}"
    summary["left_regime"] = left_regime
    summary["right_regime"] = right_regime
    return summary


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "PAIRED_METRICS_SCHEMA_VERSION",
    "paired_variant_deltas",
    "paired_regime_deltas",
    "summarize_paired_deltas",
    "summarize_regime_deltas",
]
