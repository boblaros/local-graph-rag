"""Exploratory cross-builder statistics over completed question-level metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
import math
import random
import re
from statistics import median
from typing import Any

import numpy as np

from .primary import CASCADE_COMPARISON_PLAN

EXPLORATORY_SCHEMA_VERSION = "2.0.0"
GRAPH_REGIMES = (
    "native_lightrag",
    "advanced_lightrag_er",
    "advanced_lightrag_er_rr",
)
ESTIMANDS = ("intention_to_evaluate", "complete_case")
EXPLORATORY_BINARY_METRICS = frozenset(
    {
        "retrieval_hit",
        "retrieval_hit_at_5",
        "retrieval_hit_at_10",
        "retrieval_hit_at_20",
        "complete_chain_recall_at_5",
        "complete_chain_recall_at_10",
        "complete_chain_recall_at_20",
        "answer_exact_match",
        "answer_correct",
        "unanswerable_correct",
        "hallucination",
        "over_abstention",
    }
)
_PARAMETER_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([BM])\b", re.I)
_LOWER_IS_BETTER = frozenset(
    {
        "hallucination",
        "over_abstention",
        "retrieval_latency_ms",
        "answer_latency_ms",
        "context_token_count",
        "answer_total_tokens",
    }
)


def _metric_direction(metric: str) -> str:
    return "lower_is_better" if metric in _LOWER_IS_BETTER else "higher_is_better"


def builder_metadata_from_config(builders: Sequence[Any]) -> list[dict[str, Any]]:
    """Freeze family and numeric parameter scale from configured display names."""

    result: list[dict[str, Any]] = []
    for raw in builders:
        row = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else dict(raw)
        key = str(row.get("key") or "").strip()
        display_name = str(row.get("display_name") or "").strip()
        family = str(row.get("family") or "").strip()
        match = _PARAMETER_PATTERN.search(display_name)
        if not key or not display_name or not family or match is None:
            raise ValueError(f"builder metadata is incomplete/unparseable: {key!r}")
        magnitude = float(match.group(1))
        parameter_billions = (
            magnitude / 1000.0 if match.group(2).upper() == "M" else magnitude
        )
        if parameter_billions <= 0:
            raise ValueError(f"builder parameter scale must be positive: {key}")
        if parameter_billions < 1:
            scale_band = "sub_1b"
        elif parameter_billions < 4:
            scale_band = "1_to_lt4b"
        elif parameter_billions < 8:
            scale_band = "4_to_lt8b"
        else:
            scale_band = "8b_plus"
        result.append(
            {
                "builder_key": key,
                "display_name": display_name,
                "family": family,
                "parameter_billions": parameter_billions,
                "scale_band": scale_band,
                "resolved_name": row.get("resolved_name"),
                "digest": row.get("digest"),
            }
        )
    keys = [row["builder_key"] for row in result]
    if len(result) != 12 or len(set(keys)) != 12:
        raise ValueError("exploratory analysis requires exactly 12 unique builders")
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 or not math.isfinite(value) for value in values):
        raise ValueError("p-values must be finite values in [0, 1]")
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, (original_index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[original_index] = running
    return adjusted


def exact_mcnemar(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    """Two-sided exact McNemar test for paired binary outcomes."""

    if len(left) != len(right) or not left:
        raise ValueError("McNemar requires equal non-empty paired outcomes")
    raw_pairs = [(float(a), float(b)) for a, b in zip(left, right)]
    if any(a not in {0.0, 1.0} or b not in {0.0, 1.0} for a, b in raw_pairs):
        raise ValueError("McNemar outcomes must be binary")
    pairs = [(int(a), int(b)) for a, b in raw_pairs]
    left_only = sum(a == 1 and b == 0 for a, b in pairs)
    right_only = sum(a == 0 and b == 1 for a, b in pairs)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "test": "exact_mcnemar_two_sided",
        "left_only_success": left_only,
        "right_only_success": right_only,
        "discordant_count": discordant,
        "p_value": p_value,
    }


def _row(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    raise TypeError(f"question metric row must be mapping-like: {type(value).__name__}")


def _index_rows(values: Sequence[Any], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        row = _row(value)
        question_id = str(row.get("question_id") or "").strip()
        if not question_id or question_id in result:
            raise ValueError(
                f"{label} has missing/duplicate question_id: {question_id!r}"
            )
        result[question_id] = row
    return result


def _validate_conditions(
    condition_rows: Mapping[str, Mapping[str, Sequence[Any]]],
    builder_metadata: Sequence[Mapping[str, Any]],
    expected_question_count: int,
) -> tuple[list[str], dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    builder_keys = [str(row["builder_key"]) for row in builder_metadata]
    descriptors = {str(row["builder_key"]): row for row in builder_metadata}
    if set(condition_rows) != set(builder_keys):
        raise ValueError("condition builder set differs from frozen builder metadata")
    indexed: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    reference: dict[str, tuple[str, bool]] | None = None
    question_ids: list[str] | None = None
    for builder in builder_keys:
        regimes = condition_rows[builder]
        if set(regimes) != set(GRAPH_REGIMES):
            raise ValueError(f"builder {builder} does not have exactly three regimes")
        indexed[builder] = {}
        builder_base_ids: set[str] = set()
        builder_extraction_hashes: set[str] = set()
        builder_variant_ids: set[str] = set()
        for regime in GRAPH_REGIMES:
            rows = _index_rows(regimes[regime], f"{builder}/{regime}")
            if len(rows) != expected_question_count:
                raise ValueError(
                    f"{builder}/{regime} has {len(rows)} questions, expected {expected_question_count}"
                )
            current_ids = sorted(rows)
            if question_ids is None:
                question_ids = current_ids
            elif current_ids != question_ids:
                raise ValueError("cross-model question ID sets differ")
            catalog = {
                question_id: (
                    str(row.get("question_type") or "unknown"),
                    bool(row.get("answerable")),
                )
                for question_id, row in rows.items()
            }
            if reference is None:
                reference = catalog
            elif catalog != reference:
                raise ValueError("cross-model question type/answerable mappings differ")
            for question_id, row in rows.items():
                if row.get("graph_regime") != regime:
                    raise ValueError(
                        f"graph regime mismatch for {builder}/{question_id}"
                    )
                expected_model = str(descriptors[builder].get("resolved_name") or "")
                if not expected_model or row.get("builder_model") != expected_model:
                    raise ValueError(
                        f"builder model lineage mismatch for {builder}/{question_id}"
                    )
                required_lineage = (
                    "base_run_id",
                    "variant_run_id",
                    "base_extraction_sha256",
                    "retrieval_result_id",
                    "answer_result_id",
                )
                missing = [
                    field
                    for field in required_lineage
                    if not str(row.get(field) or "").strip()
                ]
                if missing:
                    raise ValueError(
                        f"question metric lineage is incomplete for {builder}/{question_id}: {missing}"
                    )
                builder_base_ids.add(str(row["base_run_id"]))
                builder_extraction_hashes.add(str(row["base_extraction_sha256"]))
                builder_variant_ids.add(str(row["variant_run_id"]))
            indexed[builder][regime] = rows
        if len(builder_base_ids) != 1 or len(builder_extraction_hashes) != 1:
            raise ValueError(
                f"Native/ER/ER+RR base lineage differs for builder {builder}"
            )
        if len(builder_variant_ids) != 3:
            raise ValueError(f"builder {builder} must have three distinct variant IDs")
    assert question_ids is not None
    return question_ids, indexed


def _bootstrap_counts(
    question_ids: Sequence[str],
    reference_rows: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> np.ndarray:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    strata: dict[str, list[int]] = defaultdict(list)
    for index, question_id in enumerate(question_ids):
        strata[
            str(reference_rows[question_id].get("question_type") or "unknown")
        ].append(index)
    generator = random.Random(seed)
    counts = np.zeros((bootstrap_samples, len(question_ids)), dtype=np.uint16)
    for replicate in range(bootstrap_samples):
        for question_type in sorted(strata):
            indices = strata[question_type]
            for _ in indices:
                counts[replicate, indices[generator.randrange(len(indices))]] += 1
    return counts


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    raise ValueError(f"metric value must be numeric or null, got {value!r}")


def _condition_success(row: Mapping[str, Any]) -> bool:
    return not bool(row.get("retrieval_failed") or row.get("answer_failed"))


def _attach_difference_intervals(
    rows: Sequence[dict[str, Any]],
    values: Sequence[np.ndarray],
    bootstrap_counts: np.ndarray,
    confidence_level: float,
) -> None:
    if len(rows) != len(values):
        raise ValueError("bootstrap rows/value arrays differ")
    alpha = (1.0 - confidence_level) / 2.0
    available = [index for index, row in enumerate(rows) if row.get("available")]
    chunk_size = 256
    for start in range(0, len(available), chunk_size):
        positions = available[start : start + chunk_size]
        chunk_rows = [rows[index] for index in positions]
        matrix = np.vstack([values[index] for index in positions])
        observed = np.isfinite(matrix)
        filled = np.where(observed, matrix, 0.0)
        numerator = bootstrap_counts.astype(np.float64, copy=False) @ filled.T
        denominator = bootstrap_counts @ observed.T
        with np.errstate(divide="ignore", invalid="ignore"):
            means = numerator / denominator
        lower = np.nanquantile(means, alpha, axis=0)
        upper = np.nanquantile(means, 1.0 - alpha, axis=0)
        for index, row in enumerate(chunk_rows):
            row["difference_ci_lower"] = float(lower[index])
            row["difference_ci_upper"] = float(upper[index])


def _effect_size(differences: Sequence[float], *, binary: bool) -> dict[str, Any]:
    mean_difference = sum(differences) / len(differences)
    if len(differences) < 2:
        standard_deviation = 0.0
    else:
        standard_deviation = math.sqrt(
            sum((value - mean_difference) ** 2 for value in differences)
            / (len(differences) - 1)
        )
    return {
        "raw_mean_difference": mean_difference,
        "paired_standardized_mean_difference_dz": (
            mean_difference / standard_deviation if standard_deviation else None
        ),
        "zero_variance": standard_deviation == 0.0,
        "paired_risk_difference": mean_difference if binary else None,
    }


def _comparison_row(
    *,
    left_builder: str,
    right_builder: str,
    regime: str,
    metric: str,
    estimand: str,
    question_ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> tuple[dict[str, Any], np.ndarray]:
    left_values = np.full(len(question_ids), np.nan)
    right_values = np.full(len(question_ids), np.nan)
    failure_count = 0
    observed_ids: list[str] = []
    for index, question_id in enumerate(question_ids):
        left_row = indexed[left_builder][regime][question_id]
        right_row = indexed[right_builder][regime][question_id]
        if estimand == "complete_case" and not (
            _condition_success(left_row) and _condition_success(right_row)
        ):
            failure_count += 1
            continue
        left = _numeric(left_row.get(metric))
        right = _numeric(right_row.get(metric))
        if left is None or right is None:
            continue
        left_values[index] = left
        right_values[index] = right
        observed_ids.append(question_id)
        failure_count += int(
            not (_condition_success(left_row) and _condition_success(right_row))
        )
    differences_array = right_values - left_values
    observed = np.isfinite(left_values) & np.isfinite(right_values)
    if not observed.any():
        return {
            "schema_version": EXPLORATORY_SCHEMA_VERSION,
            "comparison_type": "cross_builder_within_regime",
            "estimand": estimand,
            "graph_regime": regime,
            "metric": metric,
            "left_builder": left_builder,
            "right_builder": right_builder,
            "effect_direction": "right_minus_left",
            "metric_direction": _metric_direction(metric),
            "expected_count": len(question_ids),
            "observed_count": 0,
            "missing_count": len(question_ids),
            "failure_count": failure_count,
            "available": False,
        }, differences_array
    left = left_values[observed]
    right = right_values[observed]
    differences = differences_array[observed]
    binary = metric in EXPLORATORY_BINARY_METRICS
    if binary and not (
        np.isin(left, (0.0, 1.0)).all() and np.isin(right, (0.0, 1.0)).all()
    ):
        raise ValueError(f"declared binary metric contains non-binary values: {metric}")
    mcnemar = exact_mcnemar(left.tolist(), right.tolist()) if binary else None
    direction = _metric_direction(metric)
    left_wins = differences > 0 if direction == "lower_is_better" else differences < 0
    right_wins = differences < 0 if direction == "lower_is_better" else differences > 0
    return {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "comparison_type": "cross_builder_within_regime",
        "estimand": estimand,
        "graph_regime": regime,
        "metric": metric,
        "left_builder": left_builder,
        "right_builder": right_builder,
        "effect_direction": "right_minus_left",
        "metric_direction": direction,
        "expected_count": len(question_ids),
        "observed_count": int(observed.sum()),
        "missing_count": len(question_ids) - int(observed.sum()),
        "failure_count": failure_count,
        "available": True,
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "mean_difference": float(differences.mean()),
        "left_wins": int(left_wins.sum()),
        "ties": int((differences == 0).sum()),
        "right_wins": int(right_wins.sum()),
        "practical_effect": _effect_size(differences.tolist(), binary=binary),
        "mcnemar": mcnemar,
        "holm_family_id": (
            f"cross_builder|{estimand}|{regime}|{metric}" if mcnemar else None
        ),
        "question_ids_sha256": _ids_sha256(observed_ids),
    }, differences_array


def _difference_in_differences_row(
    *,
    left_builder: str,
    right_builder: str,
    contrast_role: str,
    comparison_key: str,
    left_regime: str,
    right_regime: str,
    metric: str,
    estimand: str,
    question_ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> tuple[dict[str, Any], np.ndarray]:
    left_effect = np.full(len(question_ids), np.nan)
    right_effect = np.full(len(question_ids), np.nan)
    failure_count = 0
    observed_ids: list[str] = []
    for index, question_id in enumerate(question_ids):
        left_before = indexed[left_builder][left_regime][question_id]
        left_after = indexed[left_builder][right_regime][question_id]
        right_before = indexed[right_builder][left_regime][question_id]
        right_after = indexed[right_builder][right_regime][question_id]
        rows = (left_before, left_after, right_before, right_after)
        all_successful = all(_condition_success(row) for row in rows)
        if estimand == "complete_case" and not all_successful:
            failure_count += 1
            continue
        values = [_numeric(row.get(metric)) for row in rows]
        if any(value is None for value in values):
            continue
        left_effect[index] = float(values[1]) - float(values[0])
        right_effect[index] = float(values[3]) - float(values[2])
        observed_ids.append(question_id)
        failure_count += int(not all_successful)
    differences_array = right_effect - left_effect
    observed = np.isfinite(left_effect) & np.isfinite(right_effect)
    if not observed.any():
        return {
            "schema_version": EXPLORATORY_SCHEMA_VERSION,
            "comparison_type": "difference_in_differences",
            "estimand": estimand,
            "contrast_role": contrast_role,
            "comparison_key": comparison_key,
            "left_regime": left_regime,
            "right_regime": right_regime,
            "metric": metric,
            "left_builder": left_builder,
            "right_builder": right_builder,
            "effect_direction": "right_builder_contrast_minus_left_builder_contrast",
            "contrast_effect_direction": (
                f"{right_regime}_minus_{left_regime}"
            ),
            "metric_direction": _metric_direction(metric),
            "expected_count": len(question_ids),
            "observed_count": 0,
            "missing_count": len(question_ids),
            "failure_count": failure_count,
            "available": False,
        }, differences_array
    left = left_effect[observed]
    right = right_effect[observed]
    differences = differences_array[observed]
    direction = _metric_direction(metric)
    left_wins = differences > 0 if direction == "lower_is_better" else differences < 0
    right_wins = differences < 0 if direction == "lower_is_better" else differences > 0
    return {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "comparison_type": "difference_in_differences",
        "estimand": estimand,
        "contrast_role": contrast_role,
        "comparison_key": comparison_key,
        "left_regime": left_regime,
        "right_regime": right_regime,
        "metric": metric,
        "left_builder": left_builder,
        "right_builder": right_builder,
        "effect_direction": "right_builder_contrast_minus_left_builder_contrast",
        "contrast_effect_direction": f"{right_regime}_minus_{left_regime}",
        "metric_direction": direction,
        "expected_count": len(question_ids),
        "observed_count": int(observed.sum()),
        "missing_count": len(question_ids) - int(observed.sum()),
        "failure_count": failure_count,
        "available": True,
        "left_mean_contrast_effect": float(left.mean()),
        "right_mean_contrast_effect": float(right.mean()),
        "difference_in_differences": float(differences.mean()),
        "left_effect_wins": int(left_wins.sum()),
        "ties": int((differences == 0).sum()),
        "right_effect_wins": int(right_wins.sum()),
        "practical_effect": _effect_size(differences.tolist(), binary=False),
        "question_ids_sha256": _ids_sha256(observed_ids),
    }, differences_array


def _ids_sha256(values: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _apply_holm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = row.get("holm_family_id")
        mcnemar = row.get("mcnemar")
        if family and isinstance(mcnemar, Mapping):
            families[str(family)].append(row)
    summaries: dict[str, Any] = {}
    for family, members in sorted(families.items()):
        adjusted = holm_adjust([float(row["mcnemar"]["p_value"]) for row in members])
        for row, value in zip(members, adjusted):
            row["mcnemar"]["holm_adjusted_p_value"] = value
            row["mcnemar"]["holm_family_size"] = len(members)
            row["mcnemar"]["reject_at_0_05"] = value < 0.05
        summaries[family] = {
            "test_count": len(members),
            "raw_p_below_0_05": sum(
                float(row["mcnemar"]["p_value"]) < 0.05 for row in members
            ),
            "holm_rejections_at_0_05": sum(
                bool(row["mcnemar"]["reject_at_0_05"]) for row in members
            ),
        }
    return summaries


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    return {
        "model_count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "inference": "descriptive_only_no_node_or_question_pseudoreplication",
    }


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original, _ in ordered[index:end]:
            result[original] = rank
        index = end
    return result


def _spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    if len(x) != len(y) or len(x) < 2:
        return {"model_count": len(x), "rho": None}
    left = _ranks(x)
    right = _ranks(y)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return {
        "model_count": len(x),
        "rho": numerator / denominator if denominator else None,
        "p_value": None,
        "inference": "descriptive_rank_association",
    }


def _family_scale_summary(
    metadata: Sequence[Mapping[str, Any]],
    indexed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    question_ids: Sequence[str],
    metric_fields: Sequence[str],
) -> dict[str, Any]:
    regime_targets = {
        "native": "native_lightrag",
        "er": "advanced_lightrag_er",
        "er_rr": "advanced_lightrag_er_rr",
    }
    effect_targets = {
        f"{role}_effect": declaration
        for role, declaration in CASCADE_COMPARISON_PLAN.items()
    }
    summary_targets = (*regime_targets, *effect_targets)
    builder_metrics: list[dict[str, Any]] = []
    for descriptor in metadata:
        builder = str(descriptor["builder_key"])
        values: dict[str, Any] = {}
        for metric in metric_fields:
            target_values: dict[str, float | None] = {}
            question_counts: dict[str, int] = {}
            for target, regime in regime_targets.items():
                observed = [
                    value
                    for question_id in question_ids
                    if (
                        value := _numeric(
                            indexed[builder][regime][question_id].get(metric)
                        )
                    )
                    is not None
                ]
                target_values[target] = (
                    sum(observed) / len(observed) if observed else None
                )
                question_counts[target] = len(observed)
            for target, declaration in effect_targets.items():
                left_regime = declaration["left_regime"]
                right_regime = declaration["right_regime"]
                paired = []
                for question_id in question_ids:
                    left = _numeric(
                        indexed[builder][left_regime][question_id].get(metric)
                    )
                    right = _numeric(
                        indexed[builder][right_regime][question_id].get(metric)
                    )
                    if left is not None and right is not None:
                        paired.append((left, right))
                target_values[target] = (
                    sum(right - left for left, right in paired) / len(paired)
                    if paired
                    else None
                )
                question_counts[target] = len(paired)
            values[metric] = {
                **target_values,
                "question_counts": question_counts,
            }
        builder_metrics.append({**dict(descriptor), "metrics": values})

    def grouped(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in builder_metrics:
            groups[str(row[field])].append(row)
        return {
            group: {
                metric: {
                    target: _descriptive(
                        [
                            float(row["metrics"][metric][target])
                            for row in rows
                            if row["metrics"][metric][target] is not None
                        ]
                    )
                    for target in summary_targets
                }
                for metric in metric_fields
            }
            for group, rows in sorted(groups.items())
        }

    def correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for metric in metric_fields:
            result[metric] = {}
            for target in summary_targets:
                available = [
                    row for row in rows if row["metrics"][metric][target] is not None
                ]
                result[metric][target] = _spearman(
                    [float(row["parameter_billions"]) for row in available],
                    [float(row["metrics"][metric][target]) for row in available],
                )
        return result

    families = sorted({str(row["family"]) for row in builder_metrics})
    return {
        "unit": "builder_condition",
        "builder_metrics": builder_metrics,
        "by_family": grouped("family"),
        "by_scale_band": grouped("scale_band"),
        "scale_spearman_overall": correlations(builder_metrics),
        "scale_spearman_by_family": {
            family: correlations(
                [row for row in builder_metrics if row["family"] == family]
            )
            for family in families
        },
        "graph_level_inference": "descriptive_only_one_graph_per_builder_regime",
    }


def summarize_builder_artifacts(
    builder_summaries: Mapping[str, Mapping[str, Any]],
    *,
    builder_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe extraction/ER/graph outcomes with builder—not node—as the unit."""

    metadata = [dict(row) for row in builder_metadata]
    builder_keys = [str(row.get("builder_key") or "") for row in metadata]
    if set(builder_summaries) != set(builder_keys):
        raise ValueError("builder artifact summaries do not match metadata")

    def numeric_fields(value: Any, prefix: str = "") -> dict[str, float]:
        if not isinstance(value, Mapping):
            return {}
        result: dict[str, float] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, bool) or item is None:
                continue
            if isinstance(item, (int, float)) and math.isfinite(float(item)):
                result[name] = float(item)
        return result

    rows: list[dict[str, Any]] = []
    for descriptor in metadata:
        builder = str(descriptor["builder_key"])
        summary = builder_summaries[builder]
        extraction = summary.get("extraction")
        extraction = extraction if isinstance(extraction, Mapping) else {}
        efficiency = extraction.get("efficiency")
        native = numeric_fields(summary.get("native_topology"), "native_topology")
        advanced = numeric_fields(summary.get("er_topology"), "er_topology")
        recovered = numeric_fields(summary.get("er_rr_topology"), "er_rr_topology")
        metrics = {
            **numeric_fields(extraction, "extraction"),
            **numeric_fields(efficiency, "extraction.efficiency"),
            **numeric_fields(summary.get("entity_resolution"), "entity_resolution"),
            **native,
            **advanced,
            **recovered,
        }
        for native_name, native_value in native.items():
            suffix = native_name.removeprefix("native_topology.")
            er_name = f"er_topology.{suffix}"
            if er_name in advanced:
                metrics[f"secondary_topology_effect.{suffix}"] = (
                    advanced[er_name] - native_value
                )
            er_rr_name = f"er_rr_topology.{suffix}"
            if er_rr_name in recovered:
                metrics[f"primary_topology_effect.{suffix}"] = (
                    recovered[er_rr_name] - native_value
                )
        for er_name, er_value in advanced.items():
            suffix = er_name.removeprefix("er_topology.")
            er_rr_name = f"er_rr_topology.{suffix}"
            if er_rr_name in recovered:
                metrics[f"incremental_ablation_topology_effect.{suffix}"] = (
                    recovered[er_rr_name] - er_value
                )
        rows.append({**descriptor, "metrics": metrics})

    metric_names = sorted({name for row in rows for name in row["metrics"]})

    def grouped(group_field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[group_field])].append(row)
        return {
            group: {
                metric: _descriptive(
                    [
                        float(row["metrics"][metric])
                        for row in members
                        if metric in row["metrics"]
                    ]
                )
                for metric in metric_names
            }
            for group, members in sorted(groups.items())
        }

    def correlations(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for metric in metric_names:
            available = [row for row in members if metric in row["metrics"]]
            result[metric] = _spearman(
                [float(row["parameter_billions"]) for row in available],
                [float(row["metrics"][metric]) for row in available],
            )
        return result

    return {
        "unit": "builder_condition_or_single_graph",
        "builder_metrics": rows,
        "by_family": grouped("family"),
        "by_scale_band": grouped("scale_band"),
        "scale_spearman_overall": correlations(rows),
        "inference": "descriptive_only_no_node_edge_or_chunk_pseudoreplication",
    }


def compute_exploratory_analysis(
    condition_rows: Mapping[str, Mapping[str, Sequence[Any]]],
    *,
    builder_metadata: Sequence[Mapping[str, Any]],
    metric_fields: Sequence[str],
    expected_question_count: int,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute cross-builder views for all regimes and cascade contrasts."""

    if not metric_fields or len(metric_fields) != len(set(metric_fields)):
        raise ValueError("metric_fields must be a non-empty unique sequence")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    metadata = [dict(row) for row in builder_metadata]
    builder_keys = [str(row.get("builder_key") or "") for row in metadata]
    if len(metadata) != 12 or len(set(builder_keys)) != 12 or "" in builder_keys:
        raise ValueError(
            "exploratory analysis requires exactly 12 builder metadata rows"
        )
    question_ids, indexed = _validate_conditions(
        condition_rows, metadata, expected_question_count
    )
    reference = indexed[builder_keys[0]][GRAPH_REGIMES[0]]
    bootstrap_counts = _bootstrap_counts(
        question_ids,
        reference,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    model_pairs = list(combinations(builder_keys, 2))
    pairwise: list[dict[str, Any]] = []
    pairwise_values: list[np.ndarray] = []
    did: list[dict[str, Any]] = []
    did_values: list[np.ndarray] = []
    for estimand in ESTIMANDS:
        for regime in GRAPH_REGIMES:
            for metric in metric_fields:
                for left, right in model_pairs:
                    row, values = _comparison_row(
                        left_builder=left,
                        right_builder=right,
                        regime=regime,
                        metric=metric,
                        estimand=estimand,
                        question_ids=question_ids,
                        indexed=indexed,
                    )
                    pairwise.append(row)
                    pairwise_values.append(values)
        for contrast_role, declaration in CASCADE_COMPARISON_PLAN.items():
            for metric in metric_fields:
                for left, right in model_pairs:
                    row, values = _difference_in_differences_row(
                        left_builder=left,
                        right_builder=right,
                        contrast_role=contrast_role,
                        comparison_key=declaration["comparison_key"],
                        left_regime=declaration["left_regime"],
                        right_regime=declaration["right_regime"],
                        metric=metric,
                        estimand=estimand,
                        question_ids=question_ids,
                        indexed=indexed,
                    )
                    did.append(row)
                    did_values.append(values)
    _attach_difference_intervals(
        pairwise, pairwise_values, bootstrap_counts, confidence_level
    )
    _attach_difference_intervals(did, did_values, bootstrap_counts, confidence_level)
    holm = _apply_holm(pairwise)
    report = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "analysis_family": "exploratory_cross_builder",
        "builder_count": len(builder_keys),
        "model_pair_count": len(model_pairs),
        "within_regime_comparison_count": len(pairwise),
        "difference_in_differences_count": len(did),
        "expected_question_count": expected_question_count,
        "metric_fields": list(metric_fields),
        "estimands": list(ESTIMANDS),
        "graph_regimes": list(GRAPH_REGIMES),
        "comparison_plan": CASCADE_COMPARISON_PLAN,
        "bootstrap": {
            "method": "paired_stratified_percentile_by_question_type",
            "cluster_unit": "question_id",
            "samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "seed": seed,
            "shared_resample_plan": True,
        },
        "mcnemar": {
            "test": "exact_two_sided",
            "applicable_metrics": sorted(
                EXPLORATORY_BINARY_METRICS & set(metric_fields)
            ),
            "holm_scope": "separate_by_estimand_graph_regime_metric",
            "families": holm,
        },
        "family_and_scale": _family_scale_summary(
            metadata, indexed, question_ids, metric_fields
        ),
        "interpretation": {
            "exploratory": True,
            "raw_p_value_alone_is_not_superiority_evidence": True,
            "practical_effect_reported_separately": True,
            "graph_metrics_are_descriptive": True,
        },
    }
    return {
        "pairwise_comparisons": pairwise,
        "difference_in_differences": did,
        "global_report": report,
    }


__all__ = [
    "ESTIMANDS",
    "EXPLORATORY_BINARY_METRICS",
    "EXPLORATORY_SCHEMA_VERSION",
    "GRAPH_REGIMES",
    "builder_metadata_from_config",
    "compute_exploratory_analysis",
    "exact_mcnemar",
    "holm_adjust",
    "summarize_builder_artifacts",
]
