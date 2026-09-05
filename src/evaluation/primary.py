"""Experiment-level collection of the prespecified cascade contrasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PRIMARY_ANALYSIS_SCHEMA_VERSION = "2.0.0"
CASCADE_COMPARISON_PLAN: dict[str, dict[str, str]] = {
    "primary": {
        "comparison_key": "er_rr_minus_native",
        "hypothesis_family": "12_builder_conditions_er_rr_vs_native",
        "left_regime": "native_lightrag",
        "right_regime": "advanced_lightrag_er_rr",
        "effect_direction": "advanced_lightrag_er_rr_minus_native_lightrag",
        "multiple_comparison_scope": "12_primary_within_builder_effects",
    },
    "secondary": {
        "comparison_key": "er_minus_native",
        "hypothesis_family": "12_builder_conditions_er_vs_native",
        "left_regime": "native_lightrag",
        "right_regime": "advanced_lightrag_er",
        "effect_direction": "advanced_lightrag_er_minus_native_lightrag",
        "multiple_comparison_scope": "12_secondary_within_builder_effects",
    },
    "incremental_ablation": {
        "comparison_key": "er_rr_minus_er",
        "hypothesis_family": "12_builder_conditions_er_rr_vs_er",
        "left_regime": "advanced_lightrag_er",
        "right_regime": "advanced_lightrag_er_rr",
        "effect_direction": "advanced_lightrag_er_rr_minus_advanced_lightrag_er",
        "multiple_comparison_scope": "12_incremental_ablation_within_builder_effects",
    },
}


def collect_prespecified_cascade_comparisons(
    summaries: Sequence[Mapping[str, Any]],
    *,
    expected_builder_keys: Sequence[str],
    expected_question_count: int,
) -> dict[str, Any]:
    """Collect three prespecified within-builder contrasts for all 12 builders."""

    expected = tuple(expected_builder_keys)
    if len(expected) != 12 or len(set(expected)) != 12:
        raise ValueError("prespecified analysis requires exactly 12 unique builders")
    indexed: dict[str, dict[str, Any]] = {}
    for raw in summaries:
        summary = dict(raw)
        builder = str(summary.get("builder_key") or "")
        if not builder or builder in indexed:
            raise ValueError(f"missing/duplicate builder summary: {builder!r}")
        if int(summary.get("paired_question_count") or 0) != expected_question_count:
            raise ValueError(
                f"builder {builder} does not contain {expected_question_count} paired questions"
            )
        declarations = summary.get("comparison_plan")
        effects = summary.get("paired_comparisons")
        if not isinstance(declarations, Mapping) or not isinstance(effects, Mapping):
            raise ValueError(f"builder {builder} lacks cascade-comparison lineage")
        if set(declarations) != set(CASCADE_COMPARISON_PLAN) or set(effects) != set(
            CASCADE_COMPARISON_PLAN
        ):
            raise ValueError(f"builder {builder} has incomplete cascade comparisons")
        for role, expected_declaration in CASCADE_COMPARISON_PLAN.items():
            declaration = declarations.get(role)
            effect = effects.get(role)
            if not isinstance(declaration, Mapping) or any(
                declaration.get(field) != value
                for field, value in expected_declaration.items()
            ):
                raise ValueError(
                    f"builder {builder} has incompatible {role} comparison lineage"
                )
            if not isinstance(effect, Mapping) or effect.get("effect_direction") != (
                expected_declaration["effect_direction"]
            ):
                raise ValueError(
                    f"builder {builder} has incompatible {role} paired effects"
                )
        indexed[builder] = summary
    if set(indexed) != set(expected):
        raise ValueError(
            "prespecified comparison builder set differs: "
            f"missing={sorted(set(expected) - set(indexed))}, "
            f"unexpected={sorted(set(indexed) - set(expected))}"
        )
    question_counts = {int(item["paired_question_count"]) for item in indexed.values()}
    if question_counts != {expected_question_count}:
        raise ValueError("prespecified comparisons disagree on paired question count")

    comparison_roles: dict[str, dict[str, Any]] = {}
    for role, declaration in CASCADE_COMPARISON_PLAN.items():
        comparison_roles[role] = {
            **declaration,
            "comparison_count": len(expected),
            "builders": [
                {
                    "builder_key": builder,
                    "base_run_id": indexed[builder].get("base_run_id"),
                    "paired_question_count": indexed[builder][
                        "paired_question_count"
                    ],
                    "paired_effects": indexed[builder]["paired_comparisons"][role],
                }
                for builder in expected
            ],
        }
    return {
        "schema_version": PRIMARY_ANALYSIS_SCHEMA_VERSION,
        "analysis_family": "prespecified_12_builder_cascade_contrasts",
        "builder_count": len(expected),
        "primary_comparison_count": len(expected),
        "total_prespecified_comparison_count": len(expected)
        * len(CASCADE_COMPARISON_PLAN),
        "expected_question_count": expected_question_count,
        "comparison_roles": comparison_roles,
    }


__all__ = [
    "CASCADE_COMPARISON_PLAN",
    "PRIMARY_ANALYSIS_SCHEMA_VERSION",
    "collect_prespecified_cascade_comparisons",
]
