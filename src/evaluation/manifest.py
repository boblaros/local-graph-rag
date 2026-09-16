"""Verified notebook inputs and descriptive builder values, without pairwise tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any

from .downstream import DEFAULT_PAIRED_METRIC_FIELDS
from .paired import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
)
from .primary import CASCADE_COMPARISON_PLAN, collect_prespecified_cascade_comparisons

MANIFEST_SCHEMA_VERSION = "1.0.0"
GRAPH_REGIMES = ("native_lightrag", "advanced_lightrag_er", "advanced_lightrag_er_rr")
_PARAMETER_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([BM])\b", re.I)


def _builder_metadata(builders: Sequence[Any]) -> list[dict[str, Any]]:
    result = []
    for builder in builders:
        row = (
            builder.model_dump(mode="json")
            if hasattr(builder, "model_dump")
            else dict(builder)
        )
        match = _PARAMETER_PATTERN.search(row.get("display_name", ""))
        if not match or not row.get("key") or not row.get("family"):
            raise ValueError("builder metadata is incomplete")
        size = float(match[1]) / (1000 if match[2].upper() == "M" else 1)
        if size <= 0:
            raise ValueError("builder parameter scale must be positive")
        band = (
            "sub_1b"
            if size < 1
            else "1_to_lt4b"
            if size < 4
            else "4_to_lt8b"
            if size < 8
            else "8b_plus"
        )
        result.append(
            {
                "builder_key": row["key"],
                "display_name": row["display_name"],
                "family": row["family"],
                "parameter_billions": size,
                "scale_band": band,
                "resolved_name": row.get("resolved_name"),
                "digest": row.get("digest"),
            }
        )
    if len(result) != 12 or len({row["builder_key"] for row in result}) != 12:
        raise ValueError("analysis manifest requires exactly 12 unique builders")
    return result


def _numeric_fields(value: Any, prefix: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        f"{prefix}.{key}": float(item)
        for key, item in value.items()
        if isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(item)
    }


def _artifact_values(summary: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for key in (
        "extraction",
        "entity_resolution",
        "native_topology",
        "er_topology",
        "er_rr_topology",
    ):
        result.update(_numeric_fields(summary.get(key), key))
    result.update(
        _numeric_fields(
            summary.get("extraction", {}).get("efficiency"), "extraction.efficiency"
        )
    )
    topology = dict(
        zip(GRAPH_REGIMES, ("native_topology", "er_topology", "er_rr_topology"))
    )
    for role, declaration in CASCADE_COMPARISON_PLAN.items():
        left = summary[topology[declaration["left_regime"]]]
        right = summary[topology[declaration["right_regime"]]]
        for key in left.keys() & right.keys():
            if isinstance(left[key], (int, float)) and isinstance(
                right[key], (int, float)
            ):
                result[f"{role}_topology_effect.{key}"] = float(right[key] - left[key])
    return result


def build_analysis_manifest(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    builders: Sequence[Any],
    expected_question_count: int,
    artifact_root: Path,
) -> dict[str, Any]:
    """Validate 12 summaries and 36 question files; register portable file paths.

    Archived schema-3 question metrics remain valid inputs. New schema-4 metrics
    share the eight thesis outcomes and the same question-level failure policy.
    """
    from src.orchestration.lineage import (
        ArtifactRef,
        resolve_artifact_path,
        verify_artifact,
    )

    metadata = _builder_metadata(builders)
    required = {
        f"{row['builder_key']}/{suffix}"
        for row in metadata
        for suffix in (*GRAPH_REGIMES, "summary")
    }
    if not required.issubset(registry):
        raise ValueError(
            f"analysis inputs missing: {sorted(required - registry.keys())}"
        )
    inputs = {}
    values = {}
    for name in sorted(required):
        ref = ArtifactRef.model_validate(registry[name])
        # Normalize archived absolute paths and project-relative registry paths.
        raw = Path(ref.path)
        if raw.is_absolute() and not raw.is_file() and "runs" in raw.parts:
            raw = Path(*raw.parts[raw.parts.index("runs") + 1 :])
        elif not raw.is_absolute() and raw.parts and raw.parts[0] == "runs":
            raw = Path(*raw.parts[1:])
        ref = ref.model_copy(update={"path": str(raw)})
        path = resolve_artifact_path(ref, root=artifact_root)
        path.relative_to(artifact_root.resolve())
        valid, reasons = verify_artifact(ref, root=artifact_root)
        if not valid:
            raise ValueError(f"analysis input verification failed: {name}: {reasons}")
        if name.endswith("/summary"):
            value = json.loads(path.read_text())
            if (
                ref.schema_version not in {"5.0.0", "6.0.0"}
                or value.get("schema_version") != ref.schema_version
            ):
                raise ValueError(f"analysis summary schema mismatch: {name}")
        else:
            value = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            if (
                len(value) != expected_question_count
                or ref.record_count != expected_question_count
            ):
                raise ValueError(f"analysis question count mismatch: {name}")
            if ref.schema_version not in {"3.0.0", "4.0.0"} or any(
                row.get("schema_version") != ref.schema_version for row in value
            ):
                raise ValueError(f"analysis metric schema mismatch: {name}")
        values[name] = value
        inputs[name] = {
            **ref.model_dump(mode="json"),
            "path": "runs/" + path.relative_to(artifact_root.resolve()).as_posix(),
        }

    reference_catalog = None
    performances = []
    artifacts = []
    summaries = []
    for descriptor in metadata:
        builder = descriptor["builder_key"]
        summary = values[f"{builder}/summary"]
        if summary.get("builder_key") != builder:
            raise ValueError(f"analysis summary builder mismatch: {builder}")
        summaries.append(summary)
        per_regime = {}
        bases, extraction_hashes, variants = set(), set(), set()
        for regime in GRAPH_REGIMES:
            rows = values[f"{builder}/{regime}"]
            indexed = {row["question_id"]: row for row in rows}
            catalog = {
                key: (row["question_type"], row["answerable"])
                for key, row in indexed.items()
            }
            if len(indexed) != expected_question_count or (
                reference_catalog is not None and catalog != reference_catalog
            ):
                raise ValueError(
                    f"analysis question alignment mismatch: {builder}/{regime}"
                )
            reference_catalog = catalog
            for row in rows:
                if (
                    row.get("graph_regime") != regime
                    or row.get("builder_model") != descriptor["resolved_name"]
                ):
                    raise ValueError(
                        f"analysis model/regime mismatch: {builder}/{regime}"
                    )
                for key in (
                    "base_run_id",
                    "variant_run_id",
                    "base_extraction_sha256",
                    "retrieval_result_id",
                    "answer_result_id",
                ):
                    if not row.get(key):
                        raise ValueError(
                            f"analysis lineage missing {key}: {builder}/{regime}"
                        )
                bases.add(row["base_run_id"])
                extraction_hashes.add(row["base_extraction_sha256"])
                variants.add(row["variant_run_id"])
                if not set(DEFAULT_PAIRED_METRIC_FIELDS).issubset(row):
                    raise ValueError(
                        f"analysis thesis metrics missing: {builder}/{regime}"
                    )
            per_regime[regime] = indexed
        if (
            len(bases) != 1
            or len(extraction_hashes) != 1
            or len(variants) != 3
            or summary.get("base_run_id") not in bases
        ):
            raise ValueError(f"analysis shared extraction mismatch: {builder}")
        metrics = {}
        for field in DEFAULT_PAIRED_METRIC_FIELDS:
            targets, counts = {}, {}
            for target, regime in zip(("native", "er", "er_rr"), GRAPH_REGIMES):
                observed = [
                    float(row[field])
                    for row in per_regime[regime].values()
                    if row[field] is not None
                ]
                targets[target] = sum(observed) / len(observed) if observed else None
                counts[target] = len(observed)
            for role, declaration in CASCADE_COMPARISON_PLAN.items():
                left, right = (
                    per_regime[declaration[key]]
                    for key in ("left_regime", "right_regime")
                )
                observed = [
                    float(right[q][field]) - float(left[q][field])
                    for q in sorted(left)
                    if left[q][field] is not None and right[q][field] is not None
                ]
                target = f"{role}_effect"
                targets[target] = sum(observed) / len(observed) if observed else None
                counts[target] = len(observed)
            metrics[field] = {**targets, "question_counts": counts}
        performances.append({**descriptor, "metrics": metrics})
        artifacts.append({**descriptor, "metrics": _artifact_values(summary)})

    collect_prespecified_cascade_comparisons(
        summaries,
        expected_builder_keys=[row["builder_key"] for row in metadata],
        expected_question_count=expected_question_count,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "analysis_family": "thesis_analysis_inputs",
        "builder_count": len(metadata),
        "expected_question_count": expected_question_count,
        "graph_regimes": list(GRAPH_REGIMES),
        "comparison_plan": CASCADE_COMPARISON_PLAN,
        "bootstrap": {
            "method": "paired_stratified_percentile_by_question_type",
            "cluster_unit": "question_id",
            "samples": DEFAULT_BOOTSTRAP_SAMPLES,
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "shared_resample_plan": True,
        },
        "builder_performance": performances,
        "builder_artifacts": artifacts,
        "input_artifacts": inputs,
    }
