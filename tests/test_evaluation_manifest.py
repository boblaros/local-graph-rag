from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.downstream import DEFAULT_PAIRED_METRIC_FIELDS
from src.evaluation.manifest import GRAPH_REGIMES, build_analysis_manifest
from src.evaluation.primary import CASCADE_COMPARISON_PLAN
from src.orchestration.lineage import fingerprint_artifact


@pytest.fixture
def inputs(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    builders = [
        {
            "key": f"b{i}",
            "display_name": f"Builder {i + 1}B",
            "family": "family",
            "resolved_name": f"model:{i}",
            "digest": "a" * 64,
        }
        for i in range(12)
    ]
    registry = {}
    for builder in builders:
        key = builder["key"]
        summary = {
            "schema_version": "5.0.0",
            "builder_key": key,
            "base_run_id": f"base-{key}",
            "paired_question_count": 2,
            "comparison_plan": CASCADE_COMPARISON_PLAN,
            "paired_comparisons": {
                role: {"effect_direction": declaration["effect_direction"]}
                for role, declaration in CASCADE_COMPARISON_PLAN.items()
            },
            "native_topology": {"nodes": 4},
            "er_topology": {"nodes": 3},
            "er_rr_topology": {"nodes": 3},
        }
        path = runs / f"{key}-summary.json"
        path.write_text(json.dumps(summary))
        registry[f"{key}/summary"] = fingerprint_artifact(
            path, logical_name=f"{key}/summary", schema_version="5.0.0"
        ).model_dump(mode="json")
        for regime in GRAPH_REGIMES:
            rows = [
                {
                    "schema_version": "3.0.0",
                    "question_id": f"q{q}",
                    "question_type": "inference",
                    "answerable": True,
                    "builder_model": builder["resolved_name"],
                    "graph_regime": regime,
                    "base_run_id": f"base-{key}",
                    "base_extraction_sha256": "a" * 64,
                    "variant_run_id": f"{key}-{regime}",
                    "retrieval_result_id": f"r-{q}",
                    "answer_result_id": f"a-{q}",
                    **dict.fromkeys(DEFAULT_PAIRED_METRIC_FIELDS, float(q)),
                }
                for q in range(2)
            ]
            path = runs / f"{key}-{regime}.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            registry[f"{key}/{regime}"] = fingerprint_artifact(
                path, logical_name=f"{key}/{regime}", schema_version="3.0.0"
            ).model_dump(mode="json")
    return runs, builders, registry


def build(inputs):
    runs, builders, registry = inputs
    return build_analysis_manifest(
        registry, builders=builders, expected_question_count=2, artifact_root=runs
    )


def test_manifest_supports_archived_inputs_and_portable_roundtrip(inputs):
    manifest = build(inputs)
    assert manifest["builder_count"] == 12
    assert len(manifest["input_artifacts"]) == 48
    assert "family_and_scale" not in manifest
    assert "comparisons" not in manifest
    assert set(manifest["builder_performance"][0]["metrics"]) == set(
        DEFAULT_PAIRED_METRIC_FIELDS
    )
    assert (
        manifest["builder_artifacts"][0]["metrics"]["secondary_topology_effect.nodes"]
        == -1
    )
    assert all(
        ref["path"].startswith("runs/") for ref in manifest["input_artifacts"].values()
    )
    runs, builders, _registry = inputs
    assert (
        build_analysis_manifest(
            manifest["input_artifacts"],
            builders=builders,
            expected_question_count=2,
            artifact_root=runs,
        )
        == manifest
    )


def test_manifest_rejects_tampering_without_rehashing(inputs):
    runs, _, registry = inputs
    path = Path(registry["b0/native_lightrag"]["path"])
    path.write_text(path.read_text().replace('"q0"', '"qx"'))
    with pytest.raises(ValueError, match="verification failed"):
        build(inputs)


def test_manifest_rejects_rehashed_misaligned_questions(inputs):
    runs, _, registry = inputs
    name = "b0/native_lightrag"
    path = Path(registry[name]["path"])
    path.write_text(path.read_text().replace('"q0"', '"qx"'))
    registry[name] = fingerprint_artifact(
        path, logical_name=name, schema_version="3.0.0"
    ).model_dump(mode="json")
    with pytest.raises(ValueError, match="alignment mismatch"):
        build(inputs)


def test_manifest_rejects_missing_regime(inputs):
    del inputs[2]["b0/advanced_lightrag_er_rr"]
    with pytest.raises(ValueError, match="inputs missing"):
        build(inputs)


def test_manifest_accepts_new_question_schema(inputs):
    _runs, _, registry = inputs
    name = "b0/native_lightrag"
    path = Path(registry[name]["path"])
    path.write_text(path.read_text().replace('"3.0.0"', '"4.0.0"'))
    registry[name] = fingerprint_artifact(
        path, logical_name=name, schema_version="4.0.0"
    ).model_dump(mode="json")
    assert build(inputs)["input_artifacts"][name]["schema_version"] == "4.0.0"
