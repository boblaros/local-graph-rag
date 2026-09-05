from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.config
from src.config import LoadedExperimentConfig
from src.orchestration.cli import build_parser
from src.orchestration.harness import (
    EVALUATION_SCHEMA_VERSION,
    ExperimentHarness,
    StageOutcome,
    _condition_file,
    _evaluation_identity_sha256,
    _exploratory_identity_sha256,
    _judge_budget_gate_path,
    _metric_file,
    _primary_analysis_identity_sha256,
    _qa_artifact_path,
    _validate_native_entity_group_coverage,
)
from src.orchestration.artifacts import write_immutable_json
from src.orchestration.artifacts import write_immutable_jsonl
from src.graph import RewrittenGraph
from src.relation_recovery import OllamaRRVerifier, build_rr_plan
from src.orchestration.lineage import (
    build_base_manifest,
    build_variant_manifest,
    fingerprint_artifact,
    freeze_manifest,
    full_config_sha256,
    variant_config_sha256,
)
from src.orchestration.native import workspace_tree_sha256
from src.orchestration.paths import ConditionPaths
from src.orchestration.quality_gates import GateCheck, QualityGateReport
from src.orchestration.status import StageExpectation

from ._config_helpers import TEMPLATE, resolved_config


def _harness(tmp_path: Path) -> ExperimentHarness:
    config = resolved_config()
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"runs_root": str(tmp_path / "runs")}
            )
        }
    )
    return ExperimentHarness(LoadedExperimentConfig(config=config, path=TEMPLATE))


def test_cli_exposes_every_lifecycle_subcommand() -> None:
    parser = build_parser()
    commands = {
        action.dest: set(action.choices or {})
        for action in parser._actions
        if action.dest == "command"
    }
    assert commands["command"] == {
        "preflight",
        "smoke",
        "native-build",
        "er-plan",
        "er-quality-gate",
        "er-materialize",
        "rr-plan",
        "rr-verify",
        "rr-quality-gate",
        "rr-materialize",
        "retrieval",
        "answering",
        "evaluation",
        "primary-analysis",
        "exploratory-analysis",
        "resume",
        "status",
    }


def test_judge_budget_retry_preserves_the_failed_gate(tmp_path: Path) -> None:
    paths = ConditionPaths(
        runs_root=tmp_path,
        base_run_id="base-1",
        native_variant_run_id="native-1",
        er_variant_run_id="er-1",
        rr_variant_run_id="rr-1",
    )
    legacy = paths.er_dir / "gates" / "judge_budget.json"

    assert _judge_budget_gate_path(paths, attempt_count=1) == legacy
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}\n", encoding="utf-8")
    assert _judge_budget_gate_path(paths, attempt_count=2) == (
        paths.er_dir / "gates" / "judge_budget.attempt-0002.json"
    )


def test_exploratory_identity_is_deterministic_and_separate() -> None:
    config = resolved_config(all_builders=True)
    evaluation = _evaluation_identity_sha256(config)
    primary = _primary_analysis_identity_sha256(config)
    first = _exploratory_identity_sha256(config)
    assert first == _exploratory_identity_sha256(config)
    assert len(first) == 64
    assert EVALUATION_SCHEMA_VERSION == "5.0.0"
    assert len({evaluation, primary, first}) == 3


def test_evaluation_identity_does_not_change_upstream_qa_paths(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    paths = harness.condition_paths("qwen35_0_8b")
    qa_path = _condition_file(
        harness.config,
        paths,
        "advanced_lightrag_er_rr",
        directory="artifacts",
        stem="answers",
        suffix=".jsonl",
    )
    metric_path = _metric_file(
        harness.config,
        paths,
        "advanced_lightrag_er_rr",
        stem="question_metrics",
        suffix=".jsonl",
    )

    assert full_config_sha256(harness.config)[:16] in qa_path.name
    assert _evaluation_identity_sha256(harness.config)[:16] in metric_path.name


def test_evaluation_resolves_completed_qa_artifact_from_lifecycle(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    paths = harness.condition_paths("qwen35_0_8b")
    legacy = paths.native_dir / "artifacts" / "retrieval.legacy-config.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "{}\n" * harness.config.corpus.expected_questions, encoding="utf-8"
    )
    ref = fingerprint_artifact(
        legacy,
        logical_name="retrieval_results",
        schema_version="3.0.0",
    )
    expectation = StageExpectation(
        lineage_id=paths.native_variant_run_id,
        config_sha256="a" * 64,
    )
    claim = harness.status.claim("retrieval", paths.native_variant_run_id, expectation)
    assert claim.claimed
    harness.status.mark_completed(
        "retrieval", paths.native_variant_run_id, {"retrieval": ref}
    )

    resolved = _qa_artifact_path(
        harness.status,
        harness.config,
        paths,
        "native_lightrag",
        stage="retrieval",
    )

    assert resolved == legacy.resolve()
    assert resolved != _condition_file(
        harness.config,
        paths,
        "native_lightrag",
        directory="artifacts",
        stem="retrieval",
        suffix=".jsonl",
    )


def test_evaluation_rejects_corrupted_lifecycle_qa_artifact(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    paths = harness.condition_paths("qwen35_0_8b")
    artifact = paths.native_dir / "artifacts" / "answers.legacy-config.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        "{}\n" * harness.config.corpus.expected_questions, encoding="utf-8"
    )
    ref = fingerprint_artifact(
        artifact,
        logical_name="answer_results",
        schema_version="3.0.0",
    )
    expectation = StageExpectation(
        lineage_id=paths.native_variant_run_id,
        config_sha256="b" * 64,
    )
    harness.status.claim("answering", paths.native_variant_run_id, expectation)
    harness.status.mark_completed(
        "answering", paths.native_variant_run_id, {"answers": ref}
    )
    artifact.write_text("corrupted\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact verification failed"):
        _qa_artifact_path(
            harness.status,
            harness.config,
            paths,
            "native_lightrag",
            stage="answering",
        )


def test_workspace_hash_ignores_macos_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write_immutable_json(workspace / "store.json", {"native": True})
    expected = workspace_tree_sha256(workspace)

    (workspace / ".DS_Store").write_bytes(b"finder metadata")
    (workspace / "nested").mkdir()
    (workspace / "nested" / "._store.json").write_bytes(b"appledouble metadata")

    assert workspace_tree_sha256(workspace) == expected

    write_immutable_json(workspace / "scientific.json", {"changed": True})
    assert workspace_tree_sha256(workspace) != expected


def test_merge_only_er_requires_exact_native_node_coverage() -> None:
    mentions = [
        SimpleNamespace(original_name="Apple"),
        SimpleNamespace(original_name="Apple"),
        SimpleNamespace(original_name="Apple Inc."),
    ]
    _validate_native_entity_group_coverage(
        mentions,
        [{"node_id": "Apple"}, {"node_id": "Apple Inc."}],
    )

    with pytest.raises(RuntimeError, match="cannot reproduce exact Native"):
        _validate_native_entity_group_coverage(
            mentions,
            [{"node_id": "Apple"}],
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["native-build", "--builder", "qwen35_0_8b"],
        ["er-plan", "--builder", "qwen35_0_8b"],
        ["er-materialize", "--builder", "qwen35_0_8b"],
        ["rr-plan", "--builder", "qwen35_0_8b"],
        ["rr-verify", "--builder", "qwen35_0_8b"],
        ["rr-materialize", "--builder", "qwen35_0_8b"],
        [
            "retrieval",
            "--builder",
            "qwen35_0_8b",
            "--regime",
            "native_lightrag",
        ],
        [
            "answering",
            "--builder",
            "qwen35_0_8b",
            "--regime",
            "native_lightrag",
        ],
        ["evaluation", "--builder", "qwen35_0_8b"],
        ["resume", "--builder", "qwen35_0_8b"],
    ],
)
def test_interrupted_stage_commands_expose_explicit_reclaim(argv: list[str]) -> None:
    args = build_parser().parse_args([*argv, "--reclaim-running"])
    assert args.reclaim_running is True


def test_stage_model_identity_check_rejects_digest_drift(
    tmp_path: Path, monkeypatch
) -> None:
    harness = _harness(tmp_path)
    role = harness.config.roles.answer

    async def resolved(requested_tags, *, host):
        del host
        tag = requested_tags[0]
        return (
            {
                tag: {
                    "resolved_name": role.resolved_name,
                    "digest": role.digest,
                }
            },
            {},
        )

    monkeypatch.setattr(src.config, "resolve_ollama_inventory", resolved)
    asyncio.run(harness._verify_current_models(role))

    async def drifted(requested_tags, *, host):
        identities, inventory = await resolved(requested_tags, host=host)
        identities[requested_tags[0]]["digest"] = "f" * 64
        return identities, inventory

    monkeypatch.setattr(src.config, "resolve_ollama_inventory", drifted)
    with pytest.raises(RuntimeError, match="digest drift"):
        asyncio.run(harness._verify_current_models(role))


def test_downstream_artifacts_do_not_mix_role_configurations(tmp_path: Path) -> None:
    first = _harness(tmp_path)
    answer = first.config.roles.answer
    changed_answer = answer.model_copy(
        update={
            "generation": answer.generation.model_copy(
                update={"temperature": answer.generation.temperature + 0.1}
            )
        }
    )
    changed_config = first.config.model_copy(
        update={
            "roles": first.config.roles.model_copy(update={"answer": changed_answer})
        }
    )
    second = ExperimentHarness(
        LoadedExperimentConfig(config=changed_config, path=TEMPLATE)
    )
    first_paths = first.condition_paths("qwen35_0_8b")
    second_paths = second.condition_paths("qwen35_0_8b")

    assert first_paths.base_run_id == second_paths.base_run_id
    assert first_paths.native_variant_run_id == second_paths.native_variant_run_id
    assert _condition_file(
        first.config,
        first_paths,
        "native_lightrag",
        directory="artifacts",
        stem="answers",
        suffix=".jsonl",
    ) != _condition_file(
        second.config,
        second_paths,
        "native_lightrag",
        directory="artifacts",
        stem="answers",
        suffix=".jsonl",
    )


def test_evaluation_resume_reuses_hash_verified_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    harness = _harness(tmp_path)
    paths = harness.condition_paths("qwen35_0_8b")
    input_paths = [
        paths.base_manifest,
        paths.er_artifacts / "merge_plan.json",
        paths.rr_artifacts / "verification_results.jsonl",
        *(
            _condition_file(
                harness.config,
                paths,
                regime,
                directory="artifacts",
                stem=stem,
                suffix=".jsonl",
            )
            for regime in (
                "native_lightrag",
                "advanced_lightrag_er",
                "advanced_lightrag_er_rr",
            )
            for stem in ("retrieval", "answers")
        ),
    ]
    for path in input_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    calls = 0

    def fake_evaluate(builder_key: str) -> StageOutcome:
        nonlocal calls
        calls += 1
        assert builder_key == "qwen35_0_8b"
        metrics_dir = paths.er_dir / "metrics"
        condition_hash = _evaluation_identity_sha256(harness.config)[:16]
        report = metrics_dir / f"summary.{condition_hash}.json"
        paired_primary = (
            metrics_dir / f"paired_primary_er_rr_vs_native.{condition_hash}.jsonl"
        )
        paired_secondary = (
            metrics_dir / f"paired_secondary_er_vs_native.{condition_hash}.jsonl"
        )
        paired_incremental = (
            metrics_dir / f"paired_incremental_er_rr_vs_er.{condition_hash}.jsonl"
        )
        for output in (
            report,
            paired_primary,
            paired_secondary,
            paired_incremental,
            _metric_file(
                harness.config,
                paths,
                "native_lightrag",
                stem="question_metrics",
                suffix=".jsonl",
            ),
            _metric_file(
                harness.config,
                paths,
                "advanced_lightrag_er_rr",
                stem="question_metrics",
                suffix=".jsonl",
            ),
            _metric_file(
                harness.config,
                paths,
                "advanced_lightrag_er",
                stem="question_metrics",
                suffix=".jsonl",
            ),
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}\n", encoding="utf-8")
        return StageOutcome(
            "evaluation",
            paths.base_run_id,
            False,
            {
                "summary": str(report),
                "paired_primary_er_rr_vs_native": str(paired_primary),
            },
        )

    monkeypatch.setattr(harness, "_evaluate_once", fake_evaluate)

    first = harness.evaluation("qwen35_0_8b")
    second = harness.evaluation("qwen35_0_8b")

    assert not first.reused
    assert second.reused
    assert calls == 1


def test_rr_verification_resume_never_recalls_completed_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    harness = _harness(tmp_path)
    paths = harness.condition_paths("qwen35_0_8b")
    text = "Alpha acquired Beta."
    chunk = SimpleNamespace(
        document_id="d1",
        chunk_id="c1",
        chunk_order=0,
        text=text,
        chunk_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    nodes = [
        {
            "canonical_entity_id": "A",
            "entity_name": "Alpha",
            "aliases": ["Alpha"],
            "source_mentions": [
                {"document_id": "d1", "chunk_id": "c1", "original_name": "Alpha"}
            ],
        },
        {
            "canonical_entity_id": "B",
            "entity_name": "Beta",
            "aliases": ["Beta"],
            "source_mentions": [
                {"document_id": "d1", "chunk_id": "c1", "original_name": "Beta"}
            ],
        },
    ]
    mapping = [
        {
            "document_id": "d1",
            "chunk_id": "c1",
            "mention_id": "m1",
            "canonical_entity_id": "A",
        },
        {
            "document_id": "d1",
            "chunk_id": "c1",
            "mention_id": "m2",
            "canonical_entity_id": "B",
        },
    ]
    graph = RewrittenGraph(nodes=nodes, edges=[], mention_to_canonical=mapping)
    plan, summary = build_rr_plan(
        chunks=[chunk], nodes=nodes, edges=[], mention_to_canonical=mapping
    )
    write_immutable_jsonl(paths.rr_artifacts / "chunk_plan.jsonl", plan)
    write_immutable_json(paths.rr_artifacts / "plan_summary.json", summary)
    monkeypatch.setattr(harness, "_er_rewritten_graph", lambda paths: (graph, [chunk]))

    async def verified(*roles):
        return None

    monkeypatch.setattr(harness, "_verify_current_models", verified)

    class Client:
        calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return {"message": {"content": '{"relations":[]}'}}

    role = harness.config.roles.rr_verifier
    client = Client()
    verifier = OllamaRRVerifier(
        client=client,
        model_tag=role.resolved_name,
        model_digest=role.digest,
        prompt_version=role.prompt_version,
        temperature=role.temperature,
        seed=role.seed,
        options={"num_ctx": role.context_window, "num_predict": role.output_tokens},
    )
    monkeypatch.setattr(harness, "_rr_verifier", lambda: verifier)

    first = asyncio.run(harness.rr_verify("qwen35_0_8b"))
    second = asyncio.run(harness.rr_verify("qwen35_0_8b"))
    assert first.reused is False
    assert second.reused is True
    assert client.calls == 1


def test_materialization_publish_then_crash_is_recovered_without_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _harness(tmp_path)
    paths = harness.condition_paths("qwen35_0_8b")
    merge_path = paths.er_artifacts / "merge_plan.json"
    base_payload = paths.base_dir / "base-payload.json"
    graph_payload = paths.graph_artifacts / "graph-payload.json"
    for path, value in (
        (merge_path, {"plan": "merge"}),
        (base_payload, {"extraction": "complete"}),
        (graph_payload, {"graph": "complete"}),
    ):
        write_immutable_json(path, value)
    merge_ref = fingerprint_artifact(
        merge_path, logical_name="er/merge_plan.json", schema_version="1.0.0"
    )
    base_manifest = build_base_manifest(
        harness.config,
        "qwen35_0_8b",
        artifacts={
            "base-payload": fingerprint_artifact(
                base_payload,
                logical_name="base-payload",
                schema_version="1.0.0",
            )
        },
    )
    freeze_manifest(paths.base_manifest, base_manifest)
    advanced_manifest = build_variant_manifest(
        harness.config,
        base_manifest=base_manifest,
        graph_regime="advanced_lightrag_er",
        merge_plan=merge_ref,
        artifacts={
            "graph-payload": fingerprint_artifact(
                graph_payload,
                logical_name="graph/payload",
                schema_version="1.0.0",
            )
        },
    )
    freeze_manifest(paths.er_dir / "variant_manifest.json", advanced_manifest)
    write_immutable_json(
        paths.er_dir / "gates" / "final_workspace.json",
        QualityGateReport(
            gate="final_workspace",
            checks=(GateCheck(name="all", passed=True, detail="published"),),
        ),
    )

    native_workspace = tmp_path / "native-workspace"
    advanced_workspace = tmp_path / "advanced-workspace"
    write_immutable_json(native_workspace / "store.json", {"native": True})
    write_immutable_json(advanced_workspace / "store.json", {"advanced": True})
    native_hash = workspace_tree_sha256(native_workspace)
    advanced_hash = workspace_tree_sha256(advanced_workspace)
    write_immutable_json(
        paths.locator("native_lightrag"),
        {"workspace_dir": str(native_workspace)},
    )
    write_immutable_json(
        paths.locator("advanced_lightrag_er"),
        {
            "schema_version": "2.0.0",
            "base_run_id": paths.base_run_id,
            "variant_run_id": paths.er_variant_run_id,
            "graph_regime": "advanced_lightrag_er",
            "workspace_dir": str(advanced_workspace),
            "workspace_sha256": advanced_hash,
            "native_workspace_sha256_before": native_hash,
            "native_workspace_sha256_after": native_hash,
            "merge_plan_sha256": merge_ref.sha256,
        },
    )

    expectation = harness._expectation(
        paths.er_variant_run_id,
        config_sha256=variant_config_sha256(
            harness.config,
            base_run_id=paths.base_run_id,
            graph_regime="advanced_lightrag_er",
        ),
        inputs={
            "base_manifest": fingerprint_artifact(
                paths.base_manifest,
                logical_name="base_manifest",
                schema_version="2.0.0",
            ),
            "merge_plan": merge_ref,
        },
    )
    first = harness.status.claim(
        "er_materialization", paths.er_variant_run_id, expectation
    )
    assert first.claimed and first.record.attempt_count == 1
    # Simulate a process restart after publication but before mark_completed.
    monkeypatch.setattr(harness, "er_quality_gate", lambda _builder_key: None)
    outcome = asyncio.run(harness.er_materialize("qwen35_0_8b", reclaim_running=True))
    reuse = harness.status.claim(
        "er_materialization", paths.er_variant_run_id, expectation
    )

    assert outcome.reused
    assert outcome.payload["recovered_after_crash"] is True
    assert outcome.payload["workspace_sha256"] == advanced_hash
    assert reuse.reused
    assert reuse.record.attempt_count == 2
