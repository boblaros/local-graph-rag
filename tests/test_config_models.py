from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import EXPECTED_BUILDERS, ExperimentConfig, load_experiment_config
from src.entity_resolution import OllamaERJudge
from src.entity_resolution.models import ERConfig
from src.orchestration.lineage import sha256_json

from ._config_helpers import TEMPLATE, resolved_config


def test_committed_config_contains_exact_twelve_builders_and_is_resolved() -> None:
    config = load_experiment_config(TEMPLATE).config
    assert (
        tuple((item.key, item.requested_tag) for item in config.builders)
        == EXPECTED_BUILDERS
    )
    assert config.roles.er_judge is not None
    assert config.roles.query.resolved_name is not None
    assert config.roles.answer.resolved_name is not None
    assert config.roles.embedding.resolved_name is not None
    assert config.graph_materialization.missing_description_policy == (
        "preserve_with_audited_placeholders"
    )
    assert config.unresolved_requirements(require_all_builders=True) == []
    config.assert_ready(require_all_builders=True)


def test_builder_contract_rejects_changed_or_reordered_tag() -> None:
    config = load_experiment_config(TEMPLATE).config
    payload = config.model_dump(mode="json", exclude_none=False)
    payload["builders"][0]["requested_tag"] = "different:model"
    with pytest.raises(ValidationError, match="exact 12 ordered experiment tags"):
        ExperimentConfig.model_validate(payload)


def test_model_roles_are_independent_from_builder_and_each_other() -> None:
    config = resolved_config()
    builder = config.builders[0]
    assert (
        len(
            {
                builder.requested_tag,
                config.roles.query.requested_tag,
                config.roles.answer.requested_tag,
                config.roles.embedding.requested_tag,
                config.roles.er_judge.requested_tag,
                config.roles.rr_verifier.requested_tag,
            }
        )
        == 6
    )
    changed = config.model_copy(
        update={
            "roles": config.roles.model_copy(
                update={
                    "query": config.roles.query.model_copy(
                        update={"resolved_name": "another-query:test"}
                    )
                }
            )
        }
    )
    assert changed.builders == config.builders
    assert changed.roles.er_judge == config.roles.er_judge
    assert changed.roles.rr_verifier == config.roles.rr_verifier


def test_runtime_has_no_hidden_lightrag_storage_or_retrieval_defaults() -> None:
    config = load_experiment_config(TEMPLATE).config
    assert config.runtime.ollama_host == "http://localhost:11434"
    assert config.runtime.lightrag.storage.model_dump() == {
        "kv_storage": "JsonKVStorage",
        "vector_storage": "NanoVectorDBStorage",
        "graph_storage": "NetworkXStorage",
        "doc_status_storage": "JsonDocStatusStorage",
    }
    assert config.runtime.retrieval.mode == "hybrid"
    assert config.runtime.retrieval.top_k == 20
    flattened = config.runtime.lightrag.legacy_payload()
    assert flattened["graph_storage"] == "NetworkXStorage"
    assert flattened["llm_timeout"] == 900
    assert "storage" not in flattened
    assert "mode" not in config.runtime.retrieval.legacy_payload()
    payload = config.model_dump(mode="json", exclude_none=False)
    payload["runtime"]["lightrag"]["unknown_setting"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(payload)


def test_qwen3_8b_has_an_audited_operational_judge_budget_override() -> None:
    config = load_experiment_config(TEMPLATE).config

    default_budget, default_override = config.effective_er_judge_budget(
        "qwen3_4b"
    )
    qwen_budget, qwen_override = config.effective_er_judge_budget("qwen3_8b")

    assert default_budget == 650
    assert default_override is None
    assert qwen_budget == 1000
    assert qwen_override is not None
    assert "824" in qwen_override.reason
    assert config.entity_resolution.decision_policy.max_judge_calls_per_run == 650


def test_er_config_maps_one_to_one_to_implemented_pipeline_fields() -> None:
    config = resolved_config()
    payload = config.entity_resolution.pipeline_payload()
    assert payload["decision_policy"].reject_below_score == 0.5
    assert payload["decision_policy"].auto_merge_at_score == 0.76
    assert payload["fuzzy_candidate_threshold"] == 0.70
    assert config.entity_resolution.version == "corpus-native-merge-only-er-v8"
    assert config.entity_resolution.config_version == "8.0.0"
    assert payload["scoring_weights"] == config.entity_resolution.signal_weights
    assert set(payload) == {
        "decision_policy",
        "fuzzy_candidate_threshold",
        "containment_min_chars",
        "max_block_size",
        "er_version",
        "scoring_weights",
    }
    implemented = ERConfig(**payload)
    assert implemented.config_hash
    assert implemented.config_hash == sha256_json(payload)
    assert implemented.scoring_weights == config.entity_resolution.signal_weights
    assert config.entity_resolution.embedding_neighbor_k == 3
    assert config.entity_resolution.embedding_lsh_tables == 4
    assert "embedding_neighbor_k" not in payload
    assert config.entity_resolution.embedding_candidate_payload() == {
        "embedding_neighbor_k": 3,
        "embedding_lsh_tables": 4,
        "embedding_lsh_bits": 10,
        "embedding_lsh_max_bucket": 200,
    }


def test_removed_type_constraint_fields_are_rejected() -> None:
    payload = resolved_config().model_dump(mode="json", exclude_none=False)
    payload["entity_resolution"]["incompatible_type_families"] = [
        ["person", "organization"]
    ]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(payload)


def test_er_judge_identity_payload_exactly_matches_runtime_cache_identity() -> None:
    judge = resolved_config().roles.er_judge
    adapter = OllamaERJudge(
        client=object(),
        model_tag=judge.resolved_name,
        model_digest=judge.digest,
        prompt_version=judge.prompt_version,
        temperature=judge.temperature,
        seed=judge.seed,
        options={
            "num_ctx": judge.context_window,
            "num_predict": judge.output_tokens,
        },
    )
    assert adapter.identity.to_dict() == judge.identity_payload()
    assert adapter.identity.config_hash == sha256_json(judge.identity_payload())


def test_config_and_orchestration_do_not_depend_on_scripts_or_evaluation() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    offenders: list[str] = []
    paths = [*(root / "config").glob("*.py")]
    paths.extend(
        root / "orchestration" / name
        for name in ("lineage.py", "status.py", "preflight.py", "quality_gates.py")
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(
                "scripts" in name.split(".") or "evaluation" in name.split(".")
                for name in names
            ):
                offenders.append(f"{path.name}:{getattr(node, 'lineno', 0)}")
    assert offenders == []
