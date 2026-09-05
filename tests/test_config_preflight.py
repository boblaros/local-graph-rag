from __future__ import annotations

from src.config import LoadedExperimentConfig, load_experiment_config
from src.orchestration.preflight import validate_preflight

from ._config_helpers import TEMPLATE, inventory, loaded_resolved


def test_committed_config_preflight_still_requires_runtime_model_inventory() -> None:
    loaded = load_experiment_config(TEMPLATE)
    report = validate_preflight(
        loaded,
        builder_key="qwen35_0_8b",
        model_inventory=None,
        fail_closed=False,
    )
    assert not report.passed
    configuration = next(
        check for check in report.checks if check.name == "configuration_resolved"
    )
    assert configuration.passed
    assert any("model inventory" in error for error in report.errors)


def test_resolved_single_builder_preflight_passes_real_immutable_corpus() -> None:
    loaded = loaded_resolved()
    report = validate_preflight(
        loaded,
        builder_key="qwen35_0_8b",
        model_inventory=inventory(loaded.config),
    )
    assert report.passed
    assert report.metrics["documents"] == 155
    assert report.metrics["questions"] == 120
    assert report.metrics["question_types"] == {
        "inference": 30,
        "comparison": 30,
        "temporal": 30,
        "unanswerable": 30,
    }
    assert report.metrics["answerability"] == {
        "answerable": 90,
        "unanswerable": 30,
    }


def test_full_twelve_builder_preflight_requires_and_verifies_every_digest() -> None:
    loaded = loaded_resolved(all_builders=True)
    report = validate_preflight(
        loaded,
        require_all_builders=True,
        model_inventory=inventory(loaded.config, all_builders=True),
    )
    assert report.passed
    # 12 builders plus query, answer, embedding and the fixed ER judge.
    assert report.metrics["models_verified"] == 16


def test_preflight_rejects_configured_hash_or_model_digest_mismatch() -> None:
    loaded = loaded_resolved()
    bad_corpus = loaded.config.corpus.model_copy(update={"documents_sha256": "0" * 64})
    bad_loaded = LoadedExperimentConfig(
        config=loaded.config.model_copy(update={"corpus": bad_corpus}),
        path=loaded.path,
    )
    models = inventory(loaded.config)
    models[loaded.config.builders[0].resolved_name] = "f" * 64
    report = validate_preflight(
        bad_loaded,
        builder_key="qwen35_0_8b",
        model_inventory=models,
        fail_closed=False,
    )
    assert not report.passed
    assert any("SHA-256 mismatch" in error for error in report.errors)
    assert any("digest mismatch" in error for error in report.errors)


def test_preflight_rejects_query_thinking_ignored_by_pinned_adapter() -> None:
    loaded = loaded_resolved()
    query = loaded.config.roles.query.model_copy(
        update={
            "generation": loaded.config.roles.query.generation.model_copy(
                update={"think": True}
            )
        }
    )
    changed = LoadedExperimentConfig(
        config=loaded.config.model_copy(
            update={"roles": loaded.config.roles.model_copy(update={"query": query})}
        ),
        path=loaded.path,
    )

    report = validate_preflight(
        changed,
        builder_key="qwen35_0_8b",
        model_inventory=inventory(changed.config),
        fail_closed=False,
    )

    assert not report.passed
    assert any("ignores enable_cot" in error for error in report.errors)
