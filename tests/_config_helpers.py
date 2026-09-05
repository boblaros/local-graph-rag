from __future__ import annotations

import hashlib
from pathlib import Path

from src.config import ExperimentConfig, LoadedExperimentConfig, load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "experiment.yaml"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def resolved_config(*, all_builders: bool = False) -> ExperimentConfig:
    template = load_experiment_config(TEMPLATE).config
    payload = template.model_dump(mode="json", exclude_none=False)
    for index, builder in enumerate(payload["builders"]):
        if all_builders or index == 0:
            builder["resolved_name"] = builder["requested_tag"]
            builder["digest"] = digest(f"builder:{builder['key']}")
    payload["roles"]["query"].update(
        requested_tag="query:test",
        resolved_name="query:test",
        digest=digest("query"),
    )
    payload["roles"]["answer"].update(
        requested_tag="answer:test",
        resolved_name="answer:test",
        digest=digest("answer"),
    )
    payload["roles"]["embedding"].update(
        requested_tag="embedding:test",
        resolved_name="embedding:test",
        digest=digest("embedding"),
        dimension=8,
        max_tokens=512,
    )
    payload["roles"]["er_judge"] = {
        "requested_tag": "judge:test",
        "resolved_name": "judge:test",
        "digest": digest("judge"),
        "provider": "ollama",
        "prompt_version": "er-judge-test-v1",
        "temperature": 0.0,
        "seed": 42,
        "context_window": 4096,
        "output_tokens": 64,
    }
    payload["roles"]["rr_verifier"] = {
        "requested_tag": "rr:test",
        "resolved_name": "rr:test",
        "digest": digest("rr"),
        "provider": "ollama",
        "prompt_version": "rr-relation-only-v1",
        "temperature": 0.0,
        "seed": 42,
        "context_window": 4096,
        "output_tokens": 256,
    }
    payload["extraction"]["prompt_sha256"] = digest("effective-extraction-prompt")
    payload["graph_materialization"]["missing_description_policy"] = (
        "preserve_with_audited_placeholders"
    )
    payload["runtime"]["minimum_free_disk_gib"] = 0.0
    return ExperimentConfig.model_validate(payload)


def loaded_resolved(*, all_builders: bool = False) -> LoadedExperimentConfig:
    return LoadedExperimentConfig(
        config=resolved_config(all_builders=all_builders),
        path=TEMPLATE,
    )


def inventory(
    config: ExperimentConfig, *, all_builders: bool = False
) -> dict[str, str]:
    selected = config.builders if all_builders else (config.builders[0],)
    models = [
        *selected,
        config.roles.query,
        config.roles.answer,
        config.roles.embedding,
    ]
    if config.roles.er_judge is not None:
        models.append(config.roles.er_judge)
    if config.roles.rr_verifier is not None:
        models.append(config.roles.rr_verifier)
    return {
        str(model.resolved_name): str(model.digest)
        for model in models
        if model.resolved_name and model.digest
    }
