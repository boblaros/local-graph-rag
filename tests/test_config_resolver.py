from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from src.config import (
    resolve_experiment_config,
    resolve_extraction_prompt_sha256,
    resolve_ollama_inventory,
)

from ._config_helpers import loaded_resolved


class _FakeAsyncClient:
    def __init__(self, *, host: str) -> None:
        self.host = host

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    async def list(self):
        return {
            "models": [
                {
                    "model": "HF.CO/LMSTUDIO-COMMUNITY/MODEL:Q4_K_M",
                    "digest": "sha256:" + "a" * 64,
                    "details": {"quantization_level": "Q4_K_M"},
                }
            ]
        }

    async def show(self, model: str):
        assert model == "HF.CO/LMSTUDIO-COMMUNITY/MODEL:Q4_K_M"
        return {
            "details": {
                "family": "testarch",
                "quantization_level": "Q4_K_M",
            },
            "model_info": {
                "general.architecture": "testarch",
                "testarch.context_length": 2048,
                "testarch.embedding_length": 32,
            },
            "capabilities": ["completion", "embedding"],
        }


def test_ollama_resolution_freezes_reported_name_and_digest(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )
    requested = "hf.co/lmstudio-community/model:Q4_K_M"

    identities, inventory = asyncio.run(
        resolve_ollama_inventory([requested], host="http://local.invalid")
    )

    assert identities[requested] == {
        "requested_tag": requested,
        "resolved_name": "HF.CO/LMSTUDIO-COMMUNITY/MODEL:Q4_K_M",
        "digest": "a" * 64,
        "quantization": "Q4_K_M",
        "context_length": 2048,
        "embedding_length": 32,
        "capabilities": ["completion", "embedding"],
    }
    assert inventory["HF.CO/LMSTUDIO-COMMUNITY/MODEL:Q4_K_M"] == "a" * 64


def test_effective_extraction_prompt_bundle_hash_is_mode_specific() -> None:
    json_hash = resolve_extraction_prompt_sha256(use_json=True)
    text_hash = resolve_extraction_prompt_sha256(use_json=False)

    assert len(json_hash) == 64
    assert len(text_hash) == 64
    assert json_hash != text_hash


def test_effective_prompt_hash_includes_frozen_entity_uniqueness_guidance() -> None:
    default_hash = resolve_extraction_prompt_sha256(use_json=True)
    guided_hash = resolve_extraction_prompt_sha256(
        use_json=True,
        addon_params={"entity_types_guidance": "One unique record per referent."},
    )

    assert guided_hash != default_hash
    assert guided_hash == resolve_extraction_prompt_sha256(
        use_json=True,
        addon_params={"entity_types_guidance": "One unique record per referent."},
    )


def test_resolution_does_not_replace_an_already_frozen_digest(monkeypatch) -> None:
    loaded = loaded_resolved(all_builders=True)
    original = loaded.config.builders[0]
    current_digest = "f" * 64

    async def fake_inventory(requested_tags, *, host):  # noqa: ANN001
        del host
        identities = {
            tag: {
                "requested_tag": tag,
                "resolved_name": tag,
                "digest": current_digest,
                "quantization": "Q4_K_M",
                "context_length": 512,
                "embedding_length": 8,
                "capabilities": [],
            }
            for tag in requested_tags
        }
        return identities, {tag: current_digest for tag in requested_tags}

    monkeypatch.setattr(
        "src.config.resolver.resolve_ollama_inventory",
        fake_inventory,
    )
    resolved, inventory = asyncio.run(resolve_experiment_config(loaded))

    assert resolved.builders[0].resolved_name == original.resolved_name
    assert resolved.builders[0].digest == original.digest
    assert resolved.extraction.prompt_sha256 == loaded.config.extraction.prompt_sha256
    assert inventory[str(original.requested_tag)] == current_digest
