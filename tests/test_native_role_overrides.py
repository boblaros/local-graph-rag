from __future__ import annotations

import asyncio
import pytest

from src.extraction.native_capture import (
    NATIVE_CAPTURE_VERSION,
    _embedding_with_transient_retry,
    _normalize_runtime_role_overrides,
    _stock_json_contract_is_valid,
    inspect_extraction_response,
)
from src.orchestration.native import build_legacy_run_payload

from ._config_helpers import loaded_resolved


def test_retrieval_role_override_requires_explicit_identity_and_settings() -> None:
    override = {
        "query": {
            "model": "query:resolved",
            "model_digest": "a" * 64,
            "settings": {
                "ollama_host": "http://localhost:11434",
                "timeout_seconds": 30,
                "options": {"temperature": 0.0, "seed": 42},
            },
        }
    }

    assert _normalize_runtime_role_overrides(override) == override


def test_retrieval_role_override_cannot_replace_extraction_role() -> None:
    with pytest.raises(ValueError, match="restricted to retrieval roles"):
        _normalize_runtime_role_overrides(
            {
                "extract": {
                    "model": "different-builder",
                    "model_digest": "b" * 64,
                    "settings": {},
                }
            }
        )


def test_extraction_thinking_is_frozen_as_top_level_ollama_parameter() -> None:
    loaded = loaded_resolved()
    payload = build_legacy_run_payload(
        loaded,
        builder_key="qwen35_0_8b",
        graph_regime="native_lightrag",
        base_run_id="base-test",
        variant_run_id="variant-test",
    )

    assert payload["lightrag"]["llm_model_kwargs"]["think"] is False
    assert "think" not in payload["lightrag"]["llm_options"]
    assert loaded.config.extraction.entity_types_guidance is None
    assert "entity_types_guidance" not in payload["lightrag"]["addon_params"]


def test_stock_json_contract_matches_official_record_shape() -> None:
    payload = {
        "entities": [
            {
                "name": "Apple",
                "type": "organization",
                "description": "A technology company.",
            }
        ],
        "relationships": [
            {
                "source": "Apple",
                "target": "iPhone",
                "keywords": "manufactures",
                "description": "Apple manufactures iPhone.",
            }
        ],
    }

    assert _stock_json_contract_is_valid(payload)


def test_experiment_local_id_shape_is_not_the_stock_contract() -> None:
    payload = {
        "entities": [{"id": "e1", "name": "Apple", "type": "organization"}],
        "relationships": [
            {
                "source_id": "e1",
                "source": "e1",
                "target_id": "e2",
                "target": "e2",
            }
        ],
    }

    assert not _stock_json_contract_is_valid(payload)


def test_capture_audit_separates_strict_json_from_repaired_parse() -> None:
    raw = """```json
    {'entities':[{'name':'Apple','type':'organization','description':'Company'}],
     'relationships':[]}
    ```"""

    audit = inspect_extraction_response(raw)

    assert audit == {
        "json_valid": False,
        "schema_valid": True,
        "parse_success": True,
        "entity_count": 1,
        "relationship_count": 0,
    }
    assert NATIVE_CAPTURE_VERSION == "native-json-passive-capture-v1"


def test_native_embedding_retries_transient_eof_with_identical_operation() -> None:
    calls = 0

    async def operation() -> list[list[float]]:
        nonlocal calls
        calls += 1
        if calls < 4:
            raise RuntimeError("embedding backend EOF (status code: 400)")
        return [[1.0, 2.0]]

    result, attempts = asyncio.run(_embedding_with_transient_retry(operation))

    assert result == [[1.0, 2.0]]
    assert attempts == 4
    assert calls == 4


def test_native_embedding_transient_retry_is_bounded() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("embedding backend EOF (status code: 400)")

    with pytest.raises(RuntimeError, match="embedding backend EOF"):
        asyncio.run(_embedding_with_transient_retry(operation))
    assert calls == 6


def test_native_embedding_does_not_retry_semantic_error() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("embedding dimension mismatch")

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        asyncio.run(_embedding_with_transient_retry(operation))
    assert calls == 1
