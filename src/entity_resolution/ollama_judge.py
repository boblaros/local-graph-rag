"""Synchronous Ollama adapter for the fixed ambiguous-pair ER judge."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol

from .judge import JudgeIdentity, JudgeResult
from .models import EntityProfile, PairScore, canonical_json, content_hash


_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relationship": {
            "type": "string",
            "enum": [
                "same_entity",
                "version_or_variant",
                "role_or_character",
                "related_or_broader_narrower",
                "different_entity",
                "uncertain",
            ],
        },
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["relationship", "rationale"],
    "additionalProperties": False,
}
_MAX_PHYSICAL_ATTEMPTS = 3
_OVERFLOW_RECOVERY_VERSION = "er-judge-context-overflow-recovery-v1"
_FULL_PROMPT_MODE = "full"
_SEMANTIC_COMPACT_PROMPT_MODE = "semantic_compact"
_BOUNDED_COMPACT_PROMPT_MODE = "bounded_compact"
_COMPACT_OMITTED_PROFILE_FIELDS = (
    "mention_id",
    "document_id",
    "chunk_id",
    "provenance",
    "extraction_call_id",
    "source_mentions",
)
_BOUNDED_DESCRIPTION_CHARS = 1024
_BOUNDED_NEIGHBOUR_ITEMS = 24
_BOUNDED_NEIGHBOUR_ITEM_CHARS = 128
_BOUNDED_RELATION_CONTEXT_ITEMS = 24
_BOUNDED_RELATION_CONTEXT_ITEM_CHARS = 256

_SYSTEM_PROMPT = (
    "You are the fixed corpus-level entity-resolution pair judge. "
    "Classify the semantic relationship between two Native entity "
    "profiles. Choose exactly one relationship. Use same_entity only "
    "for aliases, alternate names, abbreviations, or titles of the "
    "exact same real-world referent. Use version_or_variant for a "
    "version, release, edition, enhanced form, feature, component, or "
    "subtype. Use role_or_character for a person versus their role or "
    "a performer versus a character. Use "
    "related_or_broader_narrower for concepts that are merely related "
    "or differ in scope. Use different_entity for distinct referents. "
    "Use uncertain when the evidence is insufficient or contradictory. "
    "Only same_entity represents identity. Treat missing evidence as "
    "unavailable, not negative evidence. Treat entity type as fallible "
    "evidence already reflected in the pair score, not an automatic "
    "veto. Return only a JSON object with exactly relationship and a "
    "non-empty rationale."
)


class SyncOllamaClient(Protocol):
    def chat(self, **kwargs: Any) -> Any: ...


class OllamaERJudge:
    """Call one explicitly configured Ollama model for ambiguous ER pairs.

    Model discovery/resolution is intentionally outside this adapter.  The
    caller must supply both the resolved tag and digest, normally obtained and
    frozen by preflight.  The digest is lineage/cache identity; Ollama receives
    the resolved tag as its public ``model`` argument.
    """

    def __init__(
        self,
        *,
        client: SyncOllamaClient,
        model_tag: str,
        model_digest: str,
        prompt_version: str,
        temperature: float,
        seed: int,
        options: Mapping[str, Any],
    ) -> None:
        if options is None:  # type: ignore[comparison-overlap]
            raise ValueError(
                "judge options must be explicit (an empty mapping is valid)"
            )
        supplied_options = dict(options)
        for name, expected in (("temperature", temperature), ("seed", seed)):
            if name in supplied_options and supplied_options[name] != expected:
                raise ValueError(f"judge option {name} conflicts with fixed {name}")
        self._client = client
        self._options = {
            **supplied_options,
            "temperature": float(temperature),
            "seed": int(seed),
        }
        self.identity = JudgeIdentity(
            model_tag=model_tag,
            model_digest=model_digest,
            prompt_version=prompt_version,
            temperature=float(temperature),
            seed=int(seed),
            generation_parameters=self._options,
        )

    @staticmethod
    def _response_content(response: Any) -> str:
        if isinstance(response, Mapping):
            message = response.get("message")
        else:
            message = getattr(response, "message", None)
        if isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama judge response has no non-empty message.content")
        return content

    @staticmethod
    def _parse_result(content: str) -> JudgeResult:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Ollama judge returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("Ollama judge JSON must be an object")
        if set(payload) != {"relationship", "rationale"}:
            raise ValueError(
                "Ollama judge JSON must contain exactly relationship and rationale"
            )
        relationship = payload["relationship"]
        rationale = payload["rationale"]
        if relationship not in {
            "same_entity",
            "version_or_variant",
            "role_or_character",
            "related_or_broader_narrower",
            "different_entity",
            "uncertain",
        }:
            raise ValueError(
                f"Ollama judge returned invalid relationship: {relationship!r}"
            )
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("Ollama judge rationale must be a non-empty string")
        return JudgeResult(
            relationship=relationship,
            rationale=rationale.strip(),
        )

    @staticmethod
    def _response_metadata(response: Any) -> dict[str, Any]:
        fields = (
            "prompt_eval_count",
            "eval_count",
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        )
        metadata: dict[str, Any] = {}
        for field in fields:
            if isinstance(response, Mapping):
                value = response.get(field)
            else:
                value = getattr(response, field, None)
            if value is not None:
                metadata[field] = value
        return metadata

    @staticmethod
    def _response_value(response: Any, field: str) -> Any:
        if isinstance(response, Mapping):
            return response.get(field)
        return getattr(response, field, None)

    @staticmethod
    def _profile_prompt_payload(profile: EntityProfile) -> dict[str, Any]:
        """Keep judge context semantic without serializing raw ANN vectors."""

        payload = profile.to_dict()
        embedding = payload.pop("embedding", None)
        payload["embedding_available"] = embedding is not None
        payload["embedding_dimension"] = (
            len(embedding) if embedding is not None else None
        )
        return payload

    @staticmethod
    def _bounded_text(value: str | None, limit: int) -> str | None:
        if value is None or len(value) <= limit:
            return value
        return value[:limit]

    @classmethod
    def _bounded_values(
        cls,
        values: tuple[str, ...] | None,
        *,
        item_limit: int,
        item_char_limit: int,
    ) -> tuple[str, ...] | None:
        if values is None:
            return None
        return tuple(
            cls._bounded_text(value, item_char_limit) or ""
            for value in values[:item_limit]
        )

    @classmethod
    def _compact_profile_prompt_payload(
        cls,
        profile: EntityProfile,
        *,
        bounded: bool,
    ) -> dict[str, Any]:
        """Keep semantic evidence while dropping redundant technical lineage.

        This projection is used only after Ollama proves that the unchanged
        full request saturated ``num_ctx`` and could not produce strict JSON.
        """

        description = profile.description
        neighbours = profile.neighbours
        relation_context = profile.relation_context
        if bounded:
            description = cls._bounded_text(
                description, _BOUNDED_DESCRIPTION_CHARS
            )
            neighbours = cls._bounded_values(
                neighbours,
                item_limit=_BOUNDED_NEIGHBOUR_ITEMS,
                item_char_limit=_BOUNDED_NEIGHBOUR_ITEM_CHARS,
            )
            relation_context = cls._bounded_values(
                relation_context,
                item_limit=_BOUNDED_RELATION_CONTEXT_ITEMS,
                item_char_limit=_BOUNDED_RELATION_CONTEXT_ITEM_CHARS,
            )
        return {
            "original_name": profile.original_name,
            "normalized_name": profile.normalized_name,
            "entity_type": profile.entity_type,
            "type_family": profile.type_family,
            "description": description,
            "neighbours": neighbours,
            "relation_context": relation_context,
            "mention_frequency": profile.mention_frequency,
            "source_diversity": profile.source_diversity,
            "embedding_available": profile.embedding is not None,
            "embedding_dimension": (
                len(profile.embedding) if profile.embedding is not None else None
            ),
        }

    def _context_saturated(self, response: Any) -> bool:
        configured = self._options.get("num_ctx")
        if isinstance(configured, bool) or not isinstance(configured, int):
            return False
        prompt_eval_count = self._response_value(response, "prompt_eval_count")
        if isinstance(prompt_eval_count, bool):
            return False
        try:
            evaluated = int(prompt_eval_count)
        except (TypeError, ValueError):
            return False
        return evaluated >= configured - 1

    def _request_kwargs(
        self,
        left: EntityProfile,
        right: EntityProfile,
        score: PairScore,
        *,
        prompt_mode: str,
    ) -> dict[str, Any]:
        if prompt_mode == _FULL_PROMPT_MODE:
            left_payload = self._profile_prompt_payload(left)
            right_payload = self._profile_prompt_payload(right)
        elif prompt_mode in {
            _SEMANTIC_COMPACT_PROMPT_MODE,
            _BOUNDED_COMPACT_PROMPT_MODE,
        }:
            bounded = prompt_mode == _BOUNDED_COMPACT_PROMPT_MODE
            left_payload = self._compact_profile_prompt_payload(
                left, bounded=bounded
            )
            right_payload = self._compact_profile_prompt_payload(
                right, bounded=bounded
            )
        else:
            raise ValueError(f"unsupported ER judge prompt mode: {prompt_mode}")
        request = {
            "prompt_version": self.identity.prompt_version,
            "left_profile": left_payload,
            "right_profile": right_payload,
            "pair_score": score.to_dict(),
        }
        return {
            "model": self.identity.model_tag,
            "stream": False,
            "format": _RESPONSE_SCHEMA,
            "options": dict(self._options),
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": canonical_json(request)},
            ],
        }

    def decide(
        self,
        left: EntityProfile,
        right: EntityProfile,
        score: PairScore,
    ) -> JudgeResult:
        last_error: ValueError | None = None
        response: Any = None
        parsed: JudgeResult | None = None
        prompt_mode = _FULL_PROMPT_MODE
        overflow_events: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {}
        for physical_attempt in range(1, _MAX_PHYSICAL_ATTEMPTS + 1):
            kwargs = self._request_kwargs(
                left,
                right,
                score,
                prompt_mode=prompt_mode,
            )
            response = self._client.chat(**kwargs)
            try:
                parsed = self._parse_result(self._response_content(response))
            except ValueError as error:
                last_error = error
                if self._context_saturated(response):
                    overflow_events.append(
                        {
                            "physical_attempt": physical_attempt,
                            "prompt_mode": prompt_mode,
                            "prompt_eval_count": self._response_value(
                                response, "prompt_eval_count"
                            ),
                        }
                    )
                    if prompt_mode == _FULL_PROMPT_MODE:
                        prompt_mode = _SEMANTIC_COMPACT_PROMPT_MODE
                    elif prompt_mode == _SEMANTIC_COMPACT_PROMPT_MODE:
                        prompt_mode = _BOUNDED_COMPACT_PROMPT_MODE
                continue
            break
        if parsed is None:
            assert last_error is not None
            raise ValueError(
                "Ollama judge returned no valid strict JSON after "
                f"{_MAX_PHYSICAL_ATTEMPTS} physical attempts"
            ) from last_error
        metadata = self._response_metadata(response)
        metadata["physical_attempts"] = physical_attempt
        if overflow_events:
            metadata.update(
                {
                    "context_overflow_recovered": True,
                    "overflow_recovery_version": _OVERFLOW_RECOVERY_VERSION,
                    "prompt_mode": prompt_mode,
                    "compact_request_sha256": content_hash(kwargs),
                    "overflow_events": overflow_events,
                    "omitted_profile_fields": list(
                        _COMPACT_OMITTED_PROFILE_FIELDS
                    ),
                }
            )
            if prompt_mode == _BOUNDED_COMPACT_PROMPT_MODE:
                metadata["bounded_profile_limits"] = {
                    "description_chars": _BOUNDED_DESCRIPTION_CHARS,
                    "neighbour_items": _BOUNDED_NEIGHBOUR_ITEMS,
                    "neighbour_item_chars": _BOUNDED_NEIGHBOUR_ITEM_CHARS,
                    "relation_context_items": _BOUNDED_RELATION_CONTEXT_ITEMS,
                    "relation_context_item_chars": (
                        _BOUNDED_RELATION_CONTEXT_ITEM_CHARS
                    ),
                }
        return JudgeResult(
            relationship=parsed.relationship,
            rationale=parsed.rationale,
            metadata=metadata,
        )
