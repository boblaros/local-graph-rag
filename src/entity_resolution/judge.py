"""Fixed ER-judge contract and content-addressed decision cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

from .models import EntityProfile, PairScore, canonical_json, content_hash


JudgeRelationship = Literal[
    "same_entity",
    "version_or_variant",
    "role_or_character",
    "related_or_broader_narrower",
    "different_entity",
    "uncertain",
]


@dataclass(frozen=True)
class JudgeIdentity:
    model_tag: str
    model_digest: str
    prompt_version: str
    temperature: float
    seed: int
    generation_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("model_tag", "model_digest", "prompt_version"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"judge {name} must be explicit")
        if self.temperature < 0:
            raise ValueError("judge temperature must be non-negative")
        object.__setattr__(
            self,
            "generation_parameters",
            dict(sorted(self.generation_parameters.items())),
        )

    @property
    def config_hash(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeResult:
    relationship: JudgeRelationship
    rationale: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relationship not in {
            "same_entity",
            "version_or_variant",
            "role_or_character",
            "related_or_broader_narrower",
            "different_entity",
            "uncertain",
        }:
            raise ValueError(f"invalid judge relationship: {self.relationship}")
        if not self.rationale.strip():
            raise ValueError("judge rationale must be non-empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def action(self) -> Literal["merge", "reject", "abstain"]:
        """Map pair semantics to policy action without delegating it to the LLM."""

        if self.relationship == "same_entity":
            return "merge"
        if self.relationship == "uncertain":
            return "abstain"
        return "reject"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JudgeResult":
        return cls(
            relationship=str(value["relationship"]),  # type: ignore[arg-type]
            rationale=str(value["rationale"]),
            metadata=dict(value.get("metadata") or {}),
        )


class FixedJudge(Protocol):
    """A judge must expose frozen identity and a deterministic call surface."""

    identity: JudgeIdentity

    def decide(
        self,
        left: EntityProfile,
        right: EntityProfile,
        score: PairScore,
    ) -> JudgeResult: ...


class JudgeCache:
    """Cache decisions by profile, prompt, model and generation hashes."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self._memory: dict[str, JudgeResult] = {}
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(
        identity: JudgeIdentity,
        left: EntityProfile,
        right: EntityProfile,
        score: PairScore,
    ) -> str:
        profiles = sorted(
            (
                {"mention_id": left.mention_id, "profile_hash": left.profile_hash},
                {"mention_id": right.mention_id, "profile_hash": right.profile_hash},
            ),
            key=lambda item: item["mention_id"],
        )
        return content_hash(
            {
                "judge": identity.to_dict(),
                "judge_config_hash": identity.config_hash,
                "profiles": profiles,
                "pair_score": score.to_dict(),
            }
        )

    def get(self, key: str) -> JudgeResult | None:
        if key in self._memory:
            return self._memory[key]
        if self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cache_key") != key:
            raise ValueError(f"judge cache key mismatch in {path}")
        result = JudgeResult.from_dict(payload["result"])
        self._memory[key] = result
        return result

    def put(self, key: str, result: JudgeResult) -> None:
        existing = self.get(key)
        if existing is not None:
            if existing != result:
                raise ValueError(f"content-addressed judge cache collision for {key}")
            return
        self._memory[key] = result
        if self.directory is None:
            return
        path = self.directory / f"{key}.json"
        payload = canonical_json({"cache_key": key, "result": result.to_dict()}) + "\n"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            on_disk = path.read_text(encoding="utf-8")
            if on_disk != payload:
                raise ValueError(f"content-addressed judge cache collision for {key}")
            return
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)

    def resolve(
        self,
        judge: FixedJudge,
        left: EntityProfile,
        right: EntityProfile,
        score: PairScore,
    ) -> tuple[JudgeResult, str, bool]:
        key = self.key_for(judge.identity, left, right, score)
        cached = self.get(key)
        if cached is not None:
            return cached, key, True
        result = judge.decide(left, right, score)
        self.put(key, result)
        return result, key, False
