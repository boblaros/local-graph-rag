"""Configuration loading and portable path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import ExperimentConfig


@dataclass(frozen=True)
class LoadedExperimentConfig:
    config: ExperimentConfig
    path: Path

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.base_dir / path).resolve()

    @property
    def manifest_path(self) -> Path:
        return self.resolve(self.config.corpus.manifest_path)

    @property
    def documents_path(self) -> Path:
        return self.resolve(self.config.corpus.documents_path)

    @property
    def questions_path(self) -> Path:
        return self.resolve(self.config.corpus.questions_path)

    @property
    def runs_root(self) -> Path:
        return self.resolve(self.config.runtime.runs_root)


def load_experiment_config(path: str | Path) -> LoadedExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"experiment config must be a YAML mapping: {config_path}")
    return LoadedExperimentConfig(
        config=ExperimentConfig.model_validate(payload),
        path=config_path,
    )


__all__ = ["LoadedExperimentConfig", "load_experiment_config"]
