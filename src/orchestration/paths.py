"""Stable filesystem layout for base extractions and derived graph variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConditionPaths:
    runs_root: Path
    base_run_id: str
    native_variant_run_id: str
    er_variant_run_id: str
    rr_variant_run_id: str

    @property
    def base_dir(self) -> Path:
        return self.runs_root / "base" / self.base_run_id

    @property
    def staging_dir(self) -> Path:
        return self.base_dir / "artifacts" / "extraction"

    @property
    def base_manifest(self) -> Path:
        return self.base_dir / "base_manifest.json"

    @property
    def native_dir(self) -> Path:
        return self.runs_root / "variants" / self.native_variant_run_id

    @property
    def er_dir(self) -> Path:
        return self.runs_root / "variants" / self.er_variant_run_id

    @property
    def rr_dir(self) -> Path:
        return self.runs_root / "variants" / self.rr_variant_run_id

    def variant_dir(self, graph_regime: str) -> Path:
        if graph_regime == "native_lightrag":
            return self.native_dir
        if graph_regime == "advanced_lightrag_er":
            return self.er_dir
        if graph_regime == "advanced_lightrag_er_rr":
            return self.rr_dir
        raise ValueError(f"unsupported graph regime: {graph_regime}")

    def locator(self, graph_regime: str) -> Path:
        return self.variant_dir(graph_regime) / "workspace_locator.json"

    @property
    def er_artifacts(self) -> Path:
        return self.er_dir / "artifacts" / "er"

    @property
    def graph_artifacts(self) -> Path:
        return self.er_dir / "artifacts" / "graph"

    @property
    def rr_artifacts(self) -> Path:
        return self.rr_dir / "artifacts" / "rr"

    @property
    def rr_graph_artifacts(self) -> Path:
        return self.rr_dir / "artifacts" / "graph"


__all__ = ["ConditionPaths"]
