"""Create Native LightRAG instances and report their extraction model."""

from src.extraction.native_capture import (
    build_lightrag,
    extract_role_runtime_metadata,
)

__all__ = ["build_lightrag", "extract_role_runtime_metadata"]
