"""Read-only Ollama identity resolution used by Native preflight."""

from src.extraction.native_capture import (
    resolve_ollama_model_digest,
    resolve_ollama_model_identity,
)

__all__ = ["resolve_ollama_model_digest", "resolve_ollama_model_identity"]
