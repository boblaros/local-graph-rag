"""Load and fix the configuration used by Native LightRAG runs."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel

from .native_support.hashes import (
    sha256_file,
    sha256_json,
    sha256_text,
    slugify,
    to_jsonable,
)
from .native_support.io_utils import (
    read_json,
    touch_append_only,
    write_json_exclusive,
)
from .native_support.schemas import (
    SCHEMA_VERSION,
    EnvironmentSnapshot,
    FrozenConfig,
    RunConfig,
)
from .native_support.status import StatusStore


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
# Retained only to resolve configuration locks created before the final harness.
PILOT_DIR = EXPERIMENT_DIR / "pilot"
DEFAULT_RUNS_ROOT = EXPERIMENT_DIR / "runs"
CONFIG_HASH_ALGORITHM = "sha256-canonical-json-v1"
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_RESERVED_CONFIG_KEYS = {"config_hash", "run_id", "frozen_at", "captured_at"}
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_key",
    "access_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "refresh_token",
    "secret",
    "session_token",
    "token",
}
DEFAULT_SAFE_ENV_KEYS: tuple[str, ...] = (
    "OLLAMA_HOST",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "KV_STORAGE",
    "VECTOR_STORAGE",
    "GRAPH_STORAGE",
    "DOC_STATUS_STORAGE",
    "MAX_ASYNC",
    "MAX_ASYNC_LLM",
    "EXTRACT_MAX_ASYNC_LLM",
    "MAX_EXTRACT_INPUT_TOKENS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP_SIZE",
    "CHUNK_F_SIZE",
    "CHUNK_F_OVERLAP_SIZE",
    "CHUNK_F_SPLIT_BY_CHARACTER",
    "CHUNK_F_SPLIT_BY_CHARACTER_ONLY",
    "TOP_K",
    "CHUNK_TOP_K",
)
DEFAULT_PACKAGE_NAMES: tuple[str, ...] = (
    "lightrag-hku",
    "ollama",
    "pydantic",
    "pandas",
    "pyarrow",
    "numpy",
    "networkx",
    "tiktoken",
    "pytest",
    "nbformat",
)


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized == "fixed_token":
        return False
    return normalized in _SENSITIVE_KEYS or any(
        normalized.endswith(f"_{marker}") for marker in _SENSITIVE_KEYS
    )


def assert_config_has_no_secrets(value: Any, *, path: str = "config") -> None:
    """Reject non-empty secret-bearing fields before a config reaches disk."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _is_sensitive_key(key_text) and item not in (None, "", "<redacted>"):
                raise ValueError(
                    "Refusing to freeze secret-bearing field "
                    f"{child_path}; use environment variables"
                )
            assert_config_has_no_secrets(item, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_config_has_no_secrets(item, path=f"{path}[{index}]")


def redact_secrets(value: Any) -> Any:
    """Return a JSON-safe copy with sensitive keyed values replaced."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>"
            if _is_sensitive_key(str(key))
            else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_secrets(item) for item in value]
    return to_jsonable(value)


def canonical_config(config: RunConfig | Mapping[str, Any]) -> RunConfig:
    input_base_dir = config._input_base_dir if isinstance(config, RunConfig) else None
    if isinstance(config, RunConfig):
        payload = config.model_dump(mode="json", exclude_none=False)
    else:
        payload = dict(config)
    for key in _RESERVED_CONFIG_KEYS:
        payload.pop(key, None)
    model = RunConfig.model_validate(payload)
    model._input_base_dir = input_base_dir
    assert_config_has_no_secrets(model)
    return model


def resolve_source_path(config_path: str | Path, source_path: str | Path) -> Path:
    """Resolve a source path relative to the JSON config containing it."""

    source = Path(source_path).expanduser()
    if not source.is_absolute():
        source = Path(config_path).resolve().parent / source
    return source.resolve()


def _fingerprint_jsonl_rows(path: Path, *, kind: str) -> dict[str, Any]:
    if kind == "documents":
        id_field, text_field, declared_hash_field = (
            "document_id",
            "text",
            "text_sha256",
        )
    elif kind == "questions":
        id_field, text_field, declared_hash_field = (
            "question_id",
            "question",
            "question_sha256",
        )
    else:
        raise ValueError(f"unsupported input kind: {kind}")

    if not path.is_file():
        raise FileNotFoundError(f"{kind} JSONL does not exist: {path}")
    row_hashes: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            item_id = row.get(id_field)
            text = row.get(text_field)
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"{path}:{line_number}: missing {id_field}")
            if item_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate {id_field} {item_id}"
                )
            seen_ids.add(item_id)
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: missing string {text_field}")
            actual_hash = sha256_text(text)
            declared_hash = row.get(declared_hash_field)
            if declared_hash is not None and declared_hash != actual_hash:
                raise ValueError(
                    f"{path}:{line_number}: {declared_hash_field} does not match {text_field}"
                )
            row_hashes.append({id_field: item_id, declared_hash_field: actual_hash})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(row_hashes),
        "row_hash_field": declared_hash_field,
        "row_hashes": row_hashes,
        "row_hashes_sha256": sha256_json(row_hashes),
    }


def fingerprint_inputs(
    config: RunConfig | Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> RunConfig:
    """Validate input JSONL files and record file and row hashes."""

    model = canonical_config(config)
    resolved_base = (
        Path(base_dir).resolve()
        if base_dir is not None
        else model._input_base_dir or Path.cwd().resolve()
    )

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (
            candidate if candidate.is_absolute() else resolved_base / candidate
        ).resolve()

    fingerprints = {
        "documents": _fingerprint_jsonl_rows(
            resolve(model.documents_path), kind="documents"
        ),
        "questions": _fingerprint_jsonl_rows(
            resolve(model.questions_path), kind="questions"
        ),
    }
    # Absolute paths describe the local environment, not the scientific
    # configuration. Keep the configured portable path in the lock/hash.
    fingerprints["documents"]["path"] = model.documents_path
    fingerprints["questions"]["path"] = model.questions_path
    existing = model.metadata.get("input_fingerprints")
    if existing is not None and existing != fingerprints:
        raise ValueError(
            "configured input_fingerprints do not match the current input data"
        )

    try:
        relative_base = resolved_base.relative_to(PILOT_DIR)
    except ValueError:
        resolution = {
            "base_kind": "absolute",
            "base_path": str(resolved_base),
            "portable": False,
        }
    else:
        resolution = {
            "base_kind": "pilot_relative",
            "base_path": relative_base.as_posix() or ".",
            "portable": True,
        }
    existing_resolution = model.metadata.get("input_resolution")
    if existing_resolution is not None and existing_resolution != resolution:
        raise ValueError("configured input_resolution does not match the config source")
    metadata = dict(model.metadata)
    metadata["input_fingerprints"] = fingerprints
    metadata["input_resolution"] = resolution
    payload = model.model_dump(mode="json", exclude_none=False)
    payload["metadata"] = metadata
    validated = RunConfig.model_validate(payload)
    validated._input_base_dir = resolved_base
    return validated


def build_run_config(
    payload: RunConfig | Mapping[str, Any],
    *,
    input_base_dir: str | Path | None = None,
    validate_inputs: bool = True,
) -> RunConfig:
    """Construct an effective config after caller-side model preflight.

    ``run_indexing`` can merge resolved Ollama digests/quantization into the
    mapping, call this helper, and then pass the result to ``prepare_run``.
    """

    model = canonical_config(payload)
    if input_base_dir is not None:
        model._input_base_dir = Path(input_base_dir).resolve()
    if validate_inputs:
        return fingerprint_inputs(model, base_dir=model._input_base_dir)
    return model


def load_run_config(path: str | Path) -> RunConfig:
    """Load JSON, resolve its relative input paths, and freeze input hashes."""

    config_path = Path(path).resolve()
    payload = read_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"run config must contain a JSON object: {config_path}")
    return build_run_config(payload, input_base_dir=config_path.parent)


def _resolved_input_base(config: RunConfig) -> Path:
    if config._input_base_dir is not None:
        return config._input_base_dir.resolve()
    resolution = config.metadata.get("input_resolution")
    if not isinstance(resolution, Mapping):
        raise ValueError(
            "frozen config lacks input_resolution; reload the source config with "
            "load_run_config and freeze it again"
        )
    kind = resolution.get("base_kind")
    raw_path = resolution.get("base_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("frozen input_resolution has no base_path")
    if kind == "pilot_relative":
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe pilot-relative input base")
        return (PILOT_DIR / relative).resolve()
    if kind == "absolute":
        absolute = Path(raw_path)
        if not absolute.is_absolute():
            raise ValueError("absolute input base is not absolute")
        return absolute.resolve()
    raise ValueError(f"unknown input base kind: {kind!r}")


def resolved_input_paths(
    config: RunConfig,
    *,
    validate: bool = True,
) -> tuple[Path, Path]:
    """Resolve frozen sources without cwd and optionally revalidate fingerprints."""

    base = _resolved_input_base(config)

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()

    documents_path = resolve(config.documents_path)
    questions_path = resolve(config.questions_path)
    if validate:
        expected = config.metadata.get("input_fingerprints")
        if not isinstance(expected, Mapping):
            raise ValueError("frozen config lacks input_fingerprints")
        actual = {
            "documents": _fingerprint_jsonl_rows(documents_path, kind="documents"),
            "questions": _fingerprint_jsonl_rows(questions_path, kind="questions"),
        }
        actual["documents"]["path"] = config.documents_path
        actual["questions"]["path"] = config.questions_path
        if actual != dict(expected):
            raise ValueError("inputs changed after config freeze")
    return documents_path, questions_path


def validate_frozen_inputs(frozen_run: FrozenRun) -> tuple[Path, Path]:
    """Resolve and re-hash both inputs from a reloaded frozen run."""

    return resolved_input_paths(frozen_run.lock.config, validate=True)


def compute_config_hash(config: RunConfig | Mapping[str, Any]) -> str:
    model = canonical_config(config)
    return sha256_json(model.model_dump(mode="json", exclude_none=False))


def make_run_id(config: RunConfig | Mapping[str, Any], *, hash_length: int = 12) -> str:
    """Encode required scientific dimensions plus the full canonical hash."""

    model = canonical_config(config)
    config_hash = compute_config_hash(model)
    return "__".join(
        (
            f"b-{slugify(model.builder_model)}",
            f"q-{slugify(model.quantization)}",
            f"g-{slugify(model.graph_regime)}",
            f"r-{slugify(model.retrieval_mode)}",
            f"s-{model.seed}",
            f"c-{config_hash[:hash_length]}",
        )
    )


@dataclass(frozen=True)
class RunPaths:
    runs_root: Path
    run_id: str
    run_dir: Path
    config_lock: Path
    environment_json: Path
    status_db: Path
    events_jsonl: Path
    workspace_dir: Path
    artifacts_dir: Path
    metrics_dir: Path

    @classmethod
    def build(cls, runs_root: str | Path, run_id: str) -> "RunPaths":
        if not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
            raise ValueError(f"unsafe run_id: {run_id!r}")
        root = Path(runs_root).resolve()
        run_dir = root / run_id
        return cls(
            runs_root=root,
            run_id=run_id,
            run_dir=run_dir,
            config_lock=run_dir / "config.lock.json",
            environment_json=run_dir / "environment.json",
            status_db=run_dir / "status.sqlite",
            events_jsonl=run_dir / "logs" / "events.jsonl",
            workspace_dir=run_dir / "workspace",
            artifacts_dir=run_dir / "artifacts",
            metrics_dir=run_dir / "metrics",
        )

    def create_directories(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.run_dir,
            self.events_jsonl.parent,
            self.workspace_dir,
            self.artifacts_dir,
            self.metrics_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for log_name in (
            "pipeline.log",
            "indexing.log",
            "retrieval.log",
            "answers.log",
            "errors.jsonl",
        ):
            touch_append_only(self.events_jsonl.parent / log_name)

    @property
    def chunks_jsonl(self) -> Path:
        return self.artifacts_dir / "chunks.jsonl"

    @property
    def extraction_calls_jsonl(self) -> Path:
        return self.artifacts_dir / "extraction_calls.jsonl"

    @property
    def graph_nodes_jsonl(self) -> Path:
        return self.artifacts_dir / "graph_nodes.jsonl"

    @property
    def graph_edges_jsonl(self) -> Path:
        return self.artifacts_dir / "graph_edges.jsonl"

    @property
    def retrieval_jsonl(self) -> Path:
        return self.artifacts_dir / "retrieval.jsonl"

    @property
    def answers_jsonl(self) -> Path:
        return self.artifacts_dir / "answers.jsonl"


@dataclass(frozen=True)
class FrozenRun:
    paths: RunPaths
    lock: FrozenConfig
    resumed: bool

    def open_status(self) -> StatusStore:
        return StatusStore(self.paths.status_db)

    def event_logger(self, *, default_stage: str = "harness") -> Any:
        from .native_support.logging_utils import EventLogger

        return EventLogger(
            self.paths.events_jsonl,
            self.lock.run_id,
            default_stage=default_stage,
            context={
                "builder_model": self.lock.config.builder_model,
                "builder_model_digest": self.lock.config.builder_model_digest,
                "keyword_model": self.lock.config.keyword_model,
                "answer_model": self.lock.config.answer_model,
                "embedding_model": self.lock.config.embedding_model,
            },
        )


def _validate_existing_lock(path: Path, expected: FrozenConfig) -> FrozenConfig:
    existing = FrozenConfig.model_validate(read_json(path))
    if (
        existing.run_id != expected.run_id
        or existing.config_hash != expected.config_hash
    ):
        raise RuntimeError(f"run directory contains a different frozen config: {path}")
    if existing.config.model_dump(
        mode="json", exclude_none=False
    ) != expected.config.model_dump(mode="json", exclude_none=False):
        raise RuntimeError(
            f"config hash collision or non-canonical config lock: {path}"
        )
    return existing


def freeze_run_config(
    config: RunConfig | Mapping[str, Any],
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    input_base_dir: str | Path | None = None,
) -> FrozenRun:
    """Freeze config and create the required run directory skeleton.

    The same deterministic run ID resumes only when the existing lock matches
    byte-for-byte at the canonical config level.
    """

    model = canonical_config(config)
    if input_base_dir is not None:
        model._input_base_dir = Path(input_base_dir).resolve()
    if model.metadata.get("input_fingerprints") is None:
        model = fingerprint_inputs(model, base_dir=model._input_base_dir)
    elif input_base_dir is not None or model._input_base_dir is not None:
        # Re-read when a source base is available; this detects any mutation
        # between config loading and the pre-model-call freeze.
        model = fingerprint_inputs(model, base_dir=model._input_base_dir)
    config_hash = compute_config_hash(model)
    run_id = make_run_id(model)
    paths = RunPaths.build(runs_root, run_id)
    paths.create_directories()
    expected = FrozenConfig(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        config_hash=config_hash,
        config_hash_algorithm=CONFIG_HASH_ALGORITHM,
        config=model,
    )
    resumed = paths.config_lock.exists()
    if resumed:
        lock = _validate_existing_lock(paths.config_lock, expected)
    else:
        try:
            write_json_exclusive(paths.config_lock, expected)
            lock = expected
        except FileExistsError:  # A concurrent initializer won the race.
            resumed = True
            lock = _validate_existing_lock(paths.config_lock, expected)
    return FrozenRun(paths=paths, lock=lock, resumed=resumed)


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    if not parsed.scheme or not parsed.netloc:
        return value
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    if port:
        hostname = f"{hostname}:{port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def safe_environment_variables(
    names: Sequence[str] = DEFAULT_SAFE_ENV_KEYS,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        if _is_sensitive_key(name):
            continue
        value = os.getenv(name)
        if value is None:
            continue
        result[name] = _safe_url(value) if name.endswith("_HOST") else value
    return result


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _python_tree_sha256(root: Path) -> str | None:
    if not root.is_dir():
        return None
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        return None
    return sha256_json(
        {str(path.relative_to(root)): sha256_file(path) for path in files}
    )


def lightrag_source_identity() -> dict[str, Any]:
    """Identify the local source used to build the LightRAG dependency."""

    checkout = (EXPERIMENT_DIR / "LightRAG").resolve()
    result: dict[str, Any] = {
        "path": str(checkout),
        "python_tree_sha256": _python_tree_sha256(checkout / "lightrag"),
    }
    if (checkout / ".git").exists():
        result.update(
            {
                "git_commit": _git_value(checkout, "rev-parse", "HEAD"),
                "git_describe": _git_value(
                    checkout, "describe", "--tags", "--always", "--dirty"
                ),
                "git_dirty": bool(
                    _git_value(
                        checkout,
                        "status",
                        "--porcelain",
                        "--untracked-files=normal",
                    )
                ),
            }
        )
    return result


def _lightrag_checkout() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        distribution = importlib.metadata.distribution("lightrag-hku")
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            result["editable"] = bool(direct_url.get("dir_info", {}).get("editable"))
            result["distribution_source"] = direct_url.get("url")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass

    spec = importlib.util.find_spec("lightrag")
    if spec and spec.origin:
        package_path = Path(spec.origin).resolve()
        result["package_path"] = str(package_path)
        result["package_tree_sha256"] = _python_tree_sha256(package_path.parent)
        checkout = package_path.parent.parent
        if (checkout / ".git").exists():
            result.update(
                {
                    "git_root": str(checkout),
                    "git_commit": _git_value(checkout, "rev-parse", "HEAD"),
                    "git_describe": _git_value(
                        checkout, "describe", "--tags", "--always"
                    ),
                    "git_commit_date": _git_value(
                        checkout, "log", "-1", "--format=%cI"
                    ),
                    "git_dirty": bool(
                        _git_value(
                            checkout,
                            "status",
                            "--porcelain",
                            "--untracked-files=normal",
                        )
                    ),
                }
            )
    return result


def capture_environment(
    run_id: str,
    config_hash: str,
    *,
    extra: Mapping[str, Any] | None = None,
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
    safe_env_names: Sequence[str] = DEFAULT_SAFE_ENV_KEYS,
) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        run_id=run_id,
        config_hash=config_hash,
        python={
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        packages=_package_versions(package_names),
        lightrag_checkout=_lightrag_checkout(),
        safe_environment=safe_environment_variables(safe_env_names),
        extra=redact_secrets(extra or {}),
    )


def freeze_environment(
    frozen_run: FrozenRun,
    *,
    extra: Mapping[str, Any] | None = None,
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
    safe_env_names: Sequence[str] = DEFAULT_SAFE_ENV_KEYS,
) -> EnvironmentSnapshot:
    path = frozen_run.paths.environment_json
    if path.exists():
        existing = EnvironmentSnapshot.model_validate(read_json(path))
        if (
            existing.run_id != frozen_run.lock.run_id
            or existing.config_hash != frozen_run.lock.config_hash
        ):
            raise RuntimeError(f"environment snapshot belongs to another run: {path}")
        return existing
    snapshot = capture_environment(
        frozen_run.lock.run_id,
        frozen_run.lock.config_hash,
        extra=extra,
        package_names=package_names,
        safe_env_names=safe_env_names,
    )
    try:
        write_json_exclusive(path, snapshot)
        return snapshot
    except FileExistsError:
        return EnvironmentSnapshot.model_validate(read_json(path))


def prepare_run(
    config: RunConfig | Mapping[str, Any],
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    environment_extra: Mapping[str, Any] | None = None,
    input_base_dir: str | Path | None = None,
) -> FrozenRun:
    """Required pre-model-call entry point for a fully initialized run."""

    frozen_run = freeze_run_config(
        config,
        runs_root=runs_root,
        input_base_dir=input_base_dir,
    )
    portable_run_config = frozen_run.paths.run_dir / "run_config.json"
    if portable_run_config.exists():
        existing = RunConfig.model_validate(read_json(portable_run_config))
        if existing.model_dump(mode="json", exclude_none=False) != (
            frozen_run.lock.config.model_dump(mode="json", exclude_none=False)
        ):
            raise RuntimeError(
                f"run_config.json differs from immutable config lock: {portable_run_config}"
            )
    else:
        try:
            write_json_exclusive(portable_run_config, frozen_run.lock.config)
        except FileExistsError:
            existing = RunConfig.model_validate(read_json(portable_run_config))
            if existing.model_dump(mode="json", exclude_none=False) != (
                frozen_run.lock.config.model_dump(mode="json", exclude_none=False)
            ):
                raise RuntimeError(
                    "concurrent run_config.json initializer wrote a different config"
                )
    freeze_environment(frozen_run, extra=environment_extra)
    with frozen_run.open_status():
        pass
    touch_append_only(frozen_run.paths.events_jsonl)
    return frozen_run


def load_frozen_run(run_dir: str | Path) -> FrozenRun:
    directory = Path(run_dir).resolve()
    lock = FrozenConfig.model_validate(read_json(directory / "config.lock.json"))
    actual_hash = compute_config_hash(lock.config)
    if actual_hash != lock.config_hash:
        raise ValueError(f"frozen config hash mismatch: {directory}")
    if make_run_id(lock.config) != lock.run_id:
        raise ValueError(f"frozen run_id does not match config dimensions: {directory}")
    paths = RunPaths.build(directory.parent, lock.run_id)
    if paths.run_dir != directory:
        raise ValueError(
            f"run directory name does not match frozen run_id: {directory}"
        )
    return FrozenRun(paths=paths, lock=lock, resumed=True)


def validate_lightrag_runtime(frozen_run: FrozenRun) -> dict[str, Any]:
    """Fail model/storage stages if the active checkout drifted after freeze."""

    stored = EnvironmentSnapshot.model_validate(
        read_json(frozen_run.paths.environment_json)
    )
    if (
        stored.run_id != frozen_run.lock.run_id
        or stored.config_hash != frozen_run.lock.config_hash
    ):
        raise RuntimeError("environment.json does not belong to the frozen run")
    current = _lightrag_checkout()
    expected_commit = frozen_run.lock.config.metadata.get(
        "expected_lightrag_git_commit"
    ) or stored.lightrag_checkout.get("git_commit")
    actual_commit = current.get("git_commit")
    if (
        expected_commit
        and actual_commit is not None
        and actual_commit != expected_commit
    ):
        raise RuntimeError(
            "active LightRAG checkout commit drifted after freeze: "
            f"expected {expected_commit}, found {actual_commit}"
        )
    stored_package_hash = stored.lightrag_checkout.get("package_tree_sha256")
    if (
        stored_package_hash
        and current.get("package_tree_sha256") != stored_package_hash
    ):
        raise RuntimeError(
            "active LightRAG installed package code drifted after freeze"
        )
    expected_source_hash = frozen_run.lock.config.metadata.get(
        "expected_lightrag_source_sha256"
    )
    expected_source_describe = frozen_run.lock.config.metadata.get(
        "expected_lightrag_git_describe"
    )
    current_source = lightrag_source_identity()
    if expected_commit and current_source.get("git_commit") != expected_commit:
        raise RuntimeError(
            "LightRAG source checkout commit drifted after freeze: "
            f"expected {expected_commit}, found {current_source.get('git_commit')}"
        )
    if (
        expected_source_describe
        and current_source.get("git_describe") != expected_source_describe
    ):
        raise RuntimeError("LightRAG source checkout dirty state drifted after freeze")
    if (
        expected_source_hash
        and current_source.get("python_tree_sha256") != expected_source_hash
    ):
        raise RuntimeError("LightRAG source checkout code drifted after freeze")
    try:
        current_version = importlib.metadata.version("lightrag-hku")
    except importlib.metadata.PackageNotFoundError:
        current_version = None
    stored_version = stored.packages.get("lightrag-hku")
    if stored_version and current_version != stored_version:
        raise RuntimeError(
            "active LightRAG package version drifted after freeze: "
            f"expected {stored_version}, found {current_version}"
        )
    return current


__all__ = [
    "CONFIG_HASH_ALGORITHM",
    "DEFAULT_PACKAGE_NAMES",
    "DEFAULT_RUNS_ROOT",
    "DEFAULT_SAFE_ENV_KEYS",
    "PILOT_DIR",
    "FrozenRun",
    "RunPaths",
    "assert_config_has_no_secrets",
    "build_run_config",
    "canonical_config",
    "capture_environment",
    "compute_config_hash",
    "fingerprint_inputs",
    "freeze_environment",
    "freeze_run_config",
    "load_frozen_run",
    "lightrag_source_identity",
    "load_run_config",
    "make_run_id",
    "prepare_run",
    "redact_secrets",
    "resolved_input_paths",
    "resolve_source_path",
    "safe_environment_variables",
    "validate_frozen_inputs",
    "validate_lightrag_runtime",
]
