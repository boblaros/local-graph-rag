"""Read-only preflight validation for inputs, model identities, and capacity."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig, LoadedExperimentConfig

from .lineage import sha256_file
from .quality_gates import QualityGateReport, _GateBuilder


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {line_number}: {exc.msg}")
                    continue
                if not isinstance(value, dict):
                    errors.append(f"line {line_number}: expected JSON object")
                    continue
                rows.append(value)
    except OSError as exc:
        errors.append(str(exc))
    return rows, errors


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _normalized_digest(value: Any) -> str:
    digest = str(value or "").strip().casefold()
    return digest.removeprefix("sha256:")


def _model_inventory_check(
    identity: Any, inventory: Mapping[str, str], label: str
) -> tuple[bool, str]:
    resolved_name = str(identity.resolved_name or "").strip()
    digest = _normalized_digest(identity.digest)
    if not resolved_name or not digest:
        return False, f"{label} identity is unresolved"
    discovered = inventory.get(resolved_name)
    if discovered is None and identity.requested_tag:
        discovered = inventory.get(identity.requested_tag)
    if discovered is None:
        return False, f"{label} resolved model {resolved_name!r} is unavailable"
    actual = _normalized_digest(discovered)
    if actual != digest:
        return (
            False,
            f"{label} digest mismatch: configured={digest}, available={actual}",
        )
    return True, f"{label} resolved to {resolved_name}@{digest}"


def validate_preflight(
    loaded: LoadedExperimentConfig,
    *,
    model_inventory: Mapping[str, str] | None,
    builder_key: str | None = None,
    require_all_builders: bool = False,
    require_er: bool = True,
    require_rr: bool = False,
    disk_path: str | Path | None = None,
    effective_prompt_sha256: str | None = None,
    fail_closed: bool = True,
) -> QualityGateReport:
    """Validate a frozen run configuration without changing external state.

    ``model_inventory`` is deliberately required and should come from the
    preflight caller's Ollama list/show calls. Keys are resolved model names
    (requested aliases are accepted as a compatibility fallback); values are
    the actually discovered digests.
    """

    config: ExperimentConfig = loaded.config
    gate = _GateBuilder("preflight")
    unresolved = config.unresolved_requirements(
        builder_key=builder_key,
        require_all_builders=require_all_builders,
        require_er=require_er,
        require_rr=require_rr,
    )
    gate.check(
        "configuration_resolved",
        not unresolved,
        f"unresolved fields={unresolved}",
    )

    input_paths = {
        "manifest": (loaded.manifest_path, config.corpus.manifest_sha256),
        "documents": (loaded.documents_path, config.corpus.documents_sha256),
        "questions": (loaded.questions_path, config.corpus.questions_sha256),
    }
    hash_errors: list[str] = []
    actual_hashes: dict[str, str] = {}
    for name, (path, expected_hash) in input_paths.items():
        if not path.is_file():
            hash_errors.append(f"{name}: missing {path}")
            continue
        actual_hashes[name] = sha256_file(path)
        if actual_hashes[name] != expected_hash:
            hash_errors.append(f"{name}: SHA-256 mismatch")
    gate.check(
        "immutable_input_hashes",
        not hash_errors,
        "; ".join(hash_errors)
        if hash_errors
        else "manifest/documents/questions match frozen hashes",
    )

    manifest: dict[str, Any] = {}
    manifest_errors: list[str] = []
    if loaded.manifest_path.is_file():
        try:
            value = json.loads(loaded.manifest_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                manifest = value
            else:
                manifest_errors.append("manifest root is not an object")
        except (OSError, json.JSONDecodeError) as exc:
            manifest_errors.append(str(exc))
    else:
        manifest_errors.append("manifest is missing")

    documents, document_errors = _read_jsonl(loaded.documents_path)
    questions, question_errors = _read_jsonl(loaded.questions_path)
    manifest_counts = (
        manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    )
    output_files = (
        manifest.get("output_files")
        if isinstance(manifest.get("output_files"), dict)
        else {}
    )
    output_document = output_files.get("documents.jsonl", {})
    output_question = output_files.get("questions.jsonl", {})
    if not isinstance(output_document, dict):
        output_document = {}
    if not isinstance(output_question, dict):
        output_question = {}
    invariants = manifest.get("invariants", {})
    failed_manifest_invariants = (
        [
            name
            for name, value in invariants.items()
            if not isinstance(value, dict) or value.get("passed") is not True
        ]
        if isinstance(invariants, dict)
        else ["invariants mapping missing"]
    )
    document_roles = Counter(str(row.get("role") or "") for row in documents)
    question_types = Counter(str(row.get("question_type") or "") for row in questions)
    answerability = Counter(bool(row.get("answerable")) for row in questions)
    document_ids = [str(row.get("document_id") or "") for row in documents]
    question_ids = [str(row.get("question_id") or "") for row in questions]
    manifest_errors.extend(document_errors)
    manifest_errors.extend(question_errors)
    expected_shape = (
        manifest.get("subset_id") == config.corpus.subset_id
        and manifest.get("immutable") is True
        and len(documents) == config.corpus.expected_documents
        and len(questions) == config.corpus.expected_questions
        and manifest_counts.get("documents_total") == config.corpus.expected_documents
        and manifest_counts.get("questions_total") == config.corpus.expected_questions
        and manifest_counts.get("gold_documents")
        == config.corpus.expected_gold_documents
        and manifest_counts.get("hard_negative_documents")
        == config.corpus.expected_hard_negatives
        and document_roles["gold"] == config.corpus.expected_gold_documents
        and document_roles["hard_negative"] == config.corpus.expected_hard_negatives
        and question_types
        == {
            "inference": 30,
            "comparison": 30,
            "temporal": 30,
            "unanswerable": 30,
        }
        and answerability == {True: 90, False: 30}
        and all(isinstance(row.get("answerable"), bool) for row in questions)
        and len(document_ids) == len(set(document_ids))
        and len(question_ids) == len(set(question_ids))
        and "" not in document_ids
        and "" not in question_ids
        and output_document.get("sha256") == config.corpus.documents_sha256
        and output_question.get("sha256") == config.corpus.questions_sha256
        and not failed_manifest_invariants
    )
    gate.check(
        "corpus_manifest_consistency",
        not manifest_errors and expected_shape,
        f"documents={len(documents)}, questions={len(questions)}, "
        f"roles={dict(document_roles)}, question_types={dict(question_types)}, "
        f"answerability={dict(answerability)}, "
        f"parse_errors={manifest_errors[:3]}, "
        f"failed_invariants={failed_manifest_invariants[:3]}",
    )

    inventory = dict(model_inventory or {})
    model_errors: list[str] = []
    model_details: list[str] = []
    if model_inventory is None:
        model_errors.append("model inventory was not supplied")
    if builder_key is None:
        selected_builders = config.builders if require_all_builders else config.builders
    elif builder_key in config.builders_by_key:
        selected_builders = (config.builders_by_key[builder_key],)
    else:
        selected_builders = ()
        model_errors.append(f"unknown builder {builder_key!r}")
    identities: list[tuple[str, Any]] = [
        (f"builder.{builder.key}", builder) for builder in selected_builders
    ]
    identities.extend(
        [
            ("roles.query", config.roles.query),
            ("roles.answer", config.roles.answer),
            ("roles.embedding", config.roles.embedding),
        ]
    )
    if require_er:
        if config.roles.er_judge is None:
            model_errors.append("roles.er_judge is not selected")
        else:
            identities.append(("roles.er_judge", config.roles.er_judge))
    if require_rr:
        if config.roles.rr_verifier is None:
            model_errors.append("roles.rr_verifier is not selected")
        else:
            identities.append(("roles.rr_verifier", config.roles.rr_verifier))
    for label, identity in identities:
        valid, detail = _model_inventory_check(identity, inventory, label)
        (model_details if valid else model_errors).append(detail)
    gate.check(
        "resolved_models_available",
        not model_errors,
        f"errors={model_errors[:5]}, verified={len(model_details)}",
    )

    # Cache settings are frozen in config, but model-result caching would make
    # an "exactly one extraction call" claim ambiguous. Keep the base call path
    # explicit and auditable.
    caches_disabled = (
        not config.runtime.lightrag.enable_llm_cache
        and not config.runtime.lightrag.enable_llm_cache_for_entity_extract
        and not config.runtime.lightrag.embedding_cache.enabled
    )
    gate.check(
        "model_caches_disabled",
        caches_disabled,
        "LLM, extraction, and embedding caches must be disabled for base runs",
    )
    configured_prompt = config.extraction.prompt_sha256
    gate.check(
        "extraction_prompt_bundle_frozen",
        bool(configured_prompt)
        and (
            effective_prompt_sha256 is None
            or configured_prompt == effective_prompt_sha256
        ),
        (
            f"configured={configured_prompt}, effective={effective_prompt_sha256}; "
            "callers executing a run must supply the effective bundle hash"
        ),
    )

    # The pinned LightRAG Ollama adapter accepts ``enable_cot`` but explicitly
    # ignores it.  Refuse a configuration that would otherwise claim a query
    # generation setting which the provider cannot reproduce.  Answering uses
    # the public Ollama client directly and may configure ``think`` separately.
    gate.check(
        "query_thinking_supported",
        config.roles.query.generation.think is False,
        "the pinned LightRAG Ollama query adapter ignores enable_cot; "
        "roles.query.generation.think must be false",
    )

    capacity_path = Path(disk_path).expanduser() if disk_path else loaded.runs_root
    capacity_path = _nearest_existing_parent(capacity_path)
    disk_error: str | None = None
    free_gib = -1.0
    try:
        free_gib = shutil.disk_usage(capacity_path).free / (1024**3)
    except OSError as exc:
        disk_error = str(exc)
    enough_disk = (
        disk_error is None and free_gib >= config.runtime.minimum_free_disk_gib
    )
    gate.check(
        "free_disk_space",
        enough_disk,
        f"path={capacity_path}, free_gib={free_gib:.3f}, "
        f"required_gib={config.runtime.minimum_free_disk_gib}, error={disk_error}",
    )

    gate.metrics.update(
        {
            "subset_id": config.corpus.subset_id,
            "documents": len(documents),
            "questions": len(questions),
            "question_types": dict(question_types),
            "answerability": {
                "answerable": answerability[True],
                "unanswerable": answerability[False],
            },
            "models_verified": len(model_details),
            "free_disk_gib": free_gib,
            "input_hashes": actual_hashes,
        }
    )
    return gate.finish(fail_closed=fail_closed)


__all__ = ["validate_preflight"]
