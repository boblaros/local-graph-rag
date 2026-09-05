#!/usr/bin/env python3
"""Build a publication-ready Hugging Face dataset bundle from final artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REGIMES = (
    "native_lightrag",
    "advanced_lightrag_er",
    "advanced_lightrag_er_rr",
)
PAIRED_PREFIXES = (
    "paired_primary_er_rr_vs_native",
    "paired_secondary_er_vs_native",
    "paired_incremental_er_rr_vs_er",
)


class BundleError(RuntimeError):
    """Raised when a source artifact is missing, ambiguous, or inconsistent."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BundleError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        count = 0
        schema_versions: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BundleError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                count += 1
                if isinstance(value, dict) and value.get("schema_version") is not None:
                    schema_versions.add(str(value["schema_version"]))
        info["record_count"] = count
        if schema_versions:
            info["schema_versions"] = sorted(schema_versions)
    elif path.suffix == ".json":
        load_json(path)
    return info


def resolve_source(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = EXPERIMENT_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise BundleError(f"Source is outside the experiment directory: {path}") from exc
    if not path.is_file():
        raise BundleError(f"Missing source artifact: {path}")
    return path


def single_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise BundleError(
            f"Expected one match for {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def render_readme(builder_keys: list[str]) -> str:
    builders = ", ".join(f"`{key}`" for key in builder_keys)
    return f"""---
license: odc-by
language:
- en
pretty_name: Local GraphRAG Cascade Artifacts
task_categories:
- question-answering
- text-retrieval
tags:
- graphrag
- knowledge-graphs
- entity-resolution
- relation-recovery
- multi-hop-reasoning
configs:
- config_name: questions
  default: true
  data_files: "corpus/questions.jsonl"
- config_name: documents
  data_files: "corpus/documents.jsonl"
- config_name: extraction_calls
  data_files: "extraction/*/raw_calls.jsonl"
- config_name: extraction_chunks
  data_files: "extraction/*/normalized_chunks.jsonl"
- config_name: extraction_entities
  data_files: "extraction/*/normalized_entities.jsonl"
- config_name: extraction_relations
  data_files: "extraction/*/normalized_relations.jsonl"
- config_name: graph_nodes
  data_files: "graphs/*/*/nodes.jsonl"
- config_name: graph_edges
  data_files: "graphs/*/*/edges.jsonl"
- config_name: retrieval
  data_files: "retrieval/*/*.jsonl"
- config_name: answers
  data_files: "answers/*/*.jsonl"
- config_name: question_metrics
  data_files: "evaluation/question_metrics/*/*.jsonl"
- config_name: paired_comparisons
  data_files: "evaluation/builders/*/paired_*.jsonl"
---

# Local GraphRAG Cascade Artifacts

This dataset contains the fixed corpus and final scientific artifacts for a
controlled GraphRAG experiment with 12 local builder models and three cumulative
graph conditions:

```text
Native -> ER -> ER+RR
```

Entity Resolution (ER) is applied to the Native graph. Relation Recovery (RR)
is then applied to the ER graph. The dataset does not contain a Native+RR
condition and does not estimate an independent RR effect.

## Dataset scope

- 120 MultiHop-RAG questions: 30 inference, 30 comparison, 30 temporal, and 30
  unanswerable.
- 155 documents: 125 gold documents and 30 hard negatives.
- 12 builder models and 36 graph conditions.
- 36 retrieval outputs, 36 answer outputs, and 4,320 question-level evaluation
  records.
- Raw and normalized extraction records, final graph node/edge exports, paired
  cascade comparisons, and exploratory cross-builder analyses.

Builders: {builders}.

## Repository structure

```text
README.md
manifest.json
checksums.sha256
corpus/
extraction/
graphs/
retrieval/
answers/
evaluation/
```

| Directory | Contents |
| --- | --- |
| `corpus/` | Questions, documents, subset manifest, audit, and selection provenance |
| `extraction/<builder>/` | Raw model calls and normalized chunks, entities, and relations |
| `graphs/<builder>/<regime>/` | Final graph nodes and edges for each cumulative condition |
| `retrieval/<builder>/` | One 120-record JSONL file per graph condition |
| `answers/<builder>/` | One 120-record JSONL file per graph condition |
| `evaluation/question_metrics/<builder>/` | Question-level retrieval and QA metrics |
| `evaluation/builders/<builder>/` | Builder summaries and three paired cascade contrasts |
| `evaluation/aggregate/` | Prespecified and exploratory cross-builder results |

The top-level `manifest.json` records every published payload file, its role,
record count, source artifact, byte size, and SHA-256 digest. Paths are relative
and contain no machine-specific workspace locations.

## Loading examples

Load a homogeneous JSONL file with 🤗 Datasets:

```python
from datasets import load_dataset

questions = load_dataset(
    "json",
    data_files="corpus/questions.jsonl",
    split="train",
)

metrics = load_dataset(
    "json",
    data_files="evaluation/question_metrics/qwen35_2b/native_lightrag.jsonl",
    split="train",
)
```

The artifact families have different schemas and should be loaded separately.
Join question-level files on `question_id`; use `builder_model`, `base_run_id`,
`variant_run_id`, and `graph_regime` for lineage across stages.

## Integrity

On Linux:

```bash
sha256sum -c checksums.sha256
```

On macOS:

```bash
shasum -a 256 -c checksums.sha256
```

## Provenance and reproducibility

The corpus is a fixed, stratified subset derived from MultiHop-RAG. Extraction
was performed once per builder. ER and RR reuse the saved extraction records,
and retrieval and answering use the same questions and fixed downstream models
across all conditions. Exact model identities, hashes, schema versions, and run
lineage are preserved in the records and `manifest.json`.

The package excludes mutable LightRAG workspaces, vector stores, SQLite state,
logs, caches, and model weights. Those files are execution infrastructure rather
than scientific exchange artifacts.

## Limitations

The subset is balanced by question type rather than sampled to reproduce the
full source distribution. Model outputs may contain extraction, retrieval, or
answering errors and should not be treated as verified facts. Graph structure
alone does not establish semantic correctness or downstream answer quality.
The corpus contains public news text and may mention identifiable people and
organizations present in the source articles.

## License

The source MultiHop-RAG data and this derived artifact collection are provided
under the Open Data Commons Attribution License (ODC-By) v1.0. Attribute Yixuan
Tang and Yi Yang and retain the license and provenance notices when redistributing
the data. Generated artifacts also retain model and source lineage for research
attribution.

ODC-By governs the database and attribution terms for this package. Underlying
article content may remain subject to rights held by its original publishers;
users should review the relevant source terms for uses beyond research and
reproducibility.

## Citation

Please cite the accompanying thesis:

```bibtex
@mastersthesis{{kutivadze2026graphrag,
  author = {{Georgii Kutivadze}},
  title = {{Knowledge Graph Construction Quality in Local GraphRAG: An Empirical Study of Model Capacity, Graph Post-Processing, and Downstream Performance}},
  school = {{Università Cattolica del Sacro Cuore}},
  year = {{2026}},
  type = {{Master's thesis}}
}}
```

Please also cite the source dataset and graph framework:

```bibtex
@misc{{tang2024multihoprag,
  title = {{MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries}},
  author = {{Yixuan Tang and Yi Yang}},
  year = {{2024}},
  eprint = {{2401.15391}},
  archivePrefix = {{arXiv}},
  primaryClass = {{cs.CL}}
}}

@article{{guo2024lightrag,
  title = {{LightRAG: Simple and Fast Retrieval-Augmented Generation}},
  author = {{Zirui Guo and Lianghao Xia and Yanhua Yu and Tu Ao and Chao Huang}},
  year = {{2024}},
  eprint = {{2410.05779}},
  archivePrefix = {{arXiv}},
  primaryClass = {{cs.IR}}
}}
```
"""


def build_bundle(report_path: Path, output_root: Path) -> None:
    report_path = resolve_source(report_path)
    report = load_json(report_path)
    if report.get("builder_count") != 12:
        raise BundleError("The selected global report does not cover 12 builders")
    if tuple(report.get("graph_regimes", [])) != REGIMES:
        raise BundleError("Unexpected graph regime order in the global report")
    if report.get("expected_question_count") != 120:
        raise BundleError("The selected global report does not cover 120 questions")

    if output_root.exists() and any(output_root.iterdir()):
        raise BundleError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    source_to_destination: dict[str, str] = {}

    def add_copy(
        source: str | Path,
        destination: str,
        *,
        category: str,
        artifact_type: str,
        builder_key: str | None = None,
        graph_regime: str | None = None,
        expected: dict[str, Any] | None = None,
        expected_records: int | None = None,
    ) -> dict[str, Any]:
        source_path = resolve_source(source)
        destination_path = output_root / destination
        if destination_path.exists():
            raise BundleError(f"Duplicate destination: {destination}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        inspected = inspect_file(destination_path)

        if expected is not None:
            if inspected["sha256"] != expected.get("sha256"):
                raise BundleError(f"SHA-256 mismatch for {source_path}")
            if inspected["size_bytes"] != expected.get("size_bytes"):
                raise BundleError(f"Size mismatch for {source_path}")
            registered_count = expected.get("record_count")
            if registered_count is not None and inspected.get("record_count") != registered_count:
                raise BundleError(f"Record-count mismatch for {source_path}")
        if expected_records is not None and inspected.get("record_count") != expected_records:
            raise BundleError(
                f"Expected {expected_records} records in {source_path}, "
                f"found {inspected.get('record_count')}"
            )

        entry: dict[str, Any] = {
            "path": destination,
            "category": category,
            "artifact_type": artifact_type,
            "source_path": source_path.relative_to(EXPERIMENT_ROOT).as_posix(),
            **inspected,
        }
        if builder_key is not None:
            entry["builder_key"] = builder_key
        if graph_regime is not None:
            entry["graph_regime"] = graph_regime
        entries.append(entry)
        source_to_destination[str(source_path)] = destination
        return entry

    def add_registered(
        reference: dict[str, Any], destination: str, **metadata: Any
    ) -> dict[str, Any]:
        return add_copy(reference["path"], destination, expected=reference, **metadata)

    validated_gates: Counter[str] = Counter()

    def validate_gate(reference: dict[str, Any], gate_type: str) -> None:
        gate_path = resolve_source(reference["path"])
        inspected = inspect_file(gate_path)
        if inspected["sha256"] != reference.get("sha256"):
            raise BundleError(f"SHA-256 mismatch for lifecycle gate {gate_path}")
        if inspected["size_bytes"] != reference.get("size_bytes"):
            raise BundleError(f"Size mismatch for lifecycle gate {gate_path}")
        gate = load_json(gate_path)
        checks = gate.get("checks")
        if not isinstance(checks, list) or not checks:
            raise BundleError(f"Lifecycle gate has no checks: {gate_path}")
        if not all(
            isinstance(check, dict) and check.get("passed") is True
            for check in checks
        ):
            raise BundleError(f"Lifecycle gate did not pass: {gate_path}")
        validated_gates[gate_type] += 1

    corpus_files = (
        ("questions.jsonl", "questions.jsonl", "questions", 120),
        ("documents.jsonl", "documents.jsonl", "documents", 155),
        ("manifest.json", "subset_manifest.json", "subset_manifest", None),
        ("audit_report.json", "audit_report.json", "subset_audit", None),
        ("selection_report.md", "selection_report.md", "selection_report", None),
    )
    for source_name, destination_name, artifact_type, expected_records in corpus_files:
        add_copy(
            EXPERIMENT_ROOT / "multihoprag_120" / source_name,
            f"corpus/{destination_name}",
            category="corpus",
            artifact_type=artifact_type,
            expected_records=expected_records,
        )
    for source_path in sorted(
        (EXPERIMENT_ROOT / "multihoprag_120" / "selection_provenance").glob("*")
    ):
        if source_path.is_file():
            add_copy(
                source_path,
                f"corpus/selection_provenance/{source_path.name}",
                category="corpus",
                artifact_type="selection_provenance",
            )

    input_artifacts = report.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise BundleError("Global report has no input_artifacts registry")
    selected_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for logical_key, reference in input_artifacts.items():
        parts = logical_key.split("/")
        if len(parts) == 2 and parts[1] in REGIMES:
            selected_metrics[(parts[0], parts[1])] = reference
    builder_keys = sorted({builder for builder, _ in selected_metrics})
    if len(builder_keys) != 12 or len(selected_metrics) != 36:
        raise BundleError("Expected a complete 12-builder x 3-regime metric registry")

    base_manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for manifest_path in sorted((EXPERIMENT_ROOT / "runs" / "base").glob("*/base_manifest.json")):
        manifest = load_json(manifest_path)
        base_run_id = manifest.get("base_run_id")
        if not isinstance(base_run_id, str):
            raise BundleError(f"Missing base_run_id in {manifest_path}")
        base_manifests[base_run_id] = (manifest_path, manifest)

    lineage_builders: dict[str, Any] = {}
    for builder_key in builder_keys:
        selected_variants: dict[str, tuple[Path, dict[str, Any]]] = {}
        for regime in REGIMES:
            metric_reference = selected_metrics[(builder_key, regime)]
            metric_source = resolve_source(metric_reference["path"])
            variant_dir = metric_source.parent.parent
            variant_manifest_path = variant_dir / "variant_manifest.json"
            variant_manifest = load_json(variant_manifest_path)
            if variant_manifest.get("graph_regime") != regime:
                raise BundleError(f"Regime mismatch in {variant_manifest_path}")
            selected_variants[regime] = (variant_dir, variant_manifest)

        base_run_ids = {item[1].get("base_run_id") for item in selected_variants.values()}
        if len(base_run_ids) != 1:
            raise BundleError(f"Variants do not share one extraction for {builder_key}")
        base_run_id = next(iter(base_run_ids))
        if base_run_id not in base_manifests:
            raise BundleError(f"Missing base manifest for {base_run_id}")
        _, base_manifest = base_manifests[base_run_id]
        if base_manifest.get("builder_key") != builder_key:
            raise BundleError(f"Builder mismatch for {base_run_id}")

        extraction_artifacts = base_manifest.get("artifacts", {})
        validate_gate(extraction_artifacts["extraction_gate"], "extraction")
        extraction_mapping = {
            "raw_extraction_calls": "raw_calls.jsonl",
            "normalized_chunks": "normalized_chunks.jsonl",
            "normalized_entities": "normalized_entities.jsonl",
            "normalized_relations": "normalized_relations.jsonl",
        }
        for artifact_key, destination_name in extraction_mapping.items():
            add_registered(
                extraction_artifacts[artifact_key],
                f"extraction/{builder_key}/{destination_name}",
                category="extraction",
                artifact_type=artifact_key,
                builder_key=builder_key,
            )

        native_artifacts_dir = resolve_source(
            extraction_artifacts["native_chunks"]["path"]
        ).parent
        native_graph_sources = {
            "nodes": native_artifacts_dir / "graph_nodes.jsonl",
            "edges": native_artifacts_dir / "graph_edges.jsonl",
        }

        lineage_builders[builder_key] = {
            "base_run_id": base_run_id,
            "builder_requested_tag": base_manifest.get("builder_requested_tag"),
            "builder_resolved_name": base_manifest.get("builder_resolved_name"),
            "builder_digest": base_manifest.get("builder_digest"),
            "variants": {},
        }

        for regime in REGIMES:
            variant_dir, variant_manifest = selected_variants[regime]
            variant_run_id = variant_manifest.get("variant_run_id")
            lineage_builders[builder_key]["variants"][regime] = variant_run_id

            if regime == "native_lightrag":
                graph_sources: dict[str, Any] = native_graph_sources
            else:
                validate_gate(
                    variant_manifest["artifacts"]["final_workspace_gate"],
                    "final_workspace",
                )
                graph_sources = {
                    "nodes": variant_manifest["artifacts"]["nodes"],
                    "edges": variant_manifest["artifacts"]["edges"],
                }
            for graph_part, source_or_reference in graph_sources.items():
                destination = f"graphs/{builder_key}/{regime}/{graph_part}.jsonl"
                if isinstance(source_or_reference, dict):
                    add_registered(
                        source_or_reference,
                        destination,
                        category="graphs",
                        artifact_type=graph_part,
                        builder_key=builder_key,
                        graph_regime=regime,
                    )
                else:
                    add_copy(
                        source_or_reference,
                        destination,
                        category="graphs",
                        artifact_type=graph_part,
                        builder_key=builder_key,
                        graph_regime=regime,
                    )

            retrieval_source = single_match(variant_dir / "artifacts", "retrieval.*.jsonl")
            answers_source = single_match(variant_dir / "artifacts", "answers.*.jsonl")
            add_copy(
                retrieval_source,
                f"retrieval/{builder_key}/{regime}.jsonl",
                category="retrieval",
                artifact_type="retrieval_results",
                builder_key=builder_key,
                graph_regime=regime,
                expected_records=120,
            )
            add_copy(
                answers_source,
                f"answers/{builder_key}/{regime}.jsonl",
                category="answers",
                artifact_type="answer_results",
                builder_key=builder_key,
                graph_regime=regime,
                expected_records=120,
            )
            add_registered(
                selected_metrics[(builder_key, regime)],
                f"evaluation/question_metrics/{builder_key}/{regime}.jsonl",
                category="evaluation",
                artifact_type="question_metrics",
                builder_key=builder_key,
                graph_regime=regime,
            )

        summary_reference = input_artifacts.get(f"{builder_key}/summary")
        if not isinstance(summary_reference, dict):
            raise BundleError(f"Missing summary registry for {builder_key}")
        add_registered(
            summary_reference,
            f"evaluation/builders/{builder_key}/summary.json",
            category="evaluation",
            artifact_type="builder_summary",
            builder_key=builder_key,
        )
        summary_source = resolve_source(summary_reference["path"])
        for prefix in PAIRED_PREFIXES:
            paired_source = single_match(summary_source.parent, f"{prefix}.*.jsonl")
            add_copy(
                paired_source,
                f"evaluation/builders/{builder_key}/{prefix}.jsonl",
                category="evaluation",
                artifact_type=prefix,
                builder_key=builder_key,
                expected_records=120,
            )

    primary_reference = input_artifacts.get("primary_analysis")
    if not isinstance(primary_reference, dict):
        raise BundleError("Missing prespecified primary-analysis registry")
    add_registered(
        primary_reference,
        "evaluation/aggregate/prespecified_cascade_comparisons.json",
        category="evaluation",
        artifact_type="prespecified_cascade_comparisons",
    )
    for artifact_key, destination_name in (
        ("model_pairs", "exploratory_model_pairs.jsonl"),
        ("difference_in_differences", "exploratory_difference_in_differences.jsonl"),
    ):
        reference = report["output_artifacts"][artifact_key]
        add_registered(
            reference,
            f"evaluation/aggregate/{destination_name}",
            category="evaluation",
            artifact_type=artifact_key,
        )

    expected_gates = Counter({"extraction": 12, "final_workspace": 24})
    if validated_gates != expected_gates:
        raise BundleError(
            f"Incomplete lifecycle-gate coverage: {dict(validated_gates)}"
        )

    def rewrite_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rewrite_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite_paths(item) for item in value]
        if isinstance(value, str) and value.startswith("/"):
            source_path = str(resolve_source(value))
            if source_path not in source_to_destination:
                raise BundleError(f"No publication path for report input: {value}")
            return source_to_destination[source_path]
        return value

    sanitized_report = rewrite_paths(report)
    sanitized_report_path = output_root / "evaluation/aggregate/global_experiment_report.json"
    sanitized_report_path.parent.mkdir(parents=True, exist_ok=True)
    with sanitized_report_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitized_report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report_info = inspect_file(sanitized_report_path)
    entries.append(
        {
            "path": "evaluation/aggregate/global_experiment_report.json",
            "category": "evaluation",
            "artifact_type": "global_experiment_report",
            "source_path": report_path.relative_to(EXPERIMENT_ROOT).as_posix(),
            "transformation": "absolute artifact paths rewritten to publication-relative paths",
            **report_info,
        }
    )

    readme_path = output_root / "README.md"
    readme_path.write_text(render_readme(builder_keys), encoding="utf-8")

    category_counts = Counter(entry["category"] for entry in entries)
    manifest = {
        "schema_version": "1.0.0",
        "dataset_name": "local_graphrag_cascade_artifacts",
        "created_at": datetime.now(UTC).isoformat(),
        "license": "ODC-By-1.0",
        "source_subset_id": "multihoprag_120_a3b015d303a201a2",
        "source_global_report": {
            "path": report_path.relative_to(EXPERIMENT_ROOT).as_posix(),
            "sha256": sha256_file(report_path),
        },
        "source_validation": {
            "extraction_gates_passed": validated_gates["extraction"],
            "final_workspace_gates_passed": validated_gates["final_workspace"],
            "jsonl_files_parsed_during_build": True,
            "registered_artifact_hashes_verified_during_build": True,
        },
        "coverage": {
            "builder_count": 12,
            "graph_regimes": list(REGIMES),
            "question_count": 120,
            "document_count": 155,
            "question_metric_records": 4320,
        },
        "lineage": {"builders": lineage_builders},
        "payload": {
            "file_count": len(entries),
            "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
            "files_by_category": dict(sorted(category_counts.items())),
        },
        "files": sorted(entries, key=lambda item: item["path"]),
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    checksum_paths = sorted(
        path for path in output_root.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}"
        for path in checksum_paths
    ]
    (output_root / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=EXPERIMENT_ROOT
        / "runs/analysis/global_experiment_report.34d2c998f8e8143d.json",
        help="Authoritative global experiment report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "huggingface_dataset",
        help="Empty output directory for the publication bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_bundle(args.report, args.output.resolve())
    print(f"Hugging Face dataset bundle created at {args.output.resolve()}")


if __name__ == "__main__":
    main()
