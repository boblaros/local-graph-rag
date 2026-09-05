"""Command-line routing; all stage implementation remains importable."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from src.config import (
    LoadedExperimentConfig,
    load_experiment_config,
    resolve_experiment_config,
    resolve_extraction_prompt_sha256,
    write_resolved_config,
)

from .artifacts import jsonable, write_immutable_json
from .harness import ExperimentHarness
from .lineage import full_config_sha256, sha256_json
from .preflight import validate_preflight
from .smoke import run_local_public_api_smoke


def _print(value: Any) -> None:
    print(
        json.dumps(
            jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


def _builder_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--builder", required=True, help="Frozen builder key")


def _reclaim_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reclaim-running",
        action="store_true",
        help="Explicitly reclaim a stage left running by an interrupted process",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment",
        description="Reproducible Native LightRAG vs ER vs ER+RR harness",
    )
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Experiment YAML"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="Resolve identities and validate inputs/models/capacity"
    )
    group = preflight.add_mutually_exclusive_group(required=True)
    group.add_argument("--builder")
    group.add_argument("--all-builders", action="store_true")
    preflight.add_argument(
        "--resolve-output",
        help="Write the resolved copy; unselected roles remain null",
    )
    preflight.add_argument("--report", help="Optional immutable report path")

    subparsers.add_parser(
        "smoke", help="Run a tiny network-free real-LightRAG public API smoke"
    ).add_argument("--report", help="Optional immutable report path")

    native = subparsers.add_parser(
        "native-build", help="Sole extraction plus Native graph build/staging"
    )
    _builder_argument(native)
    _reclaim_argument(native)

    er_plan = subparsers.add_parser(
        "er-plan", help="Offline full-corpus mention-level ER plan"
    )
    _builder_argument(er_plan)
    _reclaim_argument(er_plan)

    er_gate = subparsers.add_parser(
        "er-quality-gate", help="Revalidate saved ER artifacts without judge calls"
    )
    _builder_argument(er_gate)

    er_materialize = subparsers.add_parser(
        "er-materialize", help="Build/reopen/export a clean ER workspace"
    )
    _builder_argument(er_materialize)
    _reclaim_argument(er_materialize)

    rr_plan = subparsers.add_parser(
        "rr-plan", help="Freeze same-chunk missing-pair candidates without LLM calls"
    )
    _builder_argument(rr_plan)
    _reclaim_argument(rr_plan)

    rr_verify = subparsers.add_parser(
        "rr-verify", help="Verify eligible chunks once with the frozen RR model"
    )
    _builder_argument(rr_verify)
    _reclaim_argument(rr_verify)

    rr_gate = subparsers.add_parser(
        "rr-quality-gate", help="Revalidate RR plan/results without model calls"
    )
    _builder_argument(rr_gate)

    rr_materialize = subparsers.add_parser(
        "rr-materialize", help="Build/reopen/export a clean ER+RR workspace"
    )
    _builder_argument(rr_materialize)
    _reclaim_argument(rr_materialize)

    retrieval = subparsers.add_parser(
        "retrieval", help="Run context-only retrieval for one graph variant"
    )
    _builder_argument(retrieval)
    retrieval.add_argument(
        "--regime",
        required=True,
        choices=(
            "native_lightrag",
            "advanced_lightrag_er",
            "advanced_lightrag_er_rr",
        ),
    )
    _reclaim_argument(retrieval)

    answering = subparsers.add_parser(
        "answering", help="Answer only from saved retrieval contexts"
    )
    _builder_argument(answering)
    answering.add_argument(
        "--regime",
        required=True,
        choices=(
            "native_lightrag",
            "advanced_lightrag_er",
            "advanced_lightrag_er_rr",
        ),
    )
    _reclaim_argument(answering)

    evaluation = subparsers.add_parser(
        "evaluation", help="Compute metrics and the three cascade contrasts"
    )
    _builder_argument(evaluation)
    _reclaim_argument(evaluation)

    subparsers.add_parser(
        "primary-analysis",
        help="Collect primary, secondary and incremental effects for 12 builders",
    )
    subparsers.add_parser(
        "exploratory-analysis",
        help="Build 66 model pairs, DiD, family/scale and global report",
    )

    resume = subparsers.add_parser(
        "resume", help="Resume hash-compatible stages for one/all builders"
    )
    resume_group = resume.add_mutually_exclusive_group(required=True)
    resume_group.add_argument("--builder")
    resume_group.add_argument("--all-builders", action="store_true")
    resume.add_argument(
        "--authorize-full-experiment",
        action="store_true",
        help="Required together with metadata.full_experiment_authorized=true",
    )
    _reclaim_argument(resume)

    subparsers.add_parser("status", help="Print lifecycle stage status records")
    return parser


async def _preflight_command(
    args: argparse.Namespace, loaded: LoadedExperimentConfig
) -> int:
    resolved, inventory = await resolve_experiment_config(loaded)
    resolved_path = (
        Path(args.resolve_output).expanduser().resolve()
        if args.resolve_output
        else loaded.path
    )
    if args.resolve_output and resolved_path.parent != loaded.path.parent:
        raise ValueError(
            "--resolve-output must stay beside the source config so its relative "
            "immutable input paths retain the same meaning"
        )
    effective_loaded = LoadedExperimentConfig(
        config=resolved,
        path=resolved_path,
    )
    if args.resolve_output:
        write_resolved_config(effective_loaded.path, resolved)
    report = validate_preflight(
        effective_loaded,
        model_inventory=inventory,
        builder_key=args.builder,
        require_all_builders=bool(args.all_builders),
        require_er=True,
        require_rr=True,
        effective_prompt_sha256=resolve_extraction_prompt_sha256(
            use_json=resolved.extraction.json_extraction,
            addon_params=(
                {"entity_types_guidance": resolved.extraction.entity_types_guidance}
                if resolved.extraction.entity_types_guidance
                else None
            ),
        ),
        fail_closed=False,
    )
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else effective_loaded.runs_root
        / "preflight"
        / f"{full_config_sha256(resolved)}-{sha256_json(report)[:16]}.json"
    )
    write_immutable_json(report_path, report)
    _print(
        {
            "report": str(report_path),
            "resolved_config": str(effective_loaded.path)
            if args.resolve_output
            else None,
            "quality_gate": report,
        }
    )
    return 0 if report.passed else 2


async def _resume_all(
    args: argparse.Namespace, harness: ExperimentHarness
) -> tuple[int, dict[str, Any]]:
    if not args.authorize_full_experiment or not bool(
        harness.config.metadata.get("full_experiment_authorized")
    ):
        raise RuntimeError(
            "all-builder resume requires both --authorize-full-experiment and "
            "metadata.full_experiment_authorized=true"
        )
    completed: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for builder in harness.config.builders:
        try:
            completed[builder.key] = [
                jsonable(item)
                for item in await harness.resume(
                    builder.key, reclaim_running=args.reclaim_running
                )
            ]
        except Exception as error:  # isolate each model condition
            failed[builder.key] = f"{type(error).__name__}: {error}"
    return (1 if failed else 0), {"completed": completed, "failed": failed}


async def _async_main(args: argparse.Namespace) -> int:
    if args.command == "smoke":
        report = await run_local_public_api_smoke()
        if args.report:
            write_immutable_json(Path(args.report).expanduser().resolve(), report)
        _print(report)
        return 0 if report["passed"] else 1

    loaded = load_experiment_config(args.config)
    if args.command == "preflight":
        return await _preflight_command(args, loaded)
    harness = ExperimentHarness(loaded)
    if args.command == "status":
        _print([record.model_dump(mode="json") for record in harness.status.list()])
        return 0
    if args.command == "native-build":
        result = await harness.native_build(
            args.builder, reclaim_running=args.reclaim_running
        )
    elif args.command == "er-plan":
        result = await harness.er_plan(
            args.builder, reclaim_running=args.reclaim_running
        )
    elif args.command == "er-quality-gate":
        result = harness.er_quality_gate(args.builder)
    elif args.command == "er-materialize":
        result = await harness.er_materialize(
            args.builder, reclaim_running=args.reclaim_running
        )
    elif args.command == "rr-plan":
        result = await harness.rr_plan(
            args.builder, reclaim_running=args.reclaim_running
        )
    elif args.command == "rr-verify":
        result = await harness.rr_verify(
            args.builder, reclaim_running=args.reclaim_running
        )
    elif args.command == "rr-quality-gate":
        result = harness.rr_quality_gate(args.builder)
    elif args.command == "rr-materialize":
        result = await harness.rr_materialize(
            args.builder, reclaim_running=args.reclaim_running
        )
    elif args.command == "retrieval":
        result = await harness.retrieval(
            args.builder, args.regime, reclaim_running=args.reclaim_running
        )
    elif args.command == "answering":
        result = await harness.answering(
            args.builder, args.regime, reclaim_running=args.reclaim_running
        )
    elif args.command == "evaluation":
        result = harness.evaluation(args.builder, reclaim_running=args.reclaim_running)
    elif args.command == "primary-analysis":
        result = harness.primary_analysis()
    elif args.command == "exploratory-analysis":
        result = harness.exploratory_analysis()
    elif args.command == "resume":
        if args.all_builders:
            code, result = await _resume_all(args, harness)
            _print(result)
            return code
        result = await harness.resume(
            args.builder, reclaim_running=args.reclaim_running
        )
    else:  # pragma: no cover - argparse prevents this path.
        raise ValueError(f"unknown command {args.command}")
    _print(result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(build_parser().parse_args(argv)))


__all__ = ["build_parser", "main"]
