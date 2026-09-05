from __future__ import annotations

from src.orchestration.lineage import fingerprint_artifact
from src.orchestration.status import (
    StageExpectation,
    StageStatus,
    StageStatusStore,
)


def test_status_contract_has_all_seven_states() -> None:
    assert [status.value for status in StageStatus] == [
        "pending",
        "running",
        "completed",
        "failed",
        "blocked_by_quality_gate",
        "skipped",
        "stale",
    ]


def test_completed_stage_reuse_requires_matching_lineage_inputs_and_output_hashes(
    tmp_path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text('{"id":1}\n', encoding="utf-8")
    output_path.write_text('{"id":2}\n', encoding="utf-8")
    input_ref = fingerprint_artifact(
        input_path,
        logical_name="input",
        schema_version="2.0.0",
        root=tmp_path,
    )
    output_ref = fingerprint_artifact(
        output_path,
        logical_name="output",
        schema_version="2.0.0",
        root=tmp_path,
    )
    expectation = StageExpectation(
        lineage_id="base_123",
        config_sha256="a" * 64,
        input_artifacts={"input": input_ref},
    )
    store = StageStatusStore(tmp_path / "status.json", artifact_root=tmp_path)
    claimed = store.claim("extract", "builder-a", expectation)
    assert claimed.claimed and claimed.record.attempt_count == 1
    store.mark_completed("extract", "builder-a", {"output": output_ref})

    reused = store.claim("extract", "builder-a", expectation)
    assert reused.reused and not reused.claimed

    output_path.write_text('{"id":3}\n', encoding="utf-8")
    reclaimed = store.claim("extract", "builder-a", expectation)
    assert reclaimed.claimed and not reclaimed.reused
    assert reclaimed.record.status == StageStatus.RUNNING
    assert reclaimed.record.attempt_count == 2
    assert any("mismatch" in reason for reason in reclaimed.reasons)


def test_failed_and_quality_blocked_items_resume_without_destroying_other_items(
    tmp_path,
) -> None:
    store = StageStatusStore(tmp_path / "status.json", artifact_root=tmp_path)
    expectation = StageExpectation(lineage_id="variant_1", config_sha256="b" * 64)
    store.claim("er_plan", "builder-a", expectation)
    store.mark_blocked_by_quality_gate("er_plan", "builder-a", "cannot-link violation")
    assert (
        store.get("er_plan", "builder-a").status == StageStatus.BLOCKED_BY_QUALITY_GATE
    )
    retry = store.claim("er_plan", "builder-a", expectation)
    assert retry.claimed and retry.record.attempt_count == 2
    store.mark_failed("er_plan", "builder-a", RuntimeError("local failure"))

    store.mark_skipped("er_plan", "builder-b", expectation, "not selected")
    records = {record.item_id: record.status for record in store.list(stage="er_plan")}
    assert records == {
        "builder-a": StageStatus.FAILED,
        "builder-b": StageStatus.SKIPPED,
    }
