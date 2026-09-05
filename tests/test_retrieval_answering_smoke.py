from __future__ import annotations

import asyncio
from pathlib import Path

from src.answering.runner import INSUFFICIENT_INFORMATION, run_answering
from src.retrieval.runner import run_retrieval


class Param:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeRag:
    class Tokenizer:
        @staticmethod
        def encode(value):
            return value.split()

    tokenizer = Tokenizer()

    async def aquery_llm(self, question, *, param):
        assert param.kwargs["only_need_context"] is True
        assert param.kwargs["stream"] is False
        if "unanswerable" in question:
            return {
                "status": "failure",
                "message": "Query returned no results",
                "data": {},
                "metadata": {"failure_reason": "no_results", "mode": "hybrid"},
                "llm_response": {
                    "content": "Sorry, I'm not able to provide an answer.",
                    "response_iterator": None,
                    "is_streaming": False,
                },
            }
        context = "Apple is headquartered in Cupertino."
        chunks = [{"document_id": "doc-1", "content": context}]
        return {
            "status": "success",
            "message": "ok",
            "llm_response": {
                "content": context,
                "response_iterator": None,
                "is_streaming": False,
            },
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": chunks,
                "references": [],
            },
            "metadata": {},
        }


async def fake_answer(**kwargs):
    answer = (
        INSUFFICIENT_INFORMATION
        if not kwargs["user_prompt"].split("Retrieval context:\n", 1)[1]
        else "Cupertino"
    )
    return {
        "message": {"content": answer},
        "prompt_eval_count": 10,
        "eval_count": 1,
    }


def test_answerable_and_unanswerable_smoke(tmp_path: Path) -> None:
    questions = [
        {
            "question_id": "q1",
            "question": "Where is Apple headquartered?",
            "question_type": "inference_query",
            "answerable": True,
            "answer": "Cupertino",
        },
        {
            "question_id": "q2",
            "question": "This is unanswerable",
            "question_type": "null_query",
            "answerable": False,
            "answer": "",
        },
    ]

    async def exercise():
        retrieval = await run_retrieval(
            FakeRag(),
            questions,
            artifact_path=tmp_path / "retrieval.jsonl",
            lineage={
                "base_run_id": "base_1",
                "variant_run_id": "variant_1",
                "base_extraction_sha256": "a" * 64,
            },
            builder_model="builder",
            graph_regime="native_lightrag",
            query_model="query",
            query_model_digest="b" * 64,
            query_generation_parameters={
                "temperature": 0.0,
                "seed": 42,
                "num_ctx": 4096,
                "num_predict": 128,
                "think": False,
            },
            query_param_factory=Param,
        )
        answers = await run_answering(
            retrieval,
            questions,
            artifact_path=tmp_path / "answers.jsonl",
            answer_model="answer",
            answer_model_digest="c" * 64,
            generation_options={"temperature": 0, "seed": 42},
            answer_call=fake_answer,
        )
        return retrieval, answers

    retrieval, answers = asyncio.run(exercise())
    assert [row.raw_answer for row in answers] == [
        "Cupertino",
        INSUFFICIENT_INFORMATION,
    ]
    assert all(row.base_run_id == "base_1" for row in retrieval)
    assert [row.retrieval_outcome for row in retrieval] == ["context", "empty"]
    assert [row.context for row in retrieval] == [
        "Apple is headquartered in Cupertino.",
        "",
    ]
    assert all(row.query_generation_parameters["seed"] == 42 for row in retrieval)
    assert [row.context_token_count for row in retrieval] == [5, 0]
    assert all(row.answer_result_id.startswith("ans_") for row in answers)

    # Safe resume reuses exact records and does not call the fake providers again.
    retrieval_2, answers_2 = asyncio.run(exercise())
    assert retrieval_2 == retrieval
    assert answers_2 == answers


def test_real_query_failure_is_audited_and_not_misclassified_as_empty(
    tmp_path: Path,
) -> None:
    class FailingRag:
        async def aquery_llm(self, _question, *, param):
            assert param.kwargs["only_need_context"] is True
            return {
                "status": "failure",
                "message": "Query failed: storage unavailable",
                "data": {},
                "metadata": {},
                "llm_response": None,
            }

    records = asyncio.run(
        run_retrieval(
            FailingRag(),
            [
                {
                    "question_id": "q-failure",
                    "question": "Trigger failure",
                    "answerable": True,
                }
            ],
            artifact_path=tmp_path / "retrieval.jsonl",
            lineage={
                "base_run_id": "base_1",
                "variant_run_id": "variant_1",
                "base_extraction_sha256": "a" * 64,
            },
            builder_model="builder",
            graph_regime="native_lightrag",
            query_model="query",
            query_model_digest="b" * 64,
            query_param_factory=Param,
        )
    )
    assert records[0].retrieval_outcome == "failure"
    assert records[0].error_type == "RuntimeError"
    assert "storage unavailable" in (records[0].error_message or "")
    assert records[0].raw_result["metadata"].get("failure_reason") is None
    assert (tmp_path / "retrieval.jsonl").exists()


def test_answer_failure_and_upstream_failure_are_audited_per_question(
    tmp_path: Path,
) -> None:
    questions = [
        {"question_id": "q1", "question": "one", "answerable": True},
        {"question_id": "q2", "question": "two", "answerable": True},
    ]
    base = {
        "question_type": "inference",
        "answerable": True,
        "base_run_id": "base",
        "variant_run_id": "variant",
        "base_extraction_sha256": "a" * 64,
        "builder_model": "builder",
        "graph_regime": "native_lightrag",
        "context": "context",
        "context_sha256": __import__("hashlib").sha256(b"context").hexdigest(),
    }
    retrieval = [
        {
            **base,
            "question_id": "q1",
            "retrieval_result_id": "r1",
            "retrieval_outcome": "context",
        },
        {
            **base,
            "question_id": "q2",
            "retrieval_result_id": "r2",
            "retrieval_outcome": "failure",
        },
    ]

    async def failing_answer(**_kwargs):
        raise TimeoutError("model timeout")

    answers = asyncio.run(
        run_answering(
            retrieval,
            questions,
            artifact_path=tmp_path / "answers.jsonl",
            answer_model="answer",
            answer_model_digest="b" * 64,
            generation_options={"temperature": 0},
            answer_call=failing_answer,
        )
    )
    assert [item.answer_outcome for item in answers] == [
        "failure",
        "skipped_upstream_failure",
    ]
    assert answers[0].latency_ms is not None
    assert answers[1].latency_ms is None


def test_question_answerability_does_not_predetermine_retrieval_outcome(
    tmp_path: Path,
) -> None:
    records = asyncio.run(
        run_retrieval(
            FakeRag(),
            [
                {
                    "question_id": "q-null-with-top-k",
                    "question": "Find context for Apple despite a null gold answer",
                    "question_type": "null_query",
                    "answerable": False,
                }
            ],
            artifact_path=tmp_path / "retrieval.jsonl",
            lineage={
                "base_run_id": "base_1",
                "variant_run_id": "variant_1",
                "base_extraction_sha256": "a" * 64,
            },
            builder_model="builder",
            graph_regime="native_lightrag",
            query_model="query",
            query_model_digest="b" * 64,
            query_param_factory=Param,
        )
    )
    assert records[0].answerable is False
    assert records[0].retrieval_outcome == "context"
    assert records[0].retrieved_chunks
