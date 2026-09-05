#!/usr/bin/env python3
"""Build and audit the fixed MultiHop-RAG experiment subset.

The builder uses source metadata and local lexical statistics only. It does
not use model results or call a generative model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SUBSET_DIR = SCRIPT_PATH.parents[1]
EXPERIMENT_DIR = SUBSET_DIR.parent
SELECTION_PROVENANCE_DIR = SUBSET_DIR / "selection_provenance"
QUESTIONS_SOURCE = EXPERIMENT_DIR / "MultiHop-RAG" / "MultiHopRAG.json"
CORPUS_SOURCE = EXPERIMENT_DIR / "MultiHop-RAG" / "corpus.json"
PILOT_QUESTIONS = SELECTION_PROVENANCE_DIR / "pilot_questions.jsonl"
PILOT_DOCUMENTS = SELECTION_PROVENANCE_DIR / "pilot_documents.jsonl"

QUESTION_SOURCE_REF = "../MultiHop-RAG/MultiHopRAG.json"
CORPUS_SOURCE_REF = "../MultiHop-RAG/corpus.json"
PILOT_QUESTIONS_REF = "../pilot/pilot_questions.jsonl"
PILOT_DOCUMENTS_REF = "../pilot/pilot_documents.jsonl"
# These labels are retained so rebuilding the fixed subset reproduces the
# existing manifest byte for byte.

SOURCE_TYPE_ORDER = (
    "inference_query",
    "comparison_query",
    "temporal_query",
    "null_query",
)
SOURCE_TO_OUTPUT_TYPE = {
    "inference_query": "inference",
    "comparison_query": "comparison",
    "temporal_query": "temporal",
    "null_query": "unanswerable",
}
EXPECTED_QUESTION_FIELDS = {"query", "answer", "question_type", "evidence_list"}
EXPECTED_CORPUS_FIELDS = {
    "title",
    "author",
    "source",
    "published_at",
    "category",
    "url",
    "body",
}
EXPECTED_EVIDENCE_FIELDS = {
    "title",
    "author",
    "url",
    "source",
    "category",
    "published_at",
    "fact",
}

TARGET_CORPUS_DOCUMENTS = 155
BASE_GOLD_DOCUMENT_QUESTION_CAP = 3
BASE_HARD_NEGATIVE_SOURCE_CAP = 3
BASE_HARD_NEGATIVE_CATEGORY_CAP = 12
BUILDER_VERSION = "1.0.0"

GENERIC_ANSWERS = {
    "yes",
    "no",
    "insufficient information",
    "insufficient information.",
}

STOPWORDS = set(
    """
    a about according after again against all also am an and any are as at be
    because been before being between both but by can concerning considering
    could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in information
    into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own regarding report
    reported reports respectively same she should so some such than that the
    their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom
    why will with would you your yours yourself yourselves article single
    related whether yes
    """.split()
)


class BuildError(RuntimeError):
    """Raised when a source or generated-artifact invariant fails."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def question_id(question_text: str) -> str:
    """Derive a stable question ID from its text."""

    return f"mhq_{sha256_text(question_text)[:16]}"


def document_id(url: str) -> str:
    """Derive a stable document ID from its URL."""

    return f"mhd_{sha256_text(url)[:16]}"


def canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        sort_keys=False,
    )
    return (text + "\n").encode("utf-8")


def canonical_record_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )
    return sha256_text(payload)


def jsonl_bytes(rows: Sequence[dict[str, Any]]) -> tuple[bytes, dict[str, str]]:
    lines: list[str] = []
    hashes: dict[str, str] = {}
    for row in rows:
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        key = str(row.get("question_id") or row.get("document_id"))
        if not key or key == "None":
            raise BuildError("JSONL row is missing its stable ID")
        lines.append(line)
        hashes[key] = sha256_text(line)
    return ("".join(f"{line}\n" for line in lines).encode("utf-8"), hashes)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BuildError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def stable_tie(seed: int, namespace: str, value: int | str) -> int:
    raw = f"{seed}\0{namespace}\0{value}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def lexical_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1 and token not in STOPWORDS
    ]


def normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def direct_answer_present(answer: str, document: dict[str, Any]) -> bool:
    if answer.casefold().strip() in GENERIC_ANSWERS:
        return False
    needle = normalized_phrase(answer)
    if not needle:
        return False
    haystack = normalized_phrase(f"{document['title']} {document['body']}")
    return f" {needle} " in f" {haystack} "


def evidence_chain_present(question: dict[str, Any], document: dict[str, Any]) -> bool:
    facts = [item["fact"] for item in question["evidence_list"]]
    return bool(facts) and all(fact in document["body"] for fact in facts)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


@dataclass(frozen=True)
class SourceData:
    questions: list[dict[str, Any]]
    corpus: list[dict[str, Any]]
    corpus_indices_by_url: dict[str, tuple[int, ...]]
    pilot_questions: list[dict[str, Any]]
    pilot_documents: list[dict[str, Any]]


def load_and_validate_sources() -> SourceData:
    questions = read_json(QUESTIONS_SOURCE)
    corpus = read_json(CORPUS_SOURCE)
    pilot_questions = read_jsonl(PILOT_QUESTIONS)
    pilot_documents = read_jsonl(PILOT_DOCUMENTS)

    require(isinstance(questions, list), "MultiHopRAG.json must be a JSON array")
    require(isinstance(corpus, list), "corpus.json must be a JSON array")
    require(
        all(isinstance(row, dict) and set(row) == EXPECTED_QUESTION_FIELDS for row in questions),
        "unexpected question source schema",
    )
    require(
        all(isinstance(row, dict) and set(row) == EXPECTED_CORPUS_FIELDS for row in corpus),
        "unexpected corpus source schema",
    )

    indices_by_url: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(corpus):
        indices_by_url[document["url"]].append(index)
    frozen_indices = {url: tuple(indices) for url, indices in indices_by_url.items()}

    for source_index, question in enumerate(questions):
        require(
            question["question_type"] in SOURCE_TYPE_ORDER,
            f"QA[{source_index}] has an unknown question type",
        )
        require(
            all(set(evidence) == EXPECTED_EVIDENCE_FIELDS for evidence in question["evidence_list"]),
            f"QA[{source_index}] has an unexpected evidence schema",
        )
        if question["question_type"] == "null_query":
            require(
                question["evidence_list"] == [],
                f"QA[{source_index}] null_query must have empty evidence_list",
            )
        for evidence in question["evidence_list"]:
            matches = frozen_indices.get(evidence["url"], ())
            require(
                len(matches) == 1,
                f"QA[{source_index}] evidence URL resolves to {len(matches)} corpus rows",
            )
            document = corpus[matches[0]]
            require(
                evidence["fact"] in document["body"],
                f"QA[{source_index}] evidence fact is absent from corpus body",
            )

    source_ids = [question_id(row["query"]) for row in questions]
    require(len(source_ids) == len(set(source_ids)), "source question IDs collide")
    corpus_ids = [document_id(row["url"]) for row in corpus]
    require(len(corpus_ids) == len(set(corpus_ids)), "source document IDs collide")

    source_id_set = set(source_ids)
    for row in pilot_questions:
        require(row["question_id"] in source_id_set, "pilot question is absent from source")
        require(
            row["question_id"] == question_id(row["question"]),
            "pilot question ID does not use the established ID method",
        )
    for row in pilot_documents:
        require(
            row["document_id"] == document_id(row["url"]),
            "pilot document ID does not use the established ID method",
        )

    return SourceData(
        questions=questions,
        corpus=corpus,
        corpus_indices_by_url=frozen_indices,
        pilot_questions=pilot_questions,
        pilot_documents=pilot_documents,
    )


class BM25Index:
    """Small deterministic BM25 implementation over title + unchanged body."""

    def __init__(self, corpus: Sequence[dict[str, Any]]) -> None:
        self.doc_term_counts: list[Counter[str]] = []
        self.doc_token_sets: list[set[str]] = []
        self.doc_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        for document in corpus:
            tokens = lexical_tokens(f"{document['title']} {document['body']}")
            counts = Counter(tokens)
            self.doc_term_counts.append(counts)
            self.doc_token_sets.append(set(counts))
            self.doc_lengths.append(len(tokens))
            document_frequency.update(counts.keys())
        self.document_count = len(corpus)
        self.average_length = (
            sum(self.doc_lengths) / self.document_count if self.document_count else 0.0
        )
        self.idf = {
            term: math.log(
                1.0
                + (self.document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def scores(self, text: str, *, k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_counts = Counter(lexical_tokens(text))
        scores = [0.0] * self.document_count
        if not query_counts or not self.average_length:
            return scores
        for index, counts in enumerate(self.doc_term_counts):
            denominator_length = 1.0 - b + b * self.doc_lengths[index] / self.average_length
            score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (k1 + 1.0)
                denominator = frequency + k1 * denominator_length
                score += self.idf.get(term, 0.0) * numerator / denominator * query_frequency
            scores[index] = score
        return scores


def balanced_quotas(total: int, strata: Sequence[int], seed: int, namespace: str) -> dict[int, int]:
    require(total >= 0 and bool(strata), "invalid balanced quota request")
    base, remainder = divmod(total, len(strata))
    quotas = {stratum: base for stratum in strata}
    extra_order = sorted(
        strata,
        key=lambda value: (stable_tie(seed, f"{namespace}-quota", value), value),
    )
    for stratum in extra_order[:remainder]:
        quotas[stratum] += 1
    return quotas


def quota_schedule(quotas: dict[int, int]) -> list[int]:
    remaining = dict(quotas)
    schedule: list[int] = []
    strata = sorted(quotas)
    while any(remaining.values()):
        for stratum in strata:
            if remaining[stratum]:
                schedule.append(stratum)
                remaining[stratum] -= 1
    return schedule


def concentration_delta(counter: Counter[str], values: Iterable[str]) -> int:
    additions = Counter(values)
    return sum(
        (counter[value] + count) ** 2 - counter[value] ** 2
        for value, count in additions.items()
    )


def selected_doc_reuse_bin(value: int) -> str:
    if value <= 2:
        return "1-2"
    if value <= 9:
        return "3-9"
    if value <= 49:
        return "10-49"
    return "50+"


def length_thresholds(corpus: Sequence[dict[str, Any]]) -> tuple[int, int]:
    lengths = sorted(len(document["body"]) for document in corpus)
    return lengths[len(lengths) // 3], lengths[(2 * len(lengths)) // 3]


def body_length_bin(document: dict[str, Any], thresholds: tuple[int, int]) -> str:
    length = len(document["body"])
    if length <= thresholds[0]:
        return "short"
    if length <= thresholds[1]:
        return "medium"
    return "long"


def question_topic_terms(
    question: dict[str, Any], token_document_frequency: Counter[str]
) -> tuple[str, ...]:
    terms = set(lexical_tokens(question["query"]))
    return tuple(sorted(terms, key=lambda term: (token_document_frequency[term], term))[:8])


def answerable_evidence_strata(source_type: str) -> tuple[int, ...]:
    if source_type == "inference_query":
        return (2, 3, 4)
    if source_type in {"comparison_query", "temporal_query"}:
        return (2, 3)
    return (0,)


def select_questions(
    data: SourceData,
    bm25: BM25Index,
    *,
    questions_per_type: int,
    seed: int,
    hard_negative_count: int,
) -> dict[str, Any]:
    pilot_ids = {row["question_id"] for row in data.pilot_questions}
    corpus = data.corpus
    questions = data.questions
    thresholds = length_thresholds(corpus)
    global_url_reuse = Counter(
        evidence["url"] for question in questions for evidence in question["evidence_list"]
    )
    token_df: Counter[str] = Counter()
    for question in questions:
        token_df.update(set(lexical_tokens(question["query"])))

    candidates: dict[str, list[dict[str, Any]]] = {
        source_type: [] for source_type in SOURCE_TYPE_ORDER
    }
    exclusion_counts: Counter[str] = Counter()
    for source_index, question in enumerate(questions):
        qid = question_id(question["query"])
        if qid in pilot_ids:
            exclusion_counts["pilot_question"] += 1
            continue
        document_indices = tuple(
            data.corpus_indices_by_url[evidence["url"]][0]
            for evidence in question["evidence_list"]
        )
        if len(document_indices) != len(set(document_indices)):
            exclusion_counts["duplicate_evidence_url"] += 1
            continue
        if question["question_type"] != "null_query" and not (
            2 <= len(document_indices) <= 4
        ):
            exclusion_counts["evidence_count_outside_2_to_4"] += 1
            continue
        null_proxy_index: int | None = None
        if question["question_type"] == "null_query":
            scores = bm25.scores(question["query"])
            null_proxy_index = min(
                range(len(corpus)),
                key=lambda index: (
                    -scores[index],
                    stable_tie(seed, "null-proxy", f"{source_index}:{index}"),
                    index,
                ),
            )
        candidates[question["question_type"]].append(
            {
                "source_index": source_index,
                "question": question,
                "document_indices": document_indices,
                "topic_terms": question_topic_terms(question, token_df),
                "null_proxy_index": null_proxy_index,
            }
        )

    evidence_quotas = {
        source_type: balanced_quotas(
            questions_per_type,
            answerable_evidence_strata(source_type),
            seed,
            source_type,
        )
        for source_type in SOURCE_TYPE_ORDER
    }
    schedules = {source_type: quota_schedule(quotas) for source_type, quotas in evidence_quotas.items()}
    evidence_occurrences = sum(
        count * evidence_count
        for source_type, quotas in evidence_quotas.items()
        if source_type != "null_query"
        for evidence_count, count in quotas.items()
    )
    answerable_total = questions_per_type * 3
    minimum_union_under_base_cap = math.ceil(evidence_occurrences / BASE_GOLD_DOCUMENT_QUESTION_CAP)
    soft_gold_union_target = min(
        evidence_occurrences,
        max(minimum_union_under_base_cap, TARGET_CORPUS_DOCUMENTS - hard_negative_count),
    )

    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    selected_document_use: Counter[int] = Counter()
    selected_gold_indices: set[int] = set()
    topic_counts: Counter[str] = Counter()
    feature_counts: dict[str, Counter[str]] = {
        name: Counter()
        for name in ("source", "category", "month", "length", "global_reuse")
    }
    null_proxy_counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("source", "category", "month", "length")
    }
    selection_trace: list[dict[str, Any]] = []
    relaxations: list[dict[str, Any]] = []
    answerable_selected = 0

    def feature_values(document_indices: Sequence[int]) -> dict[str, list[str]]:
        return {
            "source": [corpus[index]["source"] for index in document_indices],
            "category": [corpus[index]["category"] for index in document_indices],
            "month": [corpus[index]["published_at"][:7] for index in document_indices],
            "length": [body_length_bin(corpus[index], thresholds) for index in document_indices],
            "global_reuse": [
                selected_doc_reuse_bin(global_url_reuse[corpus[index]["url"]])
                for index in document_indices
            ],
        }

    for round_index in range(questions_per_type):
        for source_type in SOURCE_TYPE_ORDER:
            evidence_count = schedules[source_type][round_index]
            pool = [
                candidate
                for candidate in candidates[source_type]
                if candidate["source_index"] not in selected_indices
                and len(candidate["document_indices"]) == evidence_count
            ]
            require(
                bool(pool),
                f"no eligible {source_type} candidate for evidence stratum {evidence_count}",
            )

            cap = BASE_GOLD_DOCUMENT_QUESTION_CAP
            allowed = [
                candidate
                for candidate in pool
                if all(
                    selected_document_use[index] < cap
                    for index in candidate["document_indices"]
                )
            ]
            while not allowed:
                previous_cap = cap
                cap += 1
                allowed = [
                    candidate
                    for candidate in pool
                    if all(
                        selected_document_use[index] < cap
                        for index in candidate["document_indices"]
                    )
                ]
                relaxations.append(
                    {
                        "selection_round": round_index,
                        "source_question_type": source_type,
                        "criterion": "maximum selected-question uses per gold document",
                        "from": previous_cap,
                        "to": cap,
                        "reason": "no eligible candidate remained in the required evidence stratum",
                    }
                )

            def candidate_score(candidate: dict[str, Any]) -> tuple[int, int, int]:
                nonlocal answerable_selected
                document_indices = candidate["document_indices"]
                projected_answerable = answerable_selected + (source_type != "null_query")
                projected_unique = len(selected_gold_indices | set(document_indices))
                phase_target = (
                    round(soft_gold_union_target * projected_answerable / answerable_total)
                    if projected_answerable
                    else 0
                )
                target_penalty = abs(projected_unique - phase_target)
                document_use_penalty = sum(
                    2 * selected_document_use[index] + 1 for index in document_indices
                )
                topic_penalty = concentration_delta(topic_counts, candidate["topic_terms"])
                values = feature_values(document_indices)
                feature_penalty = (
                    concentration_delta(feature_counts["source"], values["source"]) * 60
                    + concentration_delta(feature_counts["category"], values["category"]) * 80
                    + concentration_delta(feature_counts["month"], values["month"]) * 40
                    + concentration_delta(feature_counts["length"], values["length"]) * 40
                    + concentration_delta(
                        feature_counts["global_reuse"], values["global_reuse"]
                    )
                    * 30
                )
                null_proxy_penalty = 0
                if source_type == "null_query":
                    proxy_index = candidate["null_proxy_index"]
                    require(proxy_index is not None, "null candidate lacks lexical proxy")
                    proxy_values = feature_values([proxy_index])
                    null_proxy_penalty = (
                        concentration_delta(null_proxy_counts["source"], proxy_values["source"]) * 60
                        + concentration_delta(
                            null_proxy_counts["category"], proxy_values["category"]
                        )
                        * 80
                        + concentration_delta(null_proxy_counts["month"], proxy_values["month"]) * 40
                        + concentration_delta(null_proxy_counts["length"], proxy_values["length"]) * 40
                    )
                source_reuse_penalty = sum(
                    min(global_url_reuse[corpus[index]["url"]], 100)
                    for index in document_indices
                )
                aggregate = (
                    target_penalty * 100_000
                    + document_use_penalty * 5_000
                    + topic_penalty * 300
                    + feature_penalty
                    + null_proxy_penalty
                    + source_reuse_penalty * 3
                )
                return (
                    aggregate,
                    stable_tie(seed, "question", candidate["source_index"]),
                    candidate["source_index"],
                )

            chosen = min(allowed, key=candidate_score)
            chosen_score = candidate_score(chosen)[0]
            selected.append(chosen)
            selected_indices.add(chosen["source_index"])
            if source_type != "null_query":
                answerable_selected += 1
            selected_gold_indices.update(chosen["document_indices"])
            for document_index in chosen["document_indices"]:
                selected_document_use[document_index] += 1
            values = feature_values(chosen["document_indices"])
            for feature_name, feature_values_list in values.items():
                feature_counts[feature_name].update(feature_values_list)
            topic_counts.update(chosen["topic_terms"])
            if source_type == "null_query":
                proxy_index = chosen["null_proxy_index"]
                require(proxy_index is not None, "selected null candidate lacks lexical proxy")
                proxy_values = feature_values([proxy_index])
                for feature_name in null_proxy_counts:
                    null_proxy_counts[feature_name].update(proxy_values[feature_name])
            selection_trace.append(
                {
                    "selection_step": len(selection_trace) + 1,
                    "selection_round": round_index,
                    "source_index": chosen["source_index"],
                    "source_question_type": source_type,
                    "evidence_document_count": evidence_count,
                    "gold_document_cap_in_effect": cap,
                    "objective_value": chosen_score,
                }
            )

    require(len(selected) == questions_per_type * 4, "question selection count mismatch")
    require(len(selected_indices) == len(selected), "selected source question indices repeat")
    selected_by_type = {
        source_type: sorted(
            (item for item in selected if item["question"]["question_type"] == source_type),
            key=lambda item: item["source_index"],
        )
        for source_type in SOURCE_TYPE_ORDER
    }
    ordered = [item for source_type in SOURCE_TYPE_ORDER for item in selected_by_type[source_type]]

    return {
        "ordered_candidates": ordered,
        "selected_gold_indices": sorted(selected_gold_indices),
        "selected_document_use": selected_document_use,
        "evidence_quotas": evidence_quotas,
        "evidence_occurrences": evidence_occurrences,
        "soft_gold_union_target": soft_gold_union_target,
        "length_thresholds": thresholds,
        "global_url_reuse": global_url_reuse,
        "topic_counts": topic_counts,
        "feature_counts": feature_counts,
        "null_proxy_counts": null_proxy_counts,
        "selection_trace": selection_trace,
        "relaxations": relaxations,
        "eligibility": {
            "eligible_candidates_by_type": {
                source_type: len(items) for source_type, items in candidates.items()
            },
            "excluded": dict(sorted(exclusion_counts.items())),
        },
    }


def hard_negative_query_text(question: dict[str, Any]) -> str:
    titles = " ".join(evidence["title"] for evidence in question["evidence_list"])
    return f"{question['query']} {titles}".strip()


def hard_negative_pair_checks(
    question: dict[str, Any], document: dict[str, Any], question_gold_indices: set[int], document_index: int
) -> dict[str, bool]:
    return {
        "not_gold_for_question": document_index not in question_gold_indices,
        "no_complete_exact_evidence_chain": not evidence_chain_present(question, document),
        "no_normalized_direct_full_answer": not direct_answer_present(question["answer"], document),
    }


def select_hard_negatives(
    data: SourceData,
    bm25: BM25Index,
    question_selection: dict[str, Any],
    *,
    hard_negative_count: int,
    seed: int,
) -> dict[str, Any]:
    ordered_candidates = question_selection["ordered_candidates"]
    global_gold_indices = set(question_selection["selected_gold_indices"])
    corpus = data.corpus
    question_count = len(ordered_candidates)

    raw_scores: list[list[float]] = []
    normalized_scores: list[dict[int, float]] = []
    eligible_indices: list[list[int]] = []
    pair_checks: list[dict[int, dict[str, bool]]] = []
    ranked_eligible_indices: list[list[int]] = []

    for question_position, candidate in enumerate(ordered_candidates):
        question = candidate["question"]
        scores = bm25.scores(hard_negative_query_text(question))
        raw_scores.append(scores)
        question_gold_indices = set(candidate["document_indices"])
        checks_for_question: dict[int, dict[str, bool]] = {}
        eligible: list[int] = []
        for document_index, document in enumerate(corpus):
            if document_index in global_gold_indices:
                continue
            checks = hard_negative_pair_checks(
                question, document, question_gold_indices, document_index
            )
            checks_for_question[document_index] = checks
            if all(checks.values()) and scores[document_index] > 0.0:
                eligible.append(document_index)
        require(
            bool(eligible),
            f"QA[{candidate['source_index']}] has no model-free hard-negative candidate",
        )
        ranked = sorted(
            eligible,
            key=lambda index: (
                -scores[index],
                stable_tie(
                    seed,
                    "hard-negative-rank",
                    f"{candidate['source_index']}:{index}",
                ),
                index,
            ),
        )
        maximum = scores[ranked[0]]
        normalized_scores.append(
            {index: scores[index] / maximum for index in eligible}
        )
        eligible_indices.append(eligible)
        pair_checks.append(checks_for_question)
        ranked_eligible_indices.append(ranked)

    selected: list[int] = []
    selected_set: set[int] = set()
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    coverage = [0.0] * question_count
    owners: dict[int, int] = {}
    trace: list[dict[str, Any]] = []
    relaxations: list[dict[str, Any]] = []
    source_cap = BASE_HARD_NEGATIVE_SOURCE_CAP
    category_cap = BASE_HARD_NEGATIVE_CATEGORY_CAP

    candidate_universe = sorted(set(range(len(corpus))) - global_gold_indices)
    require(
        len(candidate_universe) >= hard_negative_count,
        "not enough non-gold corpus documents for requested hard negatives",
    )

    for selection_round in range(1, hard_negative_count + 1):
        allowed = [
            index
            for index in candidate_universe
            if index not in selected_set
            and source_counts[corpus[index]["source"]] < source_cap
            and category_counts[corpus[index]["category"]] < category_cap
            and any(index in normalized_scores[position] for position in range(question_count))
        ]
        while not allowed:
            previous_source_cap = source_cap
            previous_category_cap = category_cap
            source_cap += 1
            category_cap += 1
            relaxations.append(
                {
                    "selection_round": selection_round,
                    "criterion": "hard-negative source/category caps",
                    "from": {
                        "source": previous_source_cap,
                        "category": previous_category_cap,
                    },
                    "to": {"source": source_cap, "category": category_cap},
                    "reason": "no unused eligible lexical candidate remained under prior caps",
                }
            )
            allowed = [
                index
                for index in candidate_universe
                if index not in selected_set
                and source_counts[corpus[index]["source"]] < source_cap
                and category_counts[corpus[index]["category"]] < category_cap
                and any(index in normalized_scores[position] for position in range(question_count))
            ]

        def facility_key(document_index: int) -> tuple[float, int, int, int, int]:
            gain = sum(
                max(
                    0.0,
                    normalized_scores[position].get(document_index, 0.0) - coverage[position],
                )
                for position in range(question_count)
            )
            document = corpus[document_index]
            return (
                round(gain, 15),
                -source_counts[document["source"]],
                -category_counts[document["category"]],
                -question_selection["global_url_reuse"][document["url"]],
                -stable_tie(seed, "hard-negative-facility", document_index),
            )

        chosen = max(allowed, key=facility_key)
        gains = [
            max(0.0, normalized_scores[position].get(chosen, 0.0) - coverage[position])
            for position in range(question_count)
        ]
        owner = min(
            range(question_count),
            key=lambda position: (
                -gains[position],
                stable_tie(
                    seed,
                    "hard-negative-owner",
                    f"{chosen}:{ordered_candidates[position]['source_index']}",
                ),
                ordered_candidates[position]["source_index"],
            ),
        )
        if gains[owner] <= 0.0:
            owner = min(
                (
                    position
                    for position in range(question_count)
                    if chosen in normalized_scores[position]
                ),
                key=lambda position: (
                    -normalized_scores[position][chosen],
                    ordered_candidates[position]["source_index"],
                ),
            )
        owners[chosen] = owner
        selected.append(chosen)
        selected_set.add(chosen)
        source_counts[corpus[chosen]["source"]] += 1
        category_counts[corpus[chosen]["category"]] += 1
        for position in range(question_count):
            coverage[position] = max(
                coverage[position], normalized_scores[position].get(chosen, 0.0)
            )
        trace.append(
            {
                "selection_round": selection_round,
                "source_index": chosen,
                "owner_question_source_index": ordered_candidates[owner]["source_index"],
                "marginal_normalized_bm25_coverage_gain": round(sum(gains), 12),
                "source_cap_in_effect": source_cap,
                "category_cap_in_effect": category_cap,
            }
        )

    related_question_positions: dict[int, set[int]] = {
        document_index: {owners[document_index]} for document_index in selected
    }
    assigned_document_by_question: dict[int, int] = {}
    for position in range(question_count):
        eligible_selected = [
            index
            for index in selected
            if index in normalized_scores[position] and raw_scores[position][index] > 0.0
        ]
        require(
            bool(eligible_selected),
            f"QA[{ordered_candidates[position]['source_index']}] is uncovered by selected hard negatives",
        )
        assigned = min(
            eligible_selected,
            key=lambda index: (
                -raw_scores[position][index],
                stable_tie(
                    seed,
                    "hard-negative-assignment",
                    f"{ordered_candidates[position]['source_index']}:{index}",
                ),
                index,
            ),
        )
        assigned_document_by_question[position] = assigned
        related_question_positions[assigned].add(position)

    detail_by_document: dict[int, dict[str, Any]] = {}
    for document_index in selected:
        related_positions = sorted(related_question_positions[document_index])
        per_question: dict[str, dict[str, Any]] = {}
        for position in related_positions:
            candidate = ordered_candidates[position]
            qid = question_id(candidate["question"]["query"])
            rank = ranked_eligible_indices[position].index(document_index) + 1
            overlap = sorted(
                set(lexical_tokens(hard_negative_query_text(candidate["question"])))
                & bm25.doc_token_sets[document_index]
            )
            per_question[qid] = {
                "question_source_index": candidate["source_index"],
                "bm25_score": round(raw_scores[position][document_index], 12),
                "eligible_bm25_rank": rank,
                "lexical_overlap_terms": overlap[:20],
                **pair_checks[position][document_index],
            }
        detail_by_document[document_index] = {
            "source_index": document_index,
            "selection_round": next(
                row["selection_round"] for row in trace if row["source_index"] == document_index
            ),
            "owner_question_id": question_id(
                ordered_candidates[owners[document_index]]["question"]["query"]
            ),
            "related_question_ids": sorted(
                per_question,
                key=lambda qid: next(
                    position
                    for position, candidate in enumerate(ordered_candidates)
                    if question_id(candidate["question"]["query"]) == qid
                ),
            ),
            "per_question": per_question,
        }

    by_question: dict[str, list[int]] = defaultdict(list)
    for document_index, detail in detail_by_document.items():
        for qid in detail["related_question_ids"]:
            by_question[qid].append(document_index)
    for qid in by_question:
        by_question[qid].sort()

    return {
        "selected_indices": sorted(selected),
        "selection_order": selected,
        "by_question": dict(by_question),
        "detail_by_document": detail_by_document,
        "trace": trace,
        "relaxations": relaxations,
        "source_counts": source_counts,
        "category_counts": category_counts,
        "coverage": {
            "minimum_normalized_bm25": round(min(coverage), 12),
            "mean_normalized_bm25": round(sum(coverage) / len(coverage), 12),
            "questions_covered": sum(value > 0.0 for value in coverage),
        },
    }


def build_rows(
    data: SourceData,
    question_selection: dict[str, Any],
    hard_negatives: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_candidates = question_selection["ordered_candidates"]
    qid_by_source_index = {
        candidate["source_index"]: question_id(candidate["question"]["query"])
        for candidate in ordered_candidates
    }
    question_order = {
        qid_by_source_index[candidate["source_index"]]: position
        for position, candidate in enumerate(ordered_candidates)
    }

    question_rows: list[dict[str, Any]] = []
    for candidate in ordered_candidates:
        source_index = candidate["source_index"]
        source_question = candidate["question"]
        qid = qid_by_source_index[source_index]
        gold_urls = [evidence["url"] for evidence in source_question["evidence_list"]]
        hard_negative_indices = hard_negatives["by_question"].get(qid, [])
        question_rows.append(
            {
                "question_id": qid,
                "question": source_question["query"],
                "question_type": SOURCE_TO_OUTPUT_TYPE[source_question["question_type"]],
                "gold_answer": source_question["answer"],
                "answerable": source_question["question_type"] != "null_query",
                "gold_urls": gold_urls,
                "gold_document_ids": [document_id(url) for url in gold_urls],
                "hard_negative_document_ids": [
                    document_id(data.corpus[index]["url"]) for index in hard_negative_indices
                ],
                "question_sha256": sha256_text(source_question["query"]),
                "source_record_sha256": canonical_record_hash(source_question),
                "source_file": QUESTION_SOURCE_REF,
                "source_index": source_index,
                "source_question_type": source_question["question_type"],
                "selection_stratum": {
                    "evidence_document_count": len(gold_urls),
                    "topic_terms": list(candidate["topic_terms"]),
                    "null_lexical_proxy_corpus_index": candidate["null_proxy_index"],
                },
                "source_fields": source_question,
            }
        )

    gold_for_questions: dict[int, list[str]] = defaultdict(list)
    for candidate in ordered_candidates:
        qid = qid_by_source_index[candidate["source_index"]]
        for document_index in candidate["document_indices"]:
            gold_for_questions[document_index].append(qid)

    document_rows: list[dict[str, Any]] = []
    selected_document_indices = (
        question_selection["selected_gold_indices"] + hard_negatives["selected_indices"]
    )
    require(
        len(selected_document_indices) == len(set(selected_document_indices)),
        "gold and hard-negative document sets overlap",
    )
    for source_index in selected_document_indices:
        source_document = data.corpus[source_index]
        role = "gold" if source_index in gold_for_questions else "hard_negative"
        gold_question_ids = sorted(
            set(gold_for_questions.get(source_index, [])), key=question_order.get
        )
        hard_negative_question_ids = sorted(
            set(
                hard_negatives["detail_by_document"]
                .get(source_index, {})
                .get("related_question_ids", [])
            ),
            key=question_order.get,
        )
        related_question_ids = (
            gold_question_ids if role == "gold" else hard_negative_question_ids
        )
        document_rows.append(
            {
                "document_id": document_id(source_document["url"]),
                "text": source_document["body"],
                "url": source_document["url"],
                "role": role,
                "related_question_ids": related_question_ids,
                "gold_for_question_ids": gold_question_ids,
                "hard_negative_for_question_ids": hard_negative_question_ids,
                "text_sha256": sha256_text(source_document["body"]),
                "source_record_sha256": canonical_record_hash(source_document),
                "source_file": CORPUS_SOURCE_REF,
                "source_index": source_index,
                "source_fields": {
                    key: source_document[key]
                    for key in ("title", "author", "source", "published_at", "category", "url")
                },
            }
        )

    return question_rows, document_rows


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def nested_evidence_distribution(question_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for source_type in SOURCE_TYPE_ORDER:
        counts = Counter(
            len(row["gold_document_ids"])
            for row in question_rows
            if row["source_question_type"] == source_type
        )
        result[source_type] = counter_dict(counts)
    return result


def calculate_distributions(
    data: SourceData,
    question_rows: Sequence[dict[str, Any]],
    document_rows: Sequence[dict[str, Any]],
    question_selection: dict[str, Any],
    hard_negatives: dict[str, Any],
) -> dict[str, Any]:
    gold_rows = [row for row in document_rows if row["role"] == "gold"]
    hard_rows = [row for row in document_rows if row["role"] == "hard_negative"]
    gold_indices = [row["source_index"] for row in gold_rows]
    thresholds = question_selection["length_thresholds"]
    corpus = data.corpus

    evidence_source_occurrences: Counter[str] = Counter()
    evidence_category_occurrences: Counter[str] = Counter()
    evidence_month_occurrences: Counter[str] = Counter()
    evidence_length_occurrences: Counter[str] = Counter()
    evidence_global_reuse_occurrences: Counter[str] = Counter()
    for row in question_rows:
        for url in row["gold_urls"]:
            index = data.corpus_indices_by_url[url][0]
            document = corpus[index]
            evidence_source_occurrences[document["source"]] += 1
            evidence_category_occurrences[document["category"]] += 1
            evidence_month_occurrences[document["published_at"][:7]] += 1
            evidence_length_occurrences[body_length_bin(document, thresholds)] += 1
            evidence_global_reuse_occurrences[
                selected_doc_reuse_bin(question_selection["global_url_reuse"][url])
            ] += 1

    selected_document_use = question_selection["selected_document_use"]
    hard_ranks = [
        link["eligible_bm25_rank"]
        for detail in hard_negatives["detail_by_document"].values()
        for link in detail["per_question"].values()
    ]
    hard_scores = [
        link["bm25_score"]
        for detail in hard_negatives["detail_by_document"].values()
        for link in detail["per_question"].values()
    ]

    return {
        "questions_by_source_type": counter_dict(
            Counter(row["source_question_type"] for row in question_rows)
        ),
        "questions_by_output_type": counter_dict(
            Counter(row["question_type"] for row in question_rows)
        ),
        "evidence_count_by_source_type": nested_evidence_distribution(question_rows),
        "gold_evidence_occurrences": sum(len(row["gold_document_ids"]) for row in question_rows),
        "gold_document_selected_question_reuse": {
            "histogram": counter_dict(Counter(selected_document_use.values())),
            "maximum": max(selected_document_use.values(), default=0),
            "mean": round(
                sum(selected_document_use.values()) / len(selected_document_use), 6
            ),
        },
        "evidence_sources": counter_dict(evidence_source_occurrences),
        "unique_gold_document_sources": counter_dict(
            Counter(corpus[index]["source"] for index in gold_indices)
        ),
        "evidence_categories": counter_dict(evidence_category_occurrences),
        "unique_gold_document_categories": counter_dict(
            Counter(corpus[index]["category"] for index in gold_indices)
        ),
        "evidence_publication_months": counter_dict(evidence_month_occurrences),
        "unique_gold_document_publication_months": counter_dict(
            Counter(corpus[index]["published_at"][:7] for index in gold_indices)
        ),
        "evidence_body_length_bins": counter_dict(evidence_length_occurrences),
        "unique_gold_document_body_length_bins": counter_dict(
            Counter(body_length_bin(corpus[index], thresholds) for index in gold_indices)
        ),
        "body_length_bin_thresholds_characters": {
            "short_max": thresholds[0],
            "medium_max": thresholds[1],
        },
        "evidence_global_document_reuse_bins": counter_dict(
            evidence_global_reuse_occurrences
        ),
        "topic_term_selected_question_frequency": counter_dict(
            question_selection["topic_counts"]
        ),
        "null_query_lexical_proxy_sources": counter_dict(
            question_selection["null_proxy_counts"]["source"]
        ),
        "null_query_lexical_proxy_categories": counter_dict(
            question_selection["null_proxy_counts"]["category"]
        ),
        "hard_negative_sources": counter_dict(
            Counter(row["source_fields"]["source"] for row in hard_rows)
        ),
        "hard_negative_categories": counter_dict(
            Counter(row["source_fields"]["category"] for row in hard_rows)
        ),
        "hard_negative_link_count": len(hard_ranks),
        "hard_negative_eligible_bm25_rank": {
            "minimum": min(hard_ranks, default=0),
            "median": sorted(hard_ranks)[len(hard_ranks) // 2] if hard_ranks else 0,
            "maximum": max(hard_ranks, default=0),
        },
        "hard_negative_bm25_score": {
            "minimum": round(min(hard_scores), 12) if hard_scores else 0.0,
            "maximum": round(max(hard_scores), 12) if hard_scores else 0.0,
        },
    }


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def evaluate_invariants(
    data: SourceData,
    question_rows: Sequence[dict[str, Any]],
    document_rows: Sequence[dict[str, Any]],
    mapping: dict[str, dict[str, list[str]]],
    hard_negative_details: dict[str, Any],
    *,
    questions_per_type: int,
    hard_negative_count: int,
    manifest: dict[str, Any] | None = None,
    bm25: BM25Index | None = None,
    output_payloads: dict[str, bytes] | None = None,
    expected_output_hashes: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    corpus = data.corpus
    questions = data.questions
    source_question_by_index = {index: row for index, row in enumerate(questions)}
    source_document_by_index = {index: row for index, row in enumerate(corpus)}
    question_by_id = {row["question_id"]: row for row in question_rows}
    document_by_id = {row["document_id"]: row for row in document_rows}
    pilot_question_ids = {row["question_id"] for row in data.pilot_questions}
    pilot_document_ids = {row["document_id"] for row in data.pilot_documents}

    add_check(checks, "exact_question_count", len(question_rows) == questions_per_type * 4, len(question_rows))
    type_counts = Counter(row["source_question_type"] for row in question_rows)
    add_check(
        checks,
        "exact_type_balance",
        all(type_counts[source_type] == questions_per_type for source_type in SOURCE_TYPE_ORDER),
        counter_dict(type_counts),
    )
    add_check(
        checks,
        "no_duplicate_question_ids",
        len(question_by_id) == len(question_rows),
        len(question_by_id),
    )
    add_check(
        checks,
        "no_duplicate_document_ids",
        len(document_by_id) == len(document_rows),
        len(document_by_id),
    )
    add_check(
        checks,
        "no_duplicate_document_urls_or_source_indices",
        len({row["url"] for row in document_rows}) == len(document_rows)
        and len({row["source_index"] for row in document_rows}) == len(document_rows),
        len(document_rows),
    )
    add_check(
        checks,
        "no_pilot_question_overlap",
        not (set(question_by_id) & pilot_question_ids),
        sorted(set(question_by_id) & pilot_question_ids),
    )
    add_check(
        checks,
        "mapping_covers_every_question_exactly",
        set(mapping) == set(question_by_id),
        {"mapping": len(mapping), "questions": len(question_by_id)},
    )

    question_source_exact = True
    question_ids_exact = True
    question_hashes_exact = True
    evidence_integrity = True
    evidence_shape = True
    null_semantics = True
    all_gold_ids: set[str] = set()
    mapping_matches_question_rows = True
    for row in question_rows:
        source = source_question_by_index.get(row["source_index"])
        if source is None:
            question_source_exact = False
            continue
        question_source_exact &= (
            row["question"] == source["query"]
            and row["gold_answer"] == source["answer"]
            and row["source_fields"] == source
            and row["source_question_type"] == source["question_type"]
            and row["source_record_sha256"] == canonical_record_hash(source)
        )
        question_ids_exact &= row["question_id"] == question_id(source["query"])
        question_hashes_exact &= row["question_sha256"] == sha256_text(source["query"])
        urls = [evidence["url"] for evidence in source["evidence_list"]]
        expected_ids = [document_id(url) for url in urls]
        if source["question_type"] == "null_query":
            null_semantics &= (
                source["evidence_list"] == []
                and row["gold_urls"] == []
                and row["gold_document_ids"] == []
                and row["answerable"] is False
            )
        else:
            evidence_shape &= 2 <= len(urls) <= 4 and len(urls) == len(set(urls))
            for evidence in source["evidence_list"]:
                matches = data.corpus_indices_by_url.get(evidence["url"], ())
                evidence_integrity &= len(matches) == 1
                if len(matches) == 1:
                    evidence_integrity &= evidence["fact"] in corpus[matches[0]]["body"]
        mapping_matches_question_rows &= (
            row["gold_urls"] == urls
            and row["gold_document_ids"] == expected_ids
            and mapping.get(row["question_id"], {}).get("gold_document_ids") == expected_ids
            and mapping.get(row["question_id"], {}).get("hard_negative_document_ids")
            == row["hard_negative_document_ids"]
            and mapping.get(row["question_id"], {}).get("distractor_document_ids")
            == row["hard_negative_document_ids"]
        )
        all_gold_ids.update(expected_ids)

    add_check(checks, "stable_question_id_method_matches_pilot", question_ids_exact, "mhq_ + SHA-256(query)[:16]")
    add_check(checks, "question_hashes_match_exact_source_query", question_hashes_exact, len(question_rows))
    add_check(checks, "question_source_fields_preserved_exactly", question_source_exact, len(question_rows))
    add_check(checks, "answerable_questions_have_2_to_4_unique_gold_urls", evidence_shape, "selected answerable questions")
    add_check(checks, "null_queries_keep_empty_evidence_and_false_answerable", null_semantics, type_counts["null_query"])
    add_check(checks, "every_evidence_url_resolves_exactly_once_and_fact_is_in_body", evidence_integrity, "exact URL and substring checks")
    add_check(checks, "mapping_matches_question_rows", mapping_matches_question_rows, len(mapping))

    document_source_exact = True
    document_ids_exact = True
    document_hashes_exact = True
    nulls_preserved = True
    role_counts = Counter(row["role"] for row in document_rows)
    for row in document_rows:
        source = source_document_by_index.get(row["source_index"])
        if source is None:
            document_source_exact = False
            continue
        expected_fields = {
            key: source[key]
            for key in ("title", "author", "source", "published_at", "category", "url")
        }
        document_source_exact &= (
            row["text"] == source["body"]
            and row["url"] == source["url"]
            and row["source_fields"] == expected_fields
            and row["source_record_sha256"] == canonical_record_hash(source)
        )
        document_ids_exact &= row["document_id"] == document_id(source["url"])
        document_hashes_exact &= row["text_sha256"] == sha256_text(source["body"])
        nulls_preserved &= row["source_fields"]["author"] is source["author"] or (
            row["source_fields"]["author"] == source["author"]
        )

    add_check(checks, "stable_document_id_method_matches_pilot", document_ids_exact, "mhd_ + SHA-256(url)[:16]")
    add_check(checks, "document_hashes_match_exact_source_body", document_hashes_exact, len(document_rows))
    add_check(checks, "source_document_text_and_metadata_preserved_exactly", document_source_exact, len(document_rows))
    add_check(checks, "source_null_values_preserved_without_fill", nulls_preserved and question_source_exact, "deep source-field equality")
    add_check(
        checks,
        "all_gold_documents_included",
        all_gold_ids <= set(document_by_id),
        {"required": len(all_gold_ids), "included": len(all_gold_ids & set(document_by_id))},
    )
    add_check(
        checks,
        "exact_hard_negative_document_count",
        role_counts["hard_negative"] == hard_negative_count,
        role_counts["hard_negative"],
    )
    add_check(
        checks,
        "gold_and_hard_negative_roles_are_disjoint",
        all(row["role"] in {"gold", "hard_negative"} for row in document_rows)
        and not ({row["document_id"] for row in document_rows if row["role"] == "gold"}
                 & {row["document_id"] for row in document_rows if row["role"] == "hard_negative"}),
        counter_dict(role_counts),
    )

    hard_negative_pairs_valid = True
    hard_negative_details_complete = True
    every_question_has_hard_negative = True
    hard_negative_ids = {
        row["document_id"] for row in document_rows if row["role"] == "hard_negative"
    }
    for qid, row in question_by_id.items():
        linked = mapping.get(qid, {}).get("hard_negative_document_ids", [])
        every_question_has_hard_negative &= bool(linked) and set(linked) <= hard_negative_ids
        gold_indices = {
            data.corpus_indices_by_url[url][0] for url in row["gold_urls"]
        }
        source_question = questions[row["source_index"]]
        for hard_id in linked:
            document_row = document_by_id.get(hard_id)
            if document_row is None:
                hard_negative_pairs_valid = False
                continue
            document_index = document_row["source_index"]
            source_document = corpus[document_index]
            checks_for_pair = hard_negative_pair_checks(
                source_question, source_document, gold_indices, document_index
            )
            hard_negative_pairs_valid &= all(checks_for_pair.values())
            detail = hard_negative_details.get(hard_id, {}).get("per_question", {}).get(qid)
            hard_negative_details_complete &= detail is not None
            if detail is not None:
                hard_negative_details_complete &= all(
                    detail.get(name) is True for name in checks_for_pair
                )

    add_check(checks, "every_question_has_mapped_hard_negative", every_question_has_hard_negative, len(question_rows))
    add_check(
        checks,
        "hard_negatives_exclude_question_gold_complete_chain_and_direct_answer",
        hard_negative_pairs_valid,
        "per linked question/document pair; normalized exact lexical checks",
    )
    add_check(checks, "hard_negative_link_audit_details_complete", hard_negative_details_complete, len(hard_negative_details))

    reverse_links_valid = True
    for document_row in document_rows:
        if document_row["role"] != "hard_negative":
            reverse_links_valid &= not document_row["hard_negative_for_question_ids"]
            continue
        did = document_row["document_id"]
        detail = hard_negative_details.get(did)
        if detail is None:
            reverse_links_valid = False
            continue
        expected_related = set(detail["related_question_ids"])
        reverse_links_valid &= (
            set(document_row["related_question_ids"]) == expected_related
            and set(document_row["hard_negative_for_question_ids"]) == expected_related
            and all(
                did in mapping.get(qid, {}).get("hard_negative_document_ids", [])
                for qid in expected_related
            )
        )
    add_check(
        checks,
        "hard_negative_reverse_links_match_documents_and_mapping",
        reverse_links_valid,
        role_counts["hard_negative"],
    )

    if bm25 is not None and manifest is not None:
        seed = manifest["parameters"]["seed"]
        global_gold_indices = {
            row["source_index"] for row in document_rows if row["role"] == "gold"
        }
        bm25_reproducible = True
        reproduced_links = 0
        for qid, question_row in question_by_id.items():
            source_question = questions[question_row["source_index"]]
            scores = bm25.scores(hard_negative_query_text(source_question))
            question_gold_indices = {
                data.corpus_indices_by_url[url][0] for url in question_row["gold_urls"]
            }
            eligible: list[int] = []
            for document_index, source_document in enumerate(corpus):
                if document_index in global_gold_indices:
                    continue
                pair = hard_negative_pair_checks(
                    source_question,
                    source_document,
                    question_gold_indices,
                    document_index,
                )
                if all(pair.values()) and scores[document_index] > 0.0:
                    eligible.append(document_index)
            ranked = sorted(
                eligible,
                key=lambda index: (
                    -scores[index],
                    stable_tie(
                        seed,
                        "hard-negative-rank",
                        f"{question_row['source_index']}:{index}",
                    ),
                    index,
                ),
            )
            ranks = {document_index: rank for rank, document_index in enumerate(ranked, 1)}
            query_tokens = set(lexical_tokens(hard_negative_query_text(source_question)))
            for did in mapping[qid]["hard_negative_document_ids"]:
                document_row = document_by_id[did]
                document_index = document_row["source_index"]
                detail = hard_negative_details.get(did, {}).get("per_question", {}).get(qid)
                reproduced_links += 1
                if detail is None or document_index not in ranks:
                    bm25_reproducible = False
                    continue
                overlap = sorted(query_tokens & bm25.doc_token_sets[document_index])[:20]
                bm25_reproducible &= (
                    detail["bm25_score"] == round(scores[document_index], 12)
                    and detail["eligible_bm25_rank"] == ranks[document_index]
                    and detail["lexical_overlap_terms"] == overlap
                )
        add_check(
            checks,
            "hard_negative_bm25_scores_ranks_and_overlap_reproducible",
            bm25_reproducible,
            reproduced_links,
        )

    actual_pilot_document_overlap = sorted(set(document_by_id) & pilot_document_ids)
    add_check(
        checks,
        "pilot_document_overlap_computable",
        True,
        {"count": len(actual_pilot_document_overlap), "document_ids": actual_pilot_document_overlap},
    )

    if manifest is not None:
        source_hashes_match = (
            manifest["source_files"]["questions"]["sha256"] == sha256_file(QUESTIONS_SOURCE)
            and manifest["source_files"]["corpus"]["sha256"] == sha256_file(CORPUS_SOURCE)
            and manifest["source_files"]["pilot_questions"]["sha256"]
            == sha256_file(PILOT_QUESTIONS)
            and manifest["source_files"]["pilot_documents"]["sha256"]
            == sha256_file(PILOT_DOCUMENTS)
        )
        add_check(
            checks,
            "manifest_source_file_hashes_match_exact_files",
            source_hashes_match,
            {key: value["sha256"] for key, value in manifest["source_files"].items()},
        )

        hash_maps_match = (
            manifest["question_hashes"]
            == {row["question_id"]: row["question_sha256"] for row in question_rows}
            and manifest["document_hashes"]
            == {row["document_id"]: row["text_sha256"] for row in document_rows}
            and manifest["source_record_hashes"]["questions"]
            == {row["question_id"]: row["source_record_sha256"] for row in question_rows}
            and manifest["source_record_hashes"]["documents"]
            == {row["document_id"]: row["source_record_sha256"] for row in document_rows}
        )
        add_check(
            checks,
            "manifest_content_and_source_record_hash_maps_match_rows",
            hash_maps_match,
            {"questions": len(question_rows), "documents": len(document_rows)},
        )

        expected_question_indices = {
            source_type: [
                row["source_index"]
                for row in question_rows
                if row["source_question_type"] == source_type
            ]
            for source_type in SOURCE_TYPE_ORDER
        }
        expected_gold_indices = [
            row["source_index"] for row in document_rows if row["role"] == "gold"
        ]
        expected_hard_indices = [
            row["source_index"] for row in document_rows if row["role"] == "hard_negative"
        ]
        source_indices_match = (
            manifest["source_indices"]["questions_by_type"] == expected_question_indices
            and manifest["source_indices"]["gold_documents"] == expected_gold_indices
            and manifest["source_indices"]["hard_negative_documents"] == expected_hard_indices
        )
        add_check(
            checks,
            "manifest_source_index_lists_match_rows",
            source_indices_match,
            {
                "questions": sum(map(len, expected_question_indices.values())),
                "gold_documents": len(expected_gold_indices),
                "hard_negative_documents": len(expected_hard_indices),
            },
        )

        roles_match = manifest["document_roles"] == {
            row["document_id"]: row["role"] for row in document_rows
        }
        add_check(
            checks,
            "manifest_document_roles_match_rows",
            roles_match,
            counter_dict(role_counts),
        )

        actual_overlap_payload = compute_pilot_overlap(data, question_rows, document_rows)
        add_check(
            checks,
            "manifest_pilot_overlap_matches_recomputed_overlap",
            manifest["pilot_overlap"] == actual_overlap_payload,
            {
                "questions": actual_overlap_payload["questions"]["count"],
                "documents": actual_overlap_payload["documents"]["count"],
            },
        )

        expected_order = (
            sorted(expected_gold_indices) + sorted(expected_hard_indices)
        )
        add_check(
            checks,
            "stable_question_and_document_output_order",
            [row["source_index"] for row in document_rows] == expected_order
            and all(
                [
                    row["source_index"]
                    for row in question_rows
                    if row["source_question_type"] == source_type
                ]
                == sorted(expected_question_indices[source_type])
                for source_type in SOURCE_TYPE_ORDER
            ),
            "type/corpus source-index order",
        )

    if output_payloads is not None and expected_output_hashes is not None:
        hashes_match = True
        details: dict[str, Any] = {}
        for filename, metadata in expected_output_hashes.items():
            actual = sha256_bytes(output_payloads[filename])
            hashes_match &= actual == metadata["sha256"]
            details[filename] = actual
        add_check(checks, "declared_output_file_hashes_match_bytes", hashes_match, details)

        question_payload, actual_question_line_hashes = jsonl_bytes(list(question_rows))
        document_payload, actual_document_line_hashes = jsonl_bytes(list(document_rows))
        line_hashes_match = (
            question_payload == output_payloads["questions.jsonl"]
            and document_payload == output_payloads["documents.jsonl"]
            and actual_question_line_hashes
            == expected_output_hashes["questions.jsonl"]["record_line_sha256"]
            and actual_document_line_hashes
            == expected_output_hashes["documents.jsonl"]["record_line_sha256"]
        )
        add_check(checks, "jsonl_record_line_hashes_match_exact_serialized_lines", line_hashes_match, {"questions": len(actual_question_line_hashes), "documents": len(actual_document_line_hashes)})

    return checks


def mapping_from_rows(question_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    return {
        row["question_id"]: {
            "gold_document_ids": row["gold_document_ids"],
            "hard_negative_document_ids": row["hard_negative_document_ids"],
            "distractor_document_ids": row["hard_negative_document_ids"],
        }
        for row in question_rows
    }


def markdown_table(counter: dict[str, int], first_header: str) -> list[str]:
    lines = [f"| {first_header} | Count |", "|---|---:|"]
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {str(key).replace('|', chr(92) + '|')} | {value} |")
    return lines


def build_selection_report(
    data: SourceData,
    question_rows: Sequence[dict[str, Any]],
    document_rows: Sequence[dict[str, Any]],
    question_selection: dict[str, Any],
    hard_negatives: dict[str, Any],
    distributions: dict[str, Any],
    pilot_overlap: dict[str, Any],
    *,
    name: str,
    seed: int,
    questions_per_type: int,
    hard_negative_count: int,
) -> str:
    gold_count = sum(row["role"] == "gold" for row in document_rows)
    type_labels = ", ".join(
        f"{source_type}={distributions['questions_by_source_type'][source_type]}"
        for source_type in SOURCE_TYPE_ORDER
    )
    evidence_lines = []
    for source_type in SOURCE_TYPE_ORDER:
        values = distributions["evidence_count_by_source_type"][source_type]
        evidence_lines.append(
            f"- `{source_type}`: "
            + ", ".join(f"{count} docs → {number} questions" for count, number in values.items())
        )

    lines = [
        f"# MultiHop-RAG subset selection report: {name}",
        "",
        "## Result",
        "",
        f"The immutable subset contains **{len(question_rows)} questions** ({type_labels}), "
        f"**{gold_count} unique gold documents**, and **{hard_negative_count} additional hard-negative documents**. "
        f"The final corpus has **{len(document_rows)} documents**.",
        "",
        f"Selection uses `seed = {seed}` only for SHA-256-based stable tie breaking. It does not inspect pilot/model outputs or metrics. "
        "The 16 pilot question IDs are a blocklist and are not otherwise scored.",
        "",
        "## Deterministic question selection",
        "",
        f"For each of the four source types, exactly {questions_per_type} questions are selected. "
        "Evidence-count quotas are as even as possible across the counts supported by that type: inference 2/3/4, comparison 2/3, temporal 2/3, and null 0. "
        "Questions with repeated evidence URLs are ineligible because the experiment requires 2–4 unique gold URLs.",
        "",
        f"The greedy objective has a soft target of **{question_selection['soft_gold_union_target']} unique gold documents** so that the requested hard negatives lead toward a corpus near {TARGET_CORPUS_DOCUMENTS}. "
        f"No gold URL is ever dropped. A gold document may support at most {BASE_GOLD_DOCUMENT_QUESTION_CAP} selected questions before a deterministic relaxation is allowed. "
        "The objective then minimizes concentration by document, evidence source, category, publication month, body-length tertile, full-dataset reuse bin, and rare lexical topic signatures.",
        "",
        "Evidence-count distribution:",
        "",
        *evidence_lines,
        "",
        "Selected gold-document reuse:",
        "",
        f"- Maximum selected-question uses of one gold document: **{distributions['gold_document_selected_question_reuse']['maximum']}**.",
        f"- Reuse histogram (uses → documents): `{json.dumps(distributions['gold_document_selected_question_reuse']['histogram'], ensure_ascii=False, sort_keys=True)}`.",
        "",
        "## Stratification distributions",
        "",
        "Sources below count evidence occurrences; a multi-hop question contributes once for each gold document.",
        "",
        *markdown_table(distributions["evidence_sources"], "Evidence source"),
        "",
        *markdown_table(distributions["evidence_categories"], "Evidence category"),
        "",
        *markdown_table(distributions["evidence_publication_months"], "Publication month"),
        "",
        *markdown_table(distributions["evidence_body_length_bins"], "Body-length tertile"),
        "",
        "Full-dataset evidence-document reuse bins:",
        "",
        *markdown_table(distributions["evidence_global_document_reuse_bins"], "Full-dataset uses"),
        "",
        "For `null_query`, source/category/date/evidence-length stratification is not defined because `evidence_list` is empty. "
        "A deterministic top-BM25 corpus document is used only as a selection profile; it is not relabeled as gold evidence.",
        "",
        "## Hard-negative method and audit",
        "",
        f"A local BM25 index over `title + body` selects exactly **{hard_negative_count}** documents outside the global selected-gold union. "
        "The query side uses the original question plus its evidence titles (when present). A seeded greedy facility-location objective maximizes normalized lexical coverage across all selected questions while applying source/category caps. "
        "Every selected question is mapped to at least one of the hard negatives.",
        "",
        "For every mapped question/document pair, the audit verifies exact gold exclusion, absence of the complete set of exact evidence facts, and absence of the normalized exact full answer (generic answers such as Yes/No and the null sentinel are not treated as answer strings).",
        "",
        "| Corpus index | Title | Source | Category | Linked QA source indices | Eligible BM25 rank range |",
        "|---:|---|---|---|---|---:|",
    ]
    question_index_by_id = {row["question_id"]: row["source_index"] for row in question_rows}
    for source_index in hard_negatives["selected_indices"]:
        document = data.corpus[source_index]
        detail = hard_negatives["detail_by_document"][source_index]
        ranks = [item["eligible_bm25_rank"] for item in detail["per_question"].values()]
        linked = ", ".join(
            str(question_index_by_id[qid]) for qid in detail["related_question_ids"]
        )
        title = document["title"].replace("|", "\\|").replace("\n", " ")
        source = document["source"].replace("|", "\\|")
        category = document["category"].replace("|", "\\|")
        lines.append(
            f"| {source_index} | {title} | {source} | {category} | {linked} | {min(ranks)}–{max(ranks)} |"
        )

    lines.extend(
        [
            "",
            "## Pilot isolation and overlap",
            "",
            f"- Question overlap with pilot: **{pilot_overlap['questions']['count']}**.",
            f"- Document overlap with pilot: **{pilot_overlap['documents']['count']}** "
            f"({pilot_overlap['documents']['gold_in_subset']} gold, {pilot_overlap['documents']['hard_negative_in_subset']} hard negative in this subset).",
            "- Pilot questions, pilot artifacts, and pilot results were not modified.",
            "",
            "## Deterministic relaxations",
            "",
        ]
    )
    all_relaxations = question_selection["relaxations"] + hard_negatives["relaxations"]
    if all_relaxations:
        for relaxation in all_relaxations:
            lines.append(f"- `{json.dumps(relaxation, ensure_ascii=False, sort_keys=True)}`")
    else:
        lines.append("- None. Base document/source/category caps were sufficient.")

    lines.extend(
        [
            "",
            "## Integrity and reproducibility",
            "",
            "- Question and document IDs use the exact pilot formulas.",
            "- Exact source query/body strings and all decoded source fields, including JSON nulls, are preserved.",
            "- The manifest records source-file, content, source-record, exact JSONL-line, and output-file SHA-256 values.",
            "- Re-running the builder with identical parameters either produces byte-identical artifacts or reports the existing immutable subset as unchanged.",
            "- No chunking, LightRAG/Ollama indexing, model inference, or experiment execution is performed.",
            "",
            "## Limitations",
            "",
            "- The source corpus itself is imbalanced and covers only September–December 2023; stratification cannot create unavailable source/date coverage.",
            "- Entity/topic diversity is approximated by rare lexical signatures, not by a learned NER or semantic model.",
            "- BM25 is lexical. The model-free leakage audit catches exact evidence chains and normalized exact answers, but cannot prove that a paraphrase never conveys an answer.",
            "- `null_query` has no gold evidence, so its source/category/date profile is only a lexical proxy and is explicitly kept separate from gold distributions.",
            "",
        ]
    )
    return "\n".join(lines)


def hard_negative_manifest_details(
    data: SourceData, hard_negatives: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source_index in hard_negatives["selected_indices"]:
        did = document_id(data.corpus[source_index]["url"])
        result[did] = hard_negatives["detail_by_document"][source_index]
    return result


def compute_pilot_overlap(
    data: SourceData,
    question_rows: Sequence[dict[str, Any]],
    document_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    pilot_question_ids = {row["question_id"] for row in data.pilot_questions}
    pilot_documents_by_id = {row["document_id"]: row for row in data.pilot_documents}
    question_overlap = sorted(
        {row["question_id"] for row in question_rows} & pilot_question_ids
    )
    overlapping_rows = [
        row for row in document_rows if row["document_id"] in pilot_documents_by_id
    ]
    return {
        "questions": {"count": len(question_overlap), "question_ids": question_overlap},
        "documents": {
            "count": len(overlapping_rows),
            "document_ids": [row["document_id"] for row in overlapping_rows],
            "gold_in_subset": sum(row["role"] == "gold" for row in overlapping_rows),
            "hard_negative_in_subset": sum(
                row["role"] == "hard_negative" for row in overlapping_rows
            ),
            "details": {
                row["document_id"]: {
                    "subset_role": row["role"],
                    "pilot_role": pilot_documents_by_id[row["document_id"]]["role"],
                    "source_index": row["source_index"],
                }
                for row in overlapping_rows
            },
        },
    }


def build_manifest(
    data: SourceData,
    question_rows: Sequence[dict[str, Any]],
    document_rows: Sequence[dict[str, Any]],
    question_selection: dict[str, Any],
    hard_negatives: dict[str, Any],
    distributions: dict[str, Any],
    pilot_overlap: dict[str, Any],
    mapping: dict[str, dict[str, list[str]]],
    *,
    name: str,
    questions_per_type: int,
    seed: int,
    hard_negative_count: int,
) -> dict[str, Any]:
    question_indices = {
        source_type: [
            row["source_index"]
            for row in question_rows
            if row["source_question_type"] == source_type
        ]
        for source_type in SOURCE_TYPE_ORDER
    }
    document_roles = {row["document_id"]: row["role"] for row in document_rows}
    subset_fingerprint_input = {
        "dataset": "MultiHop-RAG",
        "name": name,
        "seed": seed,
        "questions_per_type": questions_per_type,
        "hard_negatives": hard_negative_count,
        "question_source_indices": question_indices,
        "document_source_indices": [row["source_index"] for row in document_rows],
        "source_hashes": {
            "questions": sha256_file(QUESTIONS_SOURCE),
            "corpus": sha256_file(CORPUS_SOURCE),
        },
    }
    subset_id = f"{name}_{canonical_record_hash(subset_fingerprint_input)[:16]}"
    return {
        "manifest_version": "1.0.0",
        "builder_version": BUILDER_VERSION,
        "dataset": "MultiHop-RAG",
        "subset_name": name,
        "subset_id": subset_id,
        "immutable": True,
        "parameters": {
            "seed": seed,
            "questions_per_type": questions_per_type,
            "hard_negatives": hard_negative_count,
            "target_corpus_documents_soft": TARGET_CORPUS_DOCUMENTS,
            "gold_union_target_soft": question_selection["soft_gold_union_target"],
        },
        "source_files": {
            "questions": {
                "path_relative_to_subset": QUESTION_SOURCE_REF,
                "project_relative_path": "experiment/MultiHop-RAG/MultiHopRAG.json",
                "sha256": sha256_file(QUESTIONS_SOURCE),
                "record_count": len(data.questions),
            },
            "corpus": {
                "path_relative_to_subset": CORPUS_SOURCE_REF,
                "project_relative_path": "experiment/MultiHop-RAG/corpus.json",
                "sha256": sha256_file(CORPUS_SOURCE),
                "record_count": len(data.corpus),
            },
            "pilot_questions": {
                "path_relative_to_subset": PILOT_QUESTIONS_REF,
                "sha256": sha256_file(PILOT_QUESTIONS),
                "record_count": len(data.pilot_questions),
                "purpose": "question-ID exclusion only; no pilot metrics or results are read",
            },
            "pilot_documents": {
                "path_relative_to_subset": PILOT_DOCUMENTS_REF,
                "sha256": sha256_file(PILOT_DOCUMENTS),
                "record_count": len(data.pilot_documents),
                "purpose": "document-overlap reporting only",
            },
        },
        "id_definitions": {
            "question_id": "mhq_ + first 16 hex characters of SHA-256(exact UTF-8 source query)",
            "document_id": "mhd_ + first 16 hex characters of SHA-256(exact UTF-8 source URL)",
            "same_as_pilot": True,
        },
        "selection": {
            "method": "deterministic stratified greedy source-only selection",
            "seed_use": "SHA-256 stable tie-breaking only; no PRNG state",
            "model_or_pilot_metrics_used": False,
            "source_indices_are_zero_based": True,
            "question_order": "source type order, then ascending source index",
            "document_order": "gold corpus source index, then hard-negative corpus source index",
            "evidence_count_quotas": {
                source_type: {str(key): value for key, value in quotas.items()}
                for source_type, quotas in question_selection["evidence_quotas"].items()
            },
            "gold_rule": "union of every exact evidence URL for every selected answerable question; never size-pruned",
            "gold_document_question_cap_base": BASE_GOLD_DOCUMENT_QUESTION_CAP,
            "features": [
                "evidence document count",
                "source",
                "category",
                "publication month",
                "body-length tertile",
                "full-dataset evidence reuse bin",
                "selected-document question reuse",
                "rare lexical entity/topic signature",
            ],
            "hard_negative_method": {
                "ranker": "local deterministic BM25(k1=1.5,b=0.75) over source title + unchanged body",
                "query": "source question + source evidence titles",
                "selection": "seeded greedy facility location over normalized BM25 coverage",
                "global_gold_documents_excluded": True,
                "source_cap_base": BASE_HARD_NEGATIVE_SOURCE_CAP,
                "category_cap_base": BASE_HARD_NEGATIVE_CATEGORY_CAP,
                "leakage_checks": [
                    "not gold for linked question",
                    "does not contain every exact evidence fact",
                    "does not contain normalized exact non-generic full answer",
                ],
                "generative_model_used": False,
            },
            "eligibility": question_selection["eligibility"],
            "question_selection_trace": question_selection["selection_trace"],
            "hard_negative_selection_trace": hard_negatives["trace"],
            "relaxations": {
                "question_selection": question_selection["relaxations"],
                "hard_negative_selection": hard_negatives["relaxations"],
            },
        },
        "source_indices": {
            "questions_by_type": question_indices,
            "gold_documents": question_selection["selected_gold_indices"],
            "hard_negative_documents": hard_negatives["selected_indices"],
        },
        "counts": {
            "questions_total": len(question_rows),
            "answerable_questions": sum(row["answerable"] for row in question_rows),
            "null_questions": sum(not row["answerable"] for row in question_rows),
            "questions_by_source_type": distributions["questions_by_source_type"],
            "questions_by_output_type": distributions["questions_by_output_type"],
            "gold_documents": sum(row["role"] == "gold" for row in document_rows),
            "hard_negative_documents": sum(
                row["role"] == "hard_negative" for row in document_rows
            ),
            "documents_total": len(document_rows),
            "gold_evidence_occurrences": distributions["gold_evidence_occurrences"],
        },
        "distributions": distributions,
        "question_document_mapping": mapping,
        "document_roles": document_roles,
        "hard_negative_links": hard_negative_manifest_details(data, hard_negatives),
        "pilot_overlap": pilot_overlap,
        "question_hashes": {
            row["question_id"]: row["question_sha256"] for row in question_rows
        },
        "document_hashes": {
            row["document_id"]: row["text_sha256"] for row in document_rows
        },
        "source_record_hashes": {
            "questions": {
                row["question_id"]: row["source_record_sha256"] for row in question_rows
            },
            "documents": {
                row["document_id"]: row["source_record_sha256"] for row in document_rows
            },
        },
        "hash_definitions": {
            "question_hashes": "SHA-256 of exact decoded source query encoded as UTF-8",
            "document_hashes": "SHA-256 of exact decoded source body encoded as UTF-8",
            "source_record_hashes": "SHA-256 of compact UTF-8 JSON serialization preserving source key order and nulls",
            "record_line_sha256": "SHA-256 of exact compact JSONL record bytes excluding the LF terminator",
            "output_file_sha256": "SHA-256 of exact file bytes including final LF",
        },
        "output_files": {},
        "invariants": {},
    }


def audit_payload(
    checks: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    output_payloads: dict[str, bytes],
) -> dict[str, Any]:
    failures = [row["name"] for row in checks if not row["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "audit_kind": "model_free",
        "dataset": manifest["dataset"],
        "subset_id": manifest["subset_id"],
        "parameters": manifest["parameters"],
        "summary": {
            "checks_total": len(checks),
            "checks_passed": len(checks) - len(failures),
            "checks_failed": len(failures),
            "failed_check_names": failures,
        },
        "checks": list(checks),
        "audited_file_hashes": {
            filename: sha256_bytes(payload)
            for filename, payload in output_payloads.items()
        },
        "source_file_hashes": {
            "MultiHopRAG.json": sha256_file(QUESTIONS_SOURCE),
            "corpus.json": sha256_file(CORPUS_SOURCE),
            "pilot_questions.jsonl": sha256_file(PILOT_QUESTIONS),
            "pilot_documents.jsonl": sha256_file(PILOT_DOCUMENTS),
        },
        "limitations": [
            "BM25 and leakage checks are lexical and model-free; semantic paraphrase leakage cannot be proven absent.",
            "null_query has no gold evidence, so its source/date/category selection profile is lexical only.",
            "The source corpus covers September through December 2023 and has native source/category imbalance.",
        ],
        "forbidden_operations_performed": {
            "generative_model_selection": False,
            "pilot_metric_selection": False,
            "lightrag_run": False,
            "ollama_indexing": False,
            "main_experiment_run": False,
        },
    }


def build_artifacts(
    *, name: str, questions_per_type: int, seed: int, hard_negative_count: int
) -> tuple[dict[str, bytes], dict[str, Any]]:
    require(bool(re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name)), "invalid subset name")
    require(questions_per_type > 0, "questions-per-type must be positive")
    require(20 <= hard_negative_count <= 30, "hard-negatives must be between 20 and 30")

    data = load_and_validate_sources()
    bm25 = BM25Index(data.corpus)
    question_selection = select_questions(
        data,
        bm25,
        questions_per_type=questions_per_type,
        seed=seed,
        hard_negative_count=hard_negative_count,
    )
    hard_negatives = select_hard_negatives(
        data,
        bm25,
        question_selection,
        hard_negative_count=hard_negative_count,
        seed=seed,
    )
    question_rows, document_rows = build_rows(data, question_selection, hard_negatives)
    mapping = mapping_from_rows(question_rows)
    distributions = calculate_distributions(
        data, question_rows, document_rows, question_selection, hard_negatives
    )
    pilot_overlap = compute_pilot_overlap(data, question_rows, document_rows)
    manifest = build_manifest(
        data,
        question_rows,
        document_rows,
        question_selection,
        hard_negatives,
        distributions,
        pilot_overlap,
        mapping,
        name=name,
        questions_per_type=questions_per_type,
        seed=seed,
        hard_negative_count=hard_negative_count,
    )

    question_payload, question_line_hashes = jsonl_bytes(question_rows)
    document_payload, document_line_hashes = jsonl_bytes(document_rows)
    report_text = build_selection_report(
        data,
        question_rows,
        document_rows,
        question_selection,
        hard_negatives,
        distributions,
        pilot_overlap,
        name=name,
        seed=seed,
        questions_per_type=questions_per_type,
        hard_negative_count=hard_negative_count,
    )
    report_payload = report_text.encode("utf-8")
    if not report_payload.endswith(b"\n"):
        report_payload += b"\n"
    payloads_without_manifest = {
        "questions.jsonl": question_payload,
        "documents.jsonl": document_payload,
        "selection_report.md": report_payload,
    }
    manifest["output_files"] = {
        "questions.jsonl": {
            "sha256": sha256_bytes(question_payload),
            "records": len(question_rows),
            "record_line_sha256": question_line_hashes,
        },
        "documents.jsonl": {
            "sha256": sha256_bytes(document_payload),
            "records": len(document_rows),
            "record_line_sha256": document_line_hashes,
        },
        "selection_report.md": {
            "sha256": sha256_bytes(report_payload),
            "bytes": len(report_payload),
        },
    }
    hard_details = manifest["hard_negative_links"]
    checks = evaluate_invariants(
        data,
        question_rows,
        document_rows,
        mapping,
        hard_details,
        questions_per_type=questions_per_type,
        hard_negative_count=hard_negative_count,
        manifest=manifest,
        bm25=bm25,
        output_payloads=payloads_without_manifest,
        expected_output_hashes=manifest["output_files"],
    )
    manifest["invariants"] = {
        row["name"]: {"passed": row["passed"], "detail": row["detail"]}
        for row in checks
    }
    failures = [row for row in checks if not row["passed"]]
    require(not failures, f"generated subset failed invariants: {[row['name'] for row in failures]}")

    manifest_payload = canonical_json_bytes(manifest, indent=2)
    audited_payloads = {
        **payloads_without_manifest,
        "manifest.json": manifest_payload,
    }
    audit = audit_payload(checks, manifest, audited_payloads)
    audit_bytes = canonical_json_bytes(audit, indent=2)
    artifacts = {
        "questions.jsonl": question_payload,
        "documents.jsonl": document_payload,
        "manifest.json": manifest_payload,
        "selection_report.md": report_payload,
        "audit_report.json": audit_bytes,
    }
    summary = {
        "subset_id": manifest["subset_id"],
        "questions": len(question_rows),
        "gold_documents": manifest["counts"]["gold_documents"],
        "hard_negative_documents": manifest["counts"]["hard_negative_documents"],
        "documents_total": len(document_rows),
        "pilot_question_overlap": pilot_overlap["questions"]["count"],
        "pilot_document_overlap": pilot_overlap["documents"]["count"],
        "checks_passed": len(checks),
        "checks_failed": 0,
    }
    return artifacts, summary


def write_immutable_artifacts(output_dir: Path, artifacts: dict[str, bytes]) -> bool:
    required_names = set(artifacts)
    existing_required = {path.name for path in output_dir.glob("*") if path.name in required_names}
    if existing_required:
        if existing_required != required_names:
            missing = sorted(required_names - existing_required)
            raise BuildError(
                f"immutable subset directory is partial; refusing to overwrite (missing {missing})"
            )
        mismatched = [
            name for name, payload in artifacts.items() if (output_dir / name).read_bytes() != payload
        ]
        if mismatched:
            raise BuildError(
                "immutable subset already exists with different bytes: " + ", ".join(mismatched)
            )
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        (output_dir / name).write_bytes(payload)
    return True


def build_subset(
    *,
    name: str,
    questions_per_type: int,
    seed: int,
    hard_negative_count: int,
    output_root: Path | None = None,
) -> dict[str, Any]:
    artifacts, summary = build_artifacts(
        name=name,
        questions_per_type=questions_per_type,
        seed=seed,
        hard_negative_count=hard_negative_count,
    )
    root = output_root if output_root is not None else EXPERIMENT_DIR
    output_dir = root / name
    created = write_immutable_artifacts(output_dir, artifacts)
    return {**summary, "output_dir": str(output_dir), "created": created, "unchanged": not created}


def audit_existing_subset(output_dir: Path) -> tuple[dict[str, Any], bool]:
    required = {
        "questions.jsonl",
        "documents.jsonl",
        "manifest.json",
        "selection_report.md",
        "audit_report.json",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    require(not missing, f"subset is missing required artifacts: {missing}")
    manifest = read_json(output_dir / "manifest.json")
    question_rows = read_jsonl(output_dir / "questions.jsonl")
    document_rows = read_jsonl(output_dir / "documents.jsonl")
    data = load_and_validate_sources()
    bm25 = BM25Index(data.corpus)
    payloads_without_audit = {
        name: (output_dir / name).read_bytes()
        for name in ("questions.jsonl", "documents.jsonl", "selection_report.md", "manifest.json")
    }
    checks = evaluate_invariants(
        data,
        question_rows,
        document_rows,
        manifest["question_document_mapping"],
        manifest["hard_negative_links"],
        questions_per_type=manifest["parameters"]["questions_per_type"],
        hard_negative_count=manifest["parameters"]["hard_negatives"],
        manifest=manifest,
        bm25=bm25,
        output_payloads={
            name: payloads_without_audit[name]
            for name in ("questions.jsonl", "documents.jsonl", "selection_report.md")
        },
        expected_output_hashes=manifest["output_files"],
    )
    recomputed = audit_payload(checks, manifest, payloads_without_audit)
    stored = read_json(output_dir / "audit_report.json")
    byte_identical = canonical_json_bytes(recomputed, indent=2) == (output_dir / "audit_report.json").read_bytes()
    recomputed["stored_audit_report_byte_identical"] = byte_identical
    recomputed["stored_audit_status"] = stored.get("status")
    return recomputed, byte_identical and recomputed["status"] == "passed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="multihoprag_120")
    parser.add_argument("--questions-per-type", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negatives", type=int, default=30)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Parent directory for named subsets (primarily for isolated tests).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit the existing named subset without writing any file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root if args.output_root is not None else EXPERIMENT_DIR
    output_dir = output_root / args.name
    try:
        if args.audit_only:
            audit, passed = audit_existing_subset(output_dir)
            print(json.dumps(audit, ensure_ascii=False, indent=2))
            return 0 if passed else 1
        summary = build_subset(
            name=args.name,
            questions_per_type=args.questions_per_type,
            seed=args.seed,
            hard_negative_count=args.hard_negatives,
            output_root=args.output_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"build_subset.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
