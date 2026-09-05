from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SUBSET_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = SUBSET_DIR.parent
SCRIPTS_DIR = SUBSET_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_subset as builder  # noqa: E402


REQUIRED_FILES = (
    "questions.jsonl",
    "documents.jsonl",
    "manifest.json",
    "selection_report.md",
    "audit_report.json",
)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeneratedSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions = load_jsonl(SUBSET_DIR / "questions.jsonl")
        cls.documents = load_jsonl(SUBSET_DIR / "documents.jsonl")
        cls.manifest = json.loads((SUBSET_DIR / "manifest.json").read_text(encoding="utf-8"))
        cls.audit = json.loads((SUBSET_DIR / "audit_report.json").read_text(encoding="utf-8"))
        cls.source_questions = json.loads(builder.QUESTIONS_SOURCE.read_text(encoding="utf-8"))
        cls.source_corpus = json.loads(builder.CORPUS_SOURCE.read_text(encoding="utf-8"))
        cls.source_corpus_by_url = {row["url"]: row for row in cls.source_corpus}
        cls.documents_by_id = {row["document_id"]: row for row in cls.documents}

    def test_required_artifacts_exist(self) -> None:
        self.assertEqual(
            {path.name for path in SUBSET_DIR.iterdir() if path.is_file()},
            set(REQUIRED_FILES),
        )

    def test_exact_question_count_type_balance_and_evidence_strata(self) -> None:
        self.assertEqual(len(self.questions), 120)
        self.assertEqual(
            Counter(row["source_question_type"] for row in self.questions),
            {
                "inference_query": 30,
                "comparison_query": 30,
                "temporal_query": 30,
                "null_query": 30,
            },
        )
        self.assertEqual(
            Counter(
                (row["source_question_type"], len(row["gold_document_ids"]))
                for row in self.questions
            ),
            {
                ("inference_query", 2): 10,
                ("inference_query", 3): 10,
                ("inference_query", 4): 10,
                ("comparison_query", 2): 15,
                ("comparison_query", 3): 15,
                ("temporal_query", 2): 15,
                ("temporal_query", 3): 15,
                ("null_query", 0): 30,
            },
        )
        for row in self.questions:
            if row["source_question_type"] == "null_query":
                self.assertFalse(row["answerable"])
                self.assertEqual(row["gold_urls"], [])
                self.assertEqual(row["source_fields"]["evidence_list"], [])
            else:
                self.assertTrue(row["answerable"])
                self.assertEqual(len(row["gold_urls"]), len(set(row["gold_urls"])))
                self.assertIn(len(row["gold_urls"]), {2, 3, 4})

    def test_pilot_questions_are_excluded_and_ids_match_pilot_method(self) -> None:
        pilot_questions = load_jsonl(builder.PILOT_QUESTIONS)
        pilot_ids = {row["question_id"] for row in pilot_questions}
        selected_ids = {row["question_id"] for row in self.questions}
        self.assertFalse(selected_ids & pilot_ids)
        self.assertEqual(len(selected_ids), 120)
        for row in self.questions:
            self.assertEqual(row["question_id"], builder.question_id(row["question"]))
        for row in self.documents:
            self.assertEqual(row["document_id"], builder.document_id(row["url"]))

    def test_source_text_metadata_nulls_and_record_hashes_are_preserved(self) -> None:
        for row in self.questions:
            source = self.source_questions[row["source_index"]]
            self.assertEqual(row["question"], source["query"])
            self.assertEqual(row["gold_answer"], source["answer"])
            self.assertEqual(row["source_fields"], source)
            self.assertEqual(row["question_sha256"], builder.sha256_text(source["query"]))
            self.assertEqual(
                row["source_record_sha256"], builder.canonical_record_hash(source)
            )
        for row in self.documents:
            source = self.source_corpus[row["source_index"]]
            self.assertEqual(row["text"], source["body"])
            self.assertEqual(row["url"], source["url"])
            self.assertEqual(row["source_fields"]["author"], source["author"])
            self.assertEqual(row["text_sha256"], builder.sha256_text(source["body"]))
            self.assertEqual(
                row["source_record_sha256"], builder.canonical_record_hash(source)
            )

    def test_evidence_url_resolution_fact_integrity_and_gold_union(self) -> None:
        source_url_counts = Counter(row["url"] for row in self.source_corpus)
        expected_gold_ids: set[str] = set()
        for row in self.questions:
            source = self.source_questions[row["source_index"]]
            for evidence in source["evidence_list"]:
                self.assertEqual(source_url_counts[evidence["url"]], 1)
                self.assertIn(evidence["fact"], self.source_corpus_by_url[evidence["url"]]["body"])
                expected_gold_ids.add(builder.document_id(evidence["url"]))
        actual_gold_ids = {
            row["document_id"] for row in self.documents if row["role"] == "gold"
        }
        self.assertEqual(actual_gold_ids, expected_gold_ids)
        self.assertEqual(len(actual_gold_ids), 125)

    def test_mapping_is_complete_and_hard_negatives_pass_model_free_checks(self) -> None:
        mapping = self.manifest["question_document_mapping"]
        self.assertEqual(set(mapping), {row["question_id"] for row in self.questions})
        hard_ids = {
            row["document_id"] for row in self.documents if row["role"] == "hard_negative"
        }
        gold_ids = {
            row["document_id"] for row in self.documents if row["role"] == "gold"
        }
        self.assertEqual(len(hard_ids), 30)
        self.assertFalse(hard_ids & gold_ids)
        for question_row in self.questions:
            qid = question_row["question_id"]
            source_question = self.source_questions[question_row["source_index"]]
            entry = mapping[qid]
            self.assertEqual(entry["gold_document_ids"], question_row["gold_document_ids"])
            self.assertEqual(
                entry["hard_negative_document_ids"],
                question_row["hard_negative_document_ids"],
            )
            self.assertEqual(entry["distractor_document_ids"], entry["hard_negative_document_ids"])
            self.assertTrue(entry["hard_negative_document_ids"])
            question_gold_indices = {
                self.documents_by_id[did]["source_index"]
                for did in question_row["gold_document_ids"]
            }
            for did in entry["hard_negative_document_ids"]:
                self.assertIn(did, hard_ids)
                document_row = self.documents_by_id[did]
                source_document = self.source_corpus[document_row["source_index"]]
                checks = builder.hard_negative_pair_checks(
                    source_question,
                    source_document,
                    question_gold_indices,
                    document_row["source_index"],
                )
                self.assertTrue(all(checks.values()))
                detail = self.manifest["hard_negative_links"][did]["per_question"][qid]
                self.assertGreater(detail["bm25_score"], 0.0)
                self.assertGreaterEqual(detail["eligible_bm25_rank"], 1)
                self.assertTrue(all(detail[name] for name in checks))

    def test_manifest_file_line_content_and_source_hashes_repeat(self) -> None:
        for filename in ("questions.jsonl", "documents.jsonl", "selection_report.md"):
            self.assertEqual(
                file_sha256(SUBSET_DIR / filename),
                self.manifest["output_files"][filename]["sha256"],
            )
        for filename, id_field in (
            ("questions.jsonl", "question_id"),
            ("documents.jsonl", "document_id"),
        ):
            expected = self.manifest["output_files"][filename]["record_line_sha256"]
            actual: dict[str, str] = {}
            for raw_line in (SUBSET_DIR / filename).read_bytes().splitlines():
                row = json.loads(raw_line)
                actual[row[id_field]] = hashlib.sha256(raw_line).hexdigest()
            self.assertEqual(actual, expected)
        self.assertEqual(
            file_sha256(builder.QUESTIONS_SOURCE),
            self.manifest["source_files"]["questions"]["sha256"],
        )
        self.assertEqual(
            file_sha256(builder.CORPUS_SOURCE),
            self.manifest["source_files"]["corpus"]["sha256"],
        )

    def test_stored_model_free_audit_passes_every_check(self) -> None:
        self.assertEqual(self.audit["status"], "passed")
        self.assertEqual(self.audit["summary"]["checks_failed"], 0)
        self.assertTrue(all(row["passed"] for row in self.audit["checks"]))
        self.assertTrue(
            all(value["passed"] for value in self.manifest["invariants"].values())
        )
        recomputed, passed = builder.audit_existing_subset(SUBSET_DIR)
        self.assertTrue(passed)
        self.assertTrue(recomputed["stored_audit_report_byte_identical"])
        self.assertEqual(recomputed["summary"]["checks_failed"], 0)


class DeterminismTests(unittest.TestCase):
    def test_two_independent_builds_are_byte_identical_and_existing_build_is_immutable(self) -> None:
        first, first_summary = builder.build_artifacts(
            name="multihoprag_120",
            questions_per_type=30,
            seed=42,
            hard_negative_count=30,
        )
        second, second_summary = builder.build_artifacts(
            name="multihoprag_120",
            questions_per_type=30,
            seed=42,
            hard_negative_count=30,
        )
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first, second)
        for filename in REQUIRED_FILES:
            self.assertEqual(first[filename], (SUBSET_DIR / filename).read_bytes())

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            created = builder.build_subset(
                name="multihoprag_120",
                questions_per_type=30,
                seed=42,
                hard_negative_count=30,
                output_root=output_root,
            )
            before = {
                filename: (output_root / "multihoprag_120" / filename).read_bytes()
                for filename in REQUIRED_FILES
            }
            unchanged = builder.build_subset(
                name="multihoprag_120",
                questions_per_type=30,
                seed=42,
                hard_negative_count=30,
                output_root=output_root,
            )
            after = {
                filename: (output_root / "multihoprag_120" / filename).read_bytes()
                for filename in REQUIRED_FILES
            }
        self.assertTrue(created["created"])
        self.assertTrue(unchanged["unchanged"])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
