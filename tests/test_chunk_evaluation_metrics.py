from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.lexical_evaluation import (
    evaluate_nuggets_lexical_at_cutoffs,
)
from evaluation.run_chunk_evaluation import (
    DEFAULT_BENCHMARK,
    build_parser,
    _gold_nuggets,
    _load_benchmark,
    _mean_nugget_scores_at_cutoffs,
    _summary,
    main,
)
from evaluation.semantic_evaluation import SemanticEvaluator


class _FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.calls += 1
        vectors = {
            "nugget one": [1.0, 0.0],
            "nugget two": [0.0, 1.0],
            "first chunk": [1.0, 0.0],
            "partial chunk": [0.6, 0.8],
            "second chunk": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


class _FakeNuggetEvaluator:
    def score_nuggets_at_cutoffs(
        self,
        nugget_texts: list[str],
        evidence_texts: list[str],
        *,
        cutoffs: tuple[int, ...],
    ) -> list[dict[int, tuple[float, str]]]:
        del nugget_texts
        return [
            {
                cutoff: (1.0, evidence_texts[0])
                for cutoff in cutoffs
            },
            {
                cutoff: (
                    (0.0, evidence_texts[0])
                    if cutoff == 1
                    else (1.0, evidence_texts[1])
                )
                for cutoff in cutoffs
            },
        ]


@dataclass
class _FakeRecord:
    evidence_text: str


@dataclass
class _FakePack:
    records: list[_FakeRecord]
    pipeline: dict[str, str]
    retrieved_papers: list[dict[str, str]]


class ChunkEvaluationMetricTests(unittest.TestCase):
    def test_runner_accepts_each_supported_reranker(self) -> None:
        parser = build_parser()

        for reranker in ("medcpt", "lexical", "hybrid"):
            with self.subTest(reranker=reranker):
                arguments = parser.parse_args(["--reranker", reranker])
                self.assertEqual(arguments.reranker, reranker)

    def test_default_semantic_loader_uses_transformers_backend(self) -> None:
        sentinel = object()
        with patch(
            "evaluation.semantic_evaluation._TransformersCLSEncoder",
            return_value=sentinel,
        ) as encoder:
            loaded = SemanticEvaluator._load_model("BAAI/bge-m3", "auto")

        self.assertIs(loaded, sentinel)
        encoder.assert_called_once_with("BAAI/bge-m3", "auto")

    def test_default_benchmark_contains_all_reviewed_nuggets(self) -> None:
        examples, metadata = _load_benchmark(DEFAULT_BENCHMARK)

        self.assertEqual(len(examples), 90)
        self.assertEqual(
            sum(len(_gold_nuggets(example)) for example in examples),
            172,
        )
        self.assertEqual(metadata["benchmark_version"], "2.1.0")

    def test_gold_nuggets_require_unique_ids_and_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate nugget id"):
            _gold_nuggets(
                {
                    "id": "QTEST",
                    "nuggets": [
                        {"id": "N1", "text": "First fact."},
                        {"id": "N1", "text": "Second fact."},
                    ],
                }
            )

    def test_lexical_nugget_scores_ignore_unrelated_extra_words(self) -> None:
        results = evaluate_nuggets_lexical_at_cutoffs(
            ["alpha beta", "gamma delta"],
            [
                "unrelated words before alpha beta and after it",
                "nothing useful here",
                "gamma delta with additional context",
            ],
            cutoffs=(1, 2, 3, 5),
        )

        self.assertEqual(results[0][1][0], 1.0)
        self.assertEqual(results[0][1][1], results[0][5][1])
        self.assertEqual(results[1][1][0], 0.0)
        self.assertEqual(results[1][2][0], 0.0)
        self.assertEqual(results[1][3][0], 1.0)
        self.assertEqual(results[1][5][0], 1.0)

    def test_semantic_scores_each_nugget_with_one_embedding_call(self) -> None:
        model = _FakeEmbeddingModel()
        evaluator = SemanticEvaluator(model=model)

        results = evaluator.score_nuggets_at_cutoffs(
            ["nugget one", "nugget two"],
            ["first chunk", "partial chunk", "second chunk"],
            cutoffs=(1, 2, 3, 5),
        )

        self.assertEqual(model.calls, 1)
        self.assertAlmostEqual(results[0][1][0], 1.0)
        self.assertAlmostEqual(results[1][1][0], 0.0)
        self.assertAlmostEqual(results[1][2][0], 0.8)
        self.assertAlmostEqual(results[1][3][0], 1.0)
        self.assertAlmostEqual(results[1][5][0], 1.0)
        self.assertEqual(results[1][3][1], "second chunk")

    def test_question_score_is_mean_of_per_nugget_best_scores(self) -> None:
        nugget_results = [
            {
                "lexical_score_at_k": {
                    str(k): 1.0 for k in (1, 3, 5, 10, 20)
                }
            },
            {
                "lexical_score_at_k": {
                    str(k): 0.0 for k in (1, 3, 5, 10, 20)
                }
            },
        ]

        aggregated = _mean_nugget_scores_at_cutoffs(
            nugget_results,
            "lexical_score_at_k",
        )

        self.assertEqual(aggregated["20"], 0.5)

    def test_summary_macro_averages_questions(self) -> None:
        first = {str(k): float(k) / 20 for k in (1, 3, 5, 10, 20)}
        second = {str(k): 0.0 for k in (1, 3, 5, 10, 20)}
        rows = [
            {
                "status": "success",
                "returned_chunks": 20,
                "nugget_count": 2,
                "lexical_nugget_score_at_k": first,
                "semantic_nugget_score_at_k": first,
            },
            {
                "status": "success",
                "returned_chunks": 10,
                "nugget_count": 1,
                "lexical_nugget_score_at_k": second,
                "semantic_nugget_score_at_k": second,
            },
            {"status": "error", "returned_chunks": 0},
        ]

        summary = _summary(rows)

        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["total_gold_nuggets"], 3)
        self.assertEqual(summary["questions_returning_20_chunks"], 1)
        self.assertAlmostEqual(
            summary["mean_lexical_nugget_score_at_k"]["10"],
            0.25,
        )
        self.assertAlmostEqual(
            summary["mean_semantic_nugget_score"],
            0.5,
        )

    def test_runner_writes_per_nugget_results(self) -> None:
        benchmark = {
            "schema_version": "test",
            "version": "test",
            "examples": [
                {
                    "id": "QTEST",
                    "question": "Test question?",
                    "category": "Test",
                    "nuggets": [
                        {"id": "QTEST_N01", "text": "alpha beta"},
                        {"id": "QTEST_N02", "text": "gamma delta"},
                    ],
                }
            ],
        }
        pack = _FakePack(
            records=[
                _FakeRecord("alpha beta with extra words"),
                _FakeRecord("gamma delta with extra words"),
            ],
            pipeline={"name": "test"},
            retrieved_papers=[],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_path = root / "benchmark.json"
            output_path = root / "results.json"
            benchmark_path.write_text(
                json.dumps(benchmark),
                encoding="utf-8",
            )
            arguments = [
                "run_chunk_evaluation",
                "--benchmark",
                str(benchmark_path),
                "--output",
                str(output_path),
                "--num-questions",
                "1",
                "--reranker",
                "lexical",
                "--no-resume",
            ]
            with (
                patch(
                    "evaluation.run_chunk_evaluation.run_pipeline",
                    return_value=pack,
                ) as run_pipeline,
                patch(
                    "evaluation.run_chunk_evaluation.SemanticEvaluator",
                    return_value=_FakeNuggetEvaluator(),
                ),
                patch.object(sys, "argv", arguments),
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            summary = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )

        row = payload["results"][0]
        self.assertEqual(
            payload["configuration"]["evaluation_unit"],
            "gold_nugget",
        )
        self.assertEqual(
            payload["configuration"]["semantic_backend"],
            "transformers_cls_pooling",
        )
        self.assertEqual(
            payload["configuration"]["semantic_max_length"],
            8192,
        )
        self.assertEqual(payload["configuration"]["reranker"], "lexical")
        self.assertEqual(
            run_pipeline.call_args.kwargs["settings"].reranker,
            "lexical",
        )
        self.assertEqual(row["nugget_count"], 2)
        self.assertEqual(len(row["nugget_results"]), 2)
        self.assertEqual(row["lexical_nugget_score_at_k"]["1"], 0.5)
        self.assertEqual(row["lexical_nugget_score"], 1.0)
        self.assertEqual(
            row["nugget_results"][1]["semantic_best_chunk_rank"],
            2,
        )
        self.assertEqual(summary["total_gold_nuggets"], 2)


if __name__ == "__main__":
    unittest.main()
