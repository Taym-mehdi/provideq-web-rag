from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

from web_rag import run_pipeline
from web_rag.config import RERANKERS, Settings
from web_rag.serializer import to_serializable

from .lexical_evaluation import evaluate_nuggets_lexical_at_cutoffs
from .semantic_evaluation import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    SemanticEvaluator,
)


DEFAULT_BENCHMARK = Path(
    "benchmark/provideq_benchmark_nuggets_v2.1.0.json"
)
DEFAULT_OUTPUT = Path("outputs/nugget_chunk_evaluation/results.json")
EVALUATION_CUTOFFS = (1, 3, 5, 10, 20)


def _load_benchmark(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(examples, list):
        raise ValueError("Benchmark must contain an 'examples' list")
    metadata = {
        "benchmark_file": path.name,
        "benchmark_version": (
            payload.get("version") if isinstance(payload, dict) else None
        ),
        "benchmark_schema_version": (
            payload.get("schema_version")
            if isinstance(payload, dict)
            else None
        ),
        "benchmark_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return (
        [item for item in examples if isinstance(item, dict)],
        metadata,
    )


def _select_examples(
    examples: list[dict[str, Any]],
    *,
    question_ids: list[str],
    number: int,
    seed: int,
) -> list[dict[str, Any]]:
    if question_ids:
        wanted = {value.strip().casefold() for value in question_ids}
        selected = [
            item
            for item in examples
            if str(item.get("id", "")).casefold() in wanted
        ]
        found = {str(item.get("id", "")).casefold() for item in selected}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(
                f"Unknown benchmark question id(s): {', '.join(missing)}"
            )
        return selected

    if not 1 <= number <= len(examples):
        raise ValueError(
            f"num_questions must be between 1 and {len(examples)}"
        )
    if number == len(examples):
        selected = list(examples)
    else:
        selected = random.Random(seed).sample(examples, number)
    return sorted(selected, key=lambda item: str(item.get("id", "")))


def _gold_nuggets(example: dict[str, Any]) -> list[dict[str, str]]:
    question_id = str(example.get("id", "<unknown>"))
    raw_nuggets = example.get("nuggets")
    if not isinstance(raw_nuggets, list) or not raw_nuggets:
        raise ValueError(f"{question_id} has no gold nuggets")

    nuggets: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for raw_nugget in raw_nuggets:
        if not isinstance(raw_nugget, dict):
            raise ValueError(f"{question_id} contains an invalid nugget")
        nugget_id = str(raw_nugget.get("id", "")).strip()
        text = str(raw_nugget.get("text", "")).strip()
        if not nugget_id or not text:
            raise ValueError(
                f"{question_id} contains a nugget without an id or text"
            )
        if nugget_id in seen_ids:
            raise ValueError(
                f"{question_id} contains duplicate nugget id {nugget_id}"
            )
        seen_ids.add(nugget_id)
        nuggets.append({"id": nugget_id, "text": text})
    return nuggets


def _best_rank(evidence_texts: list[str], best_text: str) -> int | None:
    try:
        return evidence_texts.index(best_text) + 1
    except ValueError:
        return None


def _mean_nugget_scores_at_cutoffs(
    nugget_results: list[dict[str, Any]],
    field: str,
) -> dict[str, float | None]:
    aggregated: dict[str, float | None] = {}
    for cutoff in EVALUATION_CUTOFFS:
        values = [
            float(result[field][str(cutoff)])
            for result in nugget_results
            if result.get(field, {}).get(str(cutoff)) is not None
        ]
        aggregated[str(cutoff)] = mean(values) if values else None
    return aggregated


def _write_results(
    path: Path,
    *,
    configuration: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "configuration": configuration,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_existing(
    path: Path,
    *,
    configuration: dict[str, Any],
    resume: bool,
) -> dict[str, dict[str, Any]]:
    if not resume or not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("configuration") != configuration:
        raise ValueError(
            f"{path} uses different settings. "
            "Use --no-resume or another --output path."
        )
    return {
        str(row.get("question_id", "")): row
        for row in payload.get("results", [])
        if isinstance(row, dict) and row.get("question_id")
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]

    def means_at_k(field: str) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for cutoff in EVALUATION_CUTOFFS:
            values = [
                float(row[field][str(cutoff)])
                for row in successful
                if row.get(field, {}).get(str(cutoff)) is not None
            ]
            result[str(cutoff)] = mean(values) if values else None
        return result

    lexical_at_k = means_at_k("lexical_nugget_score_at_k")
    semantic_at_k = means_at_k("semantic_nugget_score_at_k")
    return {
        "questions": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "total_gold_nuggets": sum(
            int(row.get("nugget_count", 0)) for row in successful
        ),
        "mean_lexical_nugget_score_at_k": lexical_at_k,
        "mean_semantic_nugget_score_at_k": semantic_at_k,
        "mean_lexical_nugget_score": lexical_at_k["20"],
        "mean_semantic_nugget_score": semantic_at_k["20"],
        "questions_returning_20_chunks": sum(
            int(row.get("returned_chunks", 0)) == 20
            for row in successful
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate gold nuggets against the default final 20 Web RAG "
            "chunks with lexical and semantic matching."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Evaluate a specific ID, for example Q039. Repeat as needed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--reranker",
        choices=RERANKERS,
        default=Settings().reranker,
        help=(
            "Final chunk reranker to evaluate. All retrieval, chunking, "
            "evidence-selection, and evaluation settings remain fixed."
        ),
    )
    parser.add_argument("--semantic-model", default=DEFAULT_MODEL)
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    parser.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retry-errors", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    benchmark_examples, benchmark_metadata = _load_benchmark(args.benchmark)
    examples = _select_examples(
        benchmark_examples,
        question_ids=args.question_id,
        number=args.num_questions,
        seed=args.seed,
    )
    for example in examples:
        _gold_nuggets(example)

    pipeline_settings = Settings(
        medcpt_device=args.device,
        reranker=args.reranker,
    )
    configuration = {
        "evaluation_schema_version": 4,
        "evaluation_cutoffs": list(EVALUATION_CUTOFFS),
        "evaluation_unit": "gold_nugget",
        "nugget_aggregation": "mean_of_per_nugget_best_chunk_scores",
        "lexical_metric": "mean_rouge_1_recall_and_rouge_l_recall",
        "semantic_metric": "max_cosine_similarity_per_nugget",
        **benchmark_metadata,
        "pipeline": "default",
        "retrieval_system": pipeline_settings.retrieval_system,
        "query_strategy": pipeline_settings.query_strategy,
        "paperclip_ranking": pipeline_settings.paperclip_ranking,
        "paperclip_candidate_limit": (
            pipeline_settings.paperclip_candidate_limit
        ),
        "paperclip_query_fusion": (
            pipeline_settings.paperclip_query_fusion
        ),
        "query_fusion_rrf_k": pipeline_settings.query_fusion_rrf_k,
        "reformulated_query_weight": (
            pipeline_settings.reformulated_query_weight
        ),
        "europepmc_mode": pipeline_settings.europepmc_mode,
        "europepmc_synonym": pipeline_settings.europepmc_synonym,
        "europepmc_candidate_limit": (
            pipeline_settings.europepmc_candidate_limit
        ),
        "fusion_rrf_k": pipeline_settings.fusion_rrf_k,
        "retrieval_limit": pipeline_settings.retrieval_limit,
        "chunking": pipeline_settings.chunking_method,
        "chunk_max_tokens": pipeline_settings.chunk_max_tokens,
        "chunk_overlap_fraction": (
            pipeline_settings.chunk_overlap_fraction
        ),
        "chunk_max_overlap_sentences": (
            pipeline_settings.chunk_max_overlap_sentences
        ),
        "min_chunk_words": pipeline_settings.min_chunk_words,
        "reranker": pipeline_settings.reranker,
        "reranker_model": pipeline_settings.medcpt_model,
        "bm25_k1": pipeline_settings.bm25_k1,
        "bm25_b": pipeline_settings.bm25_b,
        "hybrid_lexical_weight": (
            pipeline_settings.hybrid_lexical_weight
        ),
        "hybrid_medcpt_weight": (
            pipeline_settings.hybrid_medcpt_weight
        ),
        "top_k": pipeline_settings.top_k,
        "max_chunks_per_paper": (
            pipeline_settings.max_chunks_per_paper
        ),
        "near_duplicate_threshold": (
            pipeline_settings.near_duplicate_threshold
        ),
        "semantic_enabled": bool(args.semantic),
        "semantic_model": args.semantic_model if args.semantic else None,
        "semantic_backend": (
            "transformers_cls_pooling" if args.semantic else None
        ),
        "semantic_max_length": (
            DEFAULT_MAX_LENGTH if args.semantic else None
        ),
        "device": args.device,
    }
    rows_by_id = _load_existing(
        args.output,
        configuration=configuration,
        resume=args.resume,
    )

    pending = [
        example
        for example in examples
        if (
            str(example.get("id", "")) not in rows_by_id
            or (
                args.retry_errors
                and rows_by_id[str(example.get("id", ""))].get("status")
                == "error"
            )
        )
    ]
    semantic_evaluator = (
        SemanticEvaluator(
            args.semantic_model,
            device=args.device,
            batch_size=args.semantic_batch_size,
        )
        if args.semantic and pending
        else None
    )

    for position, example in enumerate(examples, start=1):
        question_id = str(example.get("id", ""))
        existing = rows_by_id.get(question_id)
        if (
            existing is not None
            and not (
                args.retry_errors
                and existing.get("status") == "error"
            )
        ):
            print(f"[{position}/{len(examples)}] {question_id} | skipped")
            continue

        print(f"[{position}/{len(examples)}] {question_id}")
        gold_nuggets = _gold_nuggets(example)
        nugget_texts = [nugget["text"] for nugget in gold_nuggets]
        try:
            pack = run_pipeline(
                str(example["question"]),
                settings=pipeline_settings,
            )
            evidence_texts = [
                record.evidence_text for record in pack.records
            ]
            lexical_results = evaluate_nuggets_lexical_at_cutoffs(
                nugget_texts,
                evidence_texts,
                cutoffs=EVALUATION_CUTOFFS,
            )
            if semantic_evaluator is not None:
                semantic_results = (
                    semantic_evaluator.score_nuggets_at_cutoffs(
                        nugget_texts,
                        evidence_texts,
                        cutoffs=EVALUATION_CUTOFFS,
                    )
                )
            else:
                semantic_results = [
                    {
                        cutoff: (None, "")
                        for cutoff in EVALUATION_CUTOFFS
                    }
                    for _ in gold_nuggets
                ]

            nugget_results: list[dict[str, Any]] = []
            for nugget, lexical, semantic in zip(
                gold_nuggets,
                lexical_results,
                semantic_results,
                strict=True,
            ):
                lexical_score, lexical_text = lexical[20]
                semantic_score, semantic_text = semantic[20]
                nugget_results.append(
                    {
                        "nugget_id": nugget["id"],
                        "nugget_text": nugget["text"],
                        "lexical_score_at_k": {
                            str(cutoff): lexical[cutoff][0]
                            for cutoff in EVALUATION_CUTOFFS
                        },
                        "lexical_score": lexical_score,
                        "lexical_best_chunk_rank": _best_rank(
                            evidence_texts,
                            lexical_text,
                        ),
                        "semantic_score_at_k": {
                            str(cutoff): semantic[cutoff][0]
                            for cutoff in EVALUATION_CUTOFFS
                        },
                        "semantic_score": semantic_score,
                        "semantic_best_chunk_rank": _best_rank(
                            evidence_texts,
                            semantic_text,
                        ),
                    }
                )

            lexical_at_k = _mean_nugget_scores_at_cutoffs(
                nugget_results,
                "lexical_score_at_k",
            )
            semantic_at_k = _mean_nugget_scores_at_cutoffs(
                nugget_results,
                "semantic_score_at_k",
            )

            row = {
                "question_id": question_id,
                "category": example.get("category", ""),
                "question": example.get("question", ""),
                "status": "success",
                "error": "",
                "returned_chunks": len(pack.records),
                "nugget_count": len(gold_nuggets),
                "lexical_nugget_score_at_k": lexical_at_k,
                "lexical_nugget_score": lexical_at_k["20"],
                "semantic_nugget_score_at_k": semantic_at_k,
                "semantic_nugget_score": semantic_at_k["20"],
                "nugget_results": nugget_results,
                "pipeline": to_serializable(pack.pipeline),
                "retrieved_papers": to_serializable(
                    pack.retrieved_papers
                ),
                "chunks": to_serializable(pack.records),
            }
        except Exception as exc:
            row = {
                "question_id": question_id,
                "category": example.get("category", ""),
                "question": example.get("question", ""),
                "status": "error",
                "error": str(exc),
                "returned_chunks": 0,
                "nugget_count": len(gold_nuggets),
                "lexical_nugget_score_at_k": {},
                "lexical_nugget_score": None,
                "semantic_nugget_score_at_k": {},
                "semantic_nugget_score": None,
                "nugget_results": [],
                "chunks": [],
            }
            print(f"  ERROR: {exc}")

        rows_by_id[question_id] = row
        ordered = [
            rows_by_id[str(item.get("id", ""))]
            for item in examples
            if str(item.get("id", "")) in rows_by_id
        ]
        _write_results(
            args.output,
            configuration=configuration,
            rows=ordered,
        )

    rows = [
        rows_by_id[str(item.get("id", ""))]
        for item in examples
        if str(item.get("id", "")) in rows_by_id
    ]
    summary = _summary(rows)
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output}")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
