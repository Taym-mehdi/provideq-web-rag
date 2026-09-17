from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

from web_rag import run_pipeline
from web_rag.config import Settings
from web_rag.serializer import to_serializable

from .lexical_evaluation import evaluate_lexical
from .semantic_evaluation import DEFAULT_MODEL, SemanticEvaluator


DEFAULT_BENCHMARK = Path("benchmark/provideq_benchmark.json")
DEFAULT_OUTPUT = Path("outputs/default_chunk_evaluation/results.json")


def _load_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(examples, list):
        raise ValueError("Benchmark must contain an 'examples' list")
    return [item for item in examples if isinstance(item, dict)]


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


def _gold_answers(example: dict[str, Any]) -> list[str]:
    answers = [
        str(value).strip()
        for value in example.get("gold_answers", [])
        if str(value).strip()
    ]
    for document in example.get("gold_documents", []):
        if not isinstance(document, dict):
            continue
        answers.extend(
            str(value).strip()
            for value in document.get("gold_answers", [])
            if str(value).strip()
        )
    return list(dict.fromkeys(answers))


def _best_rank(evidence_texts: list[str], best_text: str) -> int | None:
    try:
        return evidence_texts.index(best_text) + 1
    except ValueError:
        return None


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
    lexical = [
        float(row["lexical_score"])
        for row in successful
        if row.get("lexical_score") is not None
    ]
    semantic = [
        float(row["semantic_score"])
        for row in successful
        if row.get("semantic_score") is not None
    ]
    return {
        "questions": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "mean_lexical_score": mean(lexical) if lexical else None,
        "mean_semantic_score": mean(semantic) if semantic else None,
        "questions_returning_20_chunks": sum(
            int(row.get("returned_chunks", 0)) == 20
            for row in successful
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the default final 20 Web RAG chunks with lexical "
            "and semantic matching."
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
    examples = _select_examples(
        _load_examples(args.benchmark),
        question_ids=args.question_id,
        number=args.num_questions,
        seed=args.seed,
    )
    pipeline_settings = Settings(medcpt_device=args.device)
    configuration = {
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
        "reranker": pipeline_settings.medcpt_model,
        "top_k": pipeline_settings.top_k,
        "semantic_enabled": bool(args.semantic),
        "semantic_model": args.semantic_model if args.semantic else None,
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
        try:
            pack = run_pipeline(
                str(example["question"]),
                settings=pipeline_settings,
            )
            evidence_texts = [
                record.evidence_text for record in pack.records
            ]
            answers = _gold_answers(example)
            lexical_score, lexical_text = evaluate_lexical(
                answers,
                evidence_texts,
                answerable=True,
            )
            if semantic_evaluator is not None:
                semantic_score, semantic_text = semantic_evaluator.score(
                    answers,
                    evidence_texts,
                    answerable=True,
                )
            else:
                semantic_score, semantic_text = None, ""

            row = {
                "question_id": question_id,
                "category": example.get("category", ""),
                "question": example.get("question", ""),
                "status": "success",
                "error": "",
                "returned_chunks": len(pack.records),
                "lexical_score": lexical_score,
                "lexical_best_chunk_rank": _best_rank(
                    evidence_texts,
                    lexical_text,
                ),
                "semantic_score": semantic_score,
                "semantic_best_chunk_rank": _best_rank(
                    evidence_texts,
                    semantic_text,
                ),
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
                "lexical_score": None,
                "semantic_score": None,
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
