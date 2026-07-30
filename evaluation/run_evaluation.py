from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Callable

from web_rag import run_pipeline

from .lexical_evaluation import evaluate_lexical
from .llm_judge_evaluation import LLMJudgeEvaluator
from .semantic_evaluation import DEFAULT_MODEL, SemanticEvaluator


DEFAULT_BENCHMARK = Path("benchmark/provideq_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("outputs")
MAX_QUESTIONS = 97


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    examples = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(examples, list):
        raise ValueError("Benchmark must contain an 'examples' list")
    if len(examples) > MAX_QUESTIONS:
        raise ValueError(f"Benchmark contains more than {MAX_QUESTIONS} questions")

    required = {
        "id",
        "question",
        "gold_answers",
        "gold_documents",
        "category",
        "answerable",
    }
    seen_ids: set[str] = set()
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict) or not required.issubset(example):
            raise ValueError(f"Invalid benchmark item at position {index}")
        question_id = str(example["id"]).strip()
        if not question_id or question_id in seen_ids:
            raise ValueError(f"Missing or duplicate benchmark id at position {index}")
        seen_ids.add(question_id)

    return examples


def select_questions(
    examples: list[dict[str, Any]],
    number: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if number < 1 or number > MAX_QUESTIONS:
        raise ValueError(f"num_questions must be between 1 and {MAX_QUESTIONS}")
    if number > len(examples):
        raise ValueError(f"Benchmark contains only {len(examples)} questions")

    selected = random.Random(seed).sample(examples, number)
    return sorted(selected, key=lambda example: str(example["id"]))


def _evidence_texts(result: Any) -> list[str]:
    return [
        str(record.evidence_text).strip()
        for record in result.records
        if str(record.evidence_text).strip()
    ]


def _write_results(
    output_dir: Path,
    folder_name: str,
    rows: list[dict[str, Any]],
) -> Path:
    run_dir = output_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "results.csv"

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["question_id", "question", "answers", "gold_answer", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_file


def run_evaluation(
    args: argparse.Namespace,
    *,
    pipeline_fn: Callable[..., Any] = run_pipeline,
) -> list[Path]:
    examples = load_benchmark(args.benchmark)
    selected = select_questions(examples, args.num_questions, seed=args.seed)

    layers = ["lexical", "semantic", "judge"] if args.evaluation == "all" else [args.evaluation]
    semantic_evaluator = None
    judge_evaluator = None

    if "semantic" in layers:
        semantic_evaluator = SemanticEvaluator(
            args.semantic_model,
            device=args.semantic_device,
            batch_size=args.semantic_batch_size,
        )
    if "judge" in layers:
        judge_evaluator = LLMJudgeEvaluator(
            provider=args.judge_provider,
            model=args.judge_model,
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
            retries=args.judge_retries,
        )

    scores: dict[str, list[dict[str, Any]]] = {layer: [] for layer in layers}
    no_save = bool(getattr(args, "no_save", False))

    for index, example in enumerate(selected, start=1):
        question = str(example["question"])
        if not no_save:
            print(f"[{index}/{len(selected)}] {example['id']}")

        result = pipeline_fn(
            question,
            retrieval_limit=args.retrieval_limit,
            query_strategy=args.query_strategy,
            hyde_model=args.hyde_model,
            hyde_base_url=args.hyde_base_url,
            hyde_temperature=args.hyde_temperature,
            hyde_max_tokens=args.hyde_max_tokens,
            hyde_seed=args.hyde_seed,
            hyde_timeout=args.hyde_timeout,
            paperclip_source=args.paperclip_source,
            paperclip_ranking=args.paperclip_ranking,
            chunk_window_size=args.chunk_window_size,
            chunk_stride=args.chunk_stride,
            reranker=args.reranker,
            top_k=args.top_k,
            max_chunks_per_paper=args.max_chunks_per_paper,
            near_duplicate_threshold=args.near_duplicate_threshold,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            medcpt_device=args.medcpt_device,
            hybrid_lexical_weight=args.hybrid_lexical_weight,
            hybrid_medcpt_weight=args.hybrid_medcpt_weight,
        )
        evidence = _evidence_texts(result)
        gold_answers = [str(answer) for answer in example["gold_answers"]]
        answerable = bool(example["answerable"])

        if "lexical" in layers:
            score, best_evidence = evaluate_lexical(
                gold_answers, evidence, answerable=answerable
            )
            scores["lexical"].append(_result_row(example, best_evidence, score))

        if "semantic" in layers and semantic_evaluator is not None:
            score, best_evidence = semantic_evaluator.score(
                gold_answers, evidence, answerable=answerable
            )
            scores["semantic"].append(_result_row(example, best_evidence, score))

        if "judge" in layers and judge_evaluator is not None:
            score, best_evidence = judge_evaluator.score(
                question,
                gold_answers,
                evidence,
                answerable=answerable,
            )
            scores["judge"].append(_result_row(example, best_evidence, score))

        if no_save:
            _print_question_scores(str(example["id"]), layers, scores)

    if no_save:
        _print_average_scores(layers, scores)
        return []

    saved: list[Path] = []
    for layer in layers:
        folder_name = (
            f"{args.query_strategy}_{args.paperclip_ranking}_{args.reranker}_{layer}"
        )
        saved.append(_write_results(Path(args.output_dir), folder_name, scores[layer]))

    return saved


def _clean_csv_text(value: str) -> str:
    return " ".join(value.split())


def _display_score(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    return f"{float(value):.2f}"


def _print_question_scores(
    question_id: str,
    layers: list[str],
    scores: dict[str, list[dict[str, Any]]],
) -> None:
    if len(layers) == 1:
        score = scores[layers[0]][-1]["score"]
        print(f"{question_id} | score: {_display_score(score)}")
        return

    layer_scores = " | ".join(
        f"{layer}: {_display_score(scores[layer][-1]['score'])}" for layer in layers
    )
    print(f"{question_id} | {layer_scores}")


def _print_average_scores(
    layers: list[str],
    scores: dict[str, list[dict[str, Any]]],
) -> None:
    print()
    for layer in layers:
        numeric_scores = [
            float(row["score"])
            for row in scores[layer]
            if row["score"] not in (None, "")
        ]
        average = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
        label = "Average score" if len(layers) == 1 else f"Average {layer} score"
        print(f"{label}: {_display_score(average)}")


def _result_row(
    example: dict[str, Any],
    best_evidence: str,
    score: float | None,
) -> dict[str, Any]:
    return {
        "question_id": str(example["id"]),
        "question": _clean_csv_text(str(example["question"])),
        "answers": _clean_csv_text(best_evidence),
        "gold_answer": " | ".join(
            _clean_csv_text(str(answer)) for answer in example["gold_answers"]
        ),
        "score": "" if score is None else round(float(score), 4),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the ProvideQ Paperclip Web RAG pipeline")

    parser.add_argument("--evaluation", choices=("lexical", "semantic", "judge", "all"), default="lexical")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print per-question and average scores without writing results.csv files.",
    )

    parser.add_argument("--paperclip-source", default="pmc")
    parser.add_argument("--paperclip-ranking", choices=("bm25", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--retrieval-limit", type=int, default=50)
    parser.add_argument("--query-strategy", choices=("raw", "synonym", "hyde"), default="synonym")
    parser.add_argument("--hyde-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--hyde-base-url", default="http://localhost:11434")
    parser.add_argument("--hyde-temperature", type=float, default=0.0)
    parser.add_argument("--hyde-max-tokens", type=int, default=256)
    parser.add_argument("--hyde-seed", type=int, default=42)
    parser.add_argument("--hyde-timeout", type=float, default=180.0)

    parser.add_argument("--chunk-window-size", type=int, default=3)
    parser.add_argument("--chunk-stride", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chunks-per-paper", type=int, default=2)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.80)

    parser.add_argument("--reranker", choices=("lexical", "medcpt", "hybrid"), default="medcpt")
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--medcpt-device", default="auto")
    parser.add_argument("--hybrid-lexical-weight", type=float, default=0.30)
    parser.add_argument("--hybrid-medcpt-weight", type=float, default=0.70)

    parser.add_argument("--semantic-model", default=DEFAULT_MODEL)
    parser.add_argument("--semantic-device", default="auto")
    parser.add_argument("--semantic-batch-size", type=int, default=8)

    parser.add_argument("--judge-provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-retries", type=int, default=2)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = run_evaluation(args)
    for path in paths:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
