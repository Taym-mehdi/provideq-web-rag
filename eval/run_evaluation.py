from __future__ import annotations

import argparse
from pathlib import Path

from eval.benchmark_loader import load_benchmark, summarize_benchmark
from eval.comparison import run_comparison
from eval.metrics.semantic import DEFAULT_EMBEDDING_MODEL
from eval.reports import run_lexical_report, run_semantic_report
from eval.retrieval_runner import (
    build_default_run_config,
    run_retrieval_for_benchmark,
)


def parse_k_values(raw_value: str) -> list[int]:
    """
    Parse comma-separated k values.

    Example:
        "1,3,5,10" -> [1, 3, 5, 10]
    """
    values: list[int] = []

    for part in raw_value.split(","):
        part = part.strip()

        if not part:
            continue

        value = int(part)

        if value <= 0:
            raise ValueError("k values must be positive integers.")

        values.append(value)

    if not values:
        raise ValueError("At least one k value is required.")

    return sorted(set(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Web RAG evaluation."
    )

    parser.add_argument(
        "--benchmark",
        default="benchmarks/dummy_benchmark.csv",
        help="Path to benchmark CSV file."
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print loaded benchmark examples."
    )

    parser.add_argument(
        "--run-retrieval",
        action="store_true",
        help="Run Web RAG retrieval for all benchmark questions."
    )

    parser.add_argument(
        "--run-lexical-metrics",
        action="store_true",
        help="Compute lexical metrics from a raw_results.jsonl file."
    )

    parser.add_argument(
        "--run-semantic-metrics",
        action="store_true",
        help="Compute semantic metrics from a raw_results.jsonl file."
    )

    parser.add_argument(
        "--compare-runs",
        nargs="+",
        default=None,
        help=(
            "Compare evaluation run folders. Use format label=path, for example: "
            "lexical=outputs/evaluation/provideq20_lexical "
            "hybrid=outputs/evaluation/provideq20_medcpt_hybrid"
        ),
    )

    parser.add_argument(
        "--baseline-run",
        default="lexical",
        help="Baseline run label used for --compare-runs."
    )

    parser.add_argument(
        "--comparison-output",
        default="outputs/evaluation/method_comparison.csv",
        help="Path for the comparison CSV output."
    )

    parser.add_argument(
        "--raw-results",
        default="",
        help="Path to raw_results.jsonl for metric computation."
    )

    parser.add_argument(
        "--k-values",
        default="1,3,5,10",
        help="Comma-separated k values, for example: 1,3,5,10."
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformers embedding model for semantic evaluation."
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation."
    )

    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional embedding device, for example cpu or cuda."
    )

    parser.add_argument(
        "--ranker",
        choices=["lexical", "medcpt-hybrid"],
        default="lexical",
        help="Ranking method to evaluate."
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="Number of papers to retrieve per question."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of evidence records to save per question."
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Number of sentences per evidence snippet."
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory where evaluation outputs will be saved."
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for this evaluation run."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of benchmark questions to run."
    )

    return parser


def print_preview(examples) -> None:
    print("\nBenchmark preview:\n")

    for example in examples:
        print(f"{example.question_id}: {example.question}")
        print(f"  Gold answer: {example.gold_answer}")
        print("  Gold nuggets:")

        for nugget_index, nugget in enumerate(example.gold_nuggets, start=1):
            print(f"    {nugget_index}. {nugget}")

        if example.notes:
            print(f"  Notes: {example.notes}")

        print()


def run_metric_commands(args) -> bool:
    """
    Run metric-only and comparison commands.

    Returns True if a command was executed.
    """
    command_executed = False

    if args.run_lexical_metrics:
        if not args.raw_results:
            raise ValueError(
                "--raw-results is required when using --run-lexical-metrics"
            )

        k_values = parse_k_values(args.k_values)
        output_dir = Path(args.output_dir)

        print("\n=== Web RAG Lexical Evaluation ===\n")
        print(f"Raw results: {args.raw_results}")
        print(f"k values: {k_values}")
        print(f"Output directory: {output_dir}")

        report_files = run_lexical_report(
            raw_results_path=args.raw_results,
            output_dir=output_dir,
            k_values=k_values,
        )

        print("\nLexical metric reports saved:")
        print(f"Aggregate metrics: {report_files['aggregate']}")
        print(f"Per-question metrics: {report_files['question']}")
        print(f"Per-nugget details: {report_files['nugget']}")

        command_executed = True

    if args.run_semantic_metrics:
        if not args.raw_results:
            raise ValueError(
                "--raw-results is required when using --run-semantic-metrics"
            )

        k_values = parse_k_values(args.k_values)
        output_dir = Path(args.output_dir)

        print("\n=== Web RAG Semantic Evaluation ===\n")
        print(f"Raw results: {args.raw_results}")
        print(f"k values: {k_values}")
        print(f"Embedding model: {args.embedding_model}")
        print(f"Embedding batch size: {args.embedding_batch_size}")
        print(f"Embedding device: {args.embedding_device}")
        print(f"Output directory: {output_dir}")

        report_files = run_semantic_report(
            raw_results_path=args.raw_results,
            output_dir=output_dir,
            k_values=k_values,
            embedding_model_name=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_device=args.embedding_device,
        )

        print("\nSemantic metric reports saved:")
        print(f"Aggregate metrics: {report_files['aggregate']}")
        print(f"Per-question metrics: {report_files['question']}")
        print(f"Per-nugget details: {report_files['nugget']}")

        command_executed = True

    if args.compare_runs:
        print("\n=== Web RAG Method Comparison ===\n")
        print(f"Baseline run: {args.baseline_run}")
        print("Compared runs:")

        for run_spec in args.compare_runs:
            print(f"  {run_spec}")

        output_path = run_comparison(
            raw_run_specs=args.compare_runs,
            baseline_label=args.baseline_run,
            output_path=args.comparison_output,
        )

        print(f"\nComparison CSV saved: {output_path}")

        command_executed = True

    return command_executed


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    command_executed = run_metric_commands(args)

    if command_executed and not args.run_retrieval:
        return 0

    examples = load_benchmark(args.benchmark)
    summary = summarize_benchmark(examples)

    print("\n=== Web RAG Evaluation ===\n")
    print(f"Benchmark: {args.benchmark}")
    print(f"Questions: {summary['questions']}")
    print(f"Gold nuggets: {summary['gold_nuggets']}")
    print(f"Average nuggets per question: {summary['avg_nuggets_per_question']}")

    if args.preview:
        print_preview(examples)

    if args.run_retrieval:
        config = build_default_run_config(
            benchmark_path=args.benchmark,
            output_dir=args.output_dir,
            run_name=args.run_name,
            ranking_method=args.ranker,
            page_size=args.page_size,
            top_k=args.top_k,
            window_size=args.window_size,
            limit=args.limit,
        )

        print("\nRunning retrieval:")
        print(f"  Ranker: {config.ranking_method}")
        print(f"  Page size: {config.page_size}")
        print(f"  Top-k saved: {config.top_k}")
        print(f"  Window size: {config.window_size}")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Run name: {config.run_name}")

        files = run_retrieval_for_benchmark(
            examples=examples,
            config=config,
        )

        print("\nRetrieval run finished.")
        print(f"Run directory: {files.run_dir}")
        print(f"Raw JSONL: {files.raw_jsonl_path}")
        print(f"Evidence CSV: {files.evidence_csv_path}")
        print(f"Summary CSV: {files.summary_csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())