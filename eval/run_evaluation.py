from __future__ import annotations

import argparse

from eval.benchmark_loader import load_benchmark, summarize_benchmark


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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    examples = load_benchmark(args.benchmark)
    summary = summarize_benchmark(examples)

    print("\n=== Web RAG Evaluation ===\n")
    print(f"Benchmark: {args.benchmark}")
    print(f"Questions: {summary['questions']}")
    print(f"Gold nuggets: {summary['gold_nuggets']}")
    print(f"Average nuggets per question: {summary['avg_nuggets_per_question']}")

    if args.preview:
        print_preview(examples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())