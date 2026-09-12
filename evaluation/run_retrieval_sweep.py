from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalTest:
    name: str
    label: str
    arguments: tuple[str, ...]


QUERY_SETTINGS = (
    ("raw", "raw", "Raw"),
    ("anchored_hyde", "hyde", "Anchored HyDE"),
    ("anchored_llm_expansion", "llmexpand", "Anchored LLM expansion"),
)


def _build_configs() -> tuple[RetrievalTest, ...]:
    """Build the fixed 5 retrieval settings x 3 query settings matrix."""
    configs: list[RetrievalTest] = []

    for ranking in ("bm25", "vector", "hybrid"):
        for query_name, query_strategy, query_label in QUERY_SETTINGS:
            arguments = [
                "--retriever", "paperclip",
                "--query-strategy", query_strategy,
                "--paperclip-ranking", ranking,
            ]
            if query_strategy != "raw":
                # The raw ranking remains the anchor and is fused with the
                # reformulated-query ranking using the same fixed weights.
                arguments.extend(
                    [
                        "--paperclip-query-fusion",
                        "--query-fusion-rrf-k", "10",
                        "--reformulated-query-weight", "0.5",
                    ]
                )
            number = len(configs) + 1
            configs.append(
                RetrievalTest(
                    name=f"{number:02d}_paperclip_{ranking}_{query_name}",
                    label=f"Paperclip {ranking.upper()} + {query_label}",
                    arguments=tuple(arguments),
                )
            )

    for synonyms, synonym_name, synonym_label in (
        (True, "synonyms", "with synonyms"),
        (False, "no_synonyms", "without synonyms"),
    ):
        for query_name, query_strategy, query_label in QUERY_SETTINGS:
            arguments = [
                "--retriever", "europepmc",
                "--query-strategy", query_strategy,
                "--europepmc-mode", "multi",
                "--europepmc-synonym" if synonyms else "--no-europepmc-synonym",
            ]
            if query_strategy != "raw":
                # HyDE and LLM expansion already begin with the original
                # question, so Europe PMC receives an explicitly anchored query.
                arguments.append("--europepmc-use-reformulated-query")
            number = len(configs) + 1
            configs.append(
                RetrievalTest(
                    name=f"{number:02d}_europepmc_{synonym_name}_{query_name}",
                    label=f"Europe PMC {synonym_label} + {query_label}",
                    arguments=tuple(arguments),
                )
            )

    return tuple(configs)


CONFIGS = _build_configs()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the focused document-retrieval sweep.")
    parser.add_argument("--benchmark", default="benchmark/provideq_benchmark.json")
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/retrieval_15_tests")
    parser.add_argument("--retrieval-limit", type=int, default=20)
    parser.add_argument("--paperclip-candidate-limit", type=int, default=30)
    parser.add_argument("--europepmc-candidate-limit", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, rerun only rows previously saved with status=error.",
    )
    parser.add_argument(
        "--retry-warnings",
        action="store_true",
        help="When resuming, rerun rows that previously used a fallback.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print the 15 tests and exit without making retrieval requests.",
    )
    args = parser.parse_args()

    if args.list_configs:
        for index, config in enumerate(CONFIGS, start=1):
            print(f"{index:02d}. {config.name}: {config.label}")
        return 0

    output_dir = Path(args.output_dir)

    common = [
        "--benchmark", args.benchmark,
        "--num-questions", str(args.num_questions),
        "--seed", str(args.seed),
        "--retrieval-limit", str(args.retrieval_limit),
        "--paperclip-candidate-limit", str(args.paperclip_candidate_limit),
        "--europepmc-candidate-limit", str(args.europepmc_candidate_limit),
        "--rrf-k", str(args.rrf_k),
        "--output-dir", str(output_dir),
        "--europepmc-cache-dir", str(output_dir / "europepmc_cache"),
    ]
    if args.no_resume:
        common.append("--no-resume")
    if args.retry_errors:
        common.append("--retry-errors")
    if args.retry_warnings:
        common.append("--retry-warnings")

    for index, config in enumerate(CONFIGS, start=1):
        print(f"\n{'=' * 72}\n[{index}/{len(CONFIGS)}] {config.label}\n{'=' * 72}")
        command = [
            sys.executable,
            "-m",
            "evaluation.run_document_retrieval",
            *common,
            "--run-name", config.name,
            *config.arguments,
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Stopped: {config.label} failed with exit code {result.returncode}")
            return result.returncode

    print("\n15-test retrieval sweep completed.")
    summary_command = [
        sys.executable,
        "-m",
        "evaluation.summarize_document_retrieval",
        "--input-dir", str(output_dir),
        "--output", str(output_dir / "summary.csv"),
    ]
    summary_result = subprocess.run(summary_command)
    if summary_result.returncode != 0:
        return summary_result.returncode
    print(f"Results are under: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
