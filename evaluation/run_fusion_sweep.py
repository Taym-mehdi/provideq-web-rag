from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FusionTest:
    name: str
    label: str
    query_strategy: str
    arguments: tuple[str, ...]


PREVIOUS_QUERY_CACHE = Path("outputs/retrieval_15_tests_90_v6/query_cache")
DEFAULT_HYDE_CACHE = (
    PREVIOUS_QUERY_CACHE
    / "hyde_qwen3_6_35b_a3b_bf16_2026_09_biomedical_ir_v6.json"
)
DEFAULT_LLM_EXPANSION_CACHE = (
    PREVIOUS_QUERY_CACHE
    / "llmexpand_qwen3_6_35b_a3b_bf16_2026_09_biomedical_ir_v6.json"
)


def _build_configs() -> tuple[FusionTest, ...]:
    """Build the focused fusion comparison selected from the 15-run benchmark."""
    configs: list[FusionTest] = []
    query_settings = (
        (
            "raw",
            "raw",
            "Paperclip vector raw + Europe PMC raw without synonyms",
        ),
        (
            "anchored_hyde",
            "hyde",
            "Paperclip vector raw/HyDE + Europe PMC raw without synonyms",
        ),
        (
            "anchored_llm_expansion",
            "llmexpand",
            "Paperclip vector raw/LLM expansion + Europe PMC raw without synonyms",
        ),
    )

    for query_name, query_strategy, description in query_settings:
        arguments = [
            "--retriever", "fusion",
            "--query-strategy", query_strategy,
            "--paperclip-ranking", "vector",
            "--europepmc-mode", "multi",
            "--no-europepmc-synonym",
            # Europe PMC stays on the raw question. Its own multi-query builder
            # performs source-specific lexical relaxation.
            "--no-europepmc-use-reformulated-query",
        ]
        if query_strategy != "raw":
            # Only Paperclip receives the generated query. Its raw result stays
            # anchored and receives twice the weight of the reformulated result.
            arguments.extend(
                [
                    "--paperclip-query-fusion",
                    "--query-fusion-rrf-k", "10",
                    "--reformulated-query-weight", "0.5",
                ]
            )

        number = len(configs) + 1
        configs.append(
            FusionTest(
                name=(
                    f"{number:02d}_fusion_paperclip_vector_{query_name}_"
                    "europepmc_raw_no_synonyms"
                ),
                label=f"Fusion: {description}",
                query_strategy=query_strategy,
                arguments=tuple(arguments),
            )
        )

    return tuple(configs)


CONFIGS = _build_configs()


def _query_cache(config: FusionTest, args: argparse.Namespace) -> str | None:
    if config.query_strategy == "hyde":
        return args.hyde_cache
    if config.query_strategy == "llmexpand":
        return args.llm_expansion_cache
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the focused three-configuration retrieval-fusion sweep."
    )
    parser.add_argument("--benchmark", default="benchmark/provideq_benchmark.json")
    parser.add_argument("--num-questions", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/retrieval_fusion_90")
    parser.add_argument("--retrieval-limit", type=int, default=20)
    parser.add_argument("--paperclip-candidate-limit", type=int, default=30)
    parser.add_argument("--europepmc-candidate-limit", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--hyde-cache", default=str(DEFAULT_HYDE_CACHE))
    parser.add_argument(
        "--llm-expansion-cache",
        default=str(DEFAULT_LLM_EXPANSION_CACHE),
    )
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
        help="Print the three fusion tests and exit without retrieval requests.",
    )
    args = parser.parse_args()

    if args.list_configs:
        for index, config in enumerate(CONFIGS, start=1):
            print(f"{index:02d}. {config.name}: {config.label}")
        return 0

    for label, cache in (
        ("HyDE", args.hyde_cache),
        ("LLM expansion", args.llm_expansion_cache),
    ):
        if not Path(cache).is_file():
            parser.error(
                f"{label} query cache not found: {cache}. "
                "Restore the approved v6 cache or pass the correct cache path."
            )

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
        cache = _query_cache(config, args)
        if cache:
            command.extend(["--llm-cache", cache])

        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Stopped: {config.label} failed with exit code {result.returncode}")
            return result.returncode

    print("\nThree-test fusion sweep completed.")
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
