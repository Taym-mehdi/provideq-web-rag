from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    (
        "Paperclip raw BM25",
        [
            "--retriever", "paperclip",
            "--query-strategy", "raw",
            "--paperclip-ranking", "bm25",
        ],
    ),
    (
        "Paperclip raw vector",
        [
            "--retriever", "paperclip",
            "--query-strategy", "raw",
            "--paperclip-ranking", "vector",
        ],
    ),
    (
        "Paperclip raw hybrid",
        [
            "--retriever", "paperclip",
            "--query-strategy", "raw",
            "--paperclip-ranking", "hybrid",
        ],
    ),
    (
        "Paperclip anchored HyDE hybrid",
        [
            "--retriever", "paperclip",
            "--query-strategy", "hyde",
            "--paperclip-ranking", "hybrid",
            "--paperclip-query-fusion",
            "--query-fusion-rrf-k", "10",
            "--reformulated-query-weight", "0.5",
        ],
    ),
    (
        "Paperclip anchored LLM expansion hybrid",
        [
            "--retriever", "paperclip",
            "--query-strategy", "llmexpand",
            "--paperclip-ranking", "hybrid",
            "--paperclip-query-fusion",
            "--query-fusion-rrf-k", "10",
            "--reformulated-query-weight", "0.5",
        ],
    ),
    (
        "Europe PMC direct with synonyms",
        [
            "--retriever", "europepmc",
            "--query-strategy", "raw",
            "--europepmc-mode", "direct",
            "--europepmc-synonym",
        ],
    ),
    (
        "Europe PMC multi-query without synonyms",
        [
            "--retriever", "europepmc",
            "--query-strategy", "raw",
            "--europepmc-mode", "multi",
            "--no-europepmc-synonym",
        ],
    ),
    (
        "Europe PMC multi-query with synonyms",
        [
            "--retriever", "europepmc",
            "--query-strategy", "raw",
            "--europepmc-mode", "multi",
            "--europepmc-synonym",
        ],
    ),
    (
        "Paperclip and Europe PMC fusion",
        [
            "--retriever", "fusion",
            "--query-strategy", "raw",
            "--paperclip-ranking", "hybrid",
            "--europepmc-mode", "multi",
            "--no-europepmc-synonym",
        ],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the focused document-retrieval sweep.")
    parser.add_argument("--benchmark", default="benchmark/provideq_benchmark.json")
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-limit", type=int, default=20)
    parser.add_argument("--paperclip-candidate-limit", type=int, default=30)
    parser.add_argument("--europepmc-candidate-limit", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    common = [
        "--benchmark", args.benchmark,
        "--num-questions", str(args.num_questions),
        "--seed", str(args.seed),
        "--retrieval-limit", str(args.retrieval_limit),
        "--paperclip-candidate-limit", str(args.paperclip_candidate_limit),
        "--europepmc-candidate-limit", str(args.europepmc_candidate_limit),
        "--rrf-k", str(args.rrf_k),
    ]
    if args.no_resume:
        common.append("--no-resume")

    for index, (label, config) in enumerate(CONFIGS, start=1):
        print(f"\n{'=' * 72}\n[{index}/{len(CONFIGS)}] {label}\n{'=' * 72}")
        command = [
            sys.executable,
            "-m",
            "evaluation.run_document_retrieval",
            *common,
            *config,
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Stopped: {label} failed with exit code {result.returncode}")
            return result.returncode

    print("\nFocused retrieval sweep completed.")
    summary_command = [sys.executable, "-m", "evaluation.summarize_document_retrieval"]
    summary_result = subprocess.run(summary_command)
    if summary_result.returncode != 0:
        return summary_result.returncode
    print(f"Results are under: {Path('outputs/document_retrieval').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
