from __future__ import annotations

import argparse
from pathlib import Path

from eval.benchmark_loader import benchmark_summary, load_benchmark
from eval.lexical_evaluator import evaluate_lexical
from eval.semantic_evaluator import DEFAULT_SEMANTIC_MODEL, evaluate_semantic
from eval.retrieval_runner import build_config_from_args, default_run_name, run_retrieval
from web_rag.config import get_settings
from web_rag.ranker import RANKER_CHOICES


def _setting(settings: object, name: str, default: object) -> object:
    return getattr(settings, name, default)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate the ProvideQ Web RAG retrieval module.")

    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/provideq_web_rag_evidence_benchmark_20.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--preview-benchmark", action="store_true")

    parser.add_argument("--run-retrieval", action="store_true")
    parser.add_argument("--ranker", default=_setting(settings, "default_ranker", "lexical"), choices=RANKER_CHOICES)
    parser.add_argument("--page-size", type=int, default=_setting(settings, "page_size", 8))
    parser.add_argument("--top-k", type=int, default=_setting(settings, "top_k", 5))
    parser.add_argument("--snippet-window-size", type=int, default=_setting(settings, "snippet_window_size", 3))
    parser.add_argument("--snippet-stride", type=int, default=_setting(settings, "snippet_stride", 1))
    parser.add_argument("--min-snippet-word-count", type=int, default=_setting(settings, "min_snippet_word_count", 10))
    parser.add_argument("--bm25-k1", type=float, default=_setting(settings, "bm25_k1", 1.5))
    parser.add_argument("--bm25-b", type=float, default=_setting(settings, "bm25_b", 0.75))
    parser.add_argument("--medcpt-batch-size", type=int, default=_setting(settings, "medcpt_batch_size", 8))
    parser.add_argument("--medcpt-device", default=_setting(settings, "medcpt_device", None))
    parser.add_argument("--medcpt-query-model", default=_setting(settings, "medcpt_query_model", "ncbi/MedCPT-Query-Encoder"))
    parser.add_argument("--medcpt-article-model", default=_setting(settings, "medcpt_article_model", "ncbi/MedCPT-Article-Encoder"))
    parser.add_argument("--hybrid-lexical-weight", type=float, default=_setting(settings, "hybrid_lexical_weight", 0.45))
    parser.add_argument("--hybrid-medcpt-weight", type=float, default=_setting(settings, "hybrid_medcpt_weight", 0.55))

    parser.add_argument("--run-lexical", action="store_true")
    parser.add_argument("--retrieval-results", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lexical-metrics", default="all", help="all, rouge1, rougel, rouge, bm25, or comma-separated list")

    parser.add_argument("--run-semantic", action="store_true")
    parser.add_argument("--semantic-metrics", default="all", help="all, nugget, answer, or comma-separated list")
    parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    parser.add_argument("--semantic-device", default=None)
    parser.add_argument("--semantic-batch-size", type=int, default=16)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_name = args.run_name or default_run_name(args.ranker)

    if args.preview_benchmark:
        examples = load_benchmark(args.benchmark)
        summary = benchmark_summary(examples)
        print(f"Benchmark: {args.benchmark}")
        print(f"Questions: {summary['questions']}")
        print(f"Gold nuggets: {summary['gold_nuggets']}")
        print(f"Average nuggets/question: {summary['avg_nuggets_per_question']}")
        return 0

    if args.run_retrieval:
        config = build_config_from_args(args, run_name)
        output_path = run_retrieval(config)
        print(f"Retrieval CSV saved: {output_path}")
        return 0

    if args.run_lexical:
        retrieval_results = args.retrieval_results or (args.output_root / run_name / "retrieval_results.csv")
        output_path = evaluate_lexical(
            benchmark_path=args.benchmark,
            retrieval_results_path=retrieval_results,
            run_name=run_name,
            evaluation_root=args.evaluation_root,
            metrics=args.lexical_metrics,
            k=args.k,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        )
        print(f"Evaluation CSV updated: {output_path}")
        return 0

    if args.run_semantic:
        retrieval_results = args.retrieval_results or (args.output_root / run_name / "retrieval_results.csv")
        output_path = evaluate_semantic(
            benchmark_path=args.benchmark,
            retrieval_results_path=retrieval_results,
            run_name=run_name,
            evaluation_root=args.evaluation_root,
            metrics=args.semantic_metrics,
            k=args.k,
            model_name=args.semantic_model,
            device=args.semantic_device,
            batch_size=args.semantic_batch_size,
        )
        print(f"Evaluation CSV updated: {output_path}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
