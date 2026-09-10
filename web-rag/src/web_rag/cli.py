from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    CHUNKING_METHODS,
    LLM_PROVIDERS,
    PAPERCLIP_MODES,
    PAPERCLIP_RANKINGS,
    QUERY_STRATEGIES,
    RERANKERS,
    get_settings,
)
from .paperclip_retriever import PaperclipError
from .pipeline import run_pipeline
from .serializer import save_evidence_outputs, save_json, save_text, to_json


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the ProvideQ Paperclip Web RAG pipeline.")

    parser.add_argument("question", nargs="?", help="Biomedical or pre-analytical question.")
    parser.add_argument("--question", dest="question_option")
    parser.add_argument("--limit", "--retrieval-limit", dest="retrieval_limit", type=int, default=settings.retrieval_limit)
    parser.add_argument("--query-strategy", choices=QUERY_STRATEGIES, default=settings.query_strategy)
    parser.add_argument(
        "--llm-provider",
        choices=LLM_PROVIDERS,
        default=settings.llm_provider,
        help="Use openai for Interweb and other OpenAI-compatible APIs.",
    )
    parser.add_argument("--llm-model", default=settings.llm_model)
    parser.add_argument("--llm-base-url", default=settings.llm_base_url)
    parser.add_argument("--llm-api-key-env", default=settings.llm_api_key_env)
    parser.add_argument(
        "--hyde-model",
        default=None,
        help="Optional HyDE-specific model override.",
    )
    parser.add_argument(
        "--hyde-base-url",
        default=None,
        help="Optional HyDE-specific API URL override.",
    )
    parser.add_argument("--hyde-temperature", type=float, default=settings.hyde_temperature)
    parser.add_argument("--hyde-max-tokens", type=int, default=settings.hyde_max_tokens)
    parser.add_argument("--hyde-seed", type=int, default=settings.hyde_seed)
    parser.add_argument("--hyde-timeout", type=float, default=settings.hyde_timeout)

    parser.add_argument(
        "--expansion-model",
        default=None,
        help="Optional LLM-expansion-specific model override.",
    )
    parser.add_argument(
        "--expansion-base-url",
        default=None,
        help="Optional LLM-expansion-specific API URL override.",
    )
    parser.add_argument("--expansion-temperature", type=float, default=settings.expansion_temperature)
    parser.add_argument("--expansion-max-tokens", type=int, default=settings.expansion_max_tokens)
    parser.add_argument("--expansion-seed", type=int, default=settings.expansion_seed)
    parser.add_argument("--expansion-timeout", type=float, default=settings.expansion_timeout)
    parser.add_argument("--expansion-max-terms", type=int, default=settings.expansion_max_terms)
    parser.add_argument("--expansion-max-query-chars", type=int, default=settings.expansion_max_query_chars)

    parser.add_argument(
        "--paperclip-source",
        "--paperclip-corpus",
        dest="paperclip_source",
        default=settings.paperclip_source,
        help="Academic Paperclip source or comma-separated sources: pmc,biorxiv,medrxiv,arxiv.",
    )
    parser.add_argument("--paperclip-ranking", choices=PAPERCLIP_RANKINGS, default=settings.paperclip_ranking)
    parser.add_argument("--paperclip-max-lines", type=int, default=settings.paperclip_max_full_text_lines)
    parser.add_argument("--paperclip-mode", choices=PAPERCLIP_MODES)
    parser.add_argument("--paperclip-since")
    parser.add_argument("--paperclip-sort", choices=("relevance", "date"))
    parser.add_argument("--paperclip-year")
    parser.add_argument("--paperclip-journal")
    parser.add_argument("--paperclip-article-type")
    parser.add_argument("--paperclip-author")
    parser.add_argument(
        "--paperclip-full-corpus",
        action=argparse.BooleanOptionalAction,
        default=settings.paperclip_full_corpus,
        help="Search the full Paperclip corpus. Enabled by default.",
    )

    parser.add_argument("--chunking-method", choices=CHUNKING_METHODS, default=settings.chunking_method)
    parser.add_argument("--chunk-window-size", type=int, default=settings.chunk_window_size)
    parser.add_argument("--chunk-stride", type=int, default=settings.chunk_stride)
    parser.add_argument("--min-chunk-chars", type=int, default=settings.min_chunk_chars)
    parser.add_argument("--max-chunk-chars", type=int, default=settings.max_chunk_chars)
    parser.add_argument("--min-chunk-words", type=int, default=settings.min_chunk_words)
    parser.add_argument("--disable-context-backoff", action="store_true")

    parser.add_argument("--reranker", "--ranker", dest="reranker", choices=RERANKERS, default=settings.reranker)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--max-chunks-per-paper", type=int, default=settings.max_chunks_per_paper)
    parser.add_argument("--near-duplicate-threshold", type=float, default=settings.near_duplicate_threshold)
    parser.add_argument("--bm25-k1", type=float, default=settings.bm25_k1)
    parser.add_argument("--bm25-b", type=float, default=settings.bm25_b)
    parser.add_argument("--medcpt-query-model", default=settings.medcpt_query_model)
    parser.add_argument("--medcpt-article-model", default=settings.medcpt_article_model)
    parser.add_argument("--medcpt-batch-size", type=int, default=settings.medcpt_batch_size)
    parser.add_argument("--medcpt-device", default=settings.medcpt_device)
    parser.add_argument("--hybrid-lexical-weight", type=float, default=settings.hybrid_lexical_weight)
    parser.add_argument("--hybrid-medcpt-weight", type=float, default=settings.hybrid_medcpt_weight)

    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-context", type=Path)
    parser.add_argument("--show-query", action="store_true")
    parser.add_argument("--show-info", action="store_true")
    parser.add_argument("--print-context", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    question = args.question_option or args.question
    if not question:
        raise SystemExit("Provide a question as a positional argument or with --question.")

    try:
        evidence = run_pipeline(
            question,
            retrieval_limit=args.retrieval_limit,
            query_strategy=args.query_strategy,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key_env=args.llm_api_key_env,
            hyde_model=args.hyde_model,
            hyde_base_url=args.hyde_base_url,
            hyde_temperature=args.hyde_temperature,
            hyde_max_tokens=args.hyde_max_tokens,
            hyde_seed=args.hyde_seed,
            hyde_timeout=args.hyde_timeout,
            expansion_model=args.expansion_model,
            expansion_base_url=args.expansion_base_url,
            expansion_temperature=args.expansion_temperature,
            expansion_max_tokens=args.expansion_max_tokens,
            expansion_seed=args.expansion_seed,
            expansion_timeout=args.expansion_timeout,
            expansion_max_terms=args.expansion_max_terms,
            expansion_max_query_chars=args.expansion_max_query_chars,
            paperclip_source=args.paperclip_source,
            paperclip_ranking=args.paperclip_ranking,
            paperclip_max_full_text_lines=args.paperclip_max_lines,
            paperclip_mode=args.paperclip_mode,
            paperclip_since=args.paperclip_since,
            paperclip_sort=args.paperclip_sort,
            paperclip_year=args.paperclip_year,
            paperclip_journal=args.paperclip_journal,
            paperclip_article_type=args.paperclip_article_type,
            paperclip_author=args.paperclip_author,
            paperclip_full_corpus=args.paperclip_full_corpus,
            chunking_method=args.chunking_method,
            chunk_window_size=args.chunk_window_size,
            chunk_stride=args.chunk_stride,
            min_chunk_chars=args.min_chunk_chars,
            max_chunk_chars=args.max_chunk_chars,
            min_chunk_words=args.min_chunk_words,
            context_backoff=not args.disable_context_backoff,
            reranker=args.reranker,
            top_k=args.top_k,
            max_chunks_per_paper=args.max_chunks_per_paper,
            near_duplicate_threshold=args.near_duplicate_threshold,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            medcpt_query_model=args.medcpt_query_model,
            medcpt_article_model=args.medcpt_article_model,
            medcpt_batch_size=args.medcpt_batch_size,
            medcpt_device=args.medcpt_device,
            hybrid_lexical_weight=args.hybrid_lexical_weight,
            hybrid_medcpt_weight=args.hybrid_medcpt_weight,
        )
    except (PaperclipError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.show_query:
        print(f"Query strategy: {evidence.query.strategy}")
        if evidence.query.strategy == "hyde":
            print(f"Hypothetical document: {evidence.query.hypothetical_document}\n")
        else:
            print(f"Search query: {evidence.query.search_query}")
            if evidence.query.strategy == "llmexpand":
                for name, values in evidence.query.expansion_details.items():
                    print(f"{name}: {', '.join(values)}")
            print()
    if args.show_info:
        print(to_json(evidence.pipeline))
        print()

    saved_paths: list[Path] = []
    if args.output_dir:
        saved_paths.extend(save_evidence_outputs(evidence, args.output_dir).values())
    if args.output_json:
        saved_paths.append(save_json(evidence, args.output_json))
    if args.output_context:
        saved_paths.append(save_text(evidence.context_text, args.output_context))

    if saved_paths:
        for path in saved_paths:
            print(f"Saved: {path}")
    elif args.print_context:
        print(evidence.context_text)
    else:
        print(to_json(evidence))


if __name__ == "__main__":
    main()
