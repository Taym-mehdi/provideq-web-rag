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
from .serializer import (
    save_evidence_outputs,
    save_json,
    save_text,
    to_json,
)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve scientific papers and return reranked evidence chunks."
        )
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Biomedical or pre-analytical question.",
    )
    parser.add_argument("--question", dest="question_option")

    retrieval = parser.add_argument_group("Paper retrieval")
    retrieval.add_argument(
        "--limit",
        "--retrieval-limit",
        dest="retrieval_limit",
        type=int,
        default=settings.retrieval_limit,
    )
    retrieval.add_argument(
        "--query-strategy",
        choices=QUERY_STRATEGIES,
        default=settings.query_strategy,
    )
    retrieval.add_argument(
        "--paperclip-source",
        "--paperclip-corpus",
        dest="paperclip_source",
        default=settings.paperclip_source,
    )
    retrieval.add_argument(
        "--paperclip-ranking",
        choices=PAPERCLIP_RANKINGS,
        default=settings.paperclip_ranking,
    )
    retrieval.add_argument(
        "--paperclip-max-lines",
        type=int,
        default=settings.paperclip_max_full_text_lines,
    )
    retrieval.add_argument(
        "--paperclip-timeout",
        type=float,
        default=settings.paperclip_timeout,
    )
    retrieval.add_argument("--paperclip-mode", choices=PAPERCLIP_MODES)
    retrieval.add_argument("--paperclip-since")
    retrieval.add_argument(
        "--paperclip-sort",
        choices=("relevance", "date"),
    )
    retrieval.add_argument("--paperclip-year")
    retrieval.add_argument("--paperclip-journal")
    retrieval.add_argument("--paperclip-article-type")
    retrieval.add_argument("--paperclip-author")
    retrieval.add_argument(
        "--paperclip-full-corpus",
        action=argparse.BooleanOptionalAction,
        default=settings.paperclip_full_corpus,
    )

    generation = parser.add_argument_group(
        "Optional HyDE or LLM-expansion settings"
    )
    generation.add_argument(
        "--llm-provider",
        choices=LLM_PROVIDERS,
        default=settings.llm_provider,
    )
    generation.add_argument("--llm-model", default=settings.llm_model)
    generation.add_argument(
        "--llm-base-url",
        default=settings.llm_base_url,
    )
    generation.add_argument(
        "--llm-api-key-env",
        default=settings.llm_api_key_env,
    )
    generation.add_argument(
        "--hyde-max-tokens",
        type=int,
        default=settings.hyde_max_tokens,
    )
    generation.add_argument(
        "--expansion-max-tokens",
        type=int,
        default=settings.expansion_max_tokens,
    )
    generation.add_argument(
        "--expansion-max-terms",
        type=int,
        default=settings.expansion_max_terms,
    )

    chunking = parser.add_argument_group("Chunking")
    chunking.add_argument(
        "--chunking-method",
        choices=CHUNKING_METHODS,
        default=settings.chunking_method,
    )
    chunking.add_argument(
        "--chunk-tokenizer-model",
        default=settings.chunk_tokenizer_model,
    )
    chunking.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=settings.chunk_max_tokens,
    )
    chunking.add_argument(
        "--chunk-overlap",
        type=float,
        default=settings.chunk_overlap_fraction,
        help="Fraction of the previous chunk repeated as whole sentences.",
    )
    chunking.add_argument(
        "--chunk-max-overlap-sentences",
        type=int,
        default=settings.chunk_max_overlap_sentences,
    )
    chunking.add_argument(
        "--min-chunk-words",
        type=int,
        default=settings.min_chunk_words,
    )

    reranking = parser.add_argument_group("Chunk reranking")
    reranking.add_argument(
        "--reranker",
        "--ranker",
        dest="reranker",
        choices=RERANKERS,
        default=settings.reranker,
    )
    reranking.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k,
    )
    reranking.add_argument(
        "--max-chunks-per-paper",
        type=int,
        default=settings.max_chunks_per_paper,
        help="Diversity preference; relaxed only when needed to fill top-k.",
    )
    reranking.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=settings.near_duplicate_threshold,
    )
    reranking.add_argument(
        "--medcpt-model",
        default=settings.medcpt_model,
    )
    reranking.add_argument(
        "--medcpt-max-length",
        type=int,
        default=settings.medcpt_max_length,
    )
    reranking.add_argument(
        "--medcpt-batch-size",
        type=int,
        default=settings.medcpt_batch_size,
    )
    reranking.add_argument(
        "--medcpt-device",
        default=settings.medcpt_device,
    )
    reranking.add_argument(
        "--bm25-k1",
        type=float,
        default=settings.bm25_k1,
    )
    reranking.add_argument(
        "--bm25-b",
        type=float,
        default=settings.bm25_b,
    )
    reranking.add_argument(
        "--hybrid-lexical-weight",
        type=float,
        default=settings.hybrid_lexical_weight,
    )
    reranking.add_argument(
        "--hybrid-medcpt-weight",
        type=float,
        default=settings.hybrid_medcpt_weight,
    )

    output = parser.add_argument_group("Output")
    output.add_argument("--output-dir", type=Path)
    output.add_argument("--output-json", type=Path)
    output.add_argument("--output-context", type=Path)
    output.add_argument("--show-query", action="store_true")
    output.add_argument("--show-info", action="store_true")
    output.add_argument("--print-context", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    question = args.question_option or args.question
    if not question:
        raise SystemExit(
            "Provide a question as a positional argument or with --question."
        )

    try:
        evidence = run_pipeline(
            question,
            retrieval_limit=args.retrieval_limit,
            query_strategy=args.query_strategy,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key_env=args.llm_api_key_env,
            hyde_max_tokens=args.hyde_max_tokens,
            expansion_max_tokens=args.expansion_max_tokens,
            expansion_max_terms=args.expansion_max_terms,
            paperclip_source=args.paperclip_source,
            paperclip_ranking=args.paperclip_ranking,
            paperclip_max_full_text_lines=args.paperclip_max_lines,
            paperclip_timeout=args.paperclip_timeout,
            paperclip_mode=args.paperclip_mode,
            paperclip_since=args.paperclip_since,
            paperclip_sort=args.paperclip_sort,
            paperclip_year=args.paperclip_year,
            paperclip_journal=args.paperclip_journal,
            paperclip_article_type=args.paperclip_article_type,
            paperclip_author=args.paperclip_author,
            paperclip_full_corpus=args.paperclip_full_corpus,
            chunking_method=args.chunking_method,
            chunk_tokenizer_model=args.chunk_tokenizer_model,
            chunk_max_tokens=args.chunk_max_tokens,
            chunk_overlap_fraction=args.chunk_overlap,
            chunk_max_overlap_sentences=(
                args.chunk_max_overlap_sentences
            ),
            min_chunk_words=args.min_chunk_words,
            reranker=args.reranker,
            top_k=args.top_k,
            max_chunks_per_paper=args.max_chunks_per_paper,
            near_duplicate_threshold=args.near_duplicate_threshold,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            medcpt_model=args.medcpt_model,
            medcpt_max_length=args.medcpt_max_length,
            medcpt_batch_size=args.medcpt_batch_size,
            medcpt_device=args.medcpt_device,
            hybrid_lexical_weight=args.hybrid_lexical_weight,
            hybrid_medcpt_weight=args.hybrid_medcpt_weight,
        )
    except (PaperclipError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.show_query:
        print(f"Query strategy: {evidence.query.strategy}")
        print(f"Search query: {evidence.query.search_query}")
        print()
    if args.show_info:
        print(to_json(evidence.pipeline))
        print()

    saved_paths: list[Path] = []
    if args.output_dir:
        saved_paths.extend(
            save_evidence_outputs(evidence, args.output_dir).values()
        )
    if args.output_json:
        saved_paths.append(save_json(evidence, args.output_json))
    if args.output_context:
        saved_paths.append(
            save_text(evidence.context_text, args.output_context)
        )

    if saved_paths:
        for path in saved_paths:
            print(f"Saved: {path}")
    elif args.print_context:
        print(evidence.context_text)
    else:
        print(to_json(evidence))


if __name__ == "__main__":
    main()
