"""Command-line interface for the ProvideQ Web RAG retrieval baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import get_settings
from .context_builder import build_evidence_pack
from .query_builder import build_europe_pmc_query
from .ranker import RANKER_CHOICES, rank_snippets
from .serializer import evidence_pack_to_json, get_context_text, save_context_text, save_evidence_outputs, save_evidence_pack
from .snippet_extractor import extract_snippets
from .source_client import search_europe_pmc


def _search_papers(query: str, page_size: int) -> list[Any]:
    try:
        return list(search_europe_pmc(query, page_size=page_size))
    except TypeError:
        try:
            return list(search_europe_pmc(query, pageSize=page_size))
        except TypeError:
            return list(search_europe_pmc(query))


def _extract_snippets(papers: list[Any], window_size: int, stride: int) -> list[Any]:
    try:
        return list(extract_snippets(papers, window_size=window_size, stride=stride))
    except TypeError:
        try:
            return list(extract_snippets(papers, window=window_size, stride=stride))
        except TypeError:
            return list(extract_snippets(papers))


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the ProvideQ Web RAG evidence retrieval pipeline.")

    parser.add_argument("question", nargs="?", help="Biomedical/pre-analytical question.")
    parser.add_argument("--question", dest="question_flag", help="Biomedical/pre-analytical question.")
    parser.add_argument("--ranker", default=settings.default_ranker, choices=RANKER_CHOICES)

    parser.add_argument("--page-size", type=int, default=settings.page_size)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--snippet-window-size", type=int, default=settings.snippet_window_size)
    parser.add_argument("--snippet-stride", type=int, default=settings.snippet_stride)
    parser.add_argument("--min-snippet-word-count", type=int, default=settings.min_snippet_word_count)
    parser.add_argument("--max-context-chars", type=int, default=None)

    parser.add_argument("--bm25-k1", type=float, default=settings.bm25_k1)
    parser.add_argument("--bm25-b", type=float, default=settings.bm25_b)

    parser.add_argument("--medcpt-batch-size", type=int, default=settings.medcpt_batch_size)
    parser.add_argument("--medcpt-device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    parser.add_argument("--medcpt-query-model", default=settings.medcpt_query_encoder_model)
    parser.add_argument("--medcpt-article-model", default=settings.medcpt_article_encoder_model)

    parser.add_argument("--hybrid-lexical-weight", type=float, default=settings.hybrid_lexical_weight)
    parser.add_argument("--hybrid-medcpt-weight", type=float, default=settings.hybrid_medcpt_weight)

    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-context", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-query", action="store_true")
    parser.add_argument("--print-context", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question_flag or args.question
    if not question:
        raise SystemExit("Please provide a question as a positional argument or with --question.")

    query = build_europe_pmc_query(question)
    if args.show_query:
        print(f"Europe PMC query:\n{query}\n")

    papers = _search_papers(query, args.page_size)
    snippets = _extract_snippets(papers, args.snippet_window_size, args.snippet_stride)

    ranked_snippets = rank_snippets(
        question=question,
        snippets=snippets,
        ranker=args.ranker,
        top_k=args.top_k,
        min_snippet_word_count=args.min_snippet_word_count,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        lexical_weight=args.hybrid_lexical_weight,
        medcpt_weight=args.hybrid_medcpt_weight,
        query_model_name=args.medcpt_query_model,
        article_model_name=args.medcpt_article_model,
        batch_size=args.medcpt_batch_size,
        device=args.medcpt_device,
    )

    evidence_pack = build_evidence_pack(
        question=question,
        query=query,
        ranked_snippets=ranked_snippets,
        max_context_chars=args.max_context_chars,
    )

    saved_paths: list[Path] = []
    if args.output_dir:
        outputs = save_evidence_outputs(evidence_pack, args.output_dir)
        saved_paths.extend(outputs.values())
    if args.output_json:
        saved_paths.append(save_evidence_pack(evidence_pack, args.output_json))
    if args.output_context:
        saved_paths.append(save_context_text(evidence_pack, args.output_context))

    if saved_paths:
        for path in saved_paths:
            print(f"Saved: {path}")
    elif args.print_context:
        print(get_context_text(evidence_pack))
    else:
        print(evidence_pack_to_json(evidence_pack))


if __name__ == "__main__":
    main()
