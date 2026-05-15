from __future__ import annotations

import argparse
import sys
import textwrap

import requests

from web_rag.config import get_settings
from web_rag.context_builder import build_evidence_pack
from web_rag.query_builder import build_europe_pmc_query
from web_rag.ranker import rank_snippets
from web_rag.serializer import (
    evidence_pack_to_dict,
    save_context_text,
    save_json_output,
    to_pretty_json,
)
from web_rag.snippet_extractor import extract_snippets
from web_rag.source_client import search_europe_pmc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ProvideQ Web RAG command-line entry point."
    )

    parser.add_argument(
        "question",
        help="Biomedical question to search."
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="Number of papers to retrieve from Europe PMC."
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Number of sentences per evidence snippet."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of ranked evidence snippets to show."
    )

    parser.add_argument(
        "--ranker",
        choices=["lexical", "medcpt-hybrid"],
        default=None,
        help="Ranking method to use."
    )

    parser.add_argument(
        "--show-query",
        action="store_true",
        help="Print the generated Europe PMC query."
    )

    parser.add_argument(
        "--show-papers",
        action="store_true",
        help="Print retrieved papers before showing ranked evidence."
    )

    parser.add_argument(
        "--show-all-snippets",
        action="store_true",
        help="Print extracted snippets before ranking."
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the final evidence context block."
    )

    parser.add_argument(
        "--show-score-components",
        action="store_true",
        help="Print ranking score components for each evidence record."
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete result as JSON."
    )

    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save the complete JSON result."
    )

    parser.add_argument(
        "--save-context",
        default="",
        help="Optional path to save only the final evidence context text."
    )

    parser.add_argument(
        "--max-unranked-snippets",
        type=int,
        default=10,
        help="Maximum number of unranked snippets to print when --show-all-snippets is used."
    )

    return parser


def format_paper_metadata(paper) -> str:
    metadata = []

    if paper.journal:
        metadata.append(paper.journal)

    if paper.year:
        metadata.append(paper.year)

    if paper.doi:
        metadata.append(f"DOI: {paper.doi}")
    elif paper.ext_id:
        metadata.append(f"{paper.source}:{paper.ext_id}")

    return " | ".join(metadata)


def print_papers(papers: list) -> None:
    print("\nRetrieved papers:\n")

    if not papers:
        print("No papers were retrieved.")
        return

    for index, paper in enumerate(papers, start=1):
        print(f"[{index}] {paper.title}")

        metadata = format_paper_metadata(paper)
        if metadata:
            print(f"    {metadata}")

        if paper.url:
            print(f"    URL: {paper.url}")

        if paper.abstract:
            short_abstract = textwrap.shorten(
                paper.abstract,
                width=300,
                placeholder="..."
            )
            print(textwrap.fill(
                f"    Abstract: {short_abstract}",
                width=100,
                subsequent_indent="              "
            ))
        else:
            print("    Abstract: N/A")

        print()


def print_unranked_snippets(snippets: list, max_snippets: int) -> None:
    print("\nExtracted snippets before ranking:\n")

    if not snippets:
        print("No snippets could be extracted.")
        return

    shown_snippets = snippets[:max_snippets]

    for index, snippet in enumerate(shown_snippets, start=1):
        paper = snippet.paper

        print(f"[{index}] {paper.title}")

        metadata = format_paper_metadata(paper)
        if metadata:
            print(f"    {metadata}")

        print(textwrap.fill(
            f"    Evidence: {snippet.text}",
            width=100,
            subsequent_indent="              "
        ))

        print()

    remaining = len(snippets) - len(shown_snippets)

    if remaining > 0:
        print(f"... {remaining} additional unranked snippets not shown.")


def print_score_components(score_components: dict) -> None:
    if not score_components:
        return

    compact_keys = [
        "ranker",
        "final_score",
        "medcpt_raw",
        "medcpt_norm",
        "lexical_raw",
        "lexical_norm",
        "slot_score",
    ]

    visible_parts = []

    for key in compact_keys:
        if key in score_components:
            visible_parts.append(f"{key}={score_components[key]}")

    if visible_parts:
        print(f"    Components: {', '.join(visible_parts)}")


def print_evidence_pack(evidence_pack, show_score_components: bool = False) -> None:
    print("\nTop ranked evidence pack:\n")

    if not evidence_pack.records:
        print("No ranked evidence records are available.")
        return

    for record in evidence_pack.records:
        print(f"[{record.citation_id}] {record.title}")
        print(f"    Score: {record.score:.4f}")

        if show_score_components:
            print_score_components(record.score_components)

        metadata = []

        if record.journal:
            metadata.append(record.journal)

        if record.year:
            metadata.append(record.year)

        if record.doi:
            metadata.append(f"DOI: {record.doi}")
        elif record.source and record.ext_id:
            metadata.append(f"{record.source}:{record.ext_id}")

        if metadata:
            print(f"    {' | '.join(metadata)}")

        if record.url:
            print(f"    URL: {record.url}")

        print(textwrap.fill(
            f"    Evidence: {record.evidence_text}",
            width=100,
            subsequent_indent="              "
        ))

        print()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings()
    ranking_method = args.ranker or settings.default_ranker

    try:
        query = build_europe_pmc_query(args.question)

        papers = search_europe_pmc(
            query=query,
            page_size=args.page_size or settings.default_page_size,
        )

        snippets = extract_snippets(
            papers=papers,
            window_size=args.window_size or settings.snippet_window,
        )

        ranked_snippets = rank_snippets(
            question=query.original_question,
            snippets=snippets,
            top_k=args.top_k or settings.default_top_k,
            method=ranking_method,
        )

        evidence_pack = build_evidence_pack(
            question=query.original_question,
            ranked_snippets=ranked_snippets,
        )

        result_payload = evidence_pack_to_dict(
            evidence_pack=evidence_pack,
            query=query,
            retrieved_papers_count=len(papers),
            extracted_snippets_count=len(snippets),
            ranking_method=ranking_method,
        )

        if args.save_json:
            save_json_output(
                payload=result_payload,
                output_path=args.save_json,
            )

        if args.save_context:
            save_context_text(
                evidence_pack=evidence_pack,
                output_path=args.save_context,
            )

        if args.json:
            print(to_pretty_json(result_payload))
            return 0

        print("\n=== ProvideQ Web RAG: Improved Evidence Reranking ===\n")
        print(f"Question: {query.original_question}")
        print(f"Ranking method: {ranking_method}")

        if args.show_query:
            print(f"Search query: {query.search_query}")

        print(f"Retrieved papers: {len(papers)}")
        print(f"Extracted snippets: {len(snippets)}")
        print(f"Evidence records: {len(evidence_pack.records)}")

        if args.save_json:
            print(f"Saved JSON output: {args.save_json}")

        if args.save_context:
            print(f"Saved context output: {args.save_context}")

        if args.show_papers:
            print_papers(papers)

        if args.show_all_snippets:
            print_unranked_snippets(
                snippets=snippets,
                max_snippets=args.max_unranked_snippets,
            )

        print_evidence_pack(
            evidence_pack=evidence_pack,
            show_score_components=args.show_score_components,
        )

        if args.show_context:
            print("\nFinal context block:\n")
            print(evidence_pack.context_text)

        return 0

    except requests.HTTPError as error:
        print(f"HTTP error while searching Europe PMC: {error}", file=sys.stderr)

        if error.response is not None:
            print(error.response.text[:1000], file=sys.stderr)

        return 1

    except requests.RequestException as error:
        print(f"Network error while searching Europe PMC: {error}", file=sys.stderr)
        return 1

    except Exception as error:
        print(f"Unexpected error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())