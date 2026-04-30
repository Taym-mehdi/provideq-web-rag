from __future__ import annotations

import argparse
import sys
import textwrap

import requests

from web_rag.config import get_settings
from web_rag.query_builder import build_europe_pmc_query
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
        "--show-query",
        action="store_true",
        help="Print the generated Europe PMC query."
    )

    parser.add_argument(
        "--show-papers",
        action="store_true",
        help="Print retrieved papers before showing snippets."
    )

    parser.add_argument(
        "--max-snippets",
        type=int,
        default=10,
        help="Maximum number of extracted snippets to print."
    )

    return parser


def format_paper_metadata(paper) -> str:
    """
    Build a compact metadata line for one paper.
    """
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


def print_snippets(snippets: list, max_snippets: int) -> None:
    print("\nExtracted evidence snippets:\n")

    if not snippets:
        print("No snippets could be extracted from the retrieved papers.")
        return

    shown_snippets = snippets[:max_snippets]

    for index, snippet in enumerate(shown_snippets, start=1):
        paper = snippet.paper

        print(f"[{index}] {paper.title}")

        metadata = format_paper_metadata(paper)
        if metadata:
            print(f"    {metadata}")

        if paper.url:
            print(f"    URL: {paper.url}")

        print(textwrap.fill(
            f"    Evidence: {snippet.text}",
            width=100,
            subsequent_indent="              "
        ))

        print()

    remaining = len(snippets) - len(shown_snippets)

    if remaining > 0:
        print(f"... {remaining} additional snippets not shown.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings()

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

        print("\n=== ProvideQ Web RAG: Evidence Extraction ===\n")
        print(f"Question: {query.original_question}")

        if args.show_query:
            print(f"Search query: {query.search_query}")

        print(f"Retrieved papers: {len(papers)}")
        print(f"Extracted snippets: {len(snippets)}")

        if args.show_papers:
            print_papers(papers)

        print_snippets(
            snippets=snippets,
            max_snippets=args.max_snippets,
        )

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