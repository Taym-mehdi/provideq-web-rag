from __future__ import annotations

import argparse
import sys
import textwrap

import requests

from web_rag.config import get_settings
from web_rag.query_builder import build_europe_pmc_query
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
        "--show-query",
        action="store_true",
        help="Print the generated Europe PMC query."
    )

    return parser


def print_papers(question: str, search_query: str, papers: list) -> None:
    print("\n=== ProvideQ Web RAG: External Search ===\n")
    print(f"Question: {question}")

    if search_query:
        print(f"Search query: {search_query}")

    print(f"\nRetrieved papers: {len(papers)}\n")

    if not papers:
        print("No papers were retrieved.")
        return

    for index, paper in enumerate(papers, start=1):
        print(f"[{index}] {paper.title}")

        metadata = []

        if paper.journal:
            metadata.append(paper.journal)

        if paper.year:
            metadata.append(paper.year)

        if paper.doi:
            metadata.append(f"DOI: {paper.doi}")
        elif paper.ext_id:
            metadata.append(f"{paper.source}:{paper.ext_id}")

        if metadata:
            print(f"    {' | '.join(metadata)}")

        if paper.url:
            print(f"    URL: {paper.url}")

        if paper.abstract:
            short_abstract = textwrap.shorten(
                paper.abstract,
                width=350,
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

        print_papers(
            question=query.original_question,
            search_query=query.search_query if args.show_query else "",
            papers=papers,
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