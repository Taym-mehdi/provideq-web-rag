from __future__ import annotations

import argparse

from web_rag.config import get_settings
from web_rag.models import QueryBundle
from web_rag.text_utils import extract_keywords, normalize_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ProvideQ Web RAG command-line entry point."
    )
    parser.add_argument(
        "question",
        help="Biomedical question to inspect."
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings()
    normalized_question = normalize_question(args.question)
    keywords = extract_keywords(args.question)

    query = QueryBundle(
        original_question=args.question,
        normalized_question=normalized_question,
        keywords=keywords,
        search_query=""
    )

    print("=== ProvideQ Web RAG ===")
    print(f"Question: {query.original_question}")
    print(f"Normalized: {query.normalized_question}")
    print(f"Keywords: {', '.join(query.keywords) if query.keywords else 'None'}")
    print(f"Default page size: {settings.default_page_size}")
    print(f"Default top-k: {settings.default_top_k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())