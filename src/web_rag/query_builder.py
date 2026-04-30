from __future__ import annotations

from web_rag.models import QueryBundle
from web_rag.text_utils import extract_keywords, normalize_question


def escape_query_text(text: str) -> str:
    """
    Escape double quotes so the question can safely be used inside
    a quoted Europe PMC query clause.
    """
    return text.replace('"', "")


def build_keyword_clause(keywords: list[str], max_terms: int = 8) -> str:
    """
    Build a simple AND-based keyword clause.

    Example:
        ["interleukin-6", "plasma", "room", "temperature"]

    becomes:
        interleukin-6 AND plasma AND room AND temperature
    """
    selected_keywords = keywords[:max_terms]
    return " AND ".join(selected_keywords)


def build_europe_pmc_query(question: str) -> QueryBundle:
    """
    Build the first Europe PMC query representation.

    Strategy:
    - keep the original user question
    - normalize the question
    - extract simple biomedical keyword candidates
    - combine a full-question clause with a keyword clause

    This is intentionally simple and transparent. Later, we can compare this
    baseline query formulation against improved query methods.
    """
    normalized_question = normalize_question(question)
    keywords = extract_keywords(normalized_question)

    if not keywords:
        search_query = f'TITLE_ABS:"{escape_query_text(normalized_question)}"'
    else:
        quoted_question = f'TITLE_ABS:"{escape_query_text(normalized_question)}"'
        keyword_clause = build_keyword_clause(keywords)
        keyword_query = f"TITLE_ABS:({keyword_clause})"

        search_query = f"({quoted_question}) OR ({keyword_query})"

    return QueryBundle(
        original_question=question,
        normalized_question=normalized_question,
        keywords=keywords,
        search_query=search_query,
    )