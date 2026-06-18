"""BM25 lexical reranker.

Paper basis:
Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework:
BM25 and Beyond. Foundations and Trends in Information Retrieval.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, List, Sequence, Tuple

from .config import BM25_B, BM25_K1

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def tokenize_for_ranking(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in _TOKEN_PATTERN.findall(str(text))]


def get_snippet_text(snippet: Any) -> str:
    if snippet is None:
        return ""
    if isinstance(snippet, dict):
        for key in ("evidence_text", "text", "snippet", "snippet_text", "content", "passage", "abstract"):
            if snippet.get(key):
                return str(snippet[key])
        return ""
    for attr in ("evidence_text", "text", "snippet", "snippet_text", "content", "passage", "abstract"):
        value = getattr(snippet, attr, None)
        if value:
            return str(value)
    return ""


def get_snippet_title(snippet: Any) -> str:
    if snippet is None:
        return ""
    if isinstance(snippet, dict):
        paper = snippet.get("paper") or snippet.get("source") or {}
        if isinstance(paper, dict):
            return str(paper.get("title", "") or "")
        return str(getattr(paper, "title", "") or "")
    if getattr(snippet, "title", None):
        return str(getattr(snippet, "title"))
    paper = getattr(snippet, "paper", None) or getattr(snippet, "source", None)
    if isinstance(paper, dict):
        return str(paper.get("title", "") or "")
    return str(getattr(paper, "title", "") or "")


def get_ranking_document(snippet: Any) -> str:
    title = get_snippet_title(snippet)
    text = get_snippet_text(snippet)
    return f"{title} {text}".strip()


def set_ranking_metadata(snippet: Any, score: float, ranker: str) -> None:
    for attr, value in (
        ("score", float(score)),
        ("ranking_score", float(score)),
        ("ranker", ranker),
        ("ranking_method", ranker),
    ):
        try:
            setattr(snippet, attr, value)
        except Exception:
            pass


def bm25_scores(
    question: str,
    snippets: Sequence[Any],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> List[float]:
    query_terms = tokenize_for_ranking(question)
    candidates = list(snippets or [])

    if not candidates:
        return []
    if not query_terms:
        return [0.0 for _ in candidates]

    documents = [tokenize_for_ranking(get_ranking_document(snippet)) for snippet in candidates]
    lengths = [len(document) for document in documents]
    average_length = sum(lengths) / max(len(lengths), 1)
    average_length = average_length or 1.0

    document_count = len(documents)
    document_frequencies: Counter[str] = Counter()
    for document in documents:
        for term in set(document):
            document_frequencies[term] += 1

    scores: List[float] = []
    for document, length in zip(documents, lengths):
        term_frequencies = Counter(document)
        score = 0.0

        for term in query_terms:
            tf = term_frequencies.get(term, 0)
            if tf == 0:
                continue

            df = document_frequencies.get(term, 0)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * length / average_length)
            score += idf * (tf * (k1 + 1.0)) / denominator

        scores.append(float(score))

    return scores


def rank_lexical_snippets(
    question: str,
    snippets: Sequence[Any],
    *,
    top_k: int | None = None,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> List[Any]:
    candidates = list(snippets or [])
    scores = bm25_scores(question, candidates, k1=k1, b=b)

    scored: List[Tuple[float, int, Any]] = []
    for index, (snippet, score) in enumerate(zip(candidates, scores)):
        set_ranking_metadata(snippet, score, "lexical")
        scored.append((score, index, snippet))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [snippet for _, _, snippet in scored]
    return ranked[:top_k] if top_k is not None else ranked


__all__ = [
    "bm25_scores",
    "get_ranking_document",
    "get_snippet_text",
    "get_snippet_title",
    "rank_lexical_snippets",
    "set_ranking_metadata",
    "tokenize_for_ranking",
]
