"""Hybrid BM25 + MedCPT reranker.

Paper basis:
Stuhlmann, L., Saxer, M. A., and Fürst, J. (2025). Efficient and
Reproducible Biomedical Question Answering using Retrieval Augmented Generation.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from .config import (
    BM25_B,
    BM25_K1,
    HYBRID_LEXICAL_WEIGHT,
    HYBRID_MEDCPT_WEIGHT,
    MEDCPT_ARTICLE_ENCODER_MODEL,
    MEDCPT_BATCH_SIZE,
    MEDCPT_QUERY_ENCODER_MODEL,
)
from .lexical_reranker import bm25_scores
from .medcpt_embedding_reranker import medcpt_embedding_scores


def min_max_normalize(values: Sequence[float]) -> List[float]:
    values = [float(value) for value in values]
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


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


def hybrid_scores(
    question: str,
    snippets: Sequence[Any],
    *,
    lexical_weight: float = HYBRID_LEXICAL_WEIGHT,
    medcpt_weight: float = HYBRID_MEDCPT_WEIGHT,
    bm25_k1: float = BM25_K1,
    bm25_b: float = BM25_B,
    query_model_name: str = MEDCPT_QUERY_ENCODER_MODEL,
    article_model_name: str = MEDCPT_ARTICLE_ENCODER_MODEL,
    batch_size: int = MEDCPT_BATCH_SIZE,
    device: str | None = None,
) -> List[float]:
    candidates = list(snippets or [])
    if not candidates:
        return []

    if lexical_weight < 0 or medcpt_weight < 0:
        raise ValueError("Hybrid weights must be non-negative.")

    total_weight = lexical_weight + medcpt_weight
    if total_weight <= 0:
        raise ValueError("At least one hybrid weight must be greater than zero.")

    lexical_weight = lexical_weight / total_weight
    medcpt_weight = medcpt_weight / total_weight

    lexical = min_max_normalize(bm25_scores(question, candidates, k1=bm25_k1, b=bm25_b))
    medcpt = min_max_normalize(
        medcpt_embedding_scores(
            question,
            candidates,
            query_model_name=query_model_name,
            article_model_name=article_model_name,
            batch_size=batch_size,
            device=device,
        )
    )

    return [
        lexical_weight * lexical_score + medcpt_weight * medcpt_score
        for lexical_score, medcpt_score in zip(lexical, medcpt)
    ]


def rank_hybrid_snippets(
    question: str,
    snippets: Sequence[Any],
    *,
    top_k: int | None = None,
    lexical_weight: float = HYBRID_LEXICAL_WEIGHT,
    medcpt_weight: float = HYBRID_MEDCPT_WEIGHT,
    bm25_k1: float = BM25_K1,
    bm25_b: float = BM25_B,
    query_model_name: str = MEDCPT_QUERY_ENCODER_MODEL,
    article_model_name: str = MEDCPT_ARTICLE_ENCODER_MODEL,
    batch_size: int = MEDCPT_BATCH_SIZE,
    device: str | None = None,
) -> List[Any]:
    candidates = list(snippets or [])
    scores = hybrid_scores(
        question,
        candidates,
        lexical_weight=lexical_weight,
        medcpt_weight=medcpt_weight,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        query_model_name=query_model_name,
        article_model_name=article_model_name,
        batch_size=batch_size,
        device=device,
    )

    scored: List[Tuple[float, int, Any]] = []
    for index, (snippet, score) in enumerate(zip(candidates, scores)):
        set_ranking_metadata(snippet, score, "hybrid")
        scored.append((score, index, snippet))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [snippet for _, _, snippet in scored]
    return ranked[:top_k] if top_k is not None else ranked


__all__ = [
    "hybrid_scores",
    "min_max_normalize",
    "rank_hybrid_snippets",
]
