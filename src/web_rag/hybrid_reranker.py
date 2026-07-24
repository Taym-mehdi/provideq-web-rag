from __future__ import annotations

from .lexical_reranker import bm25_scores
from .medcpt_reranker import medcpt_scores
from .models import TextChunk


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.0] * len(values)
    return [(value - minimum) / (maximum - minimum) for value in values]


def rerank_hybrid(
    question: str,
    chunks: list[TextChunk],
    *,
    lexical_weight: float,
    medcpt_weight: float,
    bm25_k1: float,
    bm25_b: float,
    query_model_name: str,
    article_model_name: str,
    batch_size: int,
    device: str,
) -> list[TextChunk]:
    if lexical_weight < 0 or medcpt_weight < 0:
        raise ValueError("hybrid weights must be non-negative")
    total_weight = lexical_weight + medcpt_weight
    if total_weight <= 0:
        raise ValueError("at least one hybrid weight must be greater than 0")

    lexical_weight /= total_weight
    medcpt_weight /= total_weight
    lexical_scores = _min_max(bm25_scores(question, chunks, k1=bm25_k1, b=bm25_b))
    dense_scores = _min_max(
        medcpt_scores(
            question,
            chunks,
            query_model_name=query_model_name,
            article_model_name=article_model_name,
            batch_size=batch_size,
            device=device,
        )
    )

    for chunk, lexical_score, dense_score in zip(chunks, lexical_scores, dense_scores):
        score = lexical_weight * lexical_score + medcpt_weight * dense_score
        chunk.score = score
        chunk.score_components = {
            "bm25_normalized": lexical_score,
            "medcpt_normalized": dense_score,
            "lexical_weight": lexical_weight,
            "medcpt_weight": medcpt_weight,
            "hybrid": score,
        }
    return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
