from __future__ import annotations

from .config import RERANKERS, Settings
from .hybrid_reranker import rerank_hybrid
from .lexical_reranker import rerank_lexical
from .medcpt_reranker import rerank_medcpt
from .models import TextChunk
from .text_utils import normalize_for_deduplication, word_count


def _clean_candidates(chunks: list[TextChunk], min_words: int) -> list[TextChunk]:
    output: list[TextChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if word_count(chunk.text) < min_words:
            continue
        key = normalize_for_deduplication(chunk.text)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(chunk)
    return output


def rerank_chunks(
    question: str,
    chunks: list[TextChunk],
    *,
    settings: Settings,
) -> list[TextChunk]:
    selected = settings.reranker.strip().casefold().replace("-", "_")
    if selected not in RERANKERS:
        raise ValueError(f"Unknown reranker '{selected}'. Choose from: {', '.join(RERANKERS)}")

    candidates = _clean_candidates(chunks, settings.min_chunk_words)
    if selected == "lexical":
        return rerank_lexical(question, candidates, k1=settings.bm25_k1, b=settings.bm25_b)
    if selected == "medcpt":
        return rerank_medcpt(
            question,
            candidates,
            query_model_name=settings.medcpt_query_model,
            article_model_name=settings.medcpt_article_model,
            batch_size=settings.medcpt_batch_size,
            device=settings.medcpt_device,
        )
    return rerank_hybrid(
        question,
        candidates,
        lexical_weight=settings.hybrid_lexical_weight,
        medcpt_weight=settings.hybrid_medcpt_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        query_model_name=settings.medcpt_query_model,
        article_model_name=settings.medcpt_article_model,
        batch_size=settings.medcpt_batch_size,
        device=settings.medcpt_device,
    )
