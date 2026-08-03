from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable
from typing import Any

from .chunking import chunk_papers
from .config import Settings, get_settings, validate_settings
from .context_builder import build_evidence_pack
from .evidence_selection import select_evidence
from .models import EvidencePack, PipelineInfo
from .paperclip_retriever import retrieve_papers
from .query_reformulation import reformulate_query
from .reranking import rerank_chunks


def run_pipeline(
    question: str,
    *,
    retrieval_limit: int | None = None,
    query_strategy: str | None = None,
    hyde_model: str | None = None,
    hyde_base_url: str | None = None,
    hyde_temperature: float | None = None,
    hyde_max_tokens: int | None = None,
    hyde_seed: int | None = None,
    hyde_timeout: float | None = None,
    hyde_generator: Callable[[str], str] | None = None,
    expansion_model: str | None = None,
    expansion_base_url: str | None = None,
    expansion_temperature: float | None = None,
    expansion_max_tokens: int | None = None,
    expansion_seed: int | None = None,
    expansion_timeout: float | None = None,
    expansion_max_terms: int | None = None,
    expansion_max_query_chars: int | None = None,
    expansion_generator: Callable[[str], str] | None = None,
    paperclip_source: str | None = None,
    paperclip_ranking: str | None = None,
    paperclip_max_full_text_lines: int | None = None,
    paperclip_mode: str | None = None,
    paperclip_since: str | None = None,
    paperclip_sort: str | None = None,
    paperclip_year: int | str | None = None,
    paperclip_journal: str | None = None,
    paperclip_article_type: str | None = None,
    paperclip_author: str | None = None,
    paperclip_full_corpus: bool = False,
    chunking_method: str | None = None,
    chunk_window_size: int | None = None,
    chunk_stride: int | None = None,
    min_chunk_chars: int | None = None,
    max_chunk_chars: int | None = None,
    min_chunk_words: int | None = None,
    context_backoff: bool | None = None,
    reranker: str | None = None,
    top_k: int | None = None,
    max_chunks_per_paper: int | None = None,
    near_duplicate_threshold: float | None = None,
    bm25_k1: float | None = None,
    bm25_b: float | None = None,
    medcpt_query_model: str | None = None,
    medcpt_article_model: str | None = None,
    medcpt_batch_size: int | None = None,
    medcpt_device: str | None = None,
    hybrid_lexical_weight: float | None = None,
    hybrid_medcpt_weight: float | None = None,
    settings: Settings | None = None,
    paperclip_client: Any | None = None,
) -> EvidencePack:
    base = settings or get_settings()
    effective = replace(
        base,
        retrieval_limit=retrieval_limit if retrieval_limit is not None else base.retrieval_limit,
        query_strategy=query_strategy or base.query_strategy,
        hyde_model=hyde_model or base.hyde_model,
        hyde_base_url=hyde_base_url or base.hyde_base_url,
        hyde_temperature=(
            hyde_temperature if hyde_temperature is not None else base.hyde_temperature
        ),
        hyde_max_tokens=hyde_max_tokens if hyde_max_tokens is not None else base.hyde_max_tokens,
        hyde_seed=hyde_seed if hyde_seed is not None else base.hyde_seed,
        hyde_timeout=hyde_timeout if hyde_timeout is not None else base.hyde_timeout,
        expansion_model=expansion_model or base.expansion_model,
        expansion_base_url=expansion_base_url or base.expansion_base_url,
        expansion_temperature=(
            expansion_temperature
            if expansion_temperature is not None
            else base.expansion_temperature
        ),
        expansion_max_tokens=(
            expansion_max_tokens
            if expansion_max_tokens is not None
            else base.expansion_max_tokens
        ),
        expansion_seed=expansion_seed if expansion_seed is not None else base.expansion_seed,
        expansion_timeout=(
            expansion_timeout if expansion_timeout is not None else base.expansion_timeout
        ),
        expansion_max_terms=(
            expansion_max_terms
            if expansion_max_terms is not None
            else base.expansion_max_terms
        ),
        expansion_max_query_chars=(
            expansion_max_query_chars
            if expansion_max_query_chars is not None
            else base.expansion_max_query_chars
        ),
        paperclip_source=paperclip_source or base.paperclip_source,
        paperclip_ranking=paperclip_ranking or base.paperclip_ranking,
        paperclip_max_full_text_lines=(
            paperclip_max_full_text_lines
            if paperclip_max_full_text_lines is not None
            else base.paperclip_max_full_text_lines
        ),
        chunking_method=chunking_method or base.chunking_method,
        chunk_window_size=chunk_window_size if chunk_window_size is not None else base.chunk_window_size,
        chunk_stride=chunk_stride if chunk_stride is not None else base.chunk_stride,
        min_chunk_chars=min_chunk_chars if min_chunk_chars is not None else base.min_chunk_chars,
        max_chunk_chars=max_chunk_chars if max_chunk_chars is not None else base.max_chunk_chars,
        min_chunk_words=min_chunk_words if min_chunk_words is not None else base.min_chunk_words,
        context_backoff=context_backoff if context_backoff is not None else base.context_backoff,
        reranker=reranker or base.reranker,
        top_k=top_k if top_k is not None else base.top_k,
        max_chunks_per_paper=(
            max_chunks_per_paper if max_chunks_per_paper is not None else base.max_chunks_per_paper
        ),
        near_duplicate_threshold=(
            near_duplicate_threshold
            if near_duplicate_threshold is not None
            else base.near_duplicate_threshold
        ),
        bm25_k1=bm25_k1 if bm25_k1 is not None else base.bm25_k1,
        bm25_b=bm25_b if bm25_b is not None else base.bm25_b,
        medcpt_query_model=medcpt_query_model or base.medcpt_query_model,
        medcpt_article_model=medcpt_article_model or base.medcpt_article_model,
        medcpt_batch_size=(
            medcpt_batch_size if medcpt_batch_size is not None else base.medcpt_batch_size
        ),
        medcpt_device=medcpt_device or base.medcpt_device,
        hybrid_lexical_weight=(
            hybrid_lexical_weight
            if hybrid_lexical_weight is not None
            else base.hybrid_lexical_weight
        ),
        hybrid_medcpt_weight=(
            hybrid_medcpt_weight
            if hybrid_medcpt_weight is not None
            else base.hybrid_medcpt_weight
        ),
    )
    validate_settings(effective)

    query = reformulate_query(
        question,
        effective.query_strategy,
        hyde_model=effective.hyde_model,
        hyde_base_url=effective.hyde_base_url,
        hyde_temperature=effective.hyde_temperature,
        hyde_max_tokens=effective.hyde_max_tokens,
        hyde_seed=effective.hyde_seed,
        hyde_timeout=effective.hyde_timeout,
        hyde_generator=hyde_generator,
        expansion_model=effective.expansion_model,
        expansion_base_url=effective.expansion_base_url,
        expansion_temperature=effective.expansion_temperature,
        expansion_max_tokens=effective.expansion_max_tokens,
        expansion_seed=effective.expansion_seed,
        expansion_timeout=effective.expansion_timeout,
        expansion_max_terms=effective.expansion_max_terms,
        expansion_max_query_chars=effective.expansion_max_query_chars,
        expansion_generator=expansion_generator,
    )
    retrieval = retrieve_papers(
        query.search_query,
        limit=effective.retrieval_limit,
        source=effective.paperclip_source,
        ranking=effective.paperclip_ranking,
        max_full_text_lines=effective.paperclip_max_full_text_lines,
        mode=paperclip_mode,
        since=paperclip_since,
        sort=paperclip_sort,
        year=paperclip_year,
        journal=paperclip_journal,
        article_type=paperclip_article_type,
        author=paperclip_author,
        full_corpus=paperclip_full_corpus,
        timeout=effective.paperclip_timeout,
        client=paperclip_client,
    )
    chunks = chunk_papers(
        retrieval.papers,
        method=effective.chunking_method,
        window_size=effective.chunk_window_size,
        stride=effective.chunk_stride,
        min_chars=effective.min_chunk_chars,
        max_chars=effective.max_chunk_chars,
        min_words=effective.min_chunk_words,
        context_backoff=effective.context_backoff,
    )
    ranked = rerank_chunks(question, chunks, settings=effective)
    selected = select_evidence(
        ranked,
        top_k=effective.top_k,
        max_chunks_per_paper=effective.max_chunks_per_paper,
        near_duplicate_threshold=effective.near_duplicate_threshold,
    )

    info = PipelineInfo(
        retrieval_system="paperclip",
        paperclip_source=effective.paperclip_source,
        paperclip_ranking=effective.paperclip_ranking,
        paperclip_result_id=retrieval.result_id,
        retrieval_limit=effective.retrieval_limit,
        retrieved_papers_count=len(retrieval.papers),
        full_text_papers_count=sum(
            bool(paper.metadata.get("has_full_text")) for paper in retrieval.papers
        ),
        chunking_method=effective.chunking_method,
        extracted_chunks_count=len(chunks),
        reranker=effective.reranker,
        top_k=effective.top_k,
        returned_evidence_count=len(selected),
        parameters={
            "query_strategy": effective.query_strategy,
            "hyde_model": effective.hyde_model if effective.query_strategy == "hyde" else None,
            "hyde_temperature": (
                effective.hyde_temperature if effective.query_strategy == "hyde" else None
            ),
            "hyde_max_tokens": (
                effective.hyde_max_tokens if effective.query_strategy == "hyde" else None
            ),
            "hyde_seed": effective.hyde_seed if effective.query_strategy == "hyde" else None,
            "expansion_model": (
                effective.expansion_model
                if effective.query_strategy == "llmexpand"
                else None
            ),
            "expansion_temperature": (
                effective.expansion_temperature
                if effective.query_strategy == "llmexpand"
                else None
            ),
            "expansion_max_tokens": (
                effective.expansion_max_tokens
                if effective.query_strategy == "llmexpand"
                else None
            ),
            "expansion_seed": (
                effective.expansion_seed
                if effective.query_strategy == "llmexpand"
                else None
            ),
            "expansion_max_terms": (
                effective.expansion_max_terms
                if effective.query_strategy == "llmexpand"
                else None
            ),
            "expansion_max_query_chars": (
                effective.expansion_max_query_chars
                if effective.query_strategy == "llmexpand"
                else None
            ),
            "paperclip_mode": paperclip_mode,
            "paperclip_since": paperclip_since,
            "paperclip_sort": paperclip_sort,
            "paperclip_year": paperclip_year,
            "paperclip_journal": paperclip_journal,
            "paperclip_article_type": paperclip_article_type,
            "paperclip_author": paperclip_author,
            "paperclip_full_corpus": paperclip_full_corpus,
            "paperclip_max_full_text_lines": effective.paperclip_max_full_text_lines,
            "chunk_window_size": effective.chunk_window_size,
            "chunk_stride": effective.chunk_stride,
            "min_chunk_chars": effective.min_chunk_chars,
            "max_chunk_chars": effective.max_chunk_chars,
            "min_chunk_words": effective.min_chunk_words,
            "context_backoff": effective.context_backoff,
            "max_chunks_per_paper": effective.max_chunks_per_paper,
            "near_duplicate_threshold": effective.near_duplicate_threshold,
            "bm25_k1": effective.bm25_k1,
            "bm25_b": effective.bm25_b,
            "medcpt_query_model": effective.medcpt_query_model,
            "medcpt_article_model": effective.medcpt_article_model,
            "medcpt_batch_size": effective.medcpt_batch_size,
            "medcpt_device": effective.medcpt_device,
            "hybrid_lexical_weight": effective.hybrid_lexical_weight,
            "hybrid_medcpt_weight": effective.hybrid_medcpt_weight,
        },
    )
    return build_evidence_pack(
        question=question,
        query=query,
        pipeline=info,
        selected_chunks=selected,
        retrieved_papers=retrieval.papers,
    )
