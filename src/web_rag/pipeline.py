from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import os
from typing import Any, TypeVar

from .chunking import chunk_papers
from .config import Settings, get_settings, validate_settings
from .context_builder import build_evidence_pack
from .document_retrieval import retrieve_documents
from .evidence_selection import select_evidence
from .models import EvidencePack, PipelineInfo
from .query_reformulation import (
    QueryGenerationError,
    build_raw_query,
    make_llm_generator,
    reformulate_query,
)
from .reranking import rerank_chunks


T = TypeVar("T")


def _or_default(value: T | None, default: T) -> T:
    return default if value is None else value


def run_pipeline(
    question: str,
    *,
    retrieval_system: str | None = None,
    retrieval_limit: int | None = None,
    query_strategy: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str | None = None,
    llm_api_key: str | None = None,
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
    paperclip_candidate_limit: int | None = None,
    paperclip_query_fusion: bool | None = None,
    query_fusion_rrf_k: int | None = None,
    reformulated_query_weight: float | None = None,
    paperclip_max_full_text_lines: int | None = None,
    paperclip_timeout: float | None = None,
    paperclip_mode: str | None = None,
    paperclip_since: str | None = None,
    paperclip_sort: str | None = None,
    paperclip_year: int | str | None = None,
    paperclip_journal: str | None = None,
    paperclip_article_type: str | None = None,
    paperclip_author: str | None = None,
    paperclip_full_corpus: bool | None = None,
    europepmc_mode: str | None = None,
    europepmc_synonym: bool | None = None,
    europepmc_use_reformulated_query: bool | None = None,
    europepmc_candidate_limit: int | None = None,
    europepmc_timeout: float | None = None,
    europepmc_cache_dir: str | None = None,
    fusion_rrf_k: int | None = None,
    chunking_method: str | None = None,
    chunk_tokenizer_model: str | None = None,
    chunk_max_tokens: int | None = None,
    chunk_overlap_fraction: float | None = None,
    chunk_max_overlap_sentences: int | None = None,
    min_chunk_words: int | None = None,
    reranker: str | None = None,
    top_k: int | None = None,
    max_chunks_per_paper: int | None = None,
    near_duplicate_threshold: float | None = None,
    bm25_k1: float | None = None,
    bm25_b: float | None = None,
    medcpt_model: str | None = None,
    medcpt_max_length: int | None = None,
    medcpt_batch_size: int | None = None,
    medcpt_device: str | None = None,
    hybrid_lexical_weight: float | None = None,
    hybrid_medcpt_weight: float | None = None,
    settings: Settings | None = None,
    paperclip_client: Any | None = None,
) -> EvidencePack:
    """Run the default Web RAG and return citation-ready evidence chunks."""
    base = settings or get_settings()
    effective_llm_model = _or_default(llm_model, base.llm_model)
    effective_llm_base_url = _or_default(llm_base_url, base.llm_base_url)

    effective = replace(
        base,
        retrieval_system=_or_default(
            retrieval_system,
            base.retrieval_system,
        ),
        retrieval_limit=_or_default(retrieval_limit, base.retrieval_limit),
        query_strategy=_or_default(query_strategy, base.query_strategy),
        llm_provider=_or_default(llm_provider, base.llm_provider),
        llm_model=effective_llm_model,
        llm_base_url=effective_llm_base_url,
        llm_api_key_env=_or_default(
            llm_api_key_env,
            base.llm_api_key_env,
        ),
        hyde_model=_or_default(
            hyde_model,
            effective_llm_model if llm_model is not None else base.hyde_model,
        ),
        hyde_base_url=_or_default(
            hyde_base_url,
            effective_llm_base_url
            if llm_base_url is not None
            else base.hyde_base_url,
        ),
        hyde_temperature=_or_default(
            hyde_temperature,
            base.hyde_temperature,
        ),
        hyde_max_tokens=_or_default(
            hyde_max_tokens,
            base.hyde_max_tokens,
        ),
        hyde_seed=_or_default(hyde_seed, base.hyde_seed),
        hyde_timeout=_or_default(hyde_timeout, base.hyde_timeout),
        expansion_model=_or_default(
            expansion_model,
            effective_llm_model
            if llm_model is not None
            else base.expansion_model,
        ),
        expansion_base_url=_or_default(
            expansion_base_url,
            effective_llm_base_url
            if llm_base_url is not None
            else base.expansion_base_url,
        ),
        expansion_temperature=_or_default(
            expansion_temperature,
            base.expansion_temperature,
        ),
        expansion_max_tokens=_or_default(
            expansion_max_tokens,
            base.expansion_max_tokens,
        ),
        expansion_seed=_or_default(
            expansion_seed,
            base.expansion_seed,
        ),
        expansion_timeout=_or_default(
            expansion_timeout,
            base.expansion_timeout,
        ),
        expansion_max_terms=_or_default(
            expansion_max_terms,
            base.expansion_max_terms,
        ),
        expansion_max_query_chars=_or_default(
            expansion_max_query_chars,
            base.expansion_max_query_chars,
        ),
        paperclip_source=_or_default(
            paperclip_source,
            base.paperclip_source,
        ),
        paperclip_ranking=_or_default(
            paperclip_ranking,
            base.paperclip_ranking,
        ),
        paperclip_candidate_limit=_or_default(
            paperclip_candidate_limit,
            base.paperclip_candidate_limit,
        ),
        paperclip_query_fusion=_or_default(
            paperclip_query_fusion,
            base.paperclip_query_fusion,
        ),
        query_fusion_rrf_k=_or_default(
            query_fusion_rrf_k,
            base.query_fusion_rrf_k,
        ),
        reformulated_query_weight=_or_default(
            reformulated_query_weight,
            base.reformulated_query_weight,
        ),
        paperclip_max_full_text_lines=_or_default(
            paperclip_max_full_text_lines,
            base.paperclip_max_full_text_lines,
        ),
        paperclip_timeout=_or_default(
            paperclip_timeout,
            base.paperclip_timeout,
        ),
        paperclip_full_corpus=_or_default(
            paperclip_full_corpus,
            base.paperclip_full_corpus,
        ),
        europepmc_mode=_or_default(
            europepmc_mode,
            base.europepmc_mode,
        ),
        europepmc_synonym=_or_default(
            europepmc_synonym,
            base.europepmc_synonym,
        ),
        europepmc_use_reformulated_query=_or_default(
            europepmc_use_reformulated_query,
            base.europepmc_use_reformulated_query,
        ),
        europepmc_candidate_limit=_or_default(
            europepmc_candidate_limit,
            base.europepmc_candidate_limit,
        ),
        europepmc_timeout=_or_default(
            europepmc_timeout,
            base.europepmc_timeout,
        ),
        europepmc_cache_dir=_or_default(
            europepmc_cache_dir,
            base.europepmc_cache_dir,
        ),
        fusion_rrf_k=_or_default(
            fusion_rrf_k,
            base.fusion_rrf_k,
        ),
        chunking_method=_or_default(
            chunking_method,
            base.chunking_method,
        ),
        chunk_tokenizer_model=_or_default(
            chunk_tokenizer_model,
            base.chunk_tokenizer_model,
        ),
        chunk_max_tokens=_or_default(
            chunk_max_tokens,
            base.chunk_max_tokens,
        ),
        chunk_overlap_fraction=_or_default(
            chunk_overlap_fraction,
            base.chunk_overlap_fraction,
        ),
        chunk_max_overlap_sentences=_or_default(
            chunk_max_overlap_sentences,
            base.chunk_max_overlap_sentences,
        ),
        min_chunk_words=_or_default(
            min_chunk_words,
            base.min_chunk_words,
        ),
        reranker=_or_default(reranker, base.reranker),
        top_k=_or_default(top_k, base.top_k),
        max_chunks_per_paper=_or_default(
            max_chunks_per_paper,
            base.max_chunks_per_paper,
        ),
        near_duplicate_threshold=_or_default(
            near_duplicate_threshold,
            base.near_duplicate_threshold,
        ),
        bm25_k1=_or_default(bm25_k1, base.bm25_k1),
        bm25_b=_or_default(bm25_b, base.bm25_b),
        medcpt_model=_or_default(medcpt_model, base.medcpt_model),
        medcpt_max_length=_or_default(
            medcpt_max_length,
            base.medcpt_max_length,
        ),
        medcpt_batch_size=_or_default(
            medcpt_batch_size,
            base.medcpt_batch_size,
        ),
        medcpt_device=_or_default(
            medcpt_device,
            base.medcpt_device,
        ),
        hybrid_lexical_weight=_or_default(
            hybrid_lexical_weight,
            base.hybrid_lexical_weight,
        ),
        hybrid_medcpt_weight=_or_default(
            hybrid_medcpt_weight,
            base.hybrid_medcpt_weight,
        ),
    )
    validate_settings(effective)

    resolved_hyde_generator = hyde_generator
    resolved_expansion_generator = expansion_generator
    if effective.query_strategy in {"hyde", "llmexpand"}:
        provider = effective.llm_provider.strip().casefold()
        resolved_api_key = llm_api_key

        if effective.query_strategy == "hyde":
            if resolved_hyde_generator is None:
                if provider == "openai" and not resolved_api_key:
                    resolved_api_key = os.getenv(
                        effective.llm_api_key_env,
                        "",
                    )
                    if not resolved_api_key:
                        raise ValueError(
                            "Environment variable "
                            f"{effective.llm_api_key_env} is not set."
                        )
                resolved_hyde_generator = make_llm_generator(
                    provider,
                    model=effective.hyde_model,
                    base_url=effective.hyde_base_url,
                    api_key=resolved_api_key,
                    temperature=effective.hyde_temperature,
                    max_tokens=effective.hyde_max_tokens,
                    seed=effective.hyde_seed,
                    timeout=effective.hyde_timeout,
                    json_output=False,
                )
        elif resolved_expansion_generator is None:
            if provider == "openai" and not resolved_api_key:
                resolved_api_key = os.getenv(
                    effective.llm_api_key_env,
                    "",
                )
                if not resolved_api_key:
                    raise ValueError(
                        "Environment variable "
                        f"{effective.llm_api_key_env} is not set."
                    )
            resolved_expansion_generator = make_llm_generator(
                provider,
                model=effective.expansion_model,
                base_url=effective.expansion_base_url,
                api_key=resolved_api_key,
                temperature=effective.expansion_temperature,
                max_tokens=effective.expansion_max_tokens,
                seed=effective.expansion_seed,
                timeout=effective.expansion_timeout,
                json_output=True,
            )

    query_warnings: list[str] = []
    try:
        query = reformulate_query(
            question,
            effective.query_strategy,
            hyde_model=effective.hyde_model,
            hyde_base_url=effective.hyde_base_url,
            hyde_temperature=effective.hyde_temperature,
            hyde_max_tokens=effective.hyde_max_tokens,
            hyde_seed=effective.hyde_seed,
            hyde_timeout=effective.hyde_timeout,
            hyde_generator=resolved_hyde_generator,
            expansion_model=effective.expansion_model,
            expansion_base_url=effective.expansion_base_url,
            expansion_temperature=effective.expansion_temperature,
            expansion_max_tokens=effective.expansion_max_tokens,
            expansion_seed=effective.expansion_seed,
            expansion_timeout=effective.expansion_timeout,
            expansion_max_terms=effective.expansion_max_terms,
            expansion_max_query_chars=effective.expansion_max_query_chars,
            expansion_generator=resolved_expansion_generator,
        )
    except QueryGenerationError as exc:
        if not effective.paperclip_query_fusion:
            raise
        query = build_raw_query(question)
        query_warnings.append(
            "query reformulation failed; used raw query: " + str(exc)
        )

    retrieval = retrieve_documents(
        question,
        query,
        settings=effective,
        paperclip_mode=paperclip_mode,
        paperclip_since=paperclip_since,
        paperclip_sort=paperclip_sort,
        paperclip_year=paperclip_year,
        paperclip_journal=paperclip_journal,
        paperclip_article_type=paperclip_article_type,
        paperclip_author=paperclip_author,
        paperclip_client=paperclip_client,
        load_full_text=True,
    )

    chunks = chunk_papers(
        retrieval.papers,
        method=effective.chunking_method,
        tokenizer_model=effective.chunk_tokenizer_model,
        max_tokens=effective.chunk_max_tokens,
        overlap_fraction=effective.chunk_overlap_fraction,
        max_overlap_sentences=effective.chunk_max_overlap_sentences,
        min_words=effective.min_chunk_words,
    )
    ranked = rerank_chunks(question, chunks, settings=effective)
    selected = select_evidence(
        ranked,
        top_k=effective.top_k,
        max_chunks_per_paper=effective.max_chunks_per_paper,
        near_duplicate_threshold=effective.near_duplicate_threshold,
    )

    info = PipelineInfo(
        retrieval_system=effective.retrieval_system,
        paperclip_source=effective.paperclip_source,
        paperclip_ranking=effective.paperclip_ranking,
        paperclip_result_id=retrieval.paperclip_result_id,
        retrieval_limit=effective.retrieval_limit,
        retrieved_papers_count=len(retrieval.papers),
        full_text_papers_count=sum(
            bool(paper.metadata.get("has_full_text"))
            for paper in retrieval.papers
        ),
        chunking_method=effective.chunking_method,
        extracted_chunks_count=len(chunks),
        reranker=effective.reranker,
        top_k=effective.top_k,
        returned_evidence_count=len(selected),
        parameters={
            "query_strategy": effective.query_strategy,
            "effective_query_strategy": query.strategy,
            "warnings": [*query_warnings, *retrieval.warnings],
            "paperclip_candidate_limit": (
                effective.paperclip_candidate_limit
            ),
            "paperclip_query_fusion": (
                effective.paperclip_query_fusion
            ),
            "query_fusion_rrf_k": effective.query_fusion_rrf_k,
            "reformulated_query_weight": (
                effective.reformulated_query_weight
            ),
            "paperclip_full_corpus": effective.paperclip_full_corpus,
            "paperclip_max_full_text_lines": (
                effective.paperclip_max_full_text_lines
            ),
            "paperclip_query": retrieval.paperclip_query,
            "europepmc_query": retrieval.europepmc_query,
            "europepmc_query_variants": list(
                retrieval.europepmc_query_variants
            ),
            "europepmc_mode": effective.europepmc_mode,
            "europepmc_synonym": effective.europepmc_synonym,
            "europepmc_use_reformulated_query": (
                effective.europepmc_use_reformulated_query
            ),
            "europepmc_candidate_limit": (
                effective.europepmc_candidate_limit
            ),
            "fusion_rrf_k": effective.fusion_rrf_k,
            "retrieval_source_counts": retrieval.source_counts or {},
            "chunk_tokenizer_model": effective.chunk_tokenizer_model,
            "chunk_max_tokens": effective.chunk_max_tokens,
            "chunk_overlap_fraction": effective.chunk_overlap_fraction,
            "chunk_max_overlap_sentences": (
                effective.chunk_max_overlap_sentences
            ),
            "min_chunk_words": effective.min_chunk_words,
            "max_chunks_per_paper": effective.max_chunks_per_paper,
            "near_duplicate_threshold": (
                effective.near_duplicate_threshold
            ),
            "medcpt_model": effective.medcpt_model,
            "medcpt_max_length": effective.medcpt_max_length,
            "medcpt_batch_size": effective.medcpt_batch_size,
            "medcpt_device": effective.medcpt_device,
        },
    )
    return build_evidence_pack(
        question=question,
        query=query,
        pipeline=info,
        selected_chunks=selected,
        retrieved_papers=retrieval.papers,
    )
