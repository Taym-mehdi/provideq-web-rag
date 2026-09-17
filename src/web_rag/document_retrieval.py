from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .europepmc_retriever import retrieve_papers_europepmc
from .models import Paper, QueryBundle
from .multi_source_retriever import reciprocal_rank_fusion
from .paperclip_retriever import (
    create_client,
    load_paper_full_texts,
    retrieve_papers as retrieve_papers_paperclip,
)
from .text_utils import clean_text


@dataclass(frozen=True)
class DocumentRetrieval:
    papers: list[Paper]
    paperclip_result_id: str = ""
    paperclip_query: str = ""
    europepmc_query: str = ""
    europepmc_query_variants: tuple[str, ...] = ()
    source_counts: dict[str, int] | None = None
    warnings: tuple[str, ...] = ()


def _message(exc: Exception) -> str:
    return clean_text(str(exc)) or type(exc).__name__


def retrieve_documents(
    question: str,
    query: QueryBundle,
    *,
    settings: Settings,
    paperclip_mode: str | None = None,
    paperclip_since: str | None = None,
    paperclip_sort: str | None = None,
    paperclip_year: int | str | None = None,
    paperclip_journal: str | None = None,
    paperclip_article_type: str | None = None,
    paperclip_author: str | None = None,
    paperclip_client: Any | None = None,
    load_full_text: bool = True,
) -> DocumentRetrieval:
    """Run the tested Paperclip/Europe PMC document-retrieval configuration."""
    warnings: list[str] = []
    raw_query = clean_text(query.normalized_question)
    reformulated_query = clean_text(query.search_query)
    europepmc_query = (
        reformulated_query
        if settings.europepmc_use_reformulated_query
        else clean_text(question)
    )

    paperclip_papers: list[Paper] = []
    paperclip_result_ids: list[str] = []
    paperclip_query_log = reformulated_query
    active_client = paperclip_client

    if settings.retrieval_system in {"paperclip", "fusion"}:
        paperclip_limit = (
            settings.paperclip_candidate_limit
            if settings.retrieval_system == "fusion"
            else settings.retrieval_limit
        )
        try:
            active_client = active_client or create_client()

            def search_paperclip(search_query: str):
                return retrieve_papers_paperclip(
                    search_query,
                    limit=paperclip_limit,
                    source=settings.paperclip_source,
                    ranking=settings.paperclip_ranking,
                    max_full_text_lines=settings.paperclip_max_full_text_lines,
                    mode=paperclip_mode,
                    since=paperclip_since,
                    sort=paperclip_sort,
                    year=paperclip_year,
                    journal=paperclip_journal,
                    article_type=paperclip_article_type,
                    author=paperclip_author,
                    full_corpus=settings.paperclip_full_corpus,
                    # Candidate fusion should remain cheap. The final ranked
                    # papers are hydrated below in a single pass.
                    load_full_text=False,
                    timeout=settings.paperclip_timeout,
                    client=active_client,
                )

            if (
                settings.paperclip_query_fusion
                and reformulated_query != raw_query
            ):
                raw_result = search_paperclip(raw_query)
                reformulated_result = search_paperclip(reformulated_query)
                reformulated_name = f"paperclip_{query.strategy}"
                paperclip_papers = reciprocal_rank_fusion(
                    {
                        "paperclip_raw": list(raw_result.papers),
                        reformulated_name: list(reformulated_result.papers),
                    },
                    limit=paperclip_limit,
                    rrf_k=settings.query_fusion_rrf_k,
                    source_weights={
                        "paperclip_raw": 1.0,
                        reformulated_name: (
                            settings.reformulated_query_weight
                        ),
                    },
                ).papers
                paperclip_query_log = (
                    f"raw => {raw_query} || {query.strategy} => "
                    f"{reformulated_query}"
                )
                for label, result in (
                    ("raw", raw_result),
                    (query.strategy, reformulated_result),
                ):
                    if result.result_id:
                        paperclip_result_ids.append(
                            f"{label}:{result.result_id}"
                        )
            else:
                result = search_paperclip(reformulated_query)
                paperclip_papers = list(result.papers)
                if result.result_id:
                    paperclip_result_ids.append(result.result_id)
        except Exception as exc:
            if settings.retrieval_system != "fusion":
                raise
            warnings.append(
                "Paperclip failed; used Europe PMC only: " + _message(exc)
            )

    europepmc_papers: list[Paper] = []
    europepmc_variants: tuple[str, ...] = ()
    if settings.retrieval_system in {"europepmc", "fusion"}:
        europepmc_limit = (
            settings.europepmc_candidate_limit
            if settings.retrieval_system == "fusion"
            else settings.retrieval_limit
        )
        try:
            epmc_result = retrieve_papers_europepmc(
                europepmc_query,
                limit=europepmc_limit,
                candidate_limit=settings.europepmc_candidate_limit,
                synonym=settings.europepmc_synonym,
                mode=settings.europepmc_mode,
                rrf_k=settings.fusion_rrf_k,
                timeout=settings.europepmc_timeout,
                cache_dir=settings.europepmc_cache_dir or None,
            )
            europepmc_papers = list(epmc_result.papers)
            europepmc_variants = tuple(epmc_result.query_variants)
        except Exception as exc:
            if settings.retrieval_system != "fusion":
                raise
            warnings.append(
                "Europe PMC failed; used Paperclip only: " + _message(exc)
            )

    if (
        settings.retrieval_system == "fusion"
        and not paperclip_papers
        and not europepmc_papers
    ):
        detail = " | ".join(warnings) or "both sources returned no papers"
        raise RuntimeError(f"Fusion retrieval failed: {detail}")

    if settings.retrieval_system == "paperclip":
        papers = paperclip_papers[: settings.retrieval_limit]
        source_counts = {"paperclip": len(paperclip_papers)}
    elif settings.retrieval_system == "europepmc":
        papers = europepmc_papers[: settings.retrieval_limit]
        source_counts = {"europepmc": len(europepmc_papers)}
    else:
        fused = reciprocal_rank_fusion(
            {
                "paperclip": paperclip_papers,
                "europepmc": europepmc_papers,
            },
            limit=settings.retrieval_limit,
            rrf_k=settings.fusion_rrf_k,
        )
        papers = fused.papers
        source_counts = fused.source_counts

    for rank, paper in enumerate(papers, start=1):
        paper.retrieval_rank = rank

    if load_full_text and papers and active_client is not None:
        load_paper_full_texts(
            papers,
            max_full_text_lines=settings.paperclip_max_full_text_lines,
            client=active_client,
        )

    return DocumentRetrieval(
        papers=papers,
        paperclip_result_id="; ".join(paperclip_result_ids),
        paperclip_query=(
            paperclip_query_log
            if settings.retrieval_system in {"paperclip", "fusion"}
            else ""
        ),
        europepmc_query=(
            europepmc_query
            if settings.retrieval_system in {"europepmc", "fusion"}
            else ""
        ),
        europepmc_query_variants=europepmc_variants,
        source_counts=source_counts,
        warnings=tuple(warnings),
    )
