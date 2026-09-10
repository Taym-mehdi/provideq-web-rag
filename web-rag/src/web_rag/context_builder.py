from __future__ import annotations

from .models import (
    CitationSource,
    EvidencePack,
    EvidenceRecord,
    Paper,
    PipelineInfo,
    QueryBundle,
    RetrievedPaper,
    TextChunk,
)


def _source_line(source: CitationSource) -> str:
    parts: list[str] = []
    if source.title:
        parts.append(source.title.rstrip(" ."))
    if source.journal and source.year:
        parts.append(f"{source.journal.rstrip(' .')}, {source.year}")
    elif source.journal:
        parts.append(source.journal.rstrip(" ."))
    elif source.year:
        parts.append(source.year)
    if source.doi:
        parts.append(f"DOI: {source.doi}")
    elif source.paper_id:
        parts.append(f"ID: {source.paper_id}")
    if source.url:
        parts.append(source.url)
    return ". ".join(parts)


def _retrieved_paper(paper: Paper, fallback_rank: int) -> RetrievedPaper:
    return RetrievedPaper(
        retrieval_rank=int(paper.retrieval_rank or fallback_rank),
        paper_id=paper.paper_id,
        title=paper.title,
        source=paper.source,
        year=paper.year,
        doi=paper.doi,
        authors=paper.authors,
        journal=paper.journal,
        url=paper.url,
    )


def build_evidence_pack(
    *,
    question: str,
    query: QueryBundle,
    pipeline: PipelineInfo,
    selected_chunks: list[TextChunk],
    retrieved_papers: list[Paper] | None = None,
) -> EvidencePack:
    records: list[EvidenceRecord] = []
    context_blocks: list[str] = []

    for citation_id, chunk in enumerate(selected_chunks, start=1):
        paper = chunk.paper
        source = CitationSource(
            paper_id=paper.paper_id,
            title=paper.title,
            source=paper.source,
            year=paper.year,
            doi=paper.doi,
            authors=paper.authors,
            journal=paper.journal,
            url=paper.url,
        )
        records.append(
            EvidenceRecord(
                citation_id=citation_id,
                evidence_text=chunk.text,
                score=float(chunk.score),
                chunking_method=chunk.method,
                chunk_index=chunk.chunk_index,
                section=chunk.section,
                source=source,
                score_components=dict(chunk.score_components),
            )
        )
        context_blocks.append(f"[{citation_id}] {chunk.text}\nSource: {_source_line(source)}")

    retrieval_metadata = [
        _retrieved_paper(paper, rank)
        for rank, paper in enumerate(retrieved_papers or [], start=1)
    ]

    return EvidencePack(
        question=question,
        query=query,
        pipeline=pipeline,
        records=records,
        context_text="\n\n".join(context_blocks),
        retrieved_papers=retrieval_metadata,
    )
