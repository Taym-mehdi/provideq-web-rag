from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryBundle:
    original_question: str
    normalized_question: str
    strategy: str
    search_query: str
    keywords: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    hypothetical_document: str = ""
    expansion_details: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Paper:
    paper_id: str
    title: str
    text: str
    source: str
    year: str = ""
    doi: str = ""
    authors: str = ""
    journal: str = ""
    url: str = ""
    abstract: str = ""
    retrieval_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperclipRetrieval:
    papers: list[Paper]
    result_id: str = ""


@dataclass
class TextChunk:
    paper: Paper
    text: str
    method: str
    chunk_index: int
    section: str = ""
    start_sentence: int | None = None
    end_sentence: int | None = None
    score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CitationSource:
    paper_id: str
    title: str
    source: str
    year: str = ""
    doi: str = ""
    authors: str = ""
    journal: str = ""
    url: str = ""


@dataclass(frozen=True)
class RetrievedPaper:
    retrieval_rank: int
    paper_id: str
    title: str
    source: str
    year: str = ""
    doi: str = ""
    authors: str = ""
    journal: str = ""
    url: str = ""


@dataclass
class EvidenceRecord:
    citation_id: int
    evidence_text: str
    score: float
    chunking_method: str
    chunk_index: int
    section: str
    source: CitationSource
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineInfo:
    retrieval_system: str
    paperclip_source: str
    paperclip_ranking: str
    paperclip_result_id: str
    retrieval_limit: int
    retrieved_papers_count: int
    full_text_papers_count: int
    chunking_method: str
    extracted_chunks_count: int
    reranker: str
    top_k: int
    returned_evidence_count: int
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePack:
    question: str
    query: QueryBundle
    pipeline: PipelineInfo
    records: list[EvidenceRecord]
    context_text: str
    retrieved_papers: list[RetrievedPaper] = field(default_factory=list)
