from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Paper:
    """
    Represents one retrieved scientific paper after normalization.
    """

    title: str
    abstract: str
    year: str
    source: str
    ext_id: str
    doi: str
    authors: str
    journal: str
    url: str


@dataclass
class Snippet:
    """
    Represents one evidence snippet extracted from a paper.
    """

    paper: Paper
    text: str
    score: float = 0.0


@dataclass
class QueryBundle:
    """
    Represents the different forms of a question used for retrieval.
    """

    original_question: str
    normalized_question: str
    keywords: List[str] = field(default_factory=list)
    search_query: str = ""


@dataclass
class RetrievalResult:
    """
    Container for the output of the retrieval stage.
    """

    query: QueryBundle
    papers: List[Paper] = field(default_factory=list)
    snippets: List[Snippet] = field(default_factory=list)


@dataclass
class EvidenceRecord:
    """
    Represents one ranked evidence item prepared for answer generation.

    This is different from a raw Snippet because it already has a citation ID
    and flattened metadata. That makes it easier to print, export, evaluate,
    and pass to a grounded generator.
    """

    citation_id: int
    title: str
    evidence_text: str
    score: float
    year: str
    source: str
    ext_id: str
    doi: str
    authors: str
    journal: str
    url: str


@dataclass
class EvidencePack:
    """
    Represents the complete evidence context for one user question.
    """

    question: str
    records: List[EvidenceRecord]
    context_text: str