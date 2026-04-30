from __future__ import annotations

from collections.abc import Iterable

from web_rag.models import EvidencePack, EvidenceRecord, Paper, Snippet
from web_rag.text_utils import clean_text


def get_best_identifier(paper: Paper) -> str:
    """
    Return the best available stable identifier for a paper.

    Priority:
    1. DOI
    2. source + external ID, for example MED:123456
    3. URL
    4. title fallback

    This is used only for display and traceability. The full metadata is still
    preserved in the EvidenceRecord.
    """
    if paper.doi:
        return f"DOI: {paper.doi}"

    if paper.source and paper.ext_id:
        return f"{paper.source}:{paper.ext_id}"

    if paper.url:
        return paper.url

    return paper.title


def build_metadata_line(record: EvidenceRecord) -> str:
    """
    Build a compact metadata line for one evidence record.
    """
    metadata_parts: list[str] = []

    if record.journal:
        metadata_parts.append(record.journal)

    if record.year:
        metadata_parts.append(record.year)

    if record.doi:
        metadata_parts.append(f"DOI: {record.doi}")
    elif record.source and record.ext_id:
        metadata_parts.append(f"{record.source}:{record.ext_id}")

    return " | ".join(metadata_parts)


def snippet_to_evidence_record(snippet: Snippet, citation_id: int) -> EvidenceRecord:
    """
    Convert a ranked Snippet into a flattened EvidenceRecord.

    The Snippet object is useful internally because it keeps the full Paper
    object attached. The EvidenceRecord is better for outputs because it is
    simpler, numbered, and citation-ready.
    """
    paper = snippet.paper

    return EvidenceRecord(
        citation_id=citation_id,
        title=clean_text(paper.title),
        evidence_text=clean_text(snippet.text),
        score=snippet.score,
        year=paper.year,
        source=paper.source,
        ext_id=paper.ext_id,
        doi=paper.doi,
        authors=clean_text(paper.authors),
        journal=clean_text(paper.journal),
        url=paper.url,
    )


def build_evidence_records(snippets: Iterable[Snippet]) -> list[EvidenceRecord]:
    """
    Convert ranked snippets into numbered evidence records.
    """
    records: list[EvidenceRecord] = []

    for index, snippet in enumerate(snippets, start=1):
        records.append(
            snippet_to_evidence_record(
                snippet=snippet,
                citation_id=index,
            )
        )

    return records


def build_context_text(records: list[EvidenceRecord]) -> str:
    """
    Build a clean context block from evidence records.

    This is the text that can later be passed to a grounded answer generator.

    The format is intentionally explicit:

    [1] source metadata
    Score: ranking score
    Evidence: extracted evidence text
    URL: source URL

    This makes it easier to inspect whether an answer is actually grounded in
    the retrieved snippets.
    """
    context_blocks: list[str] = []

    for record in records:
        metadata_line = build_metadata_line(record)

        header_parts = [f"[{record.citation_id}] {record.title}"]

        if metadata_line:
            header_parts.append(metadata_line)

        block_lines = [
            " | ".join(header_parts),
            f"Score: {record.score:.2f}",
            f"Evidence: {record.evidence_text}",
        ]

        if record.url:
            block_lines.append(f"URL: {record.url}")

        context_blocks.append("\n".join(block_lines))

    return "\n\n".join(context_blocks)


def build_evidence_pack(question: str, ranked_snippets: list[Snippet]) -> EvidencePack:
    """
    Build the final evidence pack for one question.

    This is the central output of the retrieval side of the Web RAG baseline.
    It contains:

    - the original question
    - structured evidence records
    - a text context block for generation or inspection
    """
    records = build_evidence_records(ranked_snippets)
    context_text = build_context_text(records)

    return EvidencePack(
        question=question,
        records=records,
        context_text=context_text,
    )