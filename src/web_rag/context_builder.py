from __future__ import annotations

from collections.abc import Iterable

from web_rag.models import EvidencePack, EvidenceRecord, Paper, Snippet
from web_rag.text_utils import clean_text


def get_best_identifier(paper: Paper) -> str:
    if paper.doi:
        return f"DOI: {paper.doi}"

    if paper.source and paper.ext_id:
        return f"{paper.source}:{paper.ext_id}"

    if paper.url:
        return paper.url

    return paper.title


def build_metadata_line(record: EvidenceRecord) -> str:
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
        score_components=snippet.score_components,
    )


def build_evidence_records(snippets: Iterable[Snippet]) -> list[EvidenceRecord]:
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
    context_blocks: list[str] = []

    for record in records:
        metadata_line = build_metadata_line(record)

        header_parts = [f"[{record.citation_id}] {record.title}"]

        if metadata_line:
            header_parts.append(metadata_line)

        block_lines = [
            " | ".join(header_parts),
            f"Score: {record.score:.4f}",
            f"Evidence: {record.evidence_text}",
        ]

        if record.url:
            block_lines.append(f"URL: {record.url}")

        context_blocks.append("\n".join(block_lines))

    return "\n\n".join(context_blocks)


def build_evidence_pack(question: str, ranked_snippets: list[Snippet]) -> EvidencePack:
    records = build_evidence_records(ranked_snippets)
    context_text = build_context_text(records)

    return EvidencePack(
        question=question,
        records=records,
        context_text=context_text,
    )