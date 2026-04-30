from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from web_rag.models import EvidencePack, EvidenceRecord, QueryBundle


def utc_now_iso() -> str:
    """
    Return the current UTC timestamp in ISO format.

    This is useful for experiment tracking and saved outputs.
    """
    return datetime.now(timezone.utc).isoformat()


def evidence_record_to_dict(record: EvidenceRecord) -> dict[str, Any]:
    """
    Convert one EvidenceRecord into a JSON-serializable dictionary.

    The output is intentionally flat because it should later be easy to return
    from an API endpoint and easy to inspect during evaluation.
    """
    return {
        "citation_id": record.citation_id,
        "score": record.score,
        "title": record.title,
        "evidence_text": record.evidence_text,
        "metadata": {
            "year": record.year,
            "source": record.source,
            "ext_id": record.ext_id,
            "doi": record.doi,
            "authors": record.authors,
            "journal": record.journal,
            "url": record.url,
        },
    }


def evidence_pack_to_dict(
    evidence_pack: EvidencePack,
    query: QueryBundle,
    retrieved_papers_count: int,
    extracted_snippets_count: int,
) -> dict[str, Any]:
    """
    Convert the complete evidence result into a JSON-ready dictionary.

    This is the main structured output of the Web RAG baseline.

    It contains:
    - original question
    - generated search query
    - retrieval statistics
    - ranked evidence records
    - plain context text

    The downstream agent can use either:
    - the structured evidence list, or
    - the context_text field.
    """
    has_evidence = len(evidence_pack.records) > 0

    return {
        "status": "ok" if has_evidence else "no_evidence_found",
        "created_at_utc": utc_now_iso(),
        "question": evidence_pack.question,
        "query": {
            "original_question": query.original_question,
            "normalized_question": query.normalized_question,
            "keywords": query.keywords,
            "search_query": query.search_query,
        },
        "counts": {
            "retrieved_papers": retrieved_papers_count,
            "extracted_snippets": extracted_snippets_count,
            "ranked_evidence_records": len(evidence_pack.records),
        },
        "evidence": [
            evidence_record_to_dict(record)
            for record in evidence_pack.records
        ],
        "context_text": evidence_pack.context_text,
    }


def to_pretty_json(payload: dict[str, Any]) -> str:
    """
    Convert a dictionary to readable UTF-8 JSON text.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def save_json_output(payload: dict[str, Any], output_path: str) -> None:
    """
    Save JSON output to disk.

    Parent folders are created automatically.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        to_pretty_json(payload),
        encoding="utf-8",
    )


def save_context_text(evidence_pack: EvidencePack, output_path: str) -> None:
    """
    Save only the final context text to disk.

    This is useful because the context text is the exact evidence block that
    can later be passed to another agent or inspected manually.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        evidence_pack.context_text,
        encoding="utf-8",
    )