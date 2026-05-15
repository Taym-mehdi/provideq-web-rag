from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from web_rag.models import EvidencePack, EvidenceRecord, QueryBundle


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def evidence_record_to_dict(record: EvidenceRecord) -> dict[str, Any]:
    return {
        "citation_id": record.citation_id,
        "score": record.score,
        "score_components": record.score_components,
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
    ranking_method: str,
) -> dict[str, Any]:
    has_evidence = len(evidence_pack.records) > 0

    return {
        "status": "ok" if has_evidence else "no_evidence_found",
        "created_at_utc": utc_now_iso(),
        "question": evidence_pack.question,
        "ranking_method": ranking_method,
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
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def save_json_output(payload: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        to_pretty_json(payload),
        encoding="utf-8",
    )


def save_context_text(evidence_pack: EvidencePack, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        evidence_pack.context_text,
        encoding="utf-8",
    )