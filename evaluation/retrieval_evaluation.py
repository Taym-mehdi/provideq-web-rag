from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_PMCID_PATTERN = re.compile(r"\bPMC\d+\b", re.IGNORECASE)
_PMID_URL_PATTERN = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov|/pubmed/)/(\d+)", re.IGNORECASE)
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievalMRRResult:
    score: float
    rank: int | None
    matched_title: str
    matched_identifier: str = ""
    match_type: str = ""


def normalize_title(value: str) -> str:
    """Normalize a paper title for stable exact matching."""
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.replace("&", " and ")
    return " ".join(_NON_ALPHANUMERIC.sub(" ", text).split())


def normalize_doi(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.IGNORECASE)
    match = _DOI_PATTERN.search(text)
    return match.group(0).rstrip(".,;)").casefold() if match else ""


def normalize_pmcid(value: str) -> str:
    match = _PMCID_PATTERN.search(str(value or ""))
    return match.group(0).upper() if match else ""


def normalize_pmid(value: str) -> str:
    text = str(value or "").strip()
    if text.isdigit():
        return text
    match = _PMID_URL_PATTERN.search(text)
    return match.group(1) if match else ""


def _value(item: Any, field: str, default: Any = "") -> Any:
    if isinstance(item, Mapping):
        return item.get(field, default)
    return getattr(item, field, default)


def _iter_mapping_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _iter_mapping_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_mapping_values(item)


def _paper_identifiers(item: Any) -> dict[str, set[str]]:
    identifiers: dict[str, set[str]] = {
        "pmcid": set(),
        "pmid": set(),
        "doi": set(),
    }

    direct_values = {
        "paper_id": _value(item, "paper_id", ""),
        "pmcid": _value(item, "pmcid", ""),
        "pmid": _value(item, "pmid", ""),
        "doi": _value(item, "doi", ""),
        "url": _value(item, "url", ""),
    }
    metadata = _value(item, "metadata", {})

    for key, value in [*direct_values.items(), *_iter_mapping_values(metadata)]:
        normalized_key = key.casefold().replace("-", "_")
        text = str(value or "")

        pmcid = normalize_pmcid(text)
        if pmcid and normalized_key in {
            "paper_id", "pmcid", "pmc_id", "id", "document_id", "url", "path", "paper_path"
        }:
            identifiers["pmcid"].add(pmcid)

        pmid = normalize_pmid(text)
        if pmid and normalized_key in {
            "pmid", "pubmed_id", "pubmedid", "url"
        }:
            identifiers["pmid"].add(pmid)

        doi = normalize_doi(text)
        if doi and normalized_key in {"doi", "article_doi", "url"}:
            identifiers["doi"].add(doi)

    return identifiers


def _first_identifier(values: set[str]) -> str:
    return sorted(values)[0] if values else ""


def _match_paper(
    gold_document: Mapping[str, Any],
    retrieved_paper: Any,
) -> tuple[str, str] | None:
    gold_ids = _paper_identifiers(gold_document)
    retrieved_ids = _paper_identifiers(retrieved_paper)

    for identifier_type in ("pmcid", "doi", "pmid"):
        common = gold_ids[identifier_type] & retrieved_ids[identifier_type]
        if common:
            return identifier_type, _first_identifier(common)

    gold_title = normalize_title(str(gold_document.get("title", "")))
    retrieved_title = normalize_title(str(_value(retrieved_paper, "title", "")))
    if gold_title and retrieved_title and gold_title == retrieved_title:
        return "title", str(_value(retrieved_paper, "title", "")).strip()

    return None


def evaluate_retrieval_mrr_detailed(
    gold_documents: list[dict[str, Any]],
    retrieved_papers: Iterable[Any],
) -> RetrievalMRRResult:
    """Evaluate the first retrieved gold paper using identifiers before title.

    Matching priority is PMCID, DOI, PMID, then normalized exact title. The score is
    1/rank for the first match and 0 when no assigned gold paper is retrieved.
    """
    usable_gold = [
        document
        for document in gold_documents
        if isinstance(document, Mapping)
        and (
            str(document.get("title", "")).strip()
            or any(str(document.get(key, "")).strip() for key in ("pmcid", "pmid", "doi"))
        )
    ]
    if not usable_gold:
        raise ValueError(
            "gold_documents must contain a title, PMCID, PMID, or DOI"
        )

    ordered_papers = sorted(
        list(retrieved_papers),
        key=lambda paper: int(_value(paper, "retrieval_rank", 0) or 0),
    )
    for fallback_rank, paper in enumerate(ordered_papers, start=1):
        rank = int(_value(paper, "retrieval_rank", fallback_rank) or fallback_rank)
        for gold_document in usable_gold:
            match = _match_paper(gold_document, paper)
            if match is None:
                continue
            match_type, identifier = match
            return RetrievalMRRResult(
                score=1.0 / rank,
                rank=rank,
                matched_title=str(_value(paper, "title", "")).strip(),
                matched_identifier=identifier,
                match_type=match_type,
            )

    return RetrievalMRRResult(score=0.0, rank=None, matched_title="")


def evaluate_retrieval_mrr(
    gold_documents: list[dict[str, Any]],
    retrieved_papers: Iterable[Any],
) -> tuple[float, int | None, str]:
    """Backward-compatible MRR result: score, rank, matched title."""
    result = evaluate_retrieval_mrr_detailed(gold_documents, retrieved_papers)
    return result.score, result.rank, result.matched_title
