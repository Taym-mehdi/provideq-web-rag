"""Build citation-ready evidence context for the downstream ProvideQ agent."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
import re
from typing import Any, Iterable, List, Mapping, Sequence

try:
    from .models import EvidencePack, EvidenceRecord
except Exception:
    EvidencePack = None  # type: ignore[assignment]
    EvidenceRecord = None  # type: ignore[assignment]


TEXT_KEYS = ("evidence_text", "text", "snippet", "snippet_text", "content", "passage", "abstract")
TITLE_KEYS = ("title", "paper_title", "source_title")
PAPER_KEYS = ("paper", "source", "document")


def _get_value(obj: Any, names: Iterable[str], default: Any = "") -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            value = obj.get(name)
            if value not in (None, ""):
                return value
        return default
    for name in names:
        value = getattr(obj, name, None)
        if value not in (None, ""):
            return value
    return default


def _get_paper(snippet: Any) -> Any:
    return _get_value(snippet, PAPER_KEYS, None)


def _get_snippet_text(snippet: Any) -> str:
    return str(_get_value(snippet, TEXT_KEYS, "") or "").strip()


def _get_title(snippet: Any, paper: Any) -> str:
    title = _get_value(snippet, TITLE_KEYS, "")
    if title:
        return str(title).strip()
    return str(_get_value(paper, TITLE_KEYS, "") or "").strip()


def _get_score(snippet: Any) -> float:
    value = _get_value(snippet, ("score", "ranking_score", "rerank_score"), 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _to_source_dict(paper: Any, title: str) -> dict[str, Any]:
    return {
        "title": title,
        "pmid": _get_value(paper, ("pmid", "pubmed_id", "ext_id", "id"), ""),
        "doi": _get_value(paper, ("doi",), ""),
        "url": _get_value(paper, ("url", "full_text_url", "source_url"), ""),
        "journal": _get_value(paper, ("journal", "journal_title"), ""),
        "year": _get_value(paper, ("year", "publication_year", "pub_year"), ""),
        "authors": _get_value(paper, ("authors", "author_string"), ""),
    }


def _source_key(source: Mapping[str, Any]) -> str:
    for key in ("doi", "pmid", "url", "title"):
        value = str(source.get(key, "") or "").strip().lower()
        if value:
            return f"{key}:{value}"
    return ""


def _format_source_line(source: Mapping[str, Any]) -> str:
    parts: List[str] = []
    title = str(source.get("title", "") or "").strip()
    journal = str(source.get("journal", "") or "").strip()
    year = str(source.get("year", "") or "").strip()
    pmid = str(source.get("pmid", "") or "").strip()
    doi = str(source.get("doi", "") or "").strip()
    url = str(source.get("url", "") or "").strip()

    if title:
        parts.append(title)
    if journal and year:
        parts.append(f"{journal}, {year}")
    elif journal:
        parts.append(journal)
    elif year:
        parts.append(year)
    if pmid:
        parts.append(f"PMID: {pmid}")
    if doi:
        parts.append(f"DOI: {doi}")
    elif url:
        parts.append(url)

    return ". ".join(parts).strip()


def _field_default(field: Any) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    annotation = str(field.type).lower()
    if "list" in annotation or "sequence" in annotation:
        return []
    if "dict" in annotation or "mapping" in annotation:
        return {}
    if "int" in annotation:
        return 0
    if "float" in annotation:
        return 0.0
    if "bool" in annotation:
        return False
    return ""


def _construct_model(model_class: Any, values: Mapping[str, Any]) -> Any:
    if model_class is None:
        return dict(values)
    if not is_dataclass(model_class):
        try:
            return model_class(**values)
        except TypeError:
            return dict(values)

    kwargs: dict[str, Any] = {}
    for field in fields(model_class):
        if field.name in values:
            kwargs[field.name] = values[field.name]
        else:
            kwargs[field.name] = _field_default(field)

    try:
        return model_class(**kwargs)
    except TypeError:
        return dict(values)


def _make_record(snippet: Any, rank: int) -> Any | None:
    text = _get_snippet_text(snippet)
    if not text:
        return None

    paper = _get_paper(snippet)
    title = _get_title(snippet, paper)
    source = _to_source_dict(paper, title)
    source_line = _format_source_line(source)
    score = _get_score(snippet)
    ranker = _get_value(snippet, ("ranker", "ranking_method"), "")

    values = {
        "rank": rank,
        "citation_id": rank,
        "score": score,
        "ranking_score": score,
        "ranker": ranker,
        "ranking_method": ranker,
        "evidence_text": text,
        "text": text,
        "snippet": text,
        "snippet_text": text,
        "passage": text,
        "content": text,
        "title": title,
        "paper_title": title,
        "source_title": title,
        "pmid": source["pmid"],
        "ext_id": source["pmid"],
        "doi": source["doi"],
        "url": source["url"],
        "journal": source["journal"],
        "year": source["year"],
        "authors": source["authors"],
        "source": source,
        "paper": paper,
        "citation": source_line,
        "score_components": _get_value(snippet, ("score_components",), {}),
    }
    return _construct_model(EvidenceRecord, values)


def _record_get(record: Any, names: Iterable[str], default: Any = "") -> Any:
    return _get_value(record, names, default)


def _record_source(record: Any) -> Mapping[str, Any]:
    source = _record_get(record, ("source",), {})
    return source if isinstance(source, Mapping) else {}


def build_context_text(records: Sequence[Any]) -> str:
    blocks: List[str] = []
    for index, record in enumerate(records, start=1):
        rank = _record_get(record, ("rank", "citation_id"), index)
        text = str(_record_get(record, TEXT_KEYS, "") or "").strip()
        if not text:
            continue

        source = _record_source(record)
        source_line = _format_source_line(source) if source else str(_record_get(record, ("citation",), "") or "").strip()

        block_parts = [f"[{rank}] {text}"]
        if source_line:
            block_parts.append(f"Source: {source_line}")
        blocks.append("\n".join(block_parts))

    return "\n\n".join(blocks)


def _deduplicate_records(records: Sequence[Any]) -> List[Any]:
    seen_sources: set[str] = set()
    seen_texts: set[str] = set()
    output: List[Any] = []

    for record in records:
        text_key = _normalize_text(str(_record_get(record, TEXT_KEYS, "") or ""))
        if not text_key:
            continue
        source_key = _source_key(_record_source(record))
        if source_key and source_key in seen_sources:
            continue
        if text_key in seen_texts:
            continue
        if source_key:
            seen_sources.add(source_key)
        seen_texts.add(text_key)
        output.append(record)

    return output


def build_evidence_pack(
    question: str,
    ranked_snippets: Sequence[Any],
    query: str | None = None,
    *,
    max_context_chars: int | None = None,
) -> Any:
    raw_records = []
    for rank, snippet in enumerate(ranked_snippets or [], start=1):
        record = _make_record(snippet, rank)
        if record is not None:
            raw_records.append(record)

    records = _deduplicate_records(raw_records)
    context_text = build_context_text(records)
    if max_context_chars is not None and max_context_chars > 0:
        context_text = context_text[:max_context_chars].rstrip()

    values = {
        "question": question,
        "query": query or "",
        "search_query": query or "",
        "records": records,
        "evidence_records": records,
        "context_text": context_text,
        "context": context_text,
    }
    return _construct_model(EvidencePack, values)


__all__ = [
    "build_context_text",
    "build_evidence_pack",
]
