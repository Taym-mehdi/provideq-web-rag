from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import PAPERCLIP_MODES, PAPERCLIP_RANKINGS, validate_paperclip_source
from .models import Paper, PaperclipRetrieval
from .text_utils import clean_multiline_text, clean_text


class PaperclipError(RuntimeError):
    pass


def _load_client_class() -> type[Any]:
    try:
        from gxl_paperclip import PaperclipClient

        return PaperclipClient
    except ImportError:
        vendored_path = Path.home() / ".paperclip" / "lib"
        if vendored_path.is_dir() and str(vendored_path) not in sys.path:
            sys.path.insert(0, str(vendored_path))
        try:
            from gxl_paperclip import PaperclipClient

            return PaperclipClient
        except ImportError as exc:
            raise PaperclipError(
                "Paperclip SDK was not found. Install Paperclip or make ~/.paperclip/lib available."
            ) from exc


def create_client() -> Any:
    try:
        return _load_client_class().from_env()
    except Exception as exc:
        raise PaperclipError(
            "Paperclip authentication failed. Set PAPERCLIP_API_KEY or complete Paperclip login."
        ) from exc


def _to_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        data = value.model_dump()
        return dict(data) if isinstance(data, Mapping) else None
    if hasattr(value, "dict"):
        data = value.dict()
        return dict(data) if isinstance(data, Mapping) else None
    data = getattr(value, "__dict__", None)
    return dict(data) if isinstance(data, Mapping) else None


def _parse_json(text: str) -> Any | None:
    value = (text or "").strip()
    if not value:
        return None
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _extract_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for value in payload if (item := _to_mapping(value)) is not None]
    mapping = _to_mapping(payload)
    if mapping is None:
        return []
    for key in ("papers", "results", "items", "documents", "hits"):
        values = mapping.get(key)
        if isinstance(values, list):
            return [item for value in values if (item := _to_mapping(value)) is not None]
    for key in ("result_data", "data", "payload"):
        if key in mapping:
            values = _extract_list(mapping[key])
            if values:
                return values
    return []


def _extract_search_hits(client: Any, result: Any) -> list[dict[str, Any]]:
    for payload in (
        getattr(result, "papers", None),
        getattr(result, "result_data", None),
        getattr(result, "raw", None),
        _parse_json(str(getattr(result, "output", "") or "")),
    ):
        hits = _extract_list(payload)
        if hits:
            return hits

    result_id = str(getattr(result, "result_id", "") or "")
    results_api = getattr(client, "results", None)
    if result_id and results_api is not None:
        try:
            saved = results_api.get(result_id)
        except Exception:
            saved = None
        if saved is not None:
            for payload in (
                getattr(saved, "result_data", None),
                getattr(saved, "raw", None),
                _parse_json(str(getattr(saved, "output", "") or "")),
            ):
                hits = _extract_list(payload)
                if hits:
                    return hits
    return []


def _find_value(value: Any, keys: tuple[str, ...]) -> Any:
    normalized_keys = {key.casefold().replace("-", "_") for key in keys}
    mapping = _to_mapping(value)
    if mapping is not None:
        for key, item in mapping.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in normalized_keys and item not in (None, "", []):
                return item
        for item in mapping.values():
            found = _find_value(item, keys)
            if found not in (None, "", []):
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_value(item, keys)
            if found not in (None, "", []):
                return found
    return ""


_PAPER_ID_PATTERN = re.compile(
    r"(?:PMC\d+|bio_[A-Za-z0-9._-]+|med_[A-Za-z0-9._-]+|arx_[A-Za-z0-9._-]+)"
)


def _paper_id(metadata: dict[str, Any]) -> str:
    value = _find_value(
        metadata,
        ("id", "paper_id", "document_id", "pmcid", "pmc_id", "path", "paper_path"),
    )
    match = _PAPER_ID_PATTERN.search(str(value or ""))
    return match.group(0) if match else clean_text(str(value or ""))


def _execute_search(client: Any, args: list[str], timeout: float) -> Any:
    try:
        return client.execute("search", args, timeout=timeout)
    except TypeError:
        return client.execute("search", args)
    except Exception as exc:
        raise PaperclipError(f"Paperclip search failed: {exc}") from exc


def _read_file(client: Any, path: str, *, lines: int | None = None) -> Any | None:
    papers_api = getattr(client, "papers", None)
    if papers_api is not None:
        try:
            if lines is None:
                return papers_api.cat(path)
            return papers_api.head(path, lines=lines)
        except Exception:
            pass

    if hasattr(client, "execute"):
        try:
            if lines is None:
                return client.execute("cat", [path])
            return client.execute("head", ["-n", str(lines), path])
        except Exception:
            pass
    return None


def _read_metadata(client: Any, paper_id: str) -> dict[str, Any]:
    result = _read_file(client, f"/papers/{paper_id}/meta.json")
    if result is None:
        return {}
    for payload in (
        getattr(result, "result_data", None),
        getattr(result, "raw", None),
        _parse_json(str(getattr(result, "output", "") or "")),
    ):
        mapping = _to_mapping(payload)
        if mapping:
            nested = mapping.get("result_data")
            return dict(nested) if isinstance(nested, Mapping) else mapping
    return {}


def _read_full_text(client: Any, paper_id: str, max_lines: int) -> str:
    result = _read_file(client, f"/papers/{paper_id}/content.lines", lines=max_lines)
    if result is None:
        return ""
    output = str(getattr(result, "output", "") or "")
    lines: list[str] = []
    for raw_line in output.splitlines():
        line = re.sub(r"^L\d+:\s*", "", raw_line.strip())
        if not line or re.fullmatch(r"\[and \d+ more\]", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return clean_multiline_text("\n".join(lines))


def _authors(value: Any) -> str:
    if isinstance(value, str):
        return clean_text(value)
    if not isinstance(value, list):
        return ""
    names: list[str] = []
    for item in value:
        if isinstance(item, str):
            name = clean_text(item)
        else:
            name = clean_text(str(_find_value(item, ("name", "full_name", "display_name")) or ""))
        if name:
            names.append(name)
    return ", ".join(dict.fromkeys(names))


def _year(metadata: dict[str, Any]) -> str:
    value = _find_value(
        metadata,
        ("year", "publication_year", "pub_year", "date", "published", "publication_date"),
    )
    match = re.search(r"\b(?:19|20)\d{2}\b", str(value or ""))
    return match.group(0) if match else ""


def _doi(metadata: dict[str, Any], text: str) -> str:
    value = _find_value(
        metadata,
        ("doi", "article_doi", "digital_object_identifier", "article_id", "elocation_id"),
    )
    candidate = clean_text(str(value or "")) or text[:12000]
    candidate = re.sub(
        r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", candidate)
    return match.group(0).rstrip(".,;:)]}") if match else ""


def _source_name(paper_id: str, requested_source: str) -> str:
    if paper_id.startswith("PMC"):
        return "pmc"
    if paper_id.startswith("bio_"):
        return "biorxiv"
    if paper_id.startswith("med_"):
        return "medrxiv"
    if paper_id.startswith("arx_"):
        return "arxiv"
    return requested_source


def _paper_url(metadata: dict[str, Any], paper_id: str) -> str:
    value = clean_text(
        str(_find_value(metadata, ("url", "paper_url", "source_url", "landing_page_url")) or "")
    )
    if value:
        return value
    if paper_id.startswith("PMC"):
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{paper_id}/"
    return ""


def _normalize_paper(
    hit: dict[str, Any],
    file_metadata: dict[str, Any],
    full_text: str,
    requested_source: str,
    retrieval_rank: int,
    *,
    allow_metadata_only: bool = False,
) -> Paper | None:
    combined = {**hit, **file_metadata}
    paper_id = _paper_id(hit) or _paper_id(file_metadata)
    if not paper_id:
        return None

    title = clean_text(
        str(_find_value(combined, ("title", "paper_title", "document_title")) or paper_id)
    )
    abstract = clean_text(
        str(_find_value(combined, ("abstract", "summary", "snippet", "description")) or "")
    )
    text = full_text or abstract
    if not text and allow_metadata_only:
        text = title
    if not text:
        return None

    return Paper(
        paper_id=paper_id,
        title=title,
        text=text,
        source=_source_name(paper_id, requested_source),
        year=_year(combined),
        doi=_doi(combined, text),
        authors=_authors(_find_value(combined, ("authors", "author", "author_string", "creator"))),
        journal=clean_text(str(_find_value(combined, ("journal", "journal_title", "venue")) or "")),
        url=_paper_url(combined, paper_id),
        abstract=abstract,
        retrieval_rank=retrieval_rank,
        metadata={"search_hit": hit, "paperclip_file": file_metadata, "has_full_text": bool(full_text)},
    )



def _normalize_title_key(value: str) -> str:
    text = clean_text(value).casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _normalize_doi_key(value: str) -> str:
    text = clean_text(value)
    text = re.sub(
        r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return match.group(0).rstrip(".,;:)]}").casefold() if match else ""


def _deduplication_keys(paper: Paper) -> dict[str, str]:
    paper_id = clean_text(paper.paper_id)
    pmcid_match = re.search(r"\bPMC\d+\b", paper_id, flags=re.IGNORECASE)
    return {
        "doi": _normalize_doi_key(paper.doi),
        "pmcid": pmcid_match.group(0).upper() if pmcid_match else "",
        "title": _normalize_title_key(paper.title),
        "paper_id": paper_id.casefold(),
    }


def _is_duplicate_paper(
    paper: Paper,
    seen: dict[str, set[str]],
) -> bool:
    keys = _deduplication_keys(paper)
    for key_type in ("doi", "pmcid", "title", "paper_id"):
        value = keys[key_type]
        if value and value in seen[key_type]:
            return True

    for key_type, value in keys.items():
        if value:
            seen[key_type].add(value)
    return False

def retrieve_papers(
    query: str,
    *,
    limit: int = 10,
    source: str = "pmc",
    ranking: str = "hybrid",
    max_full_text_lines: int = 5000,
    mode: str | None = None,
    since: str | None = None,
    sort: str | None = None,
    year: int | str | None = None,
    journal: str | None = None,
    article_type: str | None = None,
    author: str | None = None,
    full_corpus: bool = True,
    load_full_text: bool = True,
    timeout: float = 120.0,
    client: Any | None = None,
) -> PaperclipRetrieval:
    query = clean_text(query)
    if not query:
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if max_full_text_lines <= 0:
        raise ValueError("max_full_text_lines must be greater than 0")
    validate_paperclip_source(source)
    if ranking not in PAPERCLIP_RANKINGS:
        raise ValueError(f"ranking must be one of: {', '.join(PAPERCLIP_RANKINGS)}")
    if mode is not None and mode not in PAPERCLIP_MODES:
        raise ValueError(f"mode must be one of: {', '.join(PAPERCLIP_MODES)}")

    active_client = client or create_client()
    # Ask Paperclip for extra candidates so cross-source duplicates do not reduce
    # the final number of distinct papers below the requested limit.
    search_limit = min(max(limit * 3, limit), 1000)
    args = ["--source", source, "--ranking", ranking, "-n", str(search_limit), "--json"]
    if mode:
        args.extend(["-m", mode])
    if since:
        args.extend(["--since", since])
    if sort:
        args.extend(["--sort", sort])
    if year is not None:
        args.extend(["--year", str(year)])
    if journal:
        args.extend(["--journal", journal])
    if article_type:
        args.extend(["--article-type", article_type])
    if author:
        args.extend(["--author", author])
    if full_corpus:
        args.append("--all")
    args.append(query)

    result = _execute_search(active_client, args, timeout)
    if getattr(result, "exit_code", 0) not in (None, 0):
        message = clean_text(str(getattr(result, "output", "") or "Paperclip search failed"))
        raise PaperclipError(message)

    papers: list[Paper] = []
    seen: dict[str, set[str]] = {
        "doi": set(),
        "pmcid": set(),
        "title": set(),
        "paper_id": set(),
    }
    hits = _extract_search_hits(active_client, result)
    for original_rank, hit in enumerate(hits, start=1):
        paper_id = _paper_id(hit)
        if not paper_id:
            continue
        metadata = _read_metadata(active_client, paper_id)
        full_text = (
            _read_full_text(active_client, paper_id, max_full_text_lines)
            if load_full_text
            else ""
        )
        paper = _normalize_paper(
            hit,
            metadata,
            full_text,
            source,
            len(papers) + 1,
            allow_metadata_only=not load_full_text,
        )
        if paper is None or _is_duplicate_paper(paper, seen):
            continue

        paper.metadata["paperclip_original_rank"] = original_rank
        papers.append(paper)
        if len(papers) >= limit:
            break

    return PaperclipRetrieval(
        papers=papers,
        result_id=str(getattr(result, "result_id", "") or ""),
    )
