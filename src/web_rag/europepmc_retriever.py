from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Paper
from .text_utils import clean_text


EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMCError(RuntimeError):
    """Europe PMC request failure with optional query-variant context."""

    def __init__(
        self,
        message: str,
        *,
        query: str = "",
        failed_variant: str = "",
        failed_variant_index: int | None = None,
        query_variants: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.query = query
        self.failed_variant = failed_variant
        self.failed_variant_index = failed_variant_index
        self.query_variants = list(query_variants or [])


@dataclass(frozen=True)
class EuropePMCRetrieval:
    papers: list[Paper]
    query_variants: list[str]


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "being",
    "between", "both", "by", "can", "could", "did", "do", "does", "during",
    "either", "for", "from", "had", "has", "have", "how", "if", "in", "into",
    "is", "it", "its", "may", "of", "on", "or", "should", "than", "that",
    "the", "their", "them", "there", "these", "they", "this", "to", "up",
    "was", "were", "what", "when", "where", "which", "while", "with", "without",
    "would", "i", "my", "we", "our", "you", "your", "much", "many", "well",
    "one", "two", "three", "four", "five", "several", "any", "all", "most",
    "group", "groups", "stayed", "kept", "least",
}

# Short biomedical tokens should not be discarded just because they are short.
_SHORT_BIOMEDICAL = {
    "rna", "dna", "csf", "hdl", "ldl", "edta", "nmr", "sop", "sops", "pcr",
    "mirna", "pbmc", "pbmcs", "ph", "rt", "sst", "il6", "crp",
}

_SPECIMEN_TERMS = {
    "blood", "serum", "plasma", "urine", "tissue", "csf", "saliva", "stool",
    "biospecimen", "biospecimens", "sample", "samples",
}

_PREANALYTICAL_TERMS = {
    "centrifugation", "centrifuge", "freeze", "freezing", "thaw", "thawing",
    "storage", "stored", "temperature", "delay", "delayed", "processing",
    "collection", "transport", "handling", "aliquot", "aliquots", "ischemia",
}

_GENERIC_TERMS = {
    "effect", "effects", "affect", "affected", "change", "changed", "changes",
    "stable", "stability", "measurement", "measurements", "measured", "level",
    "levels", "result", "results", "study", "studied", "evaluated", "use", "used",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")


def _tokenize(query: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(query):
        token = raw.strip(".-").casefold()
        if not token or token in _STOPWORDS:
            continue
        if len(token) < 3 and token not in _SHORT_BIOMEDICAL and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _token_score(token: str, position: int) -> tuple[int, int, int]:
    score = 0
    if token in _SPECIMEN_TERMS:
        score += 5
    if token in _PREANALYTICAL_TERMS:
        score += 5
    if token in _SHORT_BIOMEDICAL:
        score += 5
    if any(ch.isdigit() for ch in token):
        score += 3
    if len(token) >= 10:
        score += 4
    elif len(token) >= 7:
        score += 3
    elif len(token) >= 5:
        score += 2
    else:
        score += 1
    if token in _GENERIC_TERMS:
        score -= 2
    # Earlier terms in a biomedical question usually carry the central entity/condition.
    return score, -position, len(token)


def _important_terms(query: str, count: int) -> list[str]:
    tokens = _tokenize(query)
    if len(tokens) <= count:
        return tokens
    ranked_positions = sorted(
        range(len(tokens)),
        key=lambda idx: _token_score(tokens[idx], idx),
        reverse=True,
    )[:count]
    selected = set(ranked_positions)
    return [token for idx, token in enumerate(tokens) if idx in selected]


def build_query_variants(query: str, *, mode: str = "multi") -> list[str]:
    """Build Europe PMC queries from a user question.

    Europe PMC combines unqualified terms with AND. A long natural-language question can
    therefore become unnecessarily restrictive. ``multi`` uses several progressively
    relaxed title/abstract keyword queries and later fuses their rankings.
    """
    query = clean_text(query)
    if not query:
        raise ValueError("query must not be empty")
    if mode not in {"direct", "multi"}:
        raise ValueError("mode must be 'direct' or 'multi'")
    if mode == "direct":
        return [query]

    variants: list[str] = []
    # Keep the original free-text query as one signal. Europe PMC applies its own parser.
    variants.append(query)

    token_count = len(_tokenize(query))
    for count in (8, 6, 4):
        if token_count < count:
            continue
        terms = _important_terms(query, count)
        if not terms:
            continue
        variants.append("TITLE_ABS:(" + " ".join(terms) + ")")

    # If the question is already short, still add one title/abstract-focused variant.
    if len(variants) == 1:
        terms = _important_terms(query, min(6, max(1, token_count)))
        if terms:
            variants.append("TITLE_ABS:(" + " ".join(terms) + ")")

    # Stable de-duplication.
    return list(dict.fromkeys(variants))


def _request_json(
    params: dict[str, Any],
    *,
    timeout: float,
    retries: int = 2,
) -> dict[str, Any]:
    url = EUROPEPMC_SEARCH_URL + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": os.getenv(
                    "WEB_RAG_EUROPEPMC_USER_AGENT",
                    "ProvideQ-WebRAG/1.0",
                ),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise EuropePMCError("Europe PMC returned an unexpected response")
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            # The search path is fixed, so an intermittent empty 404 is normally a
            # gateway/service failure rather than a missing resource. Retrying it is
            # safe and has proven useful during benchmark sweeps.
            if exc.code not in {404, 408, 429, 500, 502, 503, 504} or attempt >= retries:
                raise EuropePMCError(
                    f"Europe PMC HTTP {exc.code}: {clean_text(body)[:300]}"
                ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                raise EuropePMCError(f"Europe PMC request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise EuropePMCError("Europe PMC returned invalid JSON") from exc

        time.sleep(min(2 ** attempt, 4))

    raise EuropePMCError(f"Europe PMC request failed: {last_error}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"true", "1", "yes", "y"}


def _authors(item: dict[str, Any]) -> str:
    direct = clean_text(item.get("authorString", ""))
    if direct:
        return direct
    author_list = item.get("authorList")
    if not isinstance(author_list, dict):
        return ""
    authors = author_list.get("author")
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        name = clean_text(
            author.get("fullName", "")
            or author.get("collectiveName", "")
            or " ".join(
                part
                for part in (
                    str(author.get("firstName", "") or "").strip(),
                    str(author.get("lastName", "") or "").strip(),
                )
                if part
            )
        )
        if name:
            names.append(name)
    return ", ".join(dict.fromkeys(names))


def _paper_id(item: dict[str, Any]) -> str:
    pmcid = clean_text(item.get("pmcid", "")).upper()
    if pmcid:
        return pmcid
    pmid = clean_text(item.get("pmid", ""))
    if pmid:
        return f"PMID:{pmid}"
    source = clean_text(item.get("source", "")).upper()
    ext_id = clean_text(item.get("id", "") or item.get("extId", ""))
    if source and ext_id:
        return f"EPMC:{source}:{ext_id}"
    doi = clean_text(item.get("doi", ""))
    if doi:
        return f"DOI:{doi}"
    return ""


def _url(item: dict[str, Any], paper_id: str) -> str:
    pmcid = clean_text(item.get("pmcid", "")).upper()
    if pmcid:
        return f"https://europepmc.org/article/PMC/{pmcid}"
    pmid = clean_text(item.get("pmid", ""))
    if pmid:
        return f"https://europepmc.org/article/MED/{pmid}"
    doi = clean_text(item.get("doi", ""))
    if doi:
        return f"https://doi.org/{doi}"
    return ""


def _normalize_item(item: dict[str, Any], rank: int, query_variant: str) -> Paper | None:
    paper_id = _paper_id(item)
    title = clean_text(item.get("title", ""))
    if not paper_id or not title:
        return None

    abstract = clean_text(item.get("abstractText", ""))
    pmcid = clean_text(item.get("pmcid", "")).upper()
    pmid = clean_text(item.get("pmid", ""))
    doi = clean_text(item.get("doi", ""))
    source_code = clean_text(item.get("source", "")).upper()
    in_epmc = _as_bool(item.get("inEPMC"))

    return Paper(
        paper_id=paper_id,
        title=title,
        text=abstract or title,
        source="europepmc",
        year=clean_text(item.get("pubYear", "")),
        doi=doi,
        authors=_authors(item),
        journal=clean_text(item.get("journalTitle", "")),
        url=_url(item, paper_id),
        abstract=abstract,
        retrieval_rank=rank,
        metadata={
            "pmcid": pmcid,
            "pmid": pmid,
            "doi": doi,
            "europepmc_source": source_code,
            "europepmc_id": clean_text(item.get("id", "") or item.get("extId", "")),
            "in_epmc": in_epmc,
            # ``inEPMC`` means full text is available from Europe PMC; this
            # search endpoint still returns only metadata and an abstract.
            "full_text_available": in_epmc,
            "has_full_text": False,
            "query_variant": query_variant,
            "raw": item,
        },
    )


def _normalize_title(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", clean_text(value).casefold()).split())


def _normalize_doi(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", text, flags=re.I)
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return match.group(0).rstrip(".,;:)]}").casefold() if match else ""


def paper_keys(paper: Paper) -> list[str]:
    keys: list[str] = []
    doi = _normalize_doi(paper.doi or str(paper.metadata.get("doi", "")))
    pmcid = clean_text(paper.metadata.get("pmcid", "")).upper()
    pmid = clean_text(paper.metadata.get("pmid", ""))
    title = _normalize_title(paper.title)
    if doi:
        keys.append("doi:" + doi)
    if pmcid:
        keys.append("pmcid:" + pmcid)
    if pmid:
        keys.append("pmid:" + pmid)
    if title:
        keys.append("title:" + title)
    return keys


def _search_variant(
    query: str,
    *,
    limit: int,
    synonym: bool,
    timeout: float,
    email: str | None,
    cache_dir: Path | None,
) -> list[Paper]:
    params: dict[str, Any] = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": limit,
        "synonym": "true" if synonym else "false",
    }
    if email:
        params["email"] = email

    payload: dict[str, Any] | None = None
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_key_payload = {k: v for k, v in params.items() if k != "email"}
        cache_key = hashlib.sha256(
            json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    payload = cached
            except (OSError, json.JSONDecodeError):
                payload = None

    if payload is None:
        payload = _request_json(params, timeout=timeout)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(cache_path)
    result_list = payload.get("resultList", {})
    raw_results = result_list.get("result", []) if isinstance(result_list, dict) else []
    if not isinstance(raw_results, list):
        return []

    papers: list[Paper] = []
    for rank, item in enumerate(raw_results, start=1):
        if not isinstance(item, dict):
            continue
        paper = _normalize_item(item, rank, query)
        if paper is not None:
            papers.append(paper)
    return papers


def retrieve_papers_europepmc(
    query: str,
    *,
    limit: int = 10,
    candidate_limit: int | None = None,
    synonym: bool = True,
    mode: str = "multi",
    rrf_k: int = 60,
    timeout: float = 45.0,
    email: str | None = None,
    cache_dir: str | Path | None = "outputs/europepmc_cache",
) -> EuropePMCRetrieval:
    """Retrieve and rank Europe PMC papers.

    ``direct`` uses one Europe PMC query. ``multi`` creates progressively relaxed
    title/abstract queries and fuses their relevance rankings with Reciprocal Rank Fusion.
    """
    query = clean_text(query)
    if not query:
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if candidate_limit is None:
        candidate_limit = max(limit * 3, 30)
    if not 1 <= candidate_limit <= 1000:
        raise ValueError("candidate_limit must be between 1 and 1000")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")

    variants = build_query_variants(query, mode=mode)
    email = email or os.getenv("EUROPEPMC_EMAIL", "").strip() or None
    resolved_cache_dir = Path(cache_dir) if cache_dir else None

    def search_variant(variant: str, variant_index: int, search_limit: int) -> list[Paper]:
        try:
            return _search_variant(
                variant,
                limit=search_limit,
                synonym=synonym,
                timeout=timeout,
                email=email,
                cache_dir=resolved_cache_dir,
            )
        except Exception as exc:
            detail = clean_text(str(exc)) or type(exc).__name__
            raise EuropePMCError(
                "Europe PMC query variant "
                f"{variant_index}/{len(variants)} failed: {variant} | {detail}",
                query=query,
                failed_variant=variant,
                failed_variant_index=variant_index,
                query_variants=variants,
            ) from exc

    if mode == "direct":
        papers = search_variant(variants[0], 1, max(limit, candidate_limit))[:limit]
        for rank, paper in enumerate(papers, start=1):
            paper.retrieval_rank = rank
        return EuropePMCRetrieval(papers=papers, query_variants=variants)

    # RRF over the source's own query variants.
    entries: list[dict[str, Any]] = []
    key_to_index: dict[str, int] = {}
    for variant_index, variant in enumerate(variants, start=1):
        papers = search_variant(variant, variant_index, candidate_limit)
        for rank, paper in enumerate(papers, start=1):
            keys = paper_keys(paper)
            existing_index = next(
                (key_to_index[key] for key in keys if key in key_to_index),
                None,
            )
            if existing_index is None:
                existing_index = len(entries)
                entries.append(
                    {
                        "paper": paper,
                        "rrf": 0.0,
                        "best_rank": rank,
                        "variant_ranks": {},
                    }
                )
            entry = entries[existing_index]
            entry["rrf"] += 1.0 / (rrf_k + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
            entry["variant_ranks"][str(variant_index)] = rank
            for key in keys:
                key_to_index[key] = existing_index
        # Avoid hammering the public service when several variants are used.
        if variant_index < len(variants):
            time.sleep(0.05)

    entries.sort(
        key=lambda entry: (
            -float(entry["rrf"]),
            int(entry["best_rank"]),
            clean_text(entry["paper"].title).casefold(),
        )
    )

    final: list[Paper] = []
    for rank, entry in enumerate(entries[:limit], start=1):
        paper = entry["paper"]
        paper.retrieval_rank = rank
        paper.metadata["europepmc_rrf_score"] = float(entry["rrf"])
        paper.metadata["europepmc_variant_ranks"] = dict(entry["variant_ranks"])
        paper.metadata["europepmc_query_variants"] = list(variants)
        final.append(paper)

    return EuropePMCRetrieval(papers=final, query_variants=variants)
