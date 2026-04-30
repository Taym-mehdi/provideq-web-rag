from __future__ import annotations

from typing import Any

import requests

from web_rag.config import get_settings
from web_rag.models import Paper, QueryBundle
from web_rag.text_utils import clean_text


def build_paper_url(item: dict[str, Any]) -> str:
    """
    Build a stable external URL for a retrieved paper.

    Priority:
    1. PMCID link if available
    2. PubMed link if this is a MED/PubMed record
    3. Europe PMC / full text URL if available
    4. empty string if no useful URL exists
    """
    source = item.get("source", "") or ""
    ext_id = item.get("id", "") or ""
    pmcid = item.get("pmcid", "") or ""

    if pmcid:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

    if source == "MED" and ext_id:
        return f"https://pubmed.ncbi.nlm.nih.gov/{ext_id}/"

    full_text_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
    if full_text_urls:
        first_url = full_text_urls[0]
        return first_url.get("url", "") or ""

    return ""


def normalize_paper(item: dict[str, Any]) -> Paper | None:
    """
    Convert one Europe PMC result item into our internal Paper model.

    Returning None means the result is not useful enough for our pipeline.
    For now, we require at least a title.
    """
    title = clean_text(item.get("title", "") or "")
    if not title:
        return None

    abstract = clean_text(item.get("abstractText", "") or "")

    return Paper(
        title=title,
        abstract=abstract,
        year=str(item.get("pubYear", "") or ""),
        source=item.get("source", "") or "",
        ext_id=item.get("id", "") or "",
        doi=item.get("doi", "") or "",
        authors=clean_text(item.get("authorString", "") or ""),
        journal=clean_text(item.get("journalTitle", "") or ""),
        url=build_paper_url(item),
    )


def search_europe_pmc(query: QueryBundle, page_size: int | None = None) -> list[Paper]:
    """
    Search Europe PMC and return normalized Paper objects.

    This function is intentionally responsible only for:
    - sending the request
    - receiving the JSON response
    - normalizing results into Paper objects

    It does not perform snippet extraction, ranking, or answer generation.
    """
    settings = get_settings()
    effective_page_size = page_size or settings.default_page_size

    params = {
        "query": query.search_query,
        "format": "json",
        "resultType": "core",
        "pageSize": effective_page_size,
    }

    headers = {
        "User-Agent": settings.user_agent,
    }

    response = requests.get(
        settings.europe_pmc_search_url,
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    raw_results = data.get("resultList", {}).get("result", [])

    papers: list[Paper] = []

    for item in raw_results:
        paper = normalize_paper(item)
        if paper is not None:
            papers.append(paper)

    return papers