"""Global reranker dispatcher for the ProvideQ Web RAG module."""

from __future__ import annotations

import re
from typing import Any, List, Sequence

from .config import (
    BM25_B,
    BM25_K1,
    DEFAULT_MIN_SNIPPET_WORD_COUNT,
    DEFAULT_RANKER,
    HYBRID_LEXICAL_WEIGHT,
    HYBRID_MEDCPT_WEIGHT,
    MEDCPT_ARTICLE_ENCODER_MODEL,
    MEDCPT_BATCH_SIZE,
    MEDCPT_QUERY_ENCODER_MODEL,
)
from .hybrid_reranker import rank_hybrid_snippets
from .lexical_reranker import rank_lexical_snippets
from .medcpt_embedding_reranker import rank_medcpt_embedding_snippets


RANKER_ALIASES = {
    "lexical": "lexical",
    "bm25": "lexical",
    "medcpt": "medcpt",
    "embedding": "medcpt",
    "dense": "medcpt",
    "medcpt_embedding": "medcpt",
    "hybrid": "hybrid",
    "medcpt_hybrid": "hybrid",
}

RANKER_CHOICES = tuple(RANKER_ALIASES.keys())
TEXT_KEYS = ("evidence_text", "text", "snippet", "snippet_text", "content", "passage", "abstract")
PAPER_KEYS = ("paper", "source", "document")


def normalize_ranker_name(ranker: str | None) -> str:
    name = (ranker or DEFAULT_RANKER).strip().lower().replace("-", "_")
    if name not in RANKER_ALIASES:
        valid = ", ".join(sorted(RANKER_ALIASES))
        raise ValueError(f"Unknown ranker '{ranker}'. Valid rankers: {valid}")
    return RANKER_ALIASES[name]


def _question_from_query_info(query_info: Any) -> str:
    if query_info is None:
        return ""
    if isinstance(query_info, dict):
        return str(query_info.get("question") or query_info.get("query") or "")
    return str(getattr(query_info, "question", None) or getattr(query_info, "query", None) or query_info)


def _get_value(obj: Any, names: Sequence[str], default: Any = "") -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
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


def _snippet_text(snippet: Any) -> str:
    return str(_get_value(snippet, TEXT_KEYS, "") or "").strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _paper_key(snippet: Any) -> str:
    paper = _get_paper(snippet)
    doi = str(_get_value(paper, ("doi",), "") or "").strip().lower()
    pmid = str(_get_value(paper, ("pmid", "pubmed_id", "ext_id", "id"), "") or "").strip().lower()
    url = str(_get_value(paper, ("url", "full_text_url", "source_url"), "") or "").strip().lower()
    title = str(_get_value(paper, ("title", "paper_title", "source_title"), "") or "").strip().lower()

    for prefix, value in (("doi", doi), ("pmid", pmid), ("url", url), ("title", title)):
        if value:
            return f"{prefix}:{value}"
    return ""


def _filter_usable_snippets(snippets: Sequence[Any], min_word_count: int) -> List[Any]:
    output: List[Any] = []
    for snippet in snippets or []:
        text = _snippet_text(snippet)
        if not text:
            continue
        if min_word_count > 0 and _word_count(text) < min_word_count:
            continue
        output.append(snippet)
    return output


def _deduplicate_and_limit(snippets: Sequence[Any], top_k: int | None) -> List[Any]:
    seen_papers: set[str] = set()
    seen_texts: set[str] = set()
    output: List[Any] = []

    for snippet in snippets or []:
        text_key = _normalize_text(_snippet_text(snippet))
        if not text_key:
            continue

        paper_key = _paper_key(snippet)
        if paper_key and paper_key in seen_papers:
            continue
        if text_key in seen_texts:
            continue

        if paper_key:
            seen_papers.add(paper_key)
        seen_texts.add(text_key)
        output.append(snippet)

        if top_k is not None and len(output) >= top_k:
            break

    return output


def rank_snippets(
    question: str | None = None,
    snippets: Sequence[Any] | None = None,
    *,
    query_info: Any = None,
    ranker: str = DEFAULT_RANKER,
    method: str | None = None,
    ranker_type: str | None = None,
    top_k: int | None = None,
    min_snippet_word_count: int = DEFAULT_MIN_SNIPPET_WORD_COUNT,
    bm25_k1: float = BM25_K1,
    bm25_b: float = BM25_B,
    lexical_weight: float = HYBRID_LEXICAL_WEIGHT,
    medcpt_weight: float = HYBRID_MEDCPT_WEIGHT,
    query_model_name: str = MEDCPT_QUERY_ENCODER_MODEL,
    article_model_name: str = MEDCPT_ARTICLE_ENCODER_MODEL,
    batch_size: int = MEDCPT_BATCH_SIZE,
    device: str | None = None,
    **_: Any,
) -> List[Any]:
    question_text = question or _question_from_query_info(query_info)
    candidates = _filter_usable_snippets(snippets or [], min_snippet_word_count)
    if not candidates:
        return []

    selected_ranker = normalize_ranker_name(method or ranker_type or ranker)

    if selected_ranker == "lexical":
        ranked = rank_lexical_snippets(
            question_text,
            candidates,
            top_k=None,
            k1=bm25_k1,
            b=bm25_b,
        )
    elif selected_ranker == "medcpt":
        ranked = rank_medcpt_embedding_snippets(
            question_text,
            candidates,
            top_k=None,
            query_model_name=query_model_name,
            article_model_name=article_model_name,
            batch_size=batch_size,
            device=device,
        )
    elif selected_ranker == "hybrid":
        ranked = rank_hybrid_snippets(
            question_text,
            candidates,
            top_k=None,
            lexical_weight=lexical_weight,
            medcpt_weight=medcpt_weight,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            query_model_name=query_model_name,
            article_model_name=article_model_name,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported ranker: {selected_ranker}")

    return _deduplicate_and_limit(ranked, top_k)


def rerank_snippets(*args: Any, **kwargs: Any) -> List[Any]:
    return rank_snippets(*args, **kwargs)


__all__ = [
    "RANKER_ALIASES",
    "RANKER_CHOICES",
    "normalize_ranker_name",
    "rank_snippets",
    "rerank_snippets",
]
