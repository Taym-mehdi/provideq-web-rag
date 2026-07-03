"""MedCPT embedding reranker.

Paper basis:
Jin, Q. et al. (2023). MedCPT: Contrastive Pre-trained Transformers with
large-scale PubMed search logs for zero-shot biomedical information retrieval.
Bioinformatics, 39(11), btad651.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Sequence, Tuple

from .config import MEDCPT_ARTICLE_ENCODER_MODEL, MEDCPT_BATCH_SIZE, MEDCPT_QUERY_ENCODER_MODEL


def get_snippet_text(snippet: Any) -> str:
    if snippet is None:
        return ""
    if isinstance(snippet, dict):
        for key in ("evidence_text", "text", "snippet", "snippet_text", "content", "passage", "abstract"):
            if snippet.get(key):
                return str(snippet[key])
        return ""
    for attr in ("evidence_text", "text", "snippet", "snippet_text", "content", "passage", "abstract"):
        value = getattr(snippet, attr, None)
        if value:
            return str(value)
    return ""


def get_snippet_title(snippet: Any) -> str:
    if snippet is None:
        return ""
    if isinstance(snippet, dict):
        paper = snippet.get("paper") or snippet.get("source") or {}
        if isinstance(paper, dict):
            return str(paper.get("title", "") or "")
        return str(getattr(paper, "title", "") or "")
    if getattr(snippet, "title", None):
        return str(getattr(snippet, "title"))
    paper = getattr(snippet, "paper", None) or getattr(snippet, "source", None)
    if isinstance(paper, dict):
        return str(paper.get("title", "") or "")
    return str(getattr(paper, "title", "") or "")


def set_ranking_metadata(snippet: Any, score: float, ranker: str) -> None:
    for attr, value in (
        ("score", float(score)),
        ("ranking_score", float(score)),
        ("ranker", ranker),
        ("ranking_method", ranker),
    ):
        try:
            setattr(snippet, attr, value)
        except Exception:
            pass


@lru_cache(maxsize=4)
def _load_medcpt_models(
    query_model_name: str,
    article_model_name: str,
    device: str | None,
):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "MedCPT reranking requires torch and transformers. Install the project requirements first."
        ) from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
    query_model = AutoModel.from_pretrained(query_model_name).to(resolved_device)
    query_model.eval()

    article_tokenizer = AutoTokenizer.from_pretrained(article_model_name)
    article_model = AutoModel.from_pretrained(article_model_name).to(resolved_device)
    article_model.eval()

    return query_tokenizer, query_model, article_tokenizer, article_model, resolved_device


def _encode_query(question: str, tokenizer: Any, model: Any, device: str):
    import torch

    encoded = tokenizer(
        [question],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=64,
    ).to(device)

    with torch.no_grad():
        return model(**encoded).last_hidden_state[:, 0, :]


def _encode_articles(
    snippets: Sequence[Any],
    tokenizer: Any,
    model: Any,
    device: str,
    *,
    batch_size: int,
):
    import torch

    pairs = [[get_snippet_title(snippet), get_snippet_text(snippet)] for snippet in snippets]
    embeddings = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        with torch.no_grad():
            embeddings.append(model(**encoded).last_hidden_state[:, 0, :])

    if not embeddings:
        return torch.empty((0, 768), device=device)

    return torch.cat(embeddings, dim=0)


def medcpt_embedding_scores(
    question: str,
    snippets: Sequence[Any],
    *,
    query_model_name: str = MEDCPT_QUERY_ENCODER_MODEL,
    article_model_name: str = MEDCPT_ARTICLE_ENCODER_MODEL,
    batch_size: int = MEDCPT_BATCH_SIZE,
    device: str | None = None,
) -> List[float]:
    candidates = list(snippets or [])
    if not candidates:
        return []

    import torch

    query_tokenizer, query_model, article_tokenizer, article_model, resolved_device = _load_medcpt_models(
        query_model_name,
        article_model_name,
        device,
    )

    query_embedding = _encode_query(question, query_tokenizer, query_model, resolved_device)
    article_embeddings = _encode_articles(
        candidates,
        article_tokenizer,
        article_model,
        resolved_device,
        batch_size=batch_size,
    )

    scores = torch.matmul(article_embeddings, query_embedding.T).squeeze(dim=1)
    return [float(score) for score in scores.detach().cpu().tolist()]


def rank_medcpt_embedding_snippets(
    question: str,
    snippets: Sequence[Any],
    *,
    top_k: int | None = None,
    query_model_name: str = MEDCPT_QUERY_ENCODER_MODEL,
    article_model_name: str = MEDCPT_ARTICLE_ENCODER_MODEL,
    batch_size: int = MEDCPT_BATCH_SIZE,
    device: str | None = None,
) -> List[Any]:
    candidates = list(snippets or [])
    scores = medcpt_embedding_scores(
        question,
        candidates,
        query_model_name=query_model_name,
        article_model_name=article_model_name,
        batch_size=batch_size,
        device=device,
    )

    scored: List[Tuple[float, int, Any]] = []
    for index, (snippet, score) in enumerate(zip(candidates, scores)):
        set_ranking_metadata(snippet, score, "medcpt")
        scored.append((score, index, snippet))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [snippet for _, _, snippet in scored]
    return ranked[:top_k] if top_k is not None else ranked


__all__ = [
    "medcpt_embedding_scores",
    "rank_medcpt_embedding_snippets",
]
