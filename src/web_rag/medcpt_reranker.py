from __future__ import annotations

from functools import lru_cache
from typing import Any

from .models import TextChunk


def resolve_device(requested: str | None = "auto") -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MedCPT requires torch and transformers") from exc

    value = (requested or "auto").strip().casefold()
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if value == "gpu":
        raise ValueError("Use 'cuda' instead of 'gpu', or use 'auto'.")
    try:
        torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device '{requested}'. Use auto, cpu, cuda, cuda:N, or mps.") from exc
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --medcpt-device cpu or auto.")
    if value == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return value


@lru_cache(maxsize=4)
def _load_models(
    query_model_name: str,
    article_model_name: str,
    device: str,
) -> tuple[Any, Any, Any, Any, str]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("MedCPT requires torch and transformers") from exc

    resolved_device = resolve_device(device)
    query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
    query_model = AutoModel.from_pretrained(query_model_name).to(resolved_device).eval()
    article_tokenizer = AutoTokenizer.from_pretrained(article_model_name)
    article_model = AutoModel.from_pretrained(article_model_name).to(resolved_device).eval()
    return query_tokenizer, query_model, article_tokenizer, article_model, resolved_device


def medcpt_scores(
    question: str,
    chunks: list[TextChunk],
    *,
    query_model_name: str,
    article_model_name: str,
    batch_size: int = 8,
    device: str = "auto",
) -> list[float]:
    if not chunks:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    import torch

    query_tokenizer, query_model, article_tokenizer, article_model, resolved_device = _load_models(
        query_model_name,
        article_model_name,
        device,
    )
    query_input = query_tokenizer(
        [question],
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt",
    ).to(resolved_device)
    with torch.inference_mode():
        query_embedding = query_model(**query_input).last_hidden_state[:, 0, :]

    embeddings = []
    document_pairs = [[chunk.paper.title, chunk.text] for chunk in chunks]
    for start in range(0, len(document_pairs), batch_size):
        article_input = article_tokenizer(
            document_pairs[start : start + batch_size],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(resolved_device)
        with torch.inference_mode():
            embeddings.append(article_model(**article_input).last_hidden_state[:, 0, :])

    document_matrix = torch.cat(embeddings, dim=0)
    values = torch.matmul(document_matrix, query_embedding.T).squeeze(1)
    return [float(value) for value in values.detach().cpu().tolist()]


def rerank_medcpt(
    question: str,
    chunks: list[TextChunk],
    *,
    query_model_name: str,
    article_model_name: str,
    batch_size: int = 8,
    device: str = "auto",
) -> list[TextChunk]:
    scores = medcpt_scores(
        question,
        chunks,
        query_model_name=query_model_name,
        article_model_name=article_model_name,
        batch_size=batch_size,
        device=device,
    )
    for chunk, score in zip(chunks, scores):
        chunk.score = score
        chunk.score_components = {"medcpt": score}
    return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
