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
        raise ValueError(
            f"Invalid torch device '{requested}'. "
            "Use auto, cpu, cuda, cuda:N, or mps."
        ) from exc
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Use --medcpt-device cpu or auto."
        )
    if value == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
    return value


@lru_cache(maxsize=4)
def _load_cross_encoder(
    model_name: str,
    device: str,
) -> tuple[Any, Any, str]:
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError("MedCPT requires torch and transformers") from exc

    resolved_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(model_name)
        .to(resolved_device)
        .eval()
    )
    return tokenizer, model, resolved_device


def _document_text(chunk: TextChunk) -> str:
    parts = [f"Title: {chunk.paper.title}"] if chunk.paper.title else []
    parts.append(chunk.text)
    return "\n".join(parts)


def medcpt_scores(
    question: str,
    chunks: list[TextChunk],
    *,
    model_name: str,
    max_length: int = 512,
    batch_size: int = 8,
    device: str = "auto",
) -> list[float]:
    """Score query-passage pairs with the MedCPT cross-encoder."""
    if not chunks:
        return []
    if max_length < 64:
        raise ValueError("max_length must be at least 64")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    import torch

    tokenizer, model, resolved_device = _load_cross_encoder(
        model_name,
        device,
    )
    documents = [_document_text(chunk) for chunk in chunks]
    scores: list[float] = []

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        encoded = tokenizer(
            [question] * len(batch),
            batch,
            truncation="only_second",
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(resolved_device)

        with torch.inference_mode():
            logits = model(**encoded).logits.squeeze(-1)
        scores.extend(
            float(value)
            for value in logits.detach().cpu().reshape(-1).tolist()
        )

    return scores


def rerank_medcpt(
    question: str,
    chunks: list[TextChunk],
    *,
    model_name: str,
    max_length: int = 512,
    batch_size: int = 8,
    device: str = "auto",
) -> list[TextChunk]:
    scores = medcpt_scores(
        question,
        chunks,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    for chunk, score in zip(chunks, scores):
        chunk.score = score
        chunk.score_components = {"medcpt_cross_encoder": score}
    return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
