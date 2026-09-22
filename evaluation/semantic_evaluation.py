# Paper: "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation" (Chen et al., Findings of ACL 2024).

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_MODEL = "BAAI/bge-m3"


class SemanticEvaluator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        batch_size: int = 8,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = model or self._load_model(model_name, device)

    @staticmethod
    def _load_model(model_name: str, device: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Semantic evaluation requires sentence-transformers."
            ) from exc

        if device == "auto":
            return SentenceTransformer(model_name)
        return SentenceTransformer(model_name, device=device)

    def score(
        self,
        gold_answers: list[str],
        evidence_texts: list[str],
        *,
        answerable: bool,
    ) -> tuple[float | None, str]:
        """Return the highest cosine similarity across all gold-answer/snippet pairs."""
        if not answerable:
            return None, ""
        if not gold_answers or not evidence_texts:
            return 0.0, ""

        texts = gold_answers + evidence_texts
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        answer_vectors = vectors[: len(gold_answers)]
        evidence_vectors = vectors[len(gold_answers) :]
        similarities = np.clip(answer_vectors @ evidence_vectors.T, 0.0, 1.0)

        best_flat_index = int(np.argmax(similarities))
        _, evidence_index = np.unravel_index(
            best_flat_index,
            similarities.shape,
        )
        return (
            float(similarities.flat[best_flat_index]),
            evidence_texts[int(evidence_index)],
        )

    def score_nuggets_at_cutoffs(
        self,
        nugget_texts: list[str],
        evidence_texts: list[str],
        *,
        cutoffs: tuple[int, ...],
    ) -> list[dict[int, tuple[float, str]]]:
        """Find the best semantic chunk match for every nugget.

        Nuggets and chunks are embedded together in one model call per
        question. The returned list stays aligned with ``nugget_texts``.
        """
        if any(cutoff <= 0 for cutoff in cutoffs):
            raise ValueError("cutoffs must contain only positive integers")

        unique_cutoffs = tuple(dict.fromkeys(cutoffs))
        if not evidence_texts:
            return [
                {cutoff: (0.0, "") for cutoff in unique_cutoffs}
                for _ in nugget_texts
            ]
        if not nugget_texts:
            return []

        texts = nugget_texts + evidence_texts
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        nugget_vectors = vectors[: len(nugget_texts)]
        evidence_vectors = vectors[len(nugget_texts) :]
        similarities = np.clip(
            nugget_vectors @ evidence_vectors.T,
            0.0,
            1.0,
        )

        results: list[dict[int, tuple[float, str]]] = []
        for nugget_index in range(len(nugget_texts)):
            nugget_results: dict[int, tuple[float, str]] = {}
            for cutoff in unique_cutoffs:
                limit = min(cutoff, len(evidence_texts))
                prefix = similarities[nugget_index, :limit]
                evidence_index = int(np.argmax(prefix))
                nugget_results[cutoff] = (
                    float(prefix[evidence_index]),
                    evidence_texts[evidence_index],
                )
            results.append(nugget_results)
        return results
