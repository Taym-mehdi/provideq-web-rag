from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class NuggetSemanticResult:
    """
    Semantic match result for one gold nugget against top-k retrieved snippets.
    """

    nugget: str
    best_similarity: float
    best_rank: int | None


@dataclass
class QuestionSemanticMetrics:
    """
    Semantic evaluation metrics for one benchmark question.
    """

    question_id: str
    question: str
    k: int
    semantic_nugget_match_at_k: float
    semantic_answer_match_at_k: float
    answer_best_rank: int | None
    nugget_count: int
    retrieved_snippet_count: int
    embedding_model: str
    nugget_results: list[NuggetSemanticResult]


class SentenceEmbeddingModel:
    """
    Thin wrapper around SentenceTransformers.

    The wrapper keeps model loading in one place and makes the evaluation code
    independent from the specific embedding model name.

    Default model:
        sentence-transformers/all-MiniLM-L6-v2

    For later experiments, the model can be changed from the CLI.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Semantic evaluation requires sentence-transformers. "
                "Run: pip install -r requirements.txt"
            ) from error

        self.model_name = model_name
        self.batch_size = batch_size

        if device:
            self.model = SentenceTransformer(model_name, device=device)
        else:
            self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into normalized vectors.

        Normalized embeddings make cosine similarity equivalent to dot product.
        """
        clean_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not clean_texts:
            return np.zeros((0, 1), dtype=np.float32)

        embeddings = self.model.encode(
            clean_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def cosine_similarity_matrix(
    left_embeddings: np.ndarray,
    right_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity matrix.

    Because embeddings are normalized during encoding, dot product gives cosine
    similarity.
    """
    if left_embeddings.size == 0 or right_embeddings.size == 0:
        return np.zeros((left_embeddings.shape[0], right_embeddings.shape[0]))

    return np.matmul(left_embeddings, right_embeddings.T)


def best_similarity_and_rank(scores: np.ndarray) -> tuple[float, int | None]:
    """
    Return best similarity score and 1-based rank.
    """
    if scores.size == 0:
        return 0.0, None

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    return best_score, best_index + 1


def evaluate_semantic_nuggets(
    gold_nuggets: list[str],
    retrieved_snippets: list[str],
    embedding_model: SentenceEmbeddingModel,
) -> list[NuggetSemanticResult]:
    """
    Evaluate semantic nugget coverage.

    For each gold nugget:
    - compare it with every retrieved top-k snippet
    - take the maximum cosine similarity
    """
    if not gold_nuggets:
        return []

    if not retrieved_snippets:
        return [
            NuggetSemanticResult(
                nugget=nugget,
                best_similarity=0.0,
                best_rank=None,
            )
            for nugget in gold_nuggets
        ]

    nugget_embeddings = embedding_model.encode(gold_nuggets)
    snippet_embeddings = embedding_model.encode(retrieved_snippets)

    similarity_matrix = cosine_similarity_matrix(
        left_embeddings=nugget_embeddings,
        right_embeddings=snippet_embeddings,
    )

    results: list[NuggetSemanticResult] = []

    for nugget_index, nugget in enumerate(gold_nuggets):
        scores = similarity_matrix[nugget_index]
        best_score, best_rank = best_similarity_and_rank(scores)

        results.append(
            NuggetSemanticResult(
                nugget=nugget,
                best_similarity=round(best_score, 4),
                best_rank=best_rank,
            )
        )

    return results


def evaluate_semantic_answer_match(
    gold_answer: str,
    retrieved_snippets: list[str],
    embedding_model: SentenceEmbeddingModel,
) -> tuple[float, int | None]:
    """
    Compare the full gold answer against retrieved snippets.

    For the answer:
    - embed the full gold answer
    - compare against each retrieved top-k snippet
    - take maximum similarity
    """
    if not gold_answer or not gold_answer.strip():
        return 0.0, None

    if not retrieved_snippets:
        return 0.0, None

    answer_embeddings = embedding_model.encode([gold_answer])
    snippet_embeddings = embedding_model.encode(retrieved_snippets)

    similarity_matrix = cosine_similarity_matrix(
        left_embeddings=answer_embeddings,
        right_embeddings=snippet_embeddings,
    )

    scores = similarity_matrix[0]
    best_score, best_rank = best_similarity_and_rank(scores)

    return round(best_score, 4), best_rank


def evaluate_question_semantic(
    question_id: str,
    question: str,
    gold_answer: str,
    gold_nuggets: list[str],
    retrieved_snippets: list[str],
    k: int,
    embedding_model: SentenceEmbeddingModel,
) -> QuestionSemanticMetrics:
    """
    Compute semantic metrics for one question.

    Metrics:
    - SemanticNuggetMatch@k
    - SemanticAnswerMatch@k
    """
    top_k_snippets = retrieved_snippets[:k]

    nugget_results = evaluate_semantic_nuggets(
        gold_nuggets=gold_nuggets,
        retrieved_snippets=top_k_snippets,
        embedding_model=embedding_model,
    )

    semantic_nugget_values = [
        result.best_similarity
        for result in nugget_results
    ]

    semantic_answer_match, answer_best_rank = evaluate_semantic_answer_match(
        gold_answer=gold_answer,
        retrieved_snippets=top_k_snippets,
        embedding_model=embedding_model,
    )

    return QuestionSemanticMetrics(
        question_id=question_id,
        question=question,
        k=k,
        semantic_nugget_match_at_k=round(safe_mean(semantic_nugget_values), 4),
        semantic_answer_match_at_k=semantic_answer_match,
        answer_best_rank=answer_best_rank,
        nugget_count=len(gold_nuggets),
        retrieved_snippet_count=len(top_k_snippets),
        embedding_model=embedding_model.model_name,
        nugget_results=nugget_results,
    )


def extract_gold_answer(raw_record: dict[str, Any]) -> str:
    return raw_record.get("gold_answer", "")


def extract_gold_nuggets(raw_record: dict[str, Any]) -> list[str]:
    nuggets = raw_record.get("gold_nuggets", [])

    if isinstance(nuggets, list):
        return [
            str(nugget).strip()
            for nugget in nuggets
            if str(nugget).strip()
        ]

    return []