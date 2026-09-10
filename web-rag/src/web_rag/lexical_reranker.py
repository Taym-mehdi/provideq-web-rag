from __future__ import annotations

from collections import Counter
import math

from .models import TextChunk
from .text_utils import extract_keywords, extract_numeric_terms, tokenize


def _query_terms(question: str) -> list[str]:
    terms = [*extract_keywords(question), *extract_numeric_terms(question)]
    return list(dict.fromkeys(token.casefold() for term in terms for token in tokenize(term)))


def bm25_scores(
    question: str,
    chunks: list[TextChunk],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not chunks:
        return []
    if k1 <= 0:
        raise ValueError("k1 must be greater than 0")
    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1")

    query_terms = _query_terms(question)
    documents = [tokenize(f"{chunk.paper.title} {chunk.text}") for chunk in chunks]
    if not query_terms:
        return [0.0] * len(documents)

    lengths = [len(document) for document in documents]
    average_length = max(sum(lengths) / len(lengths), 1.0)
    document_frequencies: Counter[str] = Counter()
    for document in documents:
        document_frequencies.update(set(document))

    document_count = len(documents)
    scores: list[float] = []
    for document, length in zip(documents, lengths):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * length / average_length)
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(float(score))
    return scores


def rerank_lexical(
    question: str,
    chunks: list[TextChunk],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[TextChunk]:
    for chunk, score in zip(chunks, bm25_scores(question, chunks, k1=k1, b=b)):
        chunk.score = score
        chunk.score_components = {"bm25": score}
    return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
