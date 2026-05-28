from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class NuggetMetricResult:
    """
    Metric result for one gold nugget against top-k retrieved snippets.
    """

    nugget: str
    best_rouge1_f1: float
    best_rougel_f1: float
    best_rouge_combined: float
    best_bm25_normalized: float
    best_rouge1_rank: int | None
    best_rougel_rank: int | None
    best_bm25_rank: int | None


@dataclass
class QuestionLexicalMetrics:
    """
    Lexical evaluation metrics for one benchmark question.
    """

    question_id: str
    question: str
    k: int
    rouge1_nugget_at_k: float
    rougel_nugget_at_k: float
    rouge_nugget_at_k: float
    bm25_nugget_at_k: float
    nugget_count: int
    retrieved_snippet_count: int
    nugget_results: list[NuggetMetricResult]


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for lexical evaluation.

    This tokenizer is intentionally simple and transparent. It lowercases text
    and keeps biomedical-like tokens containing letters, numbers, hyphens, and
    plus signs.
    """
    text = text.lower()
    text = text.replace("°", " ")
    tokens = re.findall(r"[a-z0-9\-\+]+", text)
    return tokens


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def rouge1_f1(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-1 F1 between reference text and candidate text.

    In our evaluation:
    - reference = gold nugget
    - candidate = retrieved evidence snippet
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)

    overlap = 0

    for token, ref_count in ref_counts.items():
        overlap += min(ref_count, cand_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    """
    Compute longest common subsequence length.

    Used for ROUGE-L F1.
    """
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)

    for token_a in a:
        current = [0] * (len(b) + 1)

        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])

        previous = current

    return previous[-1]


def rougel_f1(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-L F1 between reference text and candidate text.
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    lcs = lcs_length(ref_tokens, cand_tokens)

    if lcs == 0:
        return 0.0

    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def bm25_scores(
    query: str,
    documents: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """
    Compute BM25 scores for one query against a list of documents.

    This implementation is intentionally local and dependency-free. It is used
    for evaluation, not for retrieval.
    """
    query_terms = tokenize(query)
    tokenized_documents = [tokenize(document) for document in documents]

    if not query_terms or not tokenized_documents:
        return [0.0 for _ in documents]

    document_count = len(tokenized_documents)
    document_lengths = [len(document) for document in tokenized_documents]
    avg_doc_length = safe_mean(document_lengths)

    if avg_doc_length == 0:
        return [0.0 for _ in documents]

    document_frequencies: dict[str, int] = {}

    for term in set(query_terms):
        document_frequencies[term] = sum(
            1
            for document in tokenized_documents
            if term in document
        )

    scores: list[float] = []

    for document_tokens in tokenized_documents:
        term_counts = Counter(document_tokens)
        document_length = len(document_tokens)

        score = 0.0

        for term in query_terms:
            tf = term_counts.get(term, 0)

            if tf == 0:
                continue

            df = document_frequencies.get(term, 0)

            # Positive Okapi-style IDF.
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))

            denominator = tf + k1 * (
                1 - b + b * (document_length / avg_doc_length)
            )

            score += idf * ((tf * (k1 + 1)) / denominator)

        scores.append(score)

    return scores


def normalized_bm25_nugget_scores(
    nugget: str,
    snippets: list[str],
) -> list[float]:
    """
    Compute normalized BM25 scores for one nugget against retrieved snippets.

    Normalization strategy:
    - Add the nugget itself as a pseudo-perfect document.
    - Compute BM25 scores for snippets + pseudo-perfect document.
    - Normalize snippet scores by the pseudo-perfect score.
    - Cap values to [0, 1].

    This makes BM25_Nugget@k easier to compare across questions.
    """
    if not snippets:
        return []

    documents = snippets + [nugget]
    all_scores = bm25_scores(query=nugget, documents=documents)

    snippet_scores = all_scores[:-1]
    perfect_score = all_scores[-1]

    if perfect_score <= 0:
        return [0.0 for _ in snippet_scores]

    normalized_scores = [
        max(0.0, min(score / perfect_score, 1.0))
        for score in snippet_scores
    ]

    return normalized_scores


def best_score_and_rank(scores: list[float]) -> tuple[float, int | None]:
    """
    Return max score and 1-based rank position.
    """
    if not scores:
        return 0.0, None

    best_index = max(range(len(scores)), key=lambda index: scores[index])
    return scores[best_index], best_index + 1


def evaluate_nugget_against_snippets(
    nugget: str,
    snippets: list[str],
) -> NuggetMetricResult:
    """
    Evaluate one gold nugget against retrieved top-k snippets.

    For each metric, the nugget receives the best score over all retrieved
    snippets.
    """
    rouge1_scores = [
        rouge1_f1(reference=nugget, candidate=snippet)
        for snippet in snippets
    ]

    rougel_scores = [
        rougel_f1(reference=nugget, candidate=snippet)
        for snippet in snippets
    ]

    bm25_normalized_scores = normalized_bm25_nugget_scores(
        nugget=nugget,
        snippets=snippets,
    )

    best_rouge1, best_rouge1_rank = best_score_and_rank(rouge1_scores)
    best_rougel, best_rougel_rank = best_score_and_rank(rougel_scores)
    best_bm25, best_bm25_rank = best_score_and_rank(bm25_normalized_scores)

    best_rouge_combined = (best_rouge1 + best_rougel) / 2

    return NuggetMetricResult(
        nugget=nugget,
        best_rouge1_f1=round(best_rouge1, 4),
        best_rougel_f1=round(best_rougel, 4),
        best_rouge_combined=round(best_rouge_combined, 4),
        best_bm25_normalized=round(best_bm25, 4),
        best_rouge1_rank=best_rouge1_rank,
        best_rougel_rank=best_rougel_rank,
        best_bm25_rank=best_bm25_rank,
    )


def evaluate_question_lexical(
    question_id: str,
    question: str,
    gold_nuggets: list[str],
    retrieved_snippets: list[str],
    k: int,
) -> QuestionLexicalMetrics:
    """
    Compute lexical nugget metrics for one question.

    Metrics:
    - ROUGE1_Nugget@k
    - ROUGEL_Nugget@k
    - ROUGE_Nugget@k
    - BM25_Nugget@k
    """
    top_k_snippets = retrieved_snippets[:k]

    nugget_results = [
        evaluate_nugget_against_snippets(
            nugget=nugget,
            snippets=top_k_snippets,
        )
        for nugget in gold_nuggets
    ]

    rouge1_values = [
        result.best_rouge1_f1
        for result in nugget_results
    ]

    rougel_values = [
        result.best_rougel_f1
        for result in nugget_results
    ]

    rouge_combined_values = [
        result.best_rouge_combined
        for result in nugget_results
    ]

    bm25_values = [
        result.best_bm25_normalized
        for result in nugget_results
    ]

    return QuestionLexicalMetrics(
        question_id=question_id,
        question=question,
        k=k,
        rouge1_nugget_at_k=round(safe_mean(rouge1_values), 4),
        rougel_nugget_at_k=round(safe_mean(rougel_values), 4),
        rouge_nugget_at_k=round(safe_mean(rouge_combined_values), 4),
        bm25_nugget_at_k=round(safe_mean(bm25_values), 4),
        nugget_count=len(gold_nuggets),
        retrieved_snippet_count=len(top_k_snippets),
        nugget_results=nugget_results,
    )


def extract_retrieved_snippet_texts(raw_record: dict[str, Any]) -> list[str]:
    """
    Extract retrieved evidence texts from a raw_results.jsonl record.
    """
    result = raw_record.get("result", {})
    evidence = result.get("evidence", [])

    return [
        record.get("evidence_text", "")
        for record in evidence
        if record.get("evidence_text", "")
    ]