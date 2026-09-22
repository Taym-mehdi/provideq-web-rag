# RAG evaluation reference: "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation" (Ru et al., NeurIPS 2024).
# Lexical metric reference: "ROUGE: A Package for Automatic Evaluation of Summaries" (Lin, 2004).

from __future__ import annotations

import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.+-][A-Za-z0-9]+)*")


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _f1(overlap: int, reference_size: int, candidate_size: int) -> float:
    if overlap == 0 or reference_size == 0 or candidate_size == 0:
        return 0.0
    precision = overlap / candidate_size
    recall = overlap / reference_size
    return 2 * precision * recall / (precision + recall)


def _recall(overlap: int, reference_size: int) -> float:
    if overlap == 0 or reference_size == 0:
        return 0.0
    return overlap / reference_size


def rouge_1_f1(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
    return _f1(overlap, len(reference_tokens), len(candidate_tokens))


def rouge_l_f1(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    overlap = _lcs_length(reference_tokens, candidate_tokens)
    return _f1(overlap, len(reference_tokens), len(candidate_tokens))


def rouge_1_recall(reference: str, candidate: str) -> float:
    """Return unigram recall with the nugget as the reference text."""
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    overlap = sum(
        (Counter(reference_tokens) & Counter(candidate_tokens)).values()
    )
    return _recall(overlap, len(reference_tokens))


def rouge_l_recall(reference: str, candidate: str) -> float:
    """Return LCS recall with the nugget as the reference text."""
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    overlap = _lcs_length(reference_tokens, candidate_tokens)
    return _recall(overlap, len(reference_tokens))


def _pair_score(reference: str, candidate: str) -> float:
    return (rouge_1_f1(reference, candidate) + rouge_l_f1(reference, candidate)) / 2


def _nugget_pair_score(nugget: str, candidate: str) -> float:
    """Score whether a candidate contains the lexical content of a nugget."""
    return (
        rouge_1_recall(nugget, candidate)
        + rouge_l_recall(nugget, candidate)
    ) / 2


def evaluate_lexical(
    gold_answers: list[str],
    evidence_texts: list[str],
    *,
    answerable: bool,
) -> tuple[float | None, str]:
    """Return the highest lexical match across all gold-answer/snippet pairs."""
    if not answerable:
        return None, ""
    if not gold_answers or not evidence_texts:
        return 0.0, ""

    best_score = -1.0
    best_evidence = ""
    for answer in gold_answers:
        for evidence in evidence_texts:
            score = _pair_score(answer, evidence)
            if score > best_score:
                best_score = score
                best_evidence = evidence

    return max(best_score, 0.0), best_evidence


def evaluate_nuggets_lexical_at_cutoffs(
    nugget_texts: list[str],
    evidence_texts: list[str],
    *,
    cutoffs: tuple[int, ...],
) -> list[dict[int, tuple[float, str]]]:
    """Find the best lexical chunk match for every nugget at each cutoff.

    Recall is used rather than F1 because evidence chunks are intentionally
    much longer than the short gold nuggets. Extra text in a chunk therefore
    does not lower a nugget's lexical coverage score.
    """
    if any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must contain only positive integers")

    unique_cutoffs = tuple(dict.fromkeys(cutoffs))
    if not evidence_texts:
        return [
            {cutoff: (0.0, "") for cutoff in unique_cutoffs}
            for _ in nugget_texts
        ]

    results: list[dict[int, tuple[float, str]]] = []
    for nugget in nugget_texts:
        evidence_scores = [
            _nugget_pair_score(nugget, evidence)
            for evidence in evidence_texts
        ]
        nugget_results: dict[int, tuple[float, str]] = {}
        for cutoff in unique_cutoffs:
            limit = min(cutoff, len(evidence_texts))
            best_index = max(
                range(limit),
                key=evidence_scores.__getitem__,
            )
            nugget_results[cutoff] = (
                evidence_scores[best_index],
                evidence_texts[best_index],
            )
        results.append(nugget_results)
    return results
