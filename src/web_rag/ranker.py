from __future__ import annotations

import re
from collections.abc import Iterable

from web_rag.biomedical_slots import calculate_slot_score
from web_rag.config import get_settings
from web_rag.models import Snippet
from web_rag.neural_reranker import MedCPTReranker
from web_rag.text_utils import clean_text, extract_keywords


RESULT_CUES = {
    "stable",
    "stability",
    "unstable",
    "instability",
    "degradation",
    "degrade",
    "decreased",
    "decrease",
    "increased",
    "increase",
    "change",
    "changed",
    "variation",
    "variability",
    "concentration",
    "level",
    "levels",
    "measured",
    "measurement",
    "recovery",
    "loss",
    "significant",
    "not significant",
}

SAMPLE_CUES = {
    "blood",
    "plasma",
    "serum",
    "urine",
    "whole blood",
    "edta",
    "heparin",
    "citrate",
    "sample",
    "samples",
    "specimen",
    "specimens",
}

CONDITION_CUES = {
    "room temperature",
    "temperature",
    "storage",
    "stored",
    "freeze",
    "frozen",
    "freezing",
    "thaw",
    "thawed",
    "centrifugation",
    "centrifuged",
    "delay",
    "delayed",
    "hours",
    "hour",
    "minutes",
    "minute",
    "days",
    "day",
    "4 c",
    "20 c",
    "37 c",
    "-20 c",
    "-80 c",
}


def normalize_for_matching(text: str) -> str:
    text = text.lower()
    text = text.replace("°", " ")
    text = re.sub(r"[^a-z0-9\-\+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_term(text: str, term: str) -> bool:
    normalized_text = normalize_for_matching(text)
    normalized_term = normalize_for_matching(term)

    if not normalized_term:
        return False

    return normalized_term in normalized_text


def count_cue_matches(text: str, cues: set[str]) -> int:
    return sum(1 for cue in cues if contains_term(text, cue))


def snippet_length_penalty(snippet: Snippet) -> float:
    word_count = len(snippet.text.split())

    if word_count < 8:
        return -1.0

    if word_count < 15:
        return -0.4

    return 0.0


def snippet_length_bonus(snippet: Snippet) -> float:
    word_count = len(snippet.text.split())

    if 20 <= word_count <= 90:
        return 0.5

    return 0.0


def score_snippet(question: str, snippet: Snippet) -> float:
    """
    Lexical baseline score.

    This function is kept as the original baseline so that later experiments
    can compare simple lexical ranking against stronger reranking methods.
    """
    q_terms = extract_keywords(question)

    title = snippet.paper.title
    evidence = snippet.text
    combined = f"{title} {evidence}"

    score = 0.0

    for term in q_terms:
        if contains_term(evidence, term):
            score += 2.0

        if contains_term(title, term):
            score += 1.5

    result_cue_count = count_cue_matches(combined, RESULT_CUES)
    sample_cue_count = count_cue_matches(combined, SAMPLE_CUES)
    condition_cue_count = count_cue_matches(combined, CONDITION_CUES)

    score += min(result_cue_count, 4) * 0.35
    score += min(sample_cue_count, 3) * 0.25
    score += min(condition_cue_count, 4) * 0.25

    if snippet.paper.doi:
        score += 0.2

    if snippet.paper.year:
        score += 0.1

    score += snippet_length_penalty(snippet)
    score += snippet_length_bonus(snippet)

    return round(score, 4)


def min_max_normalize(values: list[float]) -> list[float]:
    """
    Normalize scores to [0, 1].

    If all scores are equal, return 0.5 for all items so the signal remains
    neutral instead of disappearing.
    """
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return [0.5 for _ in values]

    return [
        (value - min_value) / (max_value - min_value)
        for value in values
    ]


def deduplicate_ranked_snippets(snippets: Iterable[Snippet]) -> list[Snippet]:
    seen: set[tuple[str, str]] = set()
    deduplicated: list[Snippet] = []

    for snippet in snippets:
        paper_key = snippet.paper.doi or snippet.paper.ext_id or snippet.paper.title
        text_key = clean_text(snippet.text).lower()
        key = (paper_key, text_key)

        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(snippet)

    return deduplicated


def rank_snippets_lexical(
    question: str,
    snippets: list[Snippet],
    top_k: int,
) -> list[Snippet]:
    """
    Original lexical baseline ranking.
    """
    for snippet in snippets:
        lexical_score = score_snippet(question, snippet)
        snippet.score = lexical_score
        snippet.score_components = {
            "ranker": "lexical",
            "lexical_raw": lexical_score,
            "final_score": lexical_score,
        }

    snippets.sort(key=lambda item: item.score, reverse=True)
    deduplicated = deduplicate_ranked_snippets(snippets)

    return deduplicated[:top_k]


def rank_snippets_medcpt_hybrid(
    question: str,
    snippets: list[Snippet],
    top_k: int,
) -> list[Snippet]:
    """
    Hybrid biomedical reranking.

    Signals:
    - MedCPT semantic relevance score
    - lexical baseline score
    - biomedical slot-aware score

    Final score:
        0.65 * MedCPT_norm
      + 0.20 * lexical_norm
      + 0.15 * slot_score
    """
    if not snippets:
        return []

    settings = get_settings()

    lexical_raw = [
        score_snippet(question, snippet)
        for snippet in snippets
    ]

    slot_results = [
        calculate_slot_score(question, snippet)
        for snippet in snippets
    ]

    slot_scores = [score for score, _details in slot_results]
    slot_details = [details for _score, details in slot_results]

    medcpt = MedCPTReranker(
        model_name=settings.medcpt_model_name,
        batch_size=settings.medcpt_batch_size,
        max_length=settings.medcpt_max_length,
    )

    medcpt_raw = medcpt.score_snippets(
        question=question,
        snippets=snippets,
    )

    lexical_norm = min_max_normalize(lexical_raw)
    medcpt_norm = min_max_normalize(medcpt_raw)

    for index, snippet in enumerate(snippets):
        final_score = (
            settings.hybrid_weight_medcpt * medcpt_norm[index]
            + settings.hybrid_weight_lexical * lexical_norm[index]
            + settings.hybrid_weight_slots * slot_scores[index]
        )

        snippet.score = round(final_score, 4)
        snippet.score_components = {
            "ranker": "medcpt-hybrid",
            "final_score": round(final_score, 4),
            "medcpt_raw": round(medcpt_raw[index], 4),
            "medcpt_norm": round(medcpt_norm[index], 4),
            "lexical_raw": round(lexical_raw[index], 4),
            "lexical_norm": round(lexical_norm[index], 4),
            "slot_score": round(slot_scores[index], 4),
            "slot_details": slot_details[index],
            "weights": {
                "medcpt": settings.hybrid_weight_medcpt,
                "lexical": settings.hybrid_weight_lexical,
                "slots": settings.hybrid_weight_slots,
            },
        }

    snippets.sort(key=lambda item: item.score, reverse=True)
    deduplicated = deduplicate_ranked_snippets(snippets)

    return deduplicated[:top_k]


def rank_snippets(
    question: str,
    snippets: Iterable[Snippet],
    top_k: int | None = None,
    method: str | None = None,
) -> list[Snippet]:
    """
    Rank evidence snippets using the selected ranking method.

    Available methods:
    - lexical
    - medcpt-hybrid
    """
    settings = get_settings()
    effective_top_k = top_k or settings.default_top_k
    effective_method = method or settings.default_ranker

    snippet_list = list(snippets)

    if effective_method == "lexical":
        return rank_snippets_lexical(
            question=question,
            snippets=snippet_list,
            top_k=effective_top_k,
        )

    if effective_method == "medcpt-hybrid":
        return rank_snippets_medcpt_hybrid(
            question=question,
            snippets=snippet_list,
            top_k=effective_top_k,
        )

    raise ValueError(
        f"Unknown ranking method: {effective_method}. "
        "Use 'lexical' or 'medcpt-hybrid'."
    )