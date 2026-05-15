from __future__ import annotations

import re
from typing import Any

from web_rag.models import Snippet
from web_rag.text_utils import extract_keywords


SAMPLE_TERMS = {
    "blood",
    "whole blood",
    "plasma",
    "serum",
    "urine",
    "csf",
    "cerebrospinal fluid",
    "edta",
    "heparin",
    "lithium heparin",
    "citrate",
    "sample",
    "specimen",
    "biospecimen",
}

CONDITION_TERMS = {
    "room temperature",
    "ambient temperature",
    "temperature",
    "storage",
    "stored",
    "delay",
    "delayed",
    "centrifugation",
    "centrifuged",
    "freeze",
    "freezing",
    "frozen",
    "thaw",
    "thawed",
    "freeze-thaw",
    "4 c",
    "20 c",
    "25 c",
    "37 c",
    "-20 c",
    "-80 c",
    "hour",
    "hours",
    "minute",
    "minutes",
    "day",
    "days",
}

RESULT_TERMS = {
    "stable",
    "stability",
    "unstable",
    "instability",
    "degradation",
    "degraded",
    "decrease",
    "decreased",
    "increase",
    "increased",
    "change",
    "changed",
    "significant",
    "not significant",
    "concentration",
    "levels",
    "measurement",
    "recovery",
    "loss",
}

GENERIC_TERMS = SAMPLE_TERMS | CONDITION_TERMS | RESULT_TERMS | {
    "effect",
    "affect",
    "influence",
    "measure",
    "measured",
    "analysis",
    "analyte",
}


def normalize_for_slots(text: str) -> str:
    """
    Normalize text for simple biomedical slot matching.
    """
    text = text.lower()
    text = text.replace("°", " ")
    text = re.sub(r"[^a-z0-9\-\+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_term(text: str, term: str) -> bool:
    normalized_text = normalize_for_slots(text)
    normalized_term = normalize_for_slots(term)

    if not normalized_term:
        return False

    return normalized_term in normalized_text


def find_terms(text: str, vocabulary: set[str]) -> list[str]:
    """
    Return vocabulary terms found in text.
    """
    return sorted(term for term in vocabulary if contains_term(text, term))


def extract_analyte_like_terms(question: str) -> list[str]:
    """
    Approximate analyte-like terms from the question.

    This is intentionally simple. It keeps biomedical names such as IL-6,
    interleukin-6, glucose, potassium, albumin, RNA, etc., while removing
    generic sample/condition/result terms.
    """
    keywords = extract_keywords(question)
    analyte_terms: list[str] = []

    for keyword in keywords:
        normalized = normalize_for_slots(keyword)

        if not normalized:
            continue

        if normalized in {normalize_for_slots(term) for term in GENERIC_TERMS}:
            continue

        if len(normalized) <= 2:
            continue

        analyte_terms.append(keyword)

    return analyte_terms


def ratio_matched(required_terms: list[str], text: str) -> float:
    """
    Compute the fraction of required terms found in text.
    """
    if not required_terms:
        return 0.0

    matched = [term for term in required_terms if contains_term(text, term)]
    return len(matched) / len(required_terms)


def calculate_slot_score(question: str, snippet: Snippet) -> tuple[float, dict[str, Any]]:
    """
    Calculate a simple PAV-aware slot score.

    The goal is not to replace semantic reranking. The slot score acts as a
    controlled biomedical signal that rewards snippets mentioning the same
    analyte-like terms, sample material, and pre-analytical conditions as
    the question.
    """
    combined_text = f"{snippet.paper.title} {snippet.text}"

    analyte_terms = extract_analyte_like_terms(question)
    question_sample_terms = find_terms(question, SAMPLE_TERMS)
    question_condition_terms = find_terms(question, CONDITION_TERMS)

    analyte_match = ratio_matched(analyte_terms, combined_text)

    if question_sample_terms:
        sample_match = ratio_matched(question_sample_terms, combined_text)
    else:
        sample_match = 0.0

    if question_condition_terms:
        condition_match = ratio_matched(question_condition_terms, combined_text)
    else:
        condition_match = 0.0

    result_terms_found = find_terms(combined_text, RESULT_TERMS)
    result_signal = min(len(result_terms_found), 3) / 3

    score = (
        0.40 * analyte_match
        + 0.20 * sample_match
        + 0.25 * condition_match
        + 0.15 * result_signal
    )

    details: dict[str, Any] = {
        "slot_score": round(score, 4),
        "analyte_terms": analyte_terms,
        "question_sample_terms": question_sample_terms,
        "question_condition_terms": question_condition_terms,
        "result_terms_found": result_terms_found,
        "analyte_match": round(analyte_match, 4),
        "sample_match": round(sample_match, 4),
        "condition_match": round(condition_match, 4),
        "result_signal": round(result_signal, 4),
    }

    return round(score, 4), details