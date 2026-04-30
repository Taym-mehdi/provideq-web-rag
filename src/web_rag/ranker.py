from __future__ import annotations

import re
from collections.abc import Iterable

from web_rag.config import get_settings
from web_rag.models import Snippet
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
    """
    Normalize text for lightweight lexical matching.

    This is not meant to be a perfect biomedical normalizer.
    It is a transparent baseline that is easy to explain and improve later.
    """
    text = text.lower()
    text = text.replace("°", " ")
    text = re.sub(r"[^a-z0-9\-\+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_term(text: str, term: str) -> bool:
    """
    Check whether a term appears in a normalized text.

    The check uses substring matching because biomedical expressions often
    contain hyphens, numbers, and abbreviations.
    """
    normalized_text = normalize_for_matching(text)
    normalized_term = normalize_for_matching(term)

    if not normalized_term:
        return False

    return normalized_term in normalized_text


def count_cue_matches(text: str, cues: set[str]) -> int:
    """
    Count how many cue words or cue phrases occur in the text.
    """
    return sum(1 for cue in cues if contains_term(text, cue))


def snippet_length_penalty(snippet: Snippet) -> float:
    """
    Penalize snippets that are too short to provide useful evidence.

    Very short snippets are often only titles or incomplete fragments.
    We do not remove them completely because title-only records can still
    contain weak signals, but they should not dominate the ranking.
    """
    word_count = len(snippet.text.split())

    if word_count < 8:
        return -1.0

    if word_count < 15:
        return -0.4

    return 0.0


def snippet_length_bonus(snippet: Snippet) -> float:
    """
    Add a small bonus for reasonably informative snippets.

    We keep this bonus small because long text is not automatically better.
    """
    word_count = len(snippet.text.split())

    if 20 <= word_count <= 90:
        return 0.5

    return 0.0


def score_snippet(question: str, snippet: Snippet) -> float:
    """
    Score one evidence snippet for one biomedical question.

    The score is intentionally simple and inspectable.

    Main signals:
    - question keyword overlap
    - title matches
    - evidence text matches
    - biomedical result cues
    - sample/material cues
    - pre-analytical condition cues
    - snippet length penalty/bonus
    """
    keywords = extract_keywords(question)

    title = snippet.paper.title
    evidence = snippet.text
    combined = f"{title} {evidence}"

    score = 0.0

    for keyword in keywords:
        if contains_term(evidence, keyword):
            score += 2.0

        if contains_term(title, keyword):
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


def deduplicate_ranked_snippets(snippets: Iterable[Snippet]) -> list[Snippet]:
    """
    Remove exact duplicate snippets after ranking.

    We deduplicate by source paper and snippet text. This prevents repeated
    windows from the same abstract from filling the top-k evidence pack.
    """
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


def rank_snippets(
    question: str,
    snippets: Iterable[Snippet],
    top_k: int | None = None,
) -> list[Snippet]:
    """
    Score, sort, deduplicate, and return top-k snippets.

    The snippets keep their original Paper metadata, so the ranked output is
    still citation-ready and traceable.
    """
    settings = get_settings()
    effective_top_k = top_k or settings.default_top_k

    scored_snippets: list[Snippet] = []

    for snippet in snippets:
        snippet.score = score_snippet(question, snippet)
        scored_snippets.append(snippet)

    scored_snippets.sort(key=lambda item: item.score, reverse=True)
    deduplicated = deduplicate_ranked_snippets(scored_snippets)

    return deduplicated[:effective_top_k]