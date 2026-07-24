from __future__ import annotations

from collections.abc import Callable

from .models import QueryBundle
from .text_utils import clean_text, deduplicate_text, extract_keywords, extract_numeric_terms


BIOMEDICAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "serum": ("blood serum",),
    "plasma": ("blood plasma",),
    "blood": ("whole blood",),
    "gel": ("serum separator", "SST"),
    "tube": ("collection tube",),
    "tubes": ("collection tube",),
    "centrifugation": ("pre-centrifugation",),
    "centrifuge": ("centrifugation", "pre-centrifugation"),
    "delay": ("processing delay",),
    "delayed": ("delay", "processing delay"),
    "storage": ("sample storage",),
    "temperature": ("storage temperature",),
    "stability": ("stable", "unstable"),
    "stable": ("stability", "unstable"),
    "potassium": ("K+",),
    "sodium": ("Na+",),
    "phosphate": ("phosphorus",),
    "crp": ("C-reactive protein",),
}

PHRASES = (
    "serum separator tubes",
    "serum separator tube",
    "serum gel tubes",
    "serum gel tube",
    "delayed centrifugation",
    "pre-centrifugation",
    "processing delay",
    "sample storage",
    "storage temperature",
    "room temperature",
    "whole blood",
    "c-reactive protein",
)


def build_raw_query(question: str) -> QueryBundle:
    normalized = clean_text(question)
    if not normalized:
        raise ValueError("question must not be empty")
    return QueryBundle(
        original_question=question,
        normalized_question=normalized,
        strategy="raw",
        search_query=normalized,
        keywords=extract_keywords(normalized),
    )


def build_synonym_query(question: str, *, max_terms: int = 24) -> QueryBundle:
    normalized = clean_text(question)
    if not normalized:
        raise ValueError("question must not be empty")
    if max_terms <= 0:
        raise ValueError("max_terms must be greater than 0")

    keywords = extract_keywords(normalized)
    lowered = normalized.casefold()
    expanded: list[str] = [*keywords, *extract_numeric_terms(normalized)]
    expanded.extend(phrase for phrase in PHRASES if phrase in lowered)
    for keyword in keywords:
        expanded.extend(BIOMEDICAL_EXPANSIONS.get(keyword, ()))

    expanded_terms = deduplicate_text(expanded)[:max_terms]
    return QueryBundle(
        original_question=question,
        normalized_question=normalized,
        strategy="synonym",
        search_query=" ".join(expanded_terms) or normalized,
        keywords=keywords,
        expanded_terms=expanded_terms,
    )


_QUERY_REFORMULATORS: dict[str, Callable[[str], QueryBundle]] = {
    "raw": build_raw_query,
    "synonym": build_synonym_query,
}


def reformulate_query(question: str, strategy: str = "synonym") -> QueryBundle:
    selected = strategy.strip().casefold().replace("-", "_")
    try:
        reformulator = _QUERY_REFORMULATORS[selected]
    except KeyError as exc:
        choices = ", ".join(_QUERY_REFORMULATORS)
        raise ValueError(f"Unknown query strategy '{strategy}'. Choose from: {choices}") from exc
    return reformulator(question)
