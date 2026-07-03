from __future__ import annotations

# Based on query expansion/reformulation ideas from:
# Wang et al. (2023), "Query2doc: Query Expansion with Large Language Models".
# Ateia and Kruschwitz (2024), "BioRAGent: A Retrieval-Augmented Generation System for ... Scientific Q&A".

import re

from web_rag.models import QueryBundle
from web_rag.text_utils import extract_keywords, normalize_question


BIOMEDICAL_EXPANSIONS: dict[str, list[str]] = {
    "serum": ["serum", "blood serum"],
    "plasma": ["plasma", "blood plasma"],
    "gel": ["gel", "serum gel", "serum separator", "SST"],
    "tube": ["tube", "tubes", "collection tube", "blood collection tube"],
    "tubes": ["tube", "tubes", "collection tube", "blood collection tube"],
    "centrifugation": ["centrifugation", "centrifuge", "pre-centrifugation", "precentrifugation"],
    "centrifuge": ["centrifugation", "centrifuge", "pre-centrifugation", "precentrifugation"],
    "delayed": ["delayed", "delay", "processing delay", "delayed processing"],
    "delay": ["delayed", "delay", "processing delay", "delayed processing"],
    "storage": ["storage", "stored", "sample storage"],
    "temperature": ["temperature", "temperatures", "room temperature"],
    "stability": ["stability", "stable", "unstable"],
    "stable": ["stability", "stable", "unstable"],
    "potassium": ["potassium", "K+"],
    "sodium": ["sodium", "Na+"],
    "glucose": ["glucose"],
    "phosphate": ["phosphate", "phosphorus"],
    "albumin": ["albumin"],
    "crp": ["CRP", "C-reactive protein"],
    "c-reactive": ["CRP", "C-reactive protein"],
}

CORE_TERM_GROUPS = {
    "analyte": {
        "potassium", "sodium", "glucose", "phosphate", "phosphorus", "albumin", "crp",
        "c-reactive", "creatinine", "bilirubin", "calcium", "magnesium", "lactate",
        "hemoglobin", "haemoglobin", "insulin", "cortisol", "cholesterol", "triglycerides",
    },
    "material": {
        "serum", "plasma", "blood", "urine", "gel", "tube", "tubes", "sst", "edta",
        "heparin", "citrate", "separator",
    },
    "process": {
        "centrifugation", "centrifuge", "pre-centrifugation", "precentrifugation", "storage",
        "stored", "delay", "delayed", "processing", "transport", "temperature", "freeze", "frozen",
        "thaw", "stability", "stable",
    },
}


def escape_query_text(text: str) -> str:
    return (text or "").replace('"', "").strip()


def _quote(term: str) -> str:
    term = escape_query_text(term)
    if not term:
        return ""
    if re.search(r"\s", term) or any(char in term for char in "+-/"):
        return f'"{term}"'
    return term


def _deduplicate(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            output.append(normalized)
            seen.add(key)
    return output


def build_keyword_clause(keywords: list[str], max_terms: int = 8, operator: str = "AND") -> str:
    selected_keywords = _deduplicate(keywords)[:max_terms]
    return f" {operator} ".join(_quote(keyword) for keyword in selected_keywords if _quote(keyword))


def _extract_phrases(question: str) -> list[str]:
    text = question.lower()
    phrases: list[str] = []
    phrase_patterns = [
        "serum gel", "serum gel tube", "serum gel tubes", "serum separator tube", "serum separator tubes",
        "delayed centrifugation", "pre centrifugation", "pre-centrifugation", "sample processing",
        "processing delay", "sample storage", "storage temperature", "room temperature",
        "c-reactive protein", "whole blood", "blood sample", "blood samples",
    ]
    for phrase in phrase_patterns:
        if phrase in text:
            phrases.append(phrase)
    return _deduplicate(phrases)


def _expanded_terms(keywords: list[str], phrases: list[str], max_terms: int = 16) -> list[str]:
    expanded: list[str] = []
    for item in phrases + keywords:
        expanded.extend(BIOMEDICAL_EXPANSIONS.get(item.lower(), [item]))
    return _deduplicate(expanded)[:max_terms]


def _priority_terms(keywords: list[str], max_terms: int = 4) -> list[str]:
    selected: list[str] = []
    used: set[str] = set()

    for group_terms in CORE_TERM_GROUPS.values():
        for keyword in keywords:
            if keyword in group_terms and keyword not in used:
                selected.append(keyword)
                used.add(keyword)
                break

    for keyword in keywords:
        if keyword not in used:
            selected.append(keyword)
            used.add(keyword)
        if len(selected) >= max_terms:
            break

    return selected[:max_terms]


def _or_clause(terms: list[str], field: str = "TITLE_ABS") -> str:
    items = [_quote(term) for term in _deduplicate(terms)]
    items = [item for item in items if item]
    if not items:
        return ""
    return f"{field}:(" + " OR ".join(items) + ")"


def _and_clause(terms: list[str], field: str = "TITLE_ABS") -> str:
    items = [_quote(term) for term in _deduplicate(terms)]
    items = [item for item in items if item]
    if not items:
        return ""
    return f"{field}:(" + " AND ".join(items) + ")"


def build_europe_pmc_query(question: str) -> QueryBundle:
    normalized_question = normalize_question(question)
    keywords = extract_keywords(normalized_question)
    phrases = _extract_phrases(normalized_question)

    if not keywords:
        search_query = f'TITLE_ABS:"{escape_query_text(normalized_question)}"'
        return QueryBundle(question, normalized_question, keywords, search_query)

    priority_terms = _priority_terms(keywords, max_terms=4)
    expanded_terms = _expanded_terms(keywords, phrases, max_terms=32)

    clauses = [
        f'TITLE_ABS:"{escape_query_text(normalized_question)}"',
        _and_clause(priority_terms[:3]),
        _and_clause(priority_terms[:4]),
        _or_clause(phrases + expanded_terms[:24]),
    ]

    if len(keywords) > 4:
        clauses.append(_or_clause(keywords[:10]))

    clauses = [f"({clause})" for clause in _deduplicate([clause for clause in clauses if clause])]
    search_query = " OR ".join(clauses)

    return QueryBundle(
        original_question=question,
        normalized_question=normalized_question,
        keywords=keywords,
        search_query=search_query,
    )
