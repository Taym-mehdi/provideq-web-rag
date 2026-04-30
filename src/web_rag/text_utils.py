from __future__ import annotations

import re
from typing import List


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "in", "on", "for", "to",
    "from", "with", "by", "at", "is", "are", "was", "were", "be", "been", "being", "as",
    "that", "this", "these", "those", "it", "its", "into", "about", "after", "before",
    "between", "during", "under", "over", "than", "can", "could", "should", "would", "do",
    "does", "did", "done", "i", "we", "you", "they", "he", "she", "them", "their", "our",
    "question", "stable", "stability"
}


def clean_text(text: str) -> str:
    """
    Normalize whitespace in text.
    """
    return re.sub(r"\s+", " ", text).strip()


def extract_keywords(question: str) -> List[str]:
    """
    Extract simple keyword candidates from a biomedical question.

    This is lightweight for the first stage.

    """
    tokens = re.findall(r"[A-Za-z0-9\-\+°]+", question.lower())
    keywords = [token for token in tokens if len(token) > 2 and token not in STOPWORDS]
    return keywords


def normalize_question(question: str) -> str:
    """
    Produce a clean normalized question string.
    """
    return clean_text(question)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into rough sentence units.

    """
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]