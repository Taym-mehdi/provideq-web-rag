from __future__ import annotations

import html
import re
from typing import List


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "in", "on", "for", "to",
    "from", "with", "by", "at", "is", "are", "was", "were", "be", "been", "being", "as",
    "that", "this", "these", "those", "it", "its", "into", "about", "after", "before",
    "between", "during", "under", "over", "than", "can", "could", "should", "would", "do",
    "does", "did", "done", "i", "we", "you", "they", "he", "she", "them", "their", "our",
    "question", "what", "which", "who", "whom", "whose", "why", "how", "when", "where",
    "whether", "if", "up", "down", "using", "use", "used", "effect", "effects", "affect",
    "affects", "affected", "change", "changes", "valid", "validity", "result", "results",
    "measurement", "measurements", "sample", "samples", "degree", "degrees", "celsius",
    "hour", "hours", "day", "days", "minute", "minutes", "condition", "conditions", "variable",
}


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<\s*/\s*h\d\s*>", ". ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*h\d[^>]*\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    return text.strip()


def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9\-\+°]+", (question or "").lower())
    keywords = [token for token in tokens if len(token) > 2 and token not in STOPWORDS]
    output: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if keyword not in seen:
            output.append(keyword)
            seen.add(keyword)
    return output


def normalize_question(question: str) -> str:
    return clean_text(question)


def split_into_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]
