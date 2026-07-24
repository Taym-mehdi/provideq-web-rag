from __future__ import annotations

import html
import re


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "in", "on", "for", "to",
    "from", "with", "by", "at", "is", "are", "was", "were", "be", "been", "being", "as",
    "that", "this", "these", "those", "it", "its", "into", "about", "after", "before",
    "between", "during", "under", "over", "than", "can", "could", "should", "would", "do",
    "does", "did", "what", "which", "who", "why", "how", "when", "where", "whether", "up",
}


def clean_text(text: str) -> str:
    value = html.unescape(text or "")
    value = re.sub(r"<\s*/\s*h\d\s*>", ". ", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*h\d[^>]*>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([.,;:!?])", r"\1", value)
    value = re.sub(r"([.!?]){2,}", r"\1", value)
    return value.strip()


def clean_multiline_text(text: str) -> str:
    lines = [clean_text(line) for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def tokenize(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+(?:[+\-][A-Za-z0-9]*)?", text or "")
        if token
    ]


def extract_keywords(text: str, *, min_length: int = 3) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        normalized = token.rstrip("+-")
        if len(normalized) < min_length or normalized in STOPWORDS or token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def extract_numeric_terms(text: str) -> list[str]:
    values = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    units = re.findall(
        r"\b(?:hours?|hrs?|minutes?|mins?|days?|weeks?|months?|celsius|°c)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    return deduplicate_text([*values, *[unit.casefold() for unit in units]])


def split_sentences(text: str) -> list[str]:
    value = clean_text(text)
    if not value:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def normalize_for_deduplication(text: str) -> str:
    return " ".join(tokenize(clean_text(text)))


def deduplicate_text(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = clean_text(item)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def overlap_ratio(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    return intersection / min(len(left_tokens), len(right_tokens))
