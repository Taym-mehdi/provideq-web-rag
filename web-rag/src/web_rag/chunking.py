from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import re

from .models import Paper, TextChunk
from .text_utils import clean_text, normalize_for_deduplication, split_sentences, word_count


_SECTION_HEADINGS = {
    "abstract", "background", "objective", "objectives", "methods", "materials and methods",
    "patients and methods", "results", "discussion", "conclusion", "conclusions", "introduction",
    "limitations", "supplementary material", "references",
}
_CONTEXT_DEPENDENT_START = re.compile(
    r"^(?:it|this|these|those|they|such|however|therefore|thus|in contrast|as a result)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class _SentenceUnit:
    text: str
    section: str


def _is_heading(line: str) -> bool:
    value = clean_text(line).rstrip(" .:")
    if not value or len(value) > 120 or word_count(value) > 12:
        return False
    lowered = value.casefold()
    if lowered in _SECTION_HEADINGS:
        return True
    if value.endswith((".", "?", "!", ";")):
        return False
    words = value.split()
    title_case = len(words) >= 2 and sum(word[:1].isupper() for word in words) >= len(words) - 1
    return value.isupper() or title_case


def _sentence_units(text: str) -> list[_SentenceUnit]:
    units: list[_SentenceUnit] = []
    section = ""
    for raw_line in (text or "").splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if _is_heading(line):
            section = line.rstrip(" .:")
            continue
        sentences = split_sentences(line) or [line]
        units.extend(_SentenceUnit(text=sentence, section=section) for sentence in sentences)
    return units


def _fit_window(units: list[_SentenceUnit], max_chars: int) -> str:
    selected: list[str] = []
    length = 0
    for unit in units:
        additional = len(unit.text) + (1 if selected else 0)
        if selected and length + additional > max_chars:
            break
        if not selected and len(unit.text) > max_chars:
            return unit.text[:max_chars].rstrip()
        selected.append(unit.text)
        length += additional
    return clean_text(" ".join(selected))


def sentence_window_chunks(
    paper: Paper,
    *,
    window_size: int,
    stride: int,
    min_chars: int,
    max_chars: int,
    min_words: int,
    context_backoff: bool,
) -> list[TextChunk]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be greater than 0")
    if min_chars < 0 or min_words < 0:
        raise ValueError("minimum chunk thresholds must be non-negative")
    if max_chars < min_chars:
        raise ValueError("max_chars must be greater than or equal to min_chars")

    units = _sentence_units(paper.text)
    chunks: list[TextChunk] = []
    seen: set[str] = set()

    for start in range(0, len(units), stride):
        actual_start = start
        if context_backoff and start > 0 and _CONTEXT_DEPENDENT_START.match(units[start].text):
            actual_start = start - 1
        end = min(start + window_size, len(units))
        window = units[actual_start:end]
        text = _fit_window(window, max_chars)
        if len(text) < min_chars or word_count(text) < min_words:
            continue

        section = units[start].section
        if section and not text.casefold().startswith(section.casefold()):
            candidate = f"{section}: {text}"
            text = candidate if len(candidate) <= max_chars else text

        key = normalize_for_deduplication(text)
        if not key or key in seen:
            continue
        seen.add(key)
        chunks.append(
            TextChunk(
                paper=paper,
                text=text,
                method="sentence_window",
                chunk_index=len(chunks),
                section=section,
                start_sentence=actual_start,
                end_sentence=end - 1,
            )
        )
    return chunks


_CHUNKERS: dict[str, Callable[..., list[TextChunk]]] = {
    "sentence_window": sentence_window_chunks,
}


def chunk_papers(
    papers: Iterable[Paper],
    *,
    method: str = "sentence_window",
    window_size: int = 3,
    stride: int = 1,
    min_chars: int = 60,
    max_chars: int = 1200,
    min_words: int = 10,
    context_backoff: bool = True,
) -> list[TextChunk]:
    selected = method.strip().casefold().replace("-", "_")
    try:
        chunker = _CHUNKERS[selected]
    except KeyError as exc:
        choices = ", ".join(_CHUNKERS)
        raise ValueError(f"Unknown chunking method '{method}'. Choose from: {choices}") from exc

    output: list[TextChunk] = []
    for paper in papers:
        output.extend(
            chunker(
                paper,
                window_size=window_size,
                stride=stride,
                min_chars=min_chars,
                max_chars=max_chars,
                min_words=min_words,
                context_backoff=context_backoff,
            )
        )
    return output
