from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any

from .models import Paper, TextChunk
from .text_utils import (
    clean_text,
    normalize_for_deduplication,
    split_sentences,
    word_count,
)


_SECTION_HEADINGS = {
    "abstract",
    "background",
    "objective",
    "objectives",
    "introduction",
    "methods",
    "materials and methods",
    "patients and methods",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "author contributions",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "data availability",
    "supplementary material",
}
_SKIPPED_SECTIONS = {
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "author contributions",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "supplementary material",
}


@dataclass(frozen=True)
class _SentenceUnit:
    text: str
    section: str
    sentence_index: int


def _normalize_heading(value: str) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", clean_text(value).casefold()).split()
    )


def _is_heading(line: str) -> bool:
    value = clean_text(line).rstrip(" .:")
    if not value or len(value) > 120 or word_count(value) > 12:
        return False

    normalized = _normalize_heading(value)
    if normalized in _SECTION_HEADINGS:
        return True
    if value.endswith((".", "?", "!", ";")):
        return False

    words = value.split()
    title_case = (
        len(words) >= 2
        and sum(word[:1].isupper() for word in words) >= len(words) - 1
    )
    return value.isupper() or title_case


def _sentence_units(text: str) -> list[_SentenceUnit]:
    units: list[_SentenceUnit] = []
    section = ""
    sentence_index = 0

    for raw_line in (text or "").splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if _is_heading(line):
            section = line.rstrip(" .:")
            continue

        for sentence in split_sentences(line) or [line]:
            units.append(
                _SentenceUnit(
                    text=sentence,
                    section=section,
                    sentence_index=sentence_index,
                )
            )
            sentence_index += 1

    return units


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Token-aware chunking requires transformers. "
            "Install the project requirements first."
        ) from exc

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _token_count(tokenizer: Any, text: str) -> int:
    return len(_encode(tokenizer, text))


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return clean_text(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )


def _split_oversized_unit(
    unit: _SentenceUnit,
    *,
    tokenizer: Any,
    max_tokens: int,
) -> list[_SentenceUnit]:
    token_ids = _encode(tokenizer, unit.text)
    if len(token_ids) <= max_tokens:
        return [unit]

    fragments: list[_SentenceUnit] = []
    for start in range(0, len(token_ids), max_tokens):
        text = _decode(tokenizer, token_ids[start : start + max_tokens])
        if text:
            fragments.append(
                _SentenceUnit(
                    text=text,
                    section=unit.section,
                    sentence_index=unit.sentence_index,
                )
            )
    return fragments


def _section_groups(
    units: list[_SentenceUnit],
) -> list[tuple[str, list[_SentenceUnit]]]:
    groups: list[tuple[str, list[_SentenceUnit]]] = []
    for unit in units:
        if groups and groups[-1][0] == unit.section:
            groups[-1][1].append(unit)
        else:
            groups.append((unit.section, [unit]))
    return groups


def _should_skip_section(section: str) -> bool:
    normalized = _normalize_heading(section)
    return any(
        normalized == name
        or normalized.endswith(f" {name}")
        or normalized.startswith(f"{name} ")
        for name in _SKIPPED_SECTIONS
    )


def _section_prefix(section: str) -> str:
    value = clean_text(section).rstrip(" .:")
    return f"{value}: " if value else ""


def _overlap_size(
    units: list[_SentenceUnit],
    *,
    tokenizer: Any,
    overlap_fraction: float,
    max_overlap_sentences: int,
) -> int:
    if (
        overlap_fraction <= 0
        or max_overlap_sentences <= 0
        or len(units) <= 1
    ):
        return 0

    body = " ".join(unit.text for unit in units)
    target_tokens = max(
        1,
        math.ceil(_token_count(tokenizer, body) * overlap_fraction),
    )
    overlap_tokens = 0
    overlap_sentences = 0

    for unit in reversed(units):
        if overlap_sentences >= max_overlap_sentences:
            break
        overlap_tokens += _token_count(tokenizer, unit.text)
        overlap_sentences += 1
        if overlap_tokens >= target_tokens:
            break

    # A chunk must always advance by at least one unit.
    return min(overlap_sentences, len(units) - 1)


def token_aware_chunks(
    paper: Paper,
    *,
    tokenizer_model: str,
    max_tokens: int,
    overlap_fraction: float,
    max_overlap_sentences: int,
    min_words: int,
    tokenizer: Any | None = None,
) -> list[TextChunk]:
    """Merge adjacent sentences up to a token budget without crossing sections.

    Overlap is copied from complete trailing sentences. The only exception is an
    individual sentence longer than the budget, which is split by tokens so no
    article text is silently discarded.
    """
    if max_tokens < 16:
        raise ValueError("max_tokens must be at least 16")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    if max_overlap_sentences < 0 or min_words < 0:
        raise ValueError("overlap and minimum-word limits must be non-negative")

    active_tokenizer = tokenizer or _load_tokenizer(tokenizer_model)
    chunks: list[TextChunk] = []
    seen: set[str] = set()

    for section, raw_units in _section_groups(_sentence_units(paper.text)):
        if _should_skip_section(section):
            continue

        prefix = _section_prefix(section)
        prefix_tokens = _token_count(active_tokenizer, prefix)
        if prefix_tokens >= max_tokens - 8:
            prefix = ""
            prefix_tokens = 0
        body_budget = max_tokens - prefix_tokens

        units: list[_SentenceUnit] = []
        for unit in raw_units:
            units.extend(
                _split_oversized_unit(
                    unit,
                    tokenizer=active_tokenizer,
                    max_tokens=body_budget,
                )
            )

        start = 0
        while start < len(units):
            end = start
            selected: list[_SentenceUnit] = []

            while end < len(units):
                candidate = selected + [units[end]]
                body = " ".join(item.text for item in candidate)
                if (
                    selected
                    and _token_count(active_tokenizer, body) > body_budget
                ):
                    break
                selected = candidate
                end += 1

            if not selected:
                start += 1
                continue

            body = clean_text(" ".join(item.text for item in selected))
            text = clean_text(f"{prefix}{body}")
            token_count = _token_count(active_tokenizer, text)
            key = normalize_for_deduplication(text)

            if (
                key
                and key not in seen
                and word_count(text) >= min_words
                and token_count <= max_tokens
            ):
                seen.add(key)
                chunks.append(
                    TextChunk(
                        paper=paper,
                        text=text,
                        method="token_aware",
                        chunk_index=len(chunks),
                        section=section,
                        start_sentence=selected[0].sentence_index,
                        end_sentence=selected[-1].sentence_index,
                        token_count=token_count,
                    )
                )

            if end >= len(units):
                break
            overlap = _overlap_size(
                selected,
                tokenizer=active_tokenizer,
                overlap_fraction=overlap_fraction,
                max_overlap_sentences=max_overlap_sentences,
            )
            start = max(start + 1, end - overlap)

    return chunks


def chunk_papers(
    papers: Iterable[Paper],
    *,
    method: str = "token_aware",
    tokenizer_model: str = "ncbi/MedCPT-Cross-Encoder",
    max_tokens: int = 512,
    overlap_fraction: float = 0.20,
    max_overlap_sentences: int = 5,
    min_words: int = 5,
    tokenizer: Any | None = None,
) -> list[TextChunk]:
    selected = method.strip().casefold().replace("-", "_")
    if selected != "token_aware":
        raise ValueError(
            f"Unknown chunking method '{method}'. Choose from: token_aware"
        )

    output: list[TextChunk] = []
    for paper in papers:
        output.extend(
            token_aware_chunks(
                paper,
                tokenizer_model=tokenizer_model,
                max_tokens=max_tokens,
                overlap_fraction=overlap_fraction,
                max_overlap_sentences=max_overlap_sentences,
                min_words=min_words,
                tokenizer=tokenizer,
            )
        )
    return output
