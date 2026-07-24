from __future__ import annotations

from collections import Counter, defaultdict

from .models import TextChunk
from .text_utils import overlap_ratio


def _paper_key(chunk: TextChunk) -> str:
    paper = chunk.paper
    return paper.doi.casefold() or paper.paper_id.casefold() or paper.url.casefold() or paper.title.casefold()


def select_evidence(
    ranked_chunks: list[TextChunk],
    *,
    top_k: int,
    max_chunks_per_paper: int,
    near_duplicate_threshold: float,
) -> list[TextChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if max_chunks_per_paper <= 0:
        raise ValueError("max_chunks_per_paper must be greater than 0")
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")

    selected: list[TextChunk] = []
    counts: Counter[str] = Counter()
    selected_by_paper: dict[str, list[TextChunk]] = defaultdict(list)

    for chunk in ranked_chunks:
        key = _paper_key(chunk)
        if counts[key] >= max_chunks_per_paper:
            continue
        if any(
            overlap_ratio(chunk.text, previous.text) >= near_duplicate_threshold
            for previous in selected_by_paper[key]
        ):
            continue
        selected.append(chunk)
        selected_by_paper[key].append(chunk)
        counts[key] += 1
        if len(selected) >= top_k:
            break
    return selected
