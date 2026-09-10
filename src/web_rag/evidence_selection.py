from __future__ import annotations

from collections import Counter

from .models import TextChunk
from .text_utils import normalize_for_deduplication, overlap_ratio


# Abstracts and article bodies often repeat the same finding with slightly
# different wording. A lower same-paper threshold removes that repetition while
# preserving the stricter configurable threshold across independent papers.
_SAME_PAPER_NEAR_DUPLICATE_THRESHOLD = 0.85


def _paper_key(chunk: TextChunk) -> str:
    paper = chunk.paper
    return (
        paper.doi.casefold()
        or paper.paper_id.casefold()
        or paper.url.casefold()
        or paper.title.casefold()
    )


def _chunk_key(chunk: TextChunk) -> tuple[str, int, str]:
    return (
        _paper_key(chunk),
        chunk.chunk_index,
        normalize_for_deduplication(chunk.text),
    )


def _is_near_duplicate(
    chunk: TextChunk,
    selected: list[TextChunk],
    threshold: float,
) -> bool:
    paper_key = _paper_key(chunk)
    for previous in selected:
        pair_threshold = threshold
        if _paper_key(previous) == paper_key:
            pair_threshold = min(
                pair_threshold,
                _SAME_PAPER_NEAR_DUPLICATE_THRESHOLD,
            )
        if overlap_ratio(chunk.text, previous.text) >= pair_threshold:
            return True
    return False


def select_evidence(
    ranked_chunks: list[TextChunk],
    *,
    top_k: int,
    max_chunks_per_paper: int,
    near_duplicate_threshold: float,
) -> list[TextChunk]:
    """Select diverse evidence, then relax the paper cap to fill top_k.

    The paper cap is a diversity preference. Exact and near-duplicate chunks are
    never added merely to reach the requested result count.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if max_chunks_per_paper <= 0:
        raise ValueError("max_chunks_per_paper must be greater than 0")
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")

    selected: list[TextChunk] = []
    selected_keys: set[tuple[str, int, str]] = set()
    counts: Counter[str] = Counter()

    def add_candidates(*, enforce_paper_cap: bool) -> None:
        for chunk in ranked_chunks:
            if len(selected) >= top_k:
                return

            key = _chunk_key(chunk)
            paper_key = _paper_key(chunk)
            if key in selected_keys:
                continue
            if enforce_paper_cap and counts[paper_key] >= max_chunks_per_paper:
                continue
            if _is_near_duplicate(
                chunk,
                selected,
                near_duplicate_threshold,
            ):
                continue

            selected.append(chunk)
            selected_keys.add(key)
            counts[paper_key] += 1

    add_candidates(enforce_paper_cap=True)
    if len(selected) < top_k:
        add_candidates(enforce_paper_cap=False)
    return selected
