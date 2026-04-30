from __future__ import annotations

from collections.abc import Iterable

from web_rag.config import get_settings
from web_rag.models import Paper, Snippet
from web_rag.text_utils import clean_text, split_into_sentences


def build_sentence_windows(sentences: list[str], window_size: int) -> list[str]:
    """
    Build overlapping sentence windows from a list of sentences.

    Example with window_size=2:

        S1, S2, S3

    becomes:

        S1 + S2
        S2 + S3
        S3

    Overlapping windows are useful because scientific evidence often spans
    more than one sentence. A single sentence may mention the molecule, while
    the next sentence contains the stability result.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")

    windows: list[str] = []

    for index in range(len(sentences)):
        window = sentences[index:index + window_size]
        text = clean_text(" ".join(window))

        if text:
            windows.append(text)

    return windows


def extract_snippets_from_paper(
    paper: Paper,
    window_size: int | None = None,
    include_title_fallback: bool = True,
) -> list[Snippet]:
    """
    Extract evidence snippets from one paper.

    Main behavior:
    - split the abstract into sentences
    - build overlapping windows
    - attach the original Paper object to every Snippet

    Fallback behavior:
    - if a paper has no abstract, optionally use the title as a weak snippet

    The fallback is useful because some biomedical records contain useful titles
    but no abstract in Europe PMC. However, title-only snippets should later be
    treated as weaker evidence during ranking and evaluation.
    """
    settings = get_settings()
    effective_window_size = window_size or settings.snippet_window

    abstract = clean_text(paper.abstract)

    if not abstract:
        if include_title_fallback and paper.title:
            return [
                Snippet(
                    paper=paper,
                    text=clean_text(paper.title),
                    score=0.0,
                )
            ]

        return []

    sentences = split_into_sentences(abstract)
    windows = build_sentence_windows(
        sentences=sentences,
        window_size=effective_window_size,
    )

    snippets = [
        Snippet(
            paper=paper,
            text=window,
            score=0.0,
        )
        for window in windows
    ]

    return snippets


def extract_snippets(
    papers: Iterable[Paper],
    window_size: int | None = None,
    include_title_fallback: bool = True,
) -> list[Snippet]:
    """
    Extract snippets from a collection of papers.

    Each snippet keeps a reference to its source paper. This is important for
    traceability because later every ranked evidence snippet must still know:

    - title
    - year
    - DOI
    - PMID / PMCID
    - journal
    - URL
    """
    all_snippets: list[Snippet] = []

    for paper in papers:
        paper_snippets = extract_snippets_from_paper(
            paper=paper,
            window_size=window_size,
            include_title_fallback=include_title_fallback,
        )

        all_snippets.extend(paper_snippets)

    return all_snippets