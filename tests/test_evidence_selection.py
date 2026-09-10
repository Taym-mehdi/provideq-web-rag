from __future__ import annotations

from collections import Counter
import unittest

from web_rag.evidence_selection import select_evidence
from web_rag.models import Paper, TextChunk


def _chunk(paper: Paper, index: int, score: float) -> TextChunk:
    unique = " ".join(f"term{index}_{value}" for value in range(10))
    return TextChunk(
        paper=paper,
        text=f"Evidence chunk {index} {unique}",
        method="token_aware",
        chunk_index=index,
        token_count=13,
        rerank_rank=index + 1,
        score=score,
    )


class EvidenceSelectionTests(unittest.TestCase):
    def test_returns_twenty_when_twenty_unique_candidates_exist(self) -> None:
        papers = [
            Paper(f"PMC{number}", f"Paper {number}", "text", "pmc")
            for number in range(3)
        ]
        ranked = [
            _chunk(paper, paper_number * 10 + index, 100 - index)
            for paper_number, paper in enumerate(papers)
            for index in range(10)
        ]

        selected = select_evidence(
            ranked,
            top_k=20,
            max_chunks_per_paper=4,
            near_duplicate_threshold=0.95,
        )

        self.assertEqual(len(selected), 20)
        first_pass_counts = Counter(
            chunk.paper.paper_id for chunk in selected[:12]
        )
        self.assertEqual(set(first_pass_counts.values()), {4})

    def test_near_duplicates_are_not_used_to_fill_the_result(self) -> None:
        first = Paper("PMC1", "First", "text", "pmc")
        second = Paper("PMC2", "Second", "text", "pmc")
        duplicate_text = (
            "The same serum potassium evidence appears in both records."
        )
        ranked = [
            TextChunk(first, duplicate_text, "token_aware", 0, score=2),
            TextChunk(second, duplicate_text, "token_aware", 0, score=1),
        ]

        selected = select_evidence(
            ranked,
            top_k=2,
            max_chunks_per_paper=1,
            near_duplicate_threshold=0.90,
        )

        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
