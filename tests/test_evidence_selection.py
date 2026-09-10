from __future__ import annotations

from collections import Counter
import unittest

from web_rag.evidence_selection import select_evidence
from web_rag.models import Paper, TextChunk


def _chunk(paper: Paper, index: int, score: float) -> TextChunk:
    unique = " ".join(f"term{index}x{value}" for value in range(10))
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

    def test_repeated_finding_within_one_paper_is_removed(self) -> None:
        first = Paper("PMC1", "First", "text", "pmc")
        second = Paper("PMC2", "Second", "text", "pmc")
        detailed = (
            "Serum potassium remained stable after four hours of delayed "
            "centrifugation in separator gel tubes according to study results."
        )
        abbreviated = (
            "Serum potassium was stable after four hours delayed "
            "centrifugation in gel tubes according to results."
        )
        ranked = [
            TextChunk(first, detailed, "token_aware", 0, score=3),
            TextChunk(first, abbreviated, "token_aware", 1, score=2),
            TextChunk(second, abbreviated, "token_aware", 0, score=1),
        ]

        selected = select_evidence(
            ranked,
            top_k=3,
            max_chunks_per_paper=3,
            near_duplicate_threshold=0.95,
        )

        self.assertEqual(
            [(chunk.paper.paper_id, chunk.chunk_index) for chunk in selected],
            [("PMC1", 0), ("PMC2", 0)],
        )


if __name__ == "__main__":
    unittest.main()
