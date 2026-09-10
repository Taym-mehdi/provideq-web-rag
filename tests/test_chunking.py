from __future__ import annotations

import unittest

from web_rag.chunking import token_aware_chunks
from web_rag.models import Paper


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ):
        return " ".join(token_ids)


class TokenAwareChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()

    def chunk(self, paper: Paper, *, max_tokens: int = 64):
        return token_aware_chunks(
            paper,
            tokenizer_model="unused",
            max_tokens=max_tokens,
            overlap_fraction=0.20,
            max_overlap_sentences=5,
            min_words=1,
            tokenizer=self.tokenizer,
        )

    def test_respects_token_budget_sections_and_sentence_overlap(self) -> None:
        paper = Paper(
            paper_id="PMC1",
            title="Test",
            source="pmc",
            text=(
                "Methods\n"
                "Alpha beta gamma delta epsilon zeta. "
                "Eta theta iota kappa lambda mu. "
                "Nu xi omicron pi rho sigma.\n"
                "Results\n"
                "Tau upsilon phi chi psi omega. "
                "One two three four five six. "
                "Seven eight nine ten eleven twelve.\n"
                "References\n"
                "This reference section should never be emitted."
            ),
        )

        chunks = self.chunk(paper, max_tokens=16)

        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(all(chunk.token_count <= 16 for chunk in chunks))
        self.assertEqual(
            {chunk.section for chunk in chunks},
            {"Methods", "Results"},
        )
        self.assertFalse(
            any("reference section" in chunk.text for chunk in chunks)
        )

        methods = [
            chunk for chunk in chunks if chunk.section == "Methods"
        ]
        self.assertEqual(
            methods[1].start_sentence,
            methods[0].end_sentence,
        )

    def test_oversized_sentence_is_split_without_losing_the_budget(self) -> None:
        words = " ".join(f"word{index}" for index in range(40))
        paper = Paper(
            paper_id="PMC2",
            title="Long sentence",
            source="pmc",
            text=f"Results\n{words}.",
        )

        chunks = self.chunk(paper, max_tokens=16)

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk.token_count <= 16 for chunk in chunks))

    def test_removes_reference_tail_supplements_and_title_only_text(self) -> None:
        title = "Differential expression of pathways after stimulation"
        paper = Paper(
            paper_id="PMC3",
            title=title,
            source="pmc",
            text=(
                f"{title}☆\n"
                "Results\n"
                "Potassium remained stable after prompt processing.\n"
                "Conclusion\n"
                "1. Smith J, Doe A. First study. Journal. 2019; 1: 1-2. "
                "2. Jones B, Roe C. Second study. Journal. 2020; 2: 3-4. "
                "3. Brown D, White E. Third study. Journal. 2021; 3: 5-6.\n"
                "Supplementary data\n"
                "Table S1 should not be emitted."
            ),
        )

        chunks = self.chunk(paper)
        combined = " ".join(chunk.text for chunk in chunks)

        self.assertIn("Potassium remained stable", combined)
        self.assertNotIn("Differential expression", combined)
        self.assertNotIn("Smith J", combined)
        self.assertNotIn("Table S1", combined)

    def test_keeps_numbered_scientific_results(self) -> None:
        paper = Paper(
            paper_id="PMC4",
            title="Numbered results",
            source="pmc",
            text=(
                "Results\n"
                "1. Potassium increased after the delay. "
                "2. Sodium remained stable after processing. "
                "3. Calcium showed no meaningful change."
            ),
        )

        chunks = self.chunk(paper)

        self.assertTrue(chunks)
        self.assertIn("Potassium increased", chunks[0].text)

    def test_ignores_short_front_matter_lines(self) -> None:
        paper = Paper(
            paper_id="PMC5",
            title="Front matter",
            source="pmc",
            text=(
                "Accepted: 08-9-2025\n"
                "3 Population Council, Delhi, India\n"
                "Abstract\n"
                "This study evaluates stable serum processing procedures."
            ),
        )

        chunks = self.chunk(paper)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Abstract")
        self.assertNotIn("Accepted", chunks[0].text)
        self.assertNotIn("Population Council", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
