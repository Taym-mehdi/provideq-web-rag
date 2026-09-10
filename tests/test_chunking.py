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

        chunks = token_aware_chunks(
            paper,
            tokenizer_model="unused",
            max_tokens=16,
            overlap_fraction=0.20,
            max_overlap_sentences=5,
            min_words=1,
            tokenizer=self.tokenizer,
        )

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

        chunks = token_aware_chunks(
            paper,
            tokenizer_model="unused",
            max_tokens=16,
            overlap_fraction=0.20,
            max_overlap_sentences=5,
            min_words=1,
            tokenizer=self.tokenizer,
        )

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk.token_count <= 16 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
