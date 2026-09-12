from __future__ import annotations

import unittest

from web_rag.query_reformulation import (
    QUERY_PROMPT_VERSION,
    _validated_expansion_terms,
    build_hyde_query,
)


class HyDEQueryTests(unittest.TestCase):
    def test_hyde_keeps_the_original_question_as_an_anchor(self) -> None:
        question = "Which pathways contain plasma proteins affected by ex-vivo proteolysis?"
        passage = (
            "Plasma proteins affected by ex-vivo proteolysis are assessed to determine "
            "which pathways contain them. The neutral comparison measures the stated "
            "plasma protein outcome without predicting candidate pathways or a result."
        )
        prompts: list[str] = []

        def generate(prompt: str) -> str:
            prompts.append(prompt)
            return passage

        query = build_hyde_query(question, generator=generate)

        self.assertEqual(query.hypothetical_document, passage)
        self.assertTrue(query.search_query.startswith(question))
        self.assertIn("do not state a result", prompts[0])
        self.assertIn("never name candidate", prompts[0])
        self.assertNotIn("State a concise plausible finding", prompts[0])

    def test_hyde_prompt_change_has_a_new_cache_version(self) -> None:
        self.assertEqual(QUERY_PROMPT_VERSION, "2026-09-biomedical-ir-v6")


class ExpansionValidationTests(unittest.TestCase):
    def test_unknown_acronym_guesses_are_rejected(self) -> None:
        terms = _validated_expansion_terms(
            [
                "serum haptoglobin",
                "serum hemopexin",
                "hemolysis index",
                "delayed centrifugation",
            ],
            question=(
                "Can the serum HG and HI ratios identify samples centrifuged more "
                "than one hour after collection?"
            ),
            limit=4,
        )

        self.assertEqual(terms, ["delayed centrifugation"])

    def test_known_acronym_full_form_is_allowed(self) -> None:
        terms = _validated_expansion_terms(
            ["RNA integrity number RIN"],
            question="Which conditions affect RNA integrity measured by RIN?",
            limit=4,
        )

        self.assertEqual(terms, ["RNA integrity number RIN"])

    def test_expansion_cannot_reverse_an_explicit_negation(self) -> None:
        terms = _validated_expansion_terms(
            ["thawing procedure", "frozen aliquot extraction"],
            question="How can aliquots be taken without thawing the parent plasma sample?",
            limit=4,
        )

        self.assertNotIn("thawing procedure", terms)


if __name__ == "__main__":
    unittest.main()
