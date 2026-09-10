from __future__ import annotations

import unittest

from web_rag.query_reformulation import (
    _validated_expansion_terms,
    build_hyde_query,
)


class HyDEQueryTests(unittest.TestCase):
    def test_hyde_keeps_the_original_question_as_an_anchor(self) -> None:
        question = "Which pathways contain plasma proteins affected by ex-vivo proteolysis?"
        passage = (
            "Plasma proteome stability during variable processing may involve proteins "
            "from coagulation and complement pathways, with ex-vivo proteolysis altering "
            "mass-spectrometry measurements across preanalytical handling conditions."
        )

        query = build_hyde_query(question, generator=lambda _: passage)

        self.assertEqual(query.hypothetical_document, passage)
        self.assertTrue(query.search_query.startswith(question))
        self.assertIn("coagulation and complement", query.search_query)


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
