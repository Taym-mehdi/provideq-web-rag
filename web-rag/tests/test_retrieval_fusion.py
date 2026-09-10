from __future__ import annotations

import unittest

from web_rag.europepmc_retriever import build_query_variants
from web_rag.models import Paper
from web_rag.multi_source_retriever import reciprocal_rank_fusion


class EuropePMCQueryTests(unittest.TestCase):
    def test_multi_query_builds_relaxed_title_abstract_variants(self) -> None:
        variants = build_query_variants(
            "How stable are serum metabolites after repeated freeze-thaw cycles?",
            mode="multi",
        )
        self.assertGreaterEqual(len(variants), 2)
        self.assertTrue(any(value.startswith("TITLE_ABS:(") for value in variants))


class FusionTests(unittest.TestCase):
    def test_duplicate_doi_is_fused_across_sources(self) -> None:
        paperclip = Paper(
            paper_id="PMC1",
            title="Example paper",
            text="x",
            source="paperclip",
            doi="10.1234/example",
            retrieval_rank=1,
        )
        europepmc_duplicate = Paper(
            paper_id="PMID:10",
            title="Example paper",
            text="abstract",
            source="europepmc",
            doi="10.1234/example",
            retrieval_rank=2,
            metadata={"pmid": "10"},
        )
        other = Paper(
            paper_id="PMID:11",
            title="Other paper",
            text="abstract",
            source="europepmc",
            retrieval_rank=1,
            metadata={"pmid": "11"},
        )

        result = reciprocal_rank_fusion(
            {
                "paperclip": [paperclip],
                "europepmc": [other, europepmc_duplicate],
            },
            limit=10,
        )

        self.assertEqual(len(result.papers), 2)
        self.assertEqual(result.papers[0].title, "Example paper")
        self.assertEqual(
            set(result.papers[0].metadata["fusion_sources"]),
            {"paperclip", "europepmc"},
        )

    def test_source_weights_preserve_the_preferred_ranking(self) -> None:
        raw_first = Paper(
            paper_id="PMC10",
            title="Raw first",
            text="x",
            source="paperclip",
            retrieval_rank=1,
        )
        reformulated_first = Paper(
            paper_id="PMC20",
            title="Reformulated first",
            text="x",
            source="paperclip",
            retrieval_rank=1,
        )

        result = reciprocal_rank_fusion(
            {
                "raw": [raw_first],
                "reformulated": [reformulated_first],
            },
            limit=2,
            rrf_k=10,
            source_weights={"raw": 1.0, "reformulated": 0.7},
        )

        self.assertEqual(result.papers[0].title, "Raw first")
        self.assertEqual(
            result.papers[0].metadata["fusion_source_weights"]["raw"],
            1.0,
        )

    def test_source_weights_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(
                {"raw": []},
                source_weights={"raw": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
