from __future__ import annotations

import io
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from web_rag.europepmc_retriever import (
    EuropePMCError,
    _request_json,
    build_query_variants,
    retrieve_papers_europepmc,
)
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

    @patch("web_rag.europepmc_retriever._search_variant")
    def test_failed_multi_query_reports_the_exact_variant(self, search_variant) -> None:
        question = "How stable are serum metabolites after repeated freeze-thaw cycles?"
        variants = build_query_variants(question, mode="multi")
        search_variant.side_effect = [[], EuropePMCError("Europe PMC HTTP 404")]

        with self.assertRaises(EuropePMCError) as raised:
            retrieve_papers_europepmc(
                question,
                limit=20,
                candidate_limit=30,
                mode="multi",
                cache_dir=None,
            )

        error = raised.exception
        self.assertEqual(error.query, question)
        self.assertEqual(error.failed_variant, variants[1])
        self.assertEqual(error.failed_variant_index, 2)
        self.assertEqual(error.query_variants, variants)
        self.assertIn("variant 2/", str(error))

    @patch("web_rag.europepmc_retriever.time.sleep")
    @patch("web_rag.europepmc_retriever.urllib.request.urlopen")
    def test_transient_404_is_retried(self, urlopen, sleep) -> None:
        not_found = urllib.error.HTTPError(
            url="https://example.invalid",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"resultList": {"result": []}}'
        )
        urlopen.side_effect = [not_found, response]

        payload = _request_json({"query": "test"}, timeout=1.0, retries=1)

        self.assertEqual(payload, {"resultList": {"result": []}})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()


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
