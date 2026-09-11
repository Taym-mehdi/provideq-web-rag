from __future__ import annotations

import argparse
import sys
import types
import unittest
from unittest.mock import call, patch

# Keep this focused unit test independent of optional environment-loading support.
if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = dotenv

from evaluation.run_document_retrieval import _retrieve
from web_rag.europepmc_retriever import EuropePMCRetrieval
from web_rag.models import Paper, PaperclipRetrieval
from web_rag.query_reformulation import QueryGenerationError


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        query_strategy="llmexpand",
        llm_model="test-model",
        llm_base_url="https://example.invalid",
        llm_temperature=0.0,
        hyde_max_tokens=100,
        llm_timeout=10.0,
        expansion_max_tokens=120,
        expansion_max_terms=4,
        expansion_max_query_chars=400,
        retriever="fusion",
        paperclip_candidate_limit=30,
        retrieval_limit=20,
        paperclip_source="pmc",
        paperclip_ranking="hybrid",
        paperclip_full_corpus=True,
        paperclip_timeout=30.0,
        paperclip_query_fusion=True,
        query_fusion_rrf_k=10,
        reformulated_query_weight=0.5,
        europepmc_use_reformulated_query=False,
        europepmc_candidate_limit=30,
        europepmc_synonym=False,
        europepmc_mode="multi",
        rrf_k=60,
        europepmc_timeout=30.0,
        europepmc_cache_dir=None,
    )


class QueryFusionRoutingTests(unittest.TestCase):
    @patch("evaluation.run_document_retrieval.retrieve_papers_europepmc")
    @patch("evaluation.run_document_retrieval.retrieve_papers_paperclip")
    def test_raw_and_expanded_queries_only_fuse_inside_paperclip(
        self,
        retrieve_paperclip,
        retrieve_europepmc,
    ) -> None:
        question = "Is potassium stable after delayed centrifugation?"
        raw_paper = Paper("PMC1", "Raw hit", "x", "paperclip", retrieval_rank=1)
        expanded_paper = Paper(
            "PMC2", "Expanded hit", "x", "paperclip", retrieval_rank=1
        )
        epmc_paper = Paper("PMID:3", "Europe PMC hit", "x", "europepmc", retrieval_rank=1)
        retrieve_paperclip.side_effect = [
            PaperclipRetrieval([raw_paper], result_id="raw-result"),
            PaperclipRetrieval([expanded_paper], result_id="expanded-result"),
        ]
        retrieve_europepmc.return_value = EuropePMCRetrieval(
            [epmc_paper], [question]
        )

        trace = _retrieve(
            question,
            _args(),
            lambda _: '{"added_terms": ["delayed centrifugation stability"]}',
        )

        self.assertEqual(retrieve_paperclip.call_count, 2)
        self.assertEqual(retrieve_paperclip.call_args_list[0].args[0], question)
        self.assertIn(
            "delayed centrifugation stability",
            retrieve_paperclip.call_args_list[1].args[0],
        )
        self.assertEqual(retrieve_europepmc.call_args, call(question, limit=30,
            candidate_limit=30, synonym=False, mode="multi", rrf_k=60,
            timeout=30.0, cache_dir=None))
        self.assertIn("raw:raw-result", trace.paperclip_result_id)
        self.assertIn("llmexpand:expanded-result", trace.paperclip_result_id)

    @patch("evaluation.run_document_retrieval.retrieve_papers_europepmc")
    @patch("evaluation.run_document_retrieval.retrieve_papers_paperclip")
    def test_generator_outage_degrades_to_raw_retrieval(
        self,
        retrieve_paperclip,
        retrieve_europepmc,
    ) -> None:
        question = "Is potassium stable after delayed centrifugation?"
        retrieve_paperclip.return_value = PaperclipRetrieval(
            [Paper("PMC1", "Raw hit", "x", "paperclip", retrieval_rank=1)]
        )
        retrieve_europepmc.return_value = EuropePMCRetrieval([], [question])

        def unavailable(_: str) -> str:
            raise QueryGenerationError("temporary connection error")

        trace = _retrieve(question, _args(), unavailable)

        retrieve_paperclip.assert_called_once()
        self.assertEqual(retrieve_paperclip.call_args.args[0], question)
        self.assertIn("used raw query", trace.warnings[0])

    @patch("evaluation.run_document_retrieval.retrieve_papers_europepmc")
    @patch("evaluation.run_document_retrieval.retrieve_papers_paperclip")
    def test_europepmc_outage_keeps_paperclip_results(
        self,
        retrieve_paperclip,
        retrieve_europepmc,
    ) -> None:
        question = "Is potassium stable after delayed centrifugation?"
        retrieve_paperclip.side_effect = [
            PaperclipRetrieval(
                [Paper("PMC1", "Raw hit", "x", "paperclip", retrieval_rank=1)]
            ),
            PaperclipRetrieval([], result_id="expanded"),
        ]
        retrieve_europepmc.side_effect = RuntimeError("HTTP 503")

        trace = _retrieve(
            question,
            _args(),
            lambda _: '{"added_terms": ["delayed centrifugation"]}',
        )

        self.assertEqual(trace.papers[0].title, "Raw hit")
        self.assertIn("Europe PMC failed", trace.warnings[0])


if __name__ == "__main__":
    unittest.main()
