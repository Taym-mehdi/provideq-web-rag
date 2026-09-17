from __future__ import annotations

import unittest
from unittest.mock import call, patch

from web_rag.config import Settings
from web_rag.document_retrieval import retrieve_documents
from web_rag.europepmc_retriever import EuropePMCRetrieval
from web_rag.models import Paper, PaperclipRetrieval, QueryBundle
from web_rag.paperclip_retriever import load_paper_full_texts


class DefaultDocumentRetrievalTests(unittest.TestCase):
    @patch("web_rag.document_retrieval.load_paper_full_texts")
    @patch("web_rag.document_retrieval.retrieve_papers_europepmc")
    @patch("web_rag.document_retrieval.retrieve_papers_paperclip")
    @patch("web_rag.document_retrieval.create_client")
    def test_default_matches_the_winning_fusion_configuration(
        self,
        create_client,
        retrieve_paperclip,
        retrieve_europepmc,
        load_full_texts,
    ) -> None:
        question = "Is potassium stable after delayed centrifugation?"
        expanded = question + "; serum potassium concentration stability"
        query = QueryBundle(
            original_question=question,
            normalized_question=question,
            strategy="llmexpand",
            search_query=expanded,
            expanded_terms=["serum potassium concentration stability"],
        )
        client = object()
        create_client.return_value = client
        retrieve_paperclip.side_effect = [
            PaperclipRetrieval(
                [Paper("PMC1", "Raw paper", "Raw abstract", "pmc", retrieval_rank=1)],
                result_id="raw-result",
            ),
            PaperclipRetrieval(
                [
                    Paper(
                        "PMC2",
                        "Expansion paper",
                        "Expansion abstract",
                        "pmc",
                        retrieval_rank=1,
                    )
                ],
                result_id="expanded-result",
            ),
        ]
        retrieve_europepmc.return_value = EuropePMCRetrieval(
            [
                Paper(
                    "PMC3",
                    "Europe PMC paper",
                    "Europe PMC abstract",
                    "europepmc",
                    retrieval_rank=1,
                )
            ],
            [question, "TITLE_ABS:(potassium centrifugation)"],
        )
        load_full_texts.side_effect = lambda papers, **_: papers

        result = retrieve_documents(
            question,
            query,
            settings=Settings(),
            load_full_text=True,
        )

        self.assertEqual(retrieve_paperclip.call_count, 2)
        self.assertEqual(
            [item.args[0] for item in retrieve_paperclip.call_args_list],
            [question, expanded],
        )
        for item in retrieve_paperclip.call_args_list:
            self.assertEqual(item.kwargs["limit"], 30)
            self.assertEqual(item.kwargs["ranking"], "vector")
            self.assertFalse(item.kwargs["load_full_text"])
            self.assertIs(item.kwargs["client"], client)
        self.assertEqual(
            retrieve_europepmc.call_args,
            call(
                question,
                limit=30,
                candidate_limit=30,
                synonym=False,
                mode="multi",
                rrf_k=60,
                timeout=45.0,
                cache_dir=Settings().europepmc_cache_dir,
            ),
        )
        load_full_texts.assert_called_once()
        self.assertEqual(
            result.paperclip_result_id,
            "raw:raw-result; llmexpand:expanded-result",
        )
        self.assertEqual(result.source_counts, {"paperclip": 2, "europepmc": 1})
        self.assertEqual(len(result.papers), 3)
        self.assertEqual(
            [paper.retrieval_rank for paper in result.papers],
            [1, 2, 3],
        )

    @patch("web_rag.paperclip_retriever._read_metadata")
    @patch("web_rag.paperclip_retriever._read_full_text")
    def test_only_loaded_article_text_is_marked_as_full_text(
        self,
        read_full_text,
        read_metadata,
    ) -> None:
        read_full_text.side_effect = ["Complete PMC article.", ""]
        read_metadata.return_value = {}
        papers = [
            Paper(
                "PMC1",
                "Loaded paper",
                "Abstract one.",
                "paperclip+europepmc",
                metadata={"has_full_text": False},
            ),
            Paper(
                "PMC2",
                "Abstract-only paper",
                "Abstract two.",
                "europepmc",
                metadata={
                    "full_text_available": True,
                    "has_full_text": False,
                },
            ),
        ]

        load_paper_full_texts(papers, client=object())

        self.assertEqual(papers[0].text, "Complete PMC article.")
        self.assertTrue(papers[0].metadata["has_full_text"])
        self.assertEqual(papers[1].text, "Abstract two.")
        self.assertFalse(papers[1].metadata["has_full_text"])


if __name__ == "__main__":
    unittest.main()
