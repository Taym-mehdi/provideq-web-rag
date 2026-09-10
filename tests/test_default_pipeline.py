from __future__ import annotations

import unittest
from unittest.mock import patch

from web_rag.config import Settings
from web_rag.models import Paper, PaperclipRetrieval, TextChunk
from web_rag.pipeline import run_pipeline


class DefaultPipelineTests(unittest.TestCase):
    def test_defaults_match_the_aryan_compatible_pipeline(self) -> None:
        settings = Settings()
        self.assertEqual(settings.query_strategy, "raw")
        self.assertEqual(settings.paperclip_ranking, "hybrid")
        self.assertEqual(settings.chunking_method, "token_aware")
        self.assertEqual(settings.chunk_max_tokens, 512)
        self.assertEqual(settings.chunk_overlap_fraction, 0.20)
        self.assertEqual(settings.chunk_max_overlap_sentences, 5)
        self.assertEqual(settings.reranker, "medcpt")
        self.assertEqual(
            settings.medcpt_model,
            "ncbi/MedCPT-Cross-Encoder",
        )
        self.assertEqual(settings.top_k, 20)

    @patch("web_rag.pipeline.rerank_chunks")
    @patch("web_rag.pipeline.chunk_papers")
    @patch("web_rag.pipeline.retrieve_papers")
    def test_full_text_and_metadata_are_preserved(
        self,
        retrieve_papers,
        chunk_papers,
        rerank_chunks,
    ) -> None:
        papers = [
            Paper(
                paper_id=f"PMC{paper_index}",
                title=f"Paper {paper_index}",
                text="Full article text.",
                source="pmc",
                retrieval_rank=paper_index + 1,
                metadata={
                    "has_full_text": True,
                    "paperclip_original_rank": paper_index + 2,
                },
            )
            for paper_index in range(5)
        ]
        retrieve_papers.return_value = PaperclipRetrieval(
            papers,
            result_id="result-1",
        )

        chunks = []
        for index in range(20):
            paper = papers[index % len(papers)]
            unique = " ".join(
                f"term{index}_{value}" for value in range(10)
            )
            chunks.append(
                TextChunk(
                    paper=paper,
                    text=f"Evidence {index} {unique}",
                    method="token_aware",
                    chunk_index=index,
                    section="Results",
                    start_sentence=index,
                    end_sentence=index + 2,
                    token_count=100,
                    score=float(20 - index),
                )
            )
        chunk_papers.return_value = chunks

        def rank(_question, candidates, *, settings):
            for rank, chunk in enumerate(candidates, start=1):
                chunk.rerank_rank = rank
            return candidates

        rerank_chunks.side_effect = rank
        result = run_pipeline(
            "Is serum potassium stable?",
            settings=Settings(),
            paperclip_client=object(),
        )

        self.assertTrue(
            retrieve_papers.call_args.kwargs["load_full_text"]
        )
        self.assertEqual(result.pipeline.returned_evidence_count, 20)
        self.assertEqual(len(result.records), 20)
        record = result.records[0]
        self.assertTrue(record.has_full_text)
        self.assertEqual(record.token_count, 100)
        self.assertEqual(record.paper_retrieval_rank, 1)
        self.assertEqual(record.paperclip_original_rank, 2)
        self.assertEqual(record.rerank_rank, 1)


if __name__ == "__main__":
    unittest.main()
