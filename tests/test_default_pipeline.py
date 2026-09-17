from __future__ import annotations

import unittest
from unittest.mock import patch

from web_rag.config import Settings
from web_rag.document_retrieval import DocumentRetrieval
from web_rag.models import Paper, TextChunk
from web_rag.pipeline import run_pipeline
from web_rag.query_reformulation import QueryGenerationError


class DefaultPipelineTests(unittest.TestCase):
    def test_defaults_match_the_aryan_compatible_pipeline(self) -> None:
        settings = Settings()
        self.assertEqual(settings.retrieval_system, "fusion")
        self.assertEqual(settings.retrieval_limit, 20)
        self.assertEqual(settings.query_strategy, "llmexpand")
        self.assertEqual(settings.llm_model, "qwen3.6:35b-a3b-bf16")
        self.assertEqual(settings.paperclip_ranking, "vector")
        self.assertEqual(settings.paperclip_candidate_limit, 30)
        self.assertTrue(settings.paperclip_query_fusion)
        self.assertEqual(settings.query_fusion_rrf_k, 10)
        self.assertEqual(settings.reformulated_query_weight, 0.5)
        self.assertEqual(settings.europepmc_mode, "multi")
        self.assertFalse(settings.europepmc_synonym)
        self.assertFalse(settings.europepmc_use_reformulated_query)
        self.assertEqual(settings.europepmc_candidate_limit, 30)
        self.assertEqual(settings.fusion_rrf_k, 60)
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
        self.assertEqual(settings.max_chunks_per_paper, 4)
        self.assertEqual(settings.near_duplicate_threshold, 0.95)

    @patch("web_rag.pipeline.rerank_chunks")
    @patch("web_rag.pipeline.chunk_papers")
    @patch("web_rag.pipeline.retrieve_documents")
    def test_full_text_and_metadata_are_preserved(
        self,
        retrieve_documents,
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
        retrieve_documents.return_value = DocumentRetrieval(
            papers=papers,
            paperclip_result_id="raw:result-1; llmexpand:result-2",
            source_counts={"paperclip": 5, "europepmc": 0},
        )

        chunks = []
        for index in range(20):
            paper = papers[index % len(papers)]
            unique = " ".join(
                f"term{index}x{value}" for value in range(10)
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
                    score_components={
                        "medcpt_cross_encoder": float(20 - index)
                    },
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
            expansion_generator=lambda _: (
                '{"added_terms": ["potassium concentration stability"]}'
            ),
        )

        self.assertTrue(
            retrieve_documents.call_args.kwargs["load_full_text"]
        )
        effective = retrieve_documents.call_args.kwargs["settings"]
        self.assertEqual(effective.retrieval_limit, 20)
        self.assertEqual(effective.retrieval_system, "fusion")
        self.assertEqual(effective.query_strategy, "llmexpand")
        self.assertEqual(effective.paperclip_ranking, "vector")
        self.assertEqual(result.pipeline.returned_evidence_count, 20)
        self.assertEqual(len(result.records), 20)
        self.assertEqual(
            [record.citation_id for record in result.records],
            list(range(1, 21)),
        )
        self.assertEqual(len(result.retrieved_papers), 5)
        self.assertTrue(
            all(paper.has_full_text for paper in result.retrieved_papers)
        )
        self.assertIn("[1] Evidence 0", result.context_text)
        self.assertIn("[20] Evidence 19", result.context_text)
        record = result.records[0]
        self.assertTrue(record.has_full_text)
        self.assertEqual(record.token_count, 100)
        self.assertEqual(record.paper_retrieval_rank, 1)
        self.assertEqual(record.paperclip_original_rank, 2)
        self.assertEqual(record.rerank_rank, 1)
        self.assertEqual(
            record.score_components,
            {"medcpt_cross_encoder": 20.0},
        )

    @patch("web_rag.pipeline.rerank_chunks", return_value=[])
    @patch("web_rag.pipeline.chunk_papers", return_value=[])
    @patch("web_rag.pipeline.retrieve_documents")
    def test_expansion_outage_falls_back_to_the_raw_anchor(
        self,
        retrieve_documents,
        _chunk_papers,
        _rerank_chunks,
    ) -> None:
        retrieve_documents.return_value = DocumentRetrieval(papers=[])

        def unavailable(_: str) -> str:
            raise QueryGenerationError("temporary outage")

        result = run_pipeline(
            "Is serum potassium stable?",
            settings=Settings(),
            paperclip_client=object(),
            expansion_generator=unavailable,
        )

        sent_query = retrieve_documents.call_args.args[1]
        self.assertEqual(sent_query.strategy, "raw")
        self.assertEqual(
            sent_query.search_query,
            "Is serum potassium stable?",
        )
        self.assertIn(
            "used raw query",
            result.pipeline.parameters["warnings"][0],
        )


if __name__ == "__main__":
    unittest.main()
