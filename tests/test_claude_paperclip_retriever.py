from __future__ import annotations

import unittest
from unittest.mock import patch

from web_rag.claude_paperclip_retriever import (
    ClaudeAgentRun,
    ClaudePaperclipError,
    build_claude_paperclip_prompt,
    retrieve_papers_with_claude,
    validate_claude_paperclip_environment,
)


class ClaudePaperclipRetrieverTests(unittest.TestCase):
    def test_prompt_is_retrieval_only_and_budgeted(self) -> None:
        prompt = build_claude_paperclip_prompt(
            "Is potassium stable after delayed centrifugation?",
            limit=20,
            max_searches=3,
            source="pmc",
            ranking="hybrid",
        )

        self.assertIn("at most 3 Paperclip search calls", prompt)
        self.assertIn("Return at most 20 unique papers", prompt)
        self.assertIn("Do not answer or synthesize", prompt)
        self.assertIn("Do not use Paperclip map", prompt)
        self.assertIn('"papers"', prompt)

    def test_structured_response_becomes_ranked_unique_papers(self) -> None:
        async def runner(*args, **kwargs):
            return ClaudeAgentRun(
                response_text="""```json
                {
                  "searches": [
                    {"query": "serum potassium delay", "purpose": "initial"}
                  ],
                  "papers": [
                    {
                      "paper_id": "PMC9750740",
                      "title": "Impact of Time Delay",
                      "source": "pmc",
                      "year": "2022",
                      "doi": "https://doi.org/10.1000/example",
                      "pmcid": "PMC9750740",
                      "pmid": "123",
                      "relevance_reason": "Directly studies delayed analysis."
                    },
                    {
                      "paper_id": "PMC9750740",
                      "title": "Impact of Time Delay",
                      "source": "pmc",
                      "year": "2022",
                      "doi": "10.1000/example",
                      "pmcid": "PMC9750740",
                      "pmid": "123",
                      "relevance_reason": "duplicate"
                    }
                  ]
                }
                ```""",
                tool_calls=[{"name": "mcp__paperclip__search", "input": {}}],
                turns=2,
            )

        result = retrieve_papers_with_claude(
            "Is potassium stable after delayed centrifugation?",
            limit=20,
            runner=runner,
        )

        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].paper_id, "PMC9750740")
        self.assertEqual(result.papers[0].retrieval_rank, 1)
        self.assertEqual(result.papers[0].doi, "10.1000/example")
        self.assertEqual(result.papers[0].metadata["pmid"], "123")
        self.assertEqual(result.searches[0]["query"], "serum potassium delay")
        self.assertEqual(result.turns, 2)

    def test_invalid_agent_json_is_rejected(self) -> None:
        async def runner(*args, **kwargs):
            return ClaudeAgentRun(response_text="not JSON")

        with self.assertRaisesRegex(ClaudePaperclipError, "valid JSON"):
            retrieve_papers_with_claude(
                "Question",
                runner=runner,
            )

    def test_preflight_rejects_missing_anthropic_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"PAPERCLIP_API_KEY": "gxl_test"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ClaudePaperclipError,
                "ANTHROPIC_API_KEY",
            ):
                validate_claude_paperclip_environment()


if __name__ == "__main__":
    unittest.main()
