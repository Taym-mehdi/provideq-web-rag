from __future__ import annotations

import json
import unittest

from web_rag.paperclip_retriever import (
    PaperclipError,
    PaperclipMCPClient,
    load_paper_full_texts,
    retrieve_papers,
)


class _Response:
    def __init__(
        self,
        payload: dict,
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        value = self.text or json.dumps(self._payload)
        return value.encode("utf-8")


class _Opener:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, request, *, timeout: float):
        self.calls.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "json": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)


def _tool_response(text: str, *, is_error: bool = False) -> _Response:
    return _Response(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        }
    )


def _search_output() -> str:
    return json.dumps(
        {
            "search_id": "s_mcp_test",
            "query": "potassium delay",
            "papers": [
                {
                    "id": "PMC9750740",
                    "title": "Impact of Time Delay in Serum Potassium",
                    "url": "https://citations.gxl.ai/papers/PMC9750740",
                    "source": "pmc",
                    "published_date": "2022-01-01",
                    "authors": ["A. Author", "B. Author"],
                    "doi": "10.1000/example",
                    "year": 2022,
                    "journal": "Example Journal",
                    "abstract": "The study measured potassium after a delay.",
                }
            ],
        }
    )


class PaperclipMCPTests(unittest.TestCase):
    def test_search_uses_advertised_mcp_tool_and_preserves_parameters(
        self,
    ) -> None:
        opener = _Opener([_tool_response(_search_output())])
        client = PaperclipMCPClient("gxl_test", opener=opener)

        result = retrieve_papers(
            "potassium delay",
            limit=1,
            source="pmc,abstracts_only",
            ranking="vector",
            load_full_text=False,
            client=client,
        )

        self.assertEqual(result.result_id, "s_mcp_test")
        self.assertEqual(len(result.papers), 1)
        paper = result.papers[0]
        self.assertEqual(
            paper.text,
            "The study measured potassium after a delay.",
        )
        self.assertFalse(paper.metadata["has_full_text"])

        request = opener.calls[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["url"], "https://paperclip.gxl.ai/mcp")
        self.assertEqual(request["headers"]["X-api-key"], "gxl_test")
        self.assertEqual(request["json"]["method"], "tools/call")
        self.assertEqual(request["json"]["params"]["name"], "search")
        arguments = request["json"]["params"]["arguments"]
        self.assertEqual(arguments["ranking"], "vector")
        self.assertEqual(arguments["source"], "pmc,abstracts_only")
        self.assertEqual(arguments["limit"], 3)
        self.assertTrue(arguments["all_time"])
        self.assertTrue(arguments["as_json"])

    def test_formatted_mcp_search_output_is_parsed_when_json_is_ignored(
        self,
    ) -> None:
        output = (
            "Found 1 papers  [s_f012ab12]\n\n"
            "  \x1b[1m1. Time as a significant factor in potassium release\x1b[0m\n"
            "     Tom Reuter, Michael Müller\n"
            "     \x1b[2mPMC11627413 · pmc · 2024-12-09\x1b[0m\n"
            "     \x1b[36mhttps://www.ncbi.nlm.nih.gov/pmc/articles/PMC11627413/\x1b[0m\n"
            "     \x1b[2m\"Potassium increased with delayed centrifugation.\"\x1b[0m\n"
        )
        opener = _Opener([_tool_response(output)])
        client = PaperclipMCPClient("gxl_test", opener=opener)

        result = retrieve_papers(
            "potassium delay",
            limit=1,
            source="pmc",
            ranking="vector",
            load_full_text=False,
            client=client,
        )

        self.assertEqual(result.result_id, "s_f012ab12")
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].paper_id, "PMC11627413")
        self.assertEqual(result.papers[0].source, "pmc")
        self.assertIn("delayed centrifugation", result.papers[0].abstract)

    def test_full_text_is_loaded_with_head_and_metadata_with_cat(self) -> None:
        metadata = {
            "document_id": "PMC9750740",
            "title": "Impact of Time Delay in Serum Potassium",
            "abstract": "The study measured potassium after a delay.",
            "doi": "10.1000/example",
        }
        opener = _Opener(
            [
                _tool_response(_search_output()),
                _tool_response("L1: Introduction\nL2: Complete article result."),
                _tool_response(json.dumps(metadata)),
            ]
        )
        client = PaperclipMCPClient("gxl_test", opener=opener)
        result = retrieve_papers(
            "potassium delay",
            limit=1,
            source="pmc",
            ranking="vector",
            load_full_text=False,
            client=client,
        )

        load_paper_full_texts(result.papers, client=client)

        paper = result.papers[0]
        self.assertEqual(
            paper.text,
            "Introduction\nComplete article result.",
        )
        self.assertTrue(paper.metadata["has_full_text"])
        self.assertEqual(len(opener.calls), 3)
        head_call = opener.calls[1]["json"]["params"]
        self.assertEqual(head_call["name"], "head")
        self.assertEqual(
            head_call["arguments"],
            {
                "path": "/papers/PMC9750740/content.lines",
                "lines": 5000,
            },
        )
        cat_call = opener.calls[2]["json"]["params"]
        self.assertEqual(cat_call["name"], "cat")
        self.assertEqual(
            cat_call["arguments"],
            {"path": "/papers/PMC9750740/meta.json"},
        )

    def test_tool_error_is_reported_without_exposing_api_key(self) -> None:
        opener = _Opener(
            [_tool_response("Temporary Paperclip outage", is_error=True)]
        )
        client = PaperclipMCPClient("gxl_secret", opener=opener)

        with self.assertRaises(PaperclipError) as raised:
            retrieve_papers(
                "potassium delay",
                limit=1,
                source="pmc",
                client=client,
            )

        self.assertIn("Temporary Paperclip outage", str(raised.exception))
        self.assertNotIn("gxl_secret", str(raised.exception))

    def test_unavailable_article_text_does_not_abort_other_sources(self) -> None:
        opener = _Opener(
            [
                _tool_response(_search_output()),
                _tool_response("Full text unavailable", is_error=True),
            ]
        )
        client = PaperclipMCPClient("gxl_test", opener=opener)
        result = retrieve_papers(
            "potassium delay",
            limit=1,
            source="pmc",
            load_full_text=False,
            client=client,
        )

        load_paper_full_texts(result.papers, client=client)

        self.assertFalse(result.papers[0].metadata["has_full_text"])
        self.assertEqual(
            result.papers[0].text,
            "The study measured potassium after a delay.",
        )

    def test_missing_api_key_fails_before_a_request(self) -> None:
        with self.assertRaises(PaperclipError):
            PaperclipMCPClient("")


if __name__ == "__main__":
    unittest.main()
