from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import importlib.util
import json
import os
import re
from typing import Any

from .models import Paper, PaperclipRetrieval
from .text_utils import clean_text


DEFAULT_PAPERCLIP_MCP_URL = "https://paperclip.gxl.ai/mcp"


class ClaudePaperclipError(RuntimeError):
    """Raised when the optional Claude + Paperclip experiment fails."""


@dataclass(frozen=True)
class ClaudeAgentRun:
    response_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    turns: int | None = None


@dataclass
class ClaudePaperclipRetrieval(PaperclipRetrieval):
    searches: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    turns: int | None = None


AgentRunner = Callable[..., Awaitable[ClaudeAgentRun]]


def validate_claude_paperclip_environment(
    *,
    paperclip_api_key: str | None = None,
) -> str:
    """Validate optional experiment requirements before making network calls."""
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise ClaudePaperclipError(
            "ANTHROPIC_API_KEY is not set. Add it to the local .env file."
        )

    resolved_key = paperclip_api_key or os.getenv("PAPERCLIP_API_KEY", "")
    if not resolved_key.strip():
        raise ClaudePaperclipError(
            "PAPERCLIP_API_KEY is not set. Create one at "
            "https://paperclip.gxl.ai/keys and add it to the local .env file."
        )

    if importlib.util.find_spec("claude_agent_sdk") is None:
        raise ClaudePaperclipError(
            "Claude Agent SDK is not installed. Run: "
            "python -m pip install -e \".[claude]\""
        )
    return resolved_key.strip()


def build_claude_paperclip_prompt(
    question: str,
    *,
    limit: int,
    max_searches: int,
    source: str,
    ranking: str,
) -> str:
    """Build the fixed retrieval-only prompt used by the experiment."""
    question = clean_text(question)
    if not question:
        raise ValueError("question must not be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if max_searches <= 0:
        raise ValueError("max_searches must be greater than 0")

    return f"""You are evaluating document retrieval for a biomedical Web RAG system.

Original question:
{question}

Use only the connected Paperclip MCP tools. This is document retrieval only:
- Do not answer or synthesize an answer to the biomedical question.
- Do not use Paperclip map, reduce, routines, or another LLM reader.
- Do not use general web search.
- Make at most {max_searches} Paperclip search calls.
- Start from the original question. A later query may clarify terminology, but
  must preserve the original meaning, entities, conditions, and negation.
- Search source(s): {source}.
- Request Paperclip ranking: {ranking}.
- Inspect metadata, abstracts, or paper text only when needed to judge relevance.
- Return at most {limit} unique papers, best match first.
- Include only papers actually returned or inspected through Paperclip. Never
  guess a title, identifier, DOI, PMID, PMCID, source, year, or search query.

Return JSON only, with exactly this shape:
{{
  "searches": [
    {{"query": "...", "purpose": "..."}}
  ],
  "papers": [
    {{
      "paper_id": "Paperclip ID or empty string",
      "title": "exact title",
      "source": "pmc, biorxiv, medrxiv, arxiv, or abstracts_only",
      "year": "publication year or empty string",
      "doi": "DOI or empty string",
      "pmcid": "PMCID or empty string",
      "pmid": "PMID or empty string",
      "relevance_reason": "one short retrieval-focused reason"
    }}
  ]
}}"""


def _json_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if not candidate:
        raise ClaudePaperclipError("Claude returned an empty response")

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidate = fenced.group(1)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        payload = None
        for match in re.finditer(r"\{", candidate):
            try:
                payload, _ = decoder.raw_decode(candidate[match.start():])
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            raise ClaudePaperclipError(
                "Claude did not return a valid JSON object"
            )

    if not isinstance(payload, dict):
        raise ClaudePaperclipError("Claude response must be a JSON object")
    return payload


def _normalized_doi(value: Any) -> str:
    text = clean_text(str(value or ""))
    text = re.sub(
        r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return match.group(0).rstrip(".,;:)]}") if match else ""


def _normalized_pmcid(value: Any) -> str:
    match = re.search(r"\bPMC\d+\b", str(value or ""), re.IGNORECASE)
    return match.group(0).upper() if match else ""


def _paper_source(paper_id: str, supplied: Any) -> str:
    source = clean_text(str(supplied or "")).casefold()
    if source:
        return source
    if paper_id.startswith("PMC"):
        return "pmc"
    if paper_id.startswith("bio_"):
        return "biorxiv"
    if paper_id.startswith("med_"):
        return "medrxiv"
    if paper_id.startswith("arx_"):
        return "arxiv"
    return "abstracts_only"


def _papers_from_payload(
    payload: dict[str, Any],
    *,
    limit: int,
) -> list[Paper]:
    entries = payload.get("papers")
    if not isinstance(entries, list):
        raise ClaudePaperclipError("Claude JSON must contain a 'papers' list")

    papers: list[Paper] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        title = clean_text(str(entry.get("title", "")))
        doi = _normalized_doi(entry.get("doi", ""))
        pmcid = _normalized_pmcid(
            entry.get("pmcid", "") or entry.get("paper_id", "")
        )
        pmid = clean_text(str(entry.get("pmid", "")))
        paper_id = clean_text(str(entry.get("paper_id", "")))
        if pmcid:
            paper_id = pmcid
        elif not paper_id and pmid:
            paper_id = f"PMID:{pmid}"
        elif not paper_id and doi:
            paper_id = f"DOI:{doi}"

        if not title or not paper_id:
            continue

        identity_keys = {f"id:{paper_id.casefold()}"}
        if doi:
            identity_keys.add(f"doi:{doi.casefold()}")
        if pmcid:
            identity_keys.add(f"pmcid:{pmcid.casefold()}")
        if pmid:
            identity_keys.add(f"pmid:{pmid.casefold()}")
        if seen.intersection(identity_keys):
            continue
        seen.update(identity_keys)

        rank = len(papers) + 1
        papers.append(
            Paper(
                paper_id=paper_id,
                title=title,
                text=title,
                source=_paper_source(paper_id, entry.get("source", "")),
                year=clean_text(str(entry.get("year", ""))),
                doi=doi,
                retrieval_rank=rank,
                metadata={
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "claude_relevance_reason": clean_text(
                        str(entry.get("relevance_reason", ""))
                    ),
                    "agentic_retrieval": True,
                    "has_full_text": False,
                },
            )
        )
        if len(papers) >= limit:
            break
    return papers


def _searches_from_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    entries = payload.get("searches", [])
    if not isinstance(entries, list):
        return []
    searches: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        query = clean_text(str(entry.get("query", "")))
        if query:
            searches.append(
                {
                    "query": query,
                    "purpose": clean_text(str(entry.get("purpose", ""))),
                }
            )
    return searches


def _message_content(message: Any) -> list[Any]:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return content
    nested = getattr(message, "message", None)
    content = getattr(nested, "content", None)
    return content if isinstance(content, list) else []


async def _run_agent_sdk(
    prompt: str,
    *,
    model: str,
    mcp_url: str,
    paperclip_api_key: str,
    max_turns: int,
) -> ClaudeAgentRun:
    try:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ResultMessage,
            query,
        )
    except ImportError as exc:
        raise ClaudePaperclipError(
            "Claude Agent SDK is not installed. Run: "
            "python -m pip install -e \".[claude]\""
        ) from exc

    option_values: dict[str, Any] = {
        "mcp_servers": {
            "paperclip": {
                "type": "http",
                "url": mcp_url,
                "headers": {"X-API-Key": paperclip_api_key},
            }
        },
        "allowed_tools": ["mcp__paperclip__*"],
        "disallowed_tools": [
            "Bash",
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "WebSearch",
            "WebFetch",
            "Task",
        ],
        "max_turns": max_turns,
        # Auto-approve only the allowlisted Paperclip tools and deny any
        # unexpected permission request instead of pausing an unattended run.
        "permission_mode": "dontAsk",
        # Ignore MCP servers inherited from user, project, plugin, or Claude.ai
        # configuration. Paperclip above is the sole server for this test.
        "strict_mcp_config": True,
        # Explicitly ignore repository/user Claude customizations. This test is
        # controlled entirely by the prompt above and the Paperclip MCP server.
        "setting_sources": [],
    }
    if model:
        option_values["model"] = model

    response_text = ""
    tool_calls: list[dict[str, Any]] = []
    result_message: Any | None = None
    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(**option_values),
        ):
            for block in _message_content(message):
                name = getattr(block, "name", "")
                if name:
                    tool_calls.append(
                        {
                            "name": str(name),
                            "input": getattr(block, "input", {}),
                        }
                    )
            if isinstance(message, ResultMessage):
                result_message = message
                response_text = str(getattr(message, "result", "") or "")
    except Exception as exc:
        raise ClaudePaperclipError(f"Claude Agent SDK failed: {exc}") from exc

    if result_message is None or not response_text.strip():
        raise ClaudePaperclipError("Claude Agent SDK returned no final result")

    subtype = str(getattr(result_message, "subtype", "") or "")
    if bool(getattr(result_message, "is_error", False)) or (
        subtype and subtype != "success"
    ):
        raise ClaudePaperclipError(
            "Claude Agent SDK did not complete successfully "
            f"(result subtype: {subtype or 'unknown'})"
        )

    usage = getattr(result_message, "usage", {})
    return ClaudeAgentRun(
        response_text=response_text,
        tool_calls=tool_calls,
        usage=dict(usage) if isinstance(usage, dict) else {},
        total_cost_usd=getattr(result_message, "total_cost_usd", None),
        duration_ms=getattr(result_message, "duration_ms", None),
        turns=getattr(result_message, "num_turns", None),
    )


def retrieve_papers_with_claude(
    question: str,
    *,
    limit: int = 20,
    max_searches: int = 3,
    max_turns: int = 12,
    source: str = "pmc,biorxiv,medrxiv,arxiv,abstracts_only",
    ranking: str = "hybrid",
    model: str = "",
    mcp_url: str = DEFAULT_PAPERCLIP_MCP_URL,
    paperclip_api_key: str | None = None,
    runner: AgentRunner | None = None,
) -> ClaudePaperclipRetrieval:
    """Let Claude retrieve ranked Paperclip papers for a controlled experiment.

    This function is intentionally separate from ``run_pipeline`` so the tested
    default remains deterministic and unchanged.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if not 1 <= max_searches <= 10:
        raise ValueError("max_searches must be between 1 and 10")
    if max_turns <= 0:
        raise ValueError("max_turns must be greater than 0")

    prompt = build_claude_paperclip_prompt(
        question,
        limit=limit,
        max_searches=max_searches,
        source=source,
        ranking=ranking,
    )

    active_runner = runner or _run_agent_sdk
    resolved_key = paperclip_api_key or os.getenv("PAPERCLIP_API_KEY", "")
    if runner is None:
        resolved_key = validate_claude_paperclip_environment(
            paperclip_api_key=paperclip_api_key,
        )

    run = asyncio.run(
        active_runner(
            prompt,
            model=model,
            mcp_url=mcp_url,
            paperclip_api_key=resolved_key,
            max_turns=max_turns,
        )
    )
    payload = _json_object(run.response_text)
    papers = _papers_from_payload(payload, limit=limit)
    if not papers:
        raise ClaudePaperclipError(
            "Claude returned no usable Paperclip papers with IDs and titles"
        )

    return ClaudePaperclipRetrieval(
        papers=papers,
        searches=_searches_from_payload(payload),
        tool_calls=run.tool_calls,
        raw_response=run.response_text,
        usage=run.usage,
        total_cost_usd=run.total_cost_usd,
        duration_ms=run.duration_ms,
        turns=run.turns,
    )
