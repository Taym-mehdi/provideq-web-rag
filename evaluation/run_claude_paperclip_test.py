from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from web_rag.claude_paperclip_retriever import (
    ClaudePaperclipError,
    retrieve_papers_with_claude,
    validate_claude_paperclip_environment,
)
from web_rag.config import get_settings
from web_rag.paperclip_retriever import retrieve_papers
from web_rag.serializer import save_json

from .retrieval_evaluation import evaluate_retrieval_mrr_detailed
from .run_evaluation import load_benchmark


DEFAULT_BENCHMARK = Path("benchmark/provideq_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("outputs/claude_paperclip_smoke")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _select_question(
    examples: list[dict[str, Any]],
    question_id: str,
) -> dict[str, Any]:
    wanted = question_id.strip().casefold()
    for example in examples:
        if str(example.get("id", "")).strip().casefold() == wanted:
            return example
    raise ValueError(f"Question ID not found in benchmark: {question_id}")


def _metrics(gold_documents: list[dict[str, Any]], papers: list[Any]) -> dict[str, Any]:
    result = evaluate_retrieval_mrr_detailed(gold_documents, papers)
    rank = result.rank
    return {
        "first_relevant_rank": rank,
        "reciprocal_rank": float(result.score),
        "hit_at_1": bool(rank is not None and rank <= 1),
        "hit_at_3": bool(rank is not None and rank <= 3),
        "hit_at_5": bool(rank is not None and rank <= 5),
        "hit_at_10": bool(rank is not None and rank <= 10),
        "hit_at_20": bool(rank is not None and rank <= 20),
        "matched_document": result.matched_title,
        "match_type": result.match_type,
        "matched_identifier": result.matched_identifier,
    }


def _paper_dict(paper: Any) -> dict[str, Any]:
    metadata = getattr(paper, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "rank": int(getattr(paper, "retrieval_rank", 0) or 0),
        "paper_id": _clean(getattr(paper, "paper_id", "")),
        "title": _clean(getattr(paper, "title", "")),
        "source": _clean(getattr(paper, "source", "")),
        "year": _clean(getattr(paper, "year", "")),
        "doi": _clean(getattr(paper, "doi", "")),
        "pmcid": _clean(metadata.get("pmcid", "")),
        "pmid": _clean(metadata.get("pmid", "")),
        "relevance_reason": _clean(
            metadata.get("claude_relevance_reason", "")
        ),
    }


def _comparison_label(baseline_mrr: float, claude_mrr: float) -> str:
    if claude_mrr > baseline_mrr:
        return "improved"
    if claude_mrr < baseline_mrr:
        return "worse"
    return "unchanged"


def _write_papers_csv(
    path: Path,
    *,
    gold_documents: list[dict[str, Any]],
    baseline_papers: list[Any],
    claude_papers: list[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "system",
        "rank",
        "paper_id",
        "title",
        "source",
        "year",
        "doi",
        "pmcid",
        "pmid",
        "relevance_reason",
        "matches_gold",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for system, papers in (
            ("baseline", baseline_papers),
            ("claude", claude_papers),
        ):
            for paper in papers:
                row = _paper_dict(paper)
                match = evaluate_retrieval_mrr_detailed(
                    gold_documents,
                    [paper],
                )
                writer.writerow(
                    {
                        "system": system,
                        **row,
                        "matches_gold": int(match.rank is not None),
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed raw Paperclip retrieval with Claude-controlled "
            "Paperclip retrieval for one benchmark question."
        )
    )
    parser.add_argument("--question-id", default="Q003")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument(
        "--paperclip-source",
        default=settings.paperclip_source,
    )
    parser.add_argument(
        "--paperclip-ranking",
        choices=("bm25", "vector", "hybrid"),
        default=settings.paperclip_ranking,
    )
    parser.add_argument(
        "--paperclip-timeout",
        type=float,
        default=settings.paperclip_timeout,
    )
    parser.add_argument(
        "--claude-model",
        default=os.getenv("WEB_RAG_CLAUDE_MODEL", ""),
        help="Optional. Empty uses the Claude Agent SDK default model.",
    )
    parser.add_argument(
        "--paperclip-mcp-url",
        default=os.getenv(
            "WEB_RAG_PAPERCLIP_MCP_URL",
            "https://paperclip.gxl.ai/mcp",
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Fail before the fixed baseline makes a Paperclip request if the optional
    # Claude side is not configured.
    validate_claude_paperclip_environment()
    example = _select_question(
        load_benchmark(args.benchmark),
        args.question_id,
    )
    question = _clean(example["question"])
    gold_documents = [
        document
        for document in example["gold_documents"]
        if isinstance(document, dict)
    ]

    print(f"Question: {example['id']} - {question}")
    print("Running fixed baseline: raw question + Paperclip hybrid...")
    baseline = retrieve_papers(
        question,
        limit=args.limit,
        source=args.paperclip_source,
        ranking=args.paperclip_ranking,
        full_corpus=True,
        load_full_text=False,
        timeout=args.paperclip_timeout,
    )

    print("Running Claude-controlled Paperclip retrieval...")
    claude = retrieve_papers_with_claude(
        question,
        limit=args.limit,
        max_searches=args.max_searches,
        max_turns=args.max_turns,
        source=args.paperclip_source,
        ranking=args.paperclip_ranking,
        model=args.claude_model,
        mcp_url=args.paperclip_mcp_url,
    )

    baseline_metrics = _metrics(gold_documents, baseline.papers)
    claude_metrics = _metrics(gold_documents, claude.papers)
    outcome = _comparison_label(
        baseline_metrics["reciprocal_rank"],
        claude_metrics["reciprocal_rank"],
    )

    question_output = args.output_dir / str(example["id"])
    payload = {
        "question_id": example["id"],
        "category": example.get("category", ""),
        "question": question,
        "gold_documents": gold_documents,
        "settings": {
            "retrieval_limit": args.limit,
            "paperclip_source": args.paperclip_source,
            "paperclip_ranking": args.paperclip_ranking,
            "claude_model": args.claude_model or "SDK default",
            "claude_max_searches": args.max_searches,
            "claude_max_turns": args.max_turns,
            "paperclip_mcp_url": args.paperclip_mcp_url,
        },
        "baseline": {
            "method": "raw question + Paperclip",
            "paperclip_result_id": baseline.result_id,
            "metrics": baseline_metrics,
            "papers": [_paper_dict(paper) for paper in baseline.papers],
        },
        "claude": {
            "method": "Claude-controlled Paperclip MCP",
            "metrics": claude_metrics,
            "searches": claude.searches,
            "tool_calls": claude.tool_calls,
            "usage": claude.usage,
            "total_cost_usd": claude.total_cost_usd,
            "duration_ms": claude.duration_ms,
            "turns": claude.turns,
            "papers": [_paper_dict(paper) for paper in claude.papers],
            "raw_response": claude.raw_response,
        },
        "comparison": {
            "outcome_for_this_question": outcome,
            "baseline_first_relevant_rank": baseline_metrics[
                "first_relevant_rank"
            ],
            "claude_first_relevant_rank": claude_metrics[
                "first_relevant_rank"
            ],
            "reciprocal_rank_change": (
                claude_metrics["reciprocal_rank"]
                - baseline_metrics["reciprocal_rank"]
            ),
        },
    }
    json_path = save_json(payload, question_output / "comparison.json")
    csv_path = question_output / "papers.csv"
    _write_papers_csv(
        csv_path,
        gold_documents=gold_documents,
        baseline_papers=baseline.papers,
        claude_papers=claude.papers,
    )

    print()
    print(
        "Baseline first relevant rank:",
        baseline_metrics["first_relevant_rank"] or "not found",
    )
    print(
        "Claude first relevant rank:",
        claude_metrics["first_relevant_rank"] or "not found",
    )
    print(f"Result for this question: {outcome}")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except (ClaudePaperclipError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
