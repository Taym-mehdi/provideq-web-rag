from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.benchmark_loader import BenchmarkExample
from web_rag.config import get_settings
from web_rag.context_builder import build_evidence_pack
from web_rag.query_builder import build_europe_pmc_query
from web_rag.ranker import rank_snippets
from web_rag.serializer import evidence_pack_to_dict
from web_rag.snippet_extractor import extract_snippets
from web_rag.source_client import search_europe_pmc


@dataclass
class RetrievalRunConfig:
    """
    Configuration for one evaluation retrieval run.

    This does not define evaluation metrics yet.
    It only controls how the Web RAG pipeline is executed over a benchmark.
    """

    benchmark_path: str
    output_dir: str
    run_name: str
    ranking_method: str
    page_size: int
    top_k: int
    window_size: int
    limit: int | None = None


@dataclass
class RetrievalRunFiles:
    """
    Output file paths created by one retrieval run.
    """

    run_dir: Path
    raw_jsonl_path: Path
    evidence_csv_path: Path
    summary_csv_path: Path


def make_timestamp() -> str:
    """
    Return a filesystem-safe timestamp.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_files(config: RetrievalRunConfig) -> RetrievalRunFiles:
    """
    Create the run output directory and return the file paths used by the run.
    """
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return RetrievalRunFiles(
        run_dir=run_dir,
        raw_jsonl_path=run_dir / "raw_results.jsonl",
        evidence_csv_path=run_dir / "evidence_records.csv",
        summary_csv_path=run_dir / "question_summary.csv",
    )


def run_pipeline_for_question(
    question: str,
    ranking_method: str,
    page_size: int,
    top_k: int,
    window_size: int,
) -> dict[str, Any]:
    """
    Run the current Web RAG pipeline for one question.

    Output:
    - JSON-ready dictionary
    - includes query, counts, ranked evidence, context_text, and score components

    This function intentionally mirrors the CLI pipeline so that evaluation
    measures exactly the system behavior we are developing.
    """
    query = build_europe_pmc_query(question)

    papers = search_europe_pmc(
        query=query,
        page_size=page_size,
    )

    snippets = extract_snippets(
        papers=papers,
        window_size=window_size,
    )

    ranked_snippets = rank_snippets(
        question=query.original_question,
        snippets=snippets,
        top_k=top_k,
        method=ranking_method,
    )

    evidence_pack = build_evidence_pack(
        question=query.original_question,
        ranked_snippets=ranked_snippets,
    )

    result_payload = evidence_pack_to_dict(
        evidence_pack=evidence_pack,
        query=query,
        retrieved_papers_count=len(papers),
        extracted_snippets_count=len(snippets),
        ranking_method=ranking_method,
    )

    return result_payload


def build_raw_result_record(
    example: BenchmarkExample,
    result_payload: dict[str, Any],
    config: RetrievalRunConfig,
) -> dict[str, Any]:
    """
    Combine benchmark data and Web RAG output into one raw JSONL record.
    """
    return {
        "question_id": example.question_id,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "gold_nuggets": example.gold_nuggets,
        "notes": example.notes,
        "run_config": {
            "ranking_method": config.ranking_method,
            "page_size": config.page_size,
            "top_k": config.top_k,
            "window_size": config.window_size,
        },
        "result": result_payload,
    }


def build_error_record(
    example: BenchmarkExample,
    error: Exception,
    config: RetrievalRunConfig,
) -> dict[str, Any]:
    """
    Build a JSONL record for failed questions.

    We keep failures in the output instead of silently skipping them. This is
    important for reproducible evaluation and error analysis.
    """
    return {
        "question_id": example.question_id,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "gold_nuggets": example.gold_nuggets,
        "notes": example.notes,
        "run_config": {
            "ranking_method": config.ranking_method,
            "page_size": config.page_size,
            "top_k": config.top_k,
            "window_size": config.window_size,
        },
        "result": {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "evidence": [],
            "context_text": "",
        },
    }


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """
    Append one JSON object as a JSONL line.
    """
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_evidence_csv(path: Path) -> None:
    """
    Create the evidence CSV header.
    """
    fieldnames = [
        "question_id",
        "question",
        "ranking_method",
        "rank",
        "score",
        "title",
        "evidence_text",
        "year",
        "journal",
        "doi",
        "source",
        "ext_id",
        "url",
        "score_components_json",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def append_evidence_rows(
    path: Path,
    example: BenchmarkExample,
    result_payload: dict[str, Any],
) -> None:
    """
    Append one CSV row per evidence record.
    """
    fieldnames = [
        "question_id",
        "question",
        "ranking_method",
        "rank",
        "score",
        "title",
        "evidence_text",
        "year",
        "journal",
        "doi",
        "source",
        "ext_id",
        "url",
        "score_components_json",
    ]

    ranking_method = result_payload.get("ranking_method", "")

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        for record in result_payload.get("evidence", []):
            metadata = record.get("metadata", {})

            writer.writerow(
                {
                    "question_id": example.question_id,
                    "question": example.question,
                    "ranking_method": ranking_method,
                    "rank": record.get("citation_id", ""),
                    "score": record.get("score", ""),
                    "title": record.get("title", ""),
                    "evidence_text": record.get("evidence_text", ""),
                    "year": metadata.get("year", ""),
                    "journal": metadata.get("journal", ""),
                    "doi": metadata.get("doi", ""),
                    "source": metadata.get("source", ""),
                    "ext_id": metadata.get("ext_id", ""),
                    "url": metadata.get("url", ""),
                    "score_components_json": json.dumps(
                        record.get("score_components", {}),
                        ensure_ascii=False,
                    ),
                }
            )


def prepare_summary_csv(path: Path) -> None:
    """
    Create the per-question summary CSV header.
    """
    fieldnames = [
        "question_id",
        "question",
        "status",
        "ranking_method",
        "retrieved_papers",
        "extracted_snippets",
        "ranked_evidence_records",
        "top_score",
        "top_title",
        "error_type",
        "error_message",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def append_summary_row(
    path: Path,
    example: BenchmarkExample,
    result_payload: dict[str, Any],
) -> None:
    """
    Append one summary row for one benchmark question.
    """
    fieldnames = [
        "question_id",
        "question",
        "status",
        "ranking_method",
        "retrieved_papers",
        "extracted_snippets",
        "ranked_evidence_records",
        "top_score",
        "top_title",
        "error_type",
        "error_message",
    ]

    counts = result_payload.get("counts", {})
    evidence = result_payload.get("evidence", [])

    if evidence:
        top_score = evidence[0].get("score", "")
        top_title = evidence[0].get("title", "")
    else:
        top_score = ""
        top_title = ""

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writerow(
            {
                "question_id": example.question_id,
                "question": example.question,
                "status": result_payload.get("status", ""),
                "ranking_method": result_payload.get("ranking_method", ""),
                "retrieved_papers": counts.get("retrieved_papers", ""),
                "extracted_snippets": counts.get("extracted_snippets", ""),
                "ranked_evidence_records": counts.get("ranked_evidence_records", ""),
                "top_score": top_score,
                "top_title": top_title,
                "error_type": result_payload.get("error_type", ""),
                "error_message": result_payload.get("error_message", ""),
            }
        )


def run_retrieval_for_benchmark(
    examples: list[BenchmarkExample],
    config: RetrievalRunConfig,
) -> RetrievalRunFiles:
    """
    Run the Web RAG pipeline over a benchmark and save raw outputs.

    Created files:
    - raw_results.jsonl:
        full benchmark + retrieval output, one JSON object per question

    - evidence_records.csv:
        one row per retrieved evidence record

    - question_summary.csv:
        one row per benchmark question, useful for quick checking
    """
    files = create_run_files(config)

    if files.raw_jsonl_path.exists():
        files.raw_jsonl_path.unlink()

    prepare_evidence_csv(files.evidence_csv_path)
    prepare_summary_csv(files.summary_csv_path)

    selected_examples = examples

    if config.limit is not None:
        selected_examples = examples[: config.limit]

    total = len(selected_examples)

    for index, example in enumerate(selected_examples, start=1):
        print(
            f"[{index}/{total}] Running {example.question_id}: "
            f"{example.question}"
        )

        try:
            result_payload = run_pipeline_for_question(
                question=example.question,
                ranking_method=config.ranking_method,
                page_size=config.page_size,
                top_k=config.top_k,
                window_size=config.window_size,
            )

            raw_record = build_raw_result_record(
                example=example,
                result_payload=result_payload,
                config=config,
            )

        except Exception as error:
            raw_record = build_error_record(
                example=example,
                error=error,
                config=config,
            )
            result_payload = raw_record["result"]

        write_jsonl_record(files.raw_jsonl_path, raw_record)
        append_evidence_rows(files.evidence_csv_path, example, result_payload)
        append_summary_row(files.summary_csv_path, example, result_payload)

    return files


def build_default_run_config(
    benchmark_path: str,
    output_dir: str,
    run_name: str | None,
    ranking_method: str,
    page_size: int | None,
    top_k: int | None,
    window_size: int | None,
    limit: int | None,
) -> RetrievalRunConfig:
    """
    Build a retrieval run config using project defaults where needed.
    """
    settings = get_settings()

    effective_run_name = run_name or (
        f"{make_timestamp()}_{ranking_method}_k{top_k or settings.default_top_k}"
    )

    return RetrievalRunConfig(
        benchmark_path=benchmark_path,
        output_dir=output_dir,
        run_name=effective_run_name,
        ranking_method=ranking_method,
        page_size=page_size or settings.default_page_size,
        top_k=top_k or settings.default_top_k,
        window_size=window_size or settings.snippet_window,
        limit=limit,
    )