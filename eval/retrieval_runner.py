from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from eval.benchmark_loader import BenchmarkExample, load_benchmark
from web_rag.context_builder import build_evidence_pack
from web_rag.query_builder import build_europe_pmc_query
from web_rag.ranker import rank_snippets
from web_rag.serializer import evidence_pack_to_dict
from web_rag.snippet_extractor import extract_snippets
from web_rag.source_client import search_europe_pmc


RETRIEVAL_FIELDS = [
    "question_id",
    "question",
    "gold_answer",
    "gold_nuggets_json",
    "notes",
    "run_name",
    "ranker",
    "search_query",
    "keywords_json",
    "retrieved_papers_count",
    "extracted_snippets_count",
    "returned_records_count",
    "top_k",
    "context_text",
    "records_json",
    "status",
    "error_type",
    "error_message",
]


@dataclass(frozen=True)
class RetrievalConfig:
    benchmark_path: str
    output_root: str
    run_name: str
    ranker: str
    page_size: int
    top_k: int
    snippet_window_size: int
    snippet_stride: int
    min_snippet_word_count: int
    bm25_k1: float
    bm25_b: float
    medcpt_batch_size: int
    medcpt_device: str | None
    medcpt_query_model: str
    medcpt_article_model: str
    hybrid_lexical_weight: float
    hybrid_medcpt_weight: float
    limit: int | None = None


def default_run_name(ranker: str) -> str:
    return f"ranker_{ranker.strip().lower().replace('-', '_')}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _search_query_from(query: Any) -> str:
    if hasattr(query, "search_query"):
        return str(getattr(query, "search_query") or "")
    if isinstance(query, dict):
        return str(query.get("search_query") or query.get("query") or "")
    return str(query or "")


def _keywords_from(query: Any) -> list[str]:
    if hasattr(query, "keywords"):
        keywords = getattr(query, "keywords") or []
    elif isinstance(query, dict):
        keywords = query.get("keywords") or []
    else:
        keywords = []
    if isinstance(keywords, str):
        return [keywords]
    return [str(item) for item in keywords]


def _query_object(question: str, query: Any) -> Any:
    if hasattr(query, "search_query"):
        return query
    return SimpleNamespace(
        original_question=question,
        normalized_question=question,
        search_query=_search_query_from(query),
        keywords=_keywords_from(query),
    )


def _search_papers(question: str, query: Any, page_size: int) -> list[Any]:
    query_object = _query_object(question, query)
    search_query = _search_query_from(query_object)

    attempts = [
        lambda: search_europe_pmc(search_query, page_size=page_size),
        lambda: search_europe_pmc(query_object, page_size=page_size),
        lambda: search_europe_pmc(query=search_query, page_size=page_size),
        lambda: search_europe_pmc(search_query),
    ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return list(attempt())
        except (TypeError, AttributeError) as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return []


def _extract_snippets(papers: Iterable[Any], window_size: int, stride: int) -> list[Any]:
    attempts = [
        lambda: extract_snippets(papers, window_size=window_size, stride=stride),
        lambda: extract_snippets(papers=papers, window_size=window_size, stride=stride),
        lambda: extract_snippets(papers, window=window_size, stride=stride),
        lambda: extract_snippets(papers),
    ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return list(attempt())
        except TypeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return []


def _rank(question: str, snippets: list[Any], config: RetrievalConfig) -> list[Any]:
    attempts = [
        lambda: rank_snippets(
            question=question,
            snippets=snippets,
            ranker=config.ranker,
            top_k=config.top_k,
            min_snippet_word_count=config.min_snippet_word_count,
            bm25_k1=config.bm25_k1,
            bm25_b=config.bm25_b,
            lexical_weight=config.hybrid_lexical_weight,
            medcpt_weight=config.hybrid_medcpt_weight,
            query_model_name=config.medcpt_query_model,
            article_model_name=config.medcpt_article_model,
            batch_size=config.medcpt_batch_size,
            device=config.medcpt_device,
        ),
        lambda: rank_snippets(
            question=question,
            snippets=snippets,
            ranker=config.ranker,
            top_k=config.top_k,
        ),
        lambda: rank_snippets(question, snippets, config.ranker, config.top_k),
    ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return list(attempt())
        except TypeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return []


def _build_pack(question: str, query: Any, ranked_snippets: list[Any]) -> Any:
    query_object = _query_object(question, query)
    attempts = [
        lambda: build_evidence_pack(question=question, query=query_object, ranked_snippets=ranked_snippets),
        lambda: build_evidence_pack(question=question, ranked_snippets=ranked_snippets),
        lambda: build_evidence_pack(question, ranked_snippets),
    ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except (TypeError, AttributeError) as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not build evidence pack.")


def _extract_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records") or payload.get("evidence") or []
    return records if isinstance(records, list) else []


def run_question(example: BenchmarkExample, config: RetrievalConfig) -> dict[str, Any]:
    query = build_europe_pmc_query(example.question)
    query_object = _query_object(example.question, query)
    papers = _search_papers(example.question, query_object, config.page_size)
    snippets = _extract_snippets(papers, config.snippet_window_size, config.snippet_stride)
    ranked_snippets = _rank(example.question, snippets, config)

    evidence_pack = _build_pack(example.question, query_object, ranked_snippets)
    payload = evidence_pack_to_dict(evidence_pack)
    records = _extract_records(payload)

    return {
        "question_id": example.question_id,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "gold_nuggets_json": example.gold_nuggets,
        "notes": example.notes,
        "run_name": config.run_name,
        "ranker": config.ranker,
        "search_query": _search_query_from(query_object),
        "keywords_json": _keywords_from(query_object),
        "retrieved_papers_count": len(papers),
        "extracted_snippets_count": len(snippets),
        "returned_records_count": len(records),
        "top_k": config.top_k,
        "context_text": payload.get("context_text", ""),
        "records_json": records,
        "status": "ok",
        "error_type": "",
        "error_message": "",
    }


def _error_row(example: BenchmarkExample, config: RetrievalConfig, error: Exception) -> dict[str, Any]:
    return {
        "question_id": example.question_id,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "gold_nuggets_json": example.gold_nuggets,
        "notes": example.notes,
        "run_name": config.run_name,
        "ranker": config.ranker,
        "search_query": "",
        "keywords_json": [],
        "retrieved_papers_count": 0,
        "extracted_snippets_count": 0,
        "returned_records_count": 0,
        "top_k": config.top_k,
        "context_text": "",
        "records_json": [],
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RETRIEVAL_FIELDS)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field, "") for field in RETRIEVAL_FIELDS}
            output["gold_nuggets_json"] = _json(output["gold_nuggets_json"])
            output["keywords_json"] = _json(output["keywords_json"])
            output["records_json"] = _json(output["records_json"])
            writer.writerow(output)


def run_retrieval(config: RetrievalConfig) -> Path:
    examples = load_benchmark(config.benchmark_path)
    if config.limit is not None:
        examples = examples[: config.limit]

    run_dir = Path(config.output_root) / config.run_name
    output_path = run_dir / "retrieval_results.csv"
    rows: list[dict[str, Any]] = []

    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example.question_id}: {example.question}")
        try:
            rows.append(run_question(example, config))
        except Exception as error:
            rows.append(_error_row(example, config, error))

    _write_rows(output_path, rows)
    return output_path


def build_config_from_args(args: Any, run_name: str) -> RetrievalConfig:
    return RetrievalConfig(
        benchmark_path=str(args.benchmark),
        output_root=str(args.output_root),
        run_name=run_name,
        ranker=args.ranker,
        page_size=args.page_size,
        top_k=args.top_k,
        snippet_window_size=args.snippet_window_size,
        snippet_stride=args.snippet_stride,
        min_snippet_word_count=args.min_snippet_word_count,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        medcpt_batch_size=args.medcpt_batch_size,
        medcpt_device=args.medcpt_device,
        medcpt_query_model=args.medcpt_query_model,
        medcpt_article_model=args.medcpt_article_model,
        hybrid_lexical_weight=args.hybrid_lexical_weight,
        hybrid_medcpt_weight=args.hybrid_medcpt_weight,
        limit=args.limit,
    )
