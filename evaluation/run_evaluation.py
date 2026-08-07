from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from web_rag.config import get_settings
from web_rag.paperclip_retriever import retrieve_papers
from web_rag.query_reformulation import (
    QUERY_PROMPT_VERSION,
    make_llm_generator,
    reformulate_query,
)

from .retrieval_evaluation import RetrievalMRRResult, evaluate_retrieval_mrr_detailed


DEFAULT_BENCHMARK = Path("benchmark/provideq_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_SOURCE = "pmc,biorxiv,medrxiv,arxiv,abstracts_only"
DEFAULT_EMPTY_RESULT_RETRIES = 2
QUERY_STRATEGIES = ("raw", "hyde", "llmexpand")
PAPERCLIP_RANKINGS = ("bm25", "vector", "hybrid")


@dataclass(frozen=True)
class RetrievalTrace:
    query: Any
    papers: list[Any]
    result_id: str = ""


class RetrievalRunError(RuntimeError):
    def __init__(self, message: str, *, query: Any | None = None) -> None:
        super().__init__(message)
        self.query = query


RESULT_FIELDS = (
    "question_id",
    "category",
    "question",
    "status",
    "error",
    "query_strategy",
    "llm_model",
    "llm_provider",
    "llm_base_url",
    "paperclip_ranking",
    "paperclip_source",
    "paperclip_full_corpus",
    "retrieval_limit",
    "query_sent",
    "expansion_terms",
    "retrieved_count",
    "retrieved_documents",
    "retrieved_document_ids",
    "gold_documents",
    "gold_document_ids",
    "first_relevant_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "reciprocal_rank",
    "matched_document",
    "match_type",
    "matched_identifier",
    "paperclip_result_id",
)


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    examples = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(examples, list):
        raise ValueError("Benchmark must contain an 'examples' list")

    required = {
        "id",
        "question",
        "gold_answers",
        "gold_documents",
        "category",
        "answerable",
    }
    seen_ids: set[str] = set()
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict) or not required.issubset(example):
            raise ValueError(f"Invalid benchmark item at position {index}")

        question_id = str(example["id"]).strip()
        if not question_id or question_id in seen_ids:
            raise ValueError(f"Missing or duplicate benchmark id at position {index}")
        seen_ids.add(question_id)

        if not str(example["question"]).strip():
            raise ValueError(f"Question {question_id} is empty")
        if not example["gold_documents"]:
            raise ValueError(f"Question {question_id} has no gold document")

    return examples


def select_questions(
    examples: list[dict[str, Any]],
    number: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if not 1 <= number <= len(examples):
        raise ValueError(
            f"num_questions must be between 1 and {len(examples)} for this benchmark"
        )

    if number == len(examples):
        selected = list(examples)
    else:
        selected = random.Random(seed).sample(examples, number)
    return sorted(selected, key=lambda example: str(example["id"]))


def _cached_generator(
    generator: Callable[[str], str],
    cache_file: str,
    *,
    provider: str,
    model: str,
    base_url: str,
    strategy: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Callable[[str], str]:
    path = Path(cache_file)
    expected_metadata: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "base_url": base_url.rstrip("/"),
        "strategy": strategy,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "prompt_version": QUERY_PROMPT_VERSION,
    }
    payload: dict[str, Any] = {**expected_metadata, "responses": {}}

    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid LLM cache: {path}")
        for key, expected in expected_metadata.items():
            if loaded.get(key) != expected:
                raise ValueError(
                    f"LLM cache {path} was created with a different {key}. "
                    "Use another cache path or delete the old cache."
                )
        payload = loaded

    responses = payload.setdefault("responses", {})
    if not isinstance(responses, dict):
        raise ValueError(f"Invalid LLM cache responses: {path}")

    def generate(prompt: str) -> str:
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = responses.get(cache_key)
        if isinstance(cached, dict) and cached.get("prompt") == prompt:
            response = str(cached.get("response", ""))
            if response:
                return response

        response = generator(prompt)
        responses[cache_key] = {"prompt": prompt, "response": response}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return response

    return generate


def _safe_cache_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_"
        for character in value.strip().casefold()
    )
    return "_".join(part for part in cleaned.split("_") if part) or "model"


def _prepare_llm_generator(args: argparse.Namespace) -> Callable[[str], str] | None:
    if args.query_strategy == "raw":
        return None
    if not args.llm_cache:
        model_name = _safe_cache_component(args.llm_model)
        prompt_version = _safe_cache_component(QUERY_PROMPT_VERSION)
        args.llm_cache = str(
            Path(args.output_dir)
            / "query_cache"
            / f"{args.query_strategy}_{model_name}_{prompt_version}.json"
        )

    provider = str(args.llm_provider).strip().casefold()
    api_key = None
    if provider == "openai":
        api_key = os.getenv(args.llm_api_key_env, "")
        if not api_key:
            raise ValueError(f"Environment variable {args.llm_api_key_env} is not set")

    max_tokens = (
        args.hyde_max_tokens
        if args.query_strategy == "hyde"
        else args.expansion_max_tokens
    )
    generator = make_llm_generator(
        provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=api_key,
        temperature=args.llm_temperature,
        max_tokens=max_tokens,
        seed=args.seed,
        timeout=args.llm_timeout,
        json_output=args.query_strategy == "llmexpand",
    )
    return _cached_generator(
        generator,
        args.llm_cache,
        provider=provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        strategy=args.query_strategy,
        temperature=args.llm_temperature,
        max_tokens=max_tokens,
        seed=args.seed,
    )


def _retrieve(
    question: str,
    args: argparse.Namespace,
    generator: Callable[[str], str] | None,
) -> RetrievalTrace:
    query = reformulate_query(
        question,
        args.query_strategy,
        hyde_model=args.llm_model,
        hyde_base_url=args.llm_base_url,
        hyde_temperature=args.llm_temperature,
        hyde_max_tokens=args.hyde_max_tokens,
        hyde_timeout=args.llm_timeout,
        hyde_generator=generator if args.query_strategy == "hyde" else None,
        expansion_model=args.llm_model,
        expansion_base_url=args.llm_base_url,
        expansion_temperature=args.llm_temperature,
        expansion_max_tokens=args.expansion_max_tokens,
        expansion_timeout=args.llm_timeout,
        expansion_max_terms=args.expansion_max_terms,
        expansion_max_query_chars=args.expansion_max_query_chars,
        expansion_generator=(
            generator if args.query_strategy == "llmexpand" else None
        ),
    )
    retrieval = None
    papers: list[Any] = []
    total_attempts = args.empty_result_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            retrieval = retrieve_papers(
                query.search_query,
                limit=args.retrieval_limit,
                source=args.paperclip_source,
                ranking=args.paperclip_ranking,
                full_corpus=args.paperclip_full_corpus,
                load_full_text=False,
                timeout=args.paperclip_timeout,
            )
        except Exception as exc:
            raise RetrievalRunError(str(exc), query=query) from exc

        papers = list(getattr(retrieval, "papers", []) or [])
        if papers or attempt == total_attempts:
            break

        retry_number = attempt
        delay_seconds = min(2 ** (attempt - 1), 4)
        print(
            "  Paperclip returned 0 papers; "
            f"empty-result retry {retry_number}/{args.empty_result_retries} "
            f"in {delay_seconds}s"
        )
        time.sleep(delay_seconds)

    return RetrievalTrace(
        query=query,
        papers=papers,
        result_id=str(getattr(retrieval, "result_id", "") or ""),
    )


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _paper_value(paper: Any, field: str, default: Any = "") -> Any:
    if isinstance(paper, dict):
        return paper.get(field, default)
    return getattr(paper, field, default)


def _paper_identifier(paper: Any) -> str:
    values: list[str] = []
    paper_id = _clean(_paper_value(paper, "paper_id", ""))
    doi = _clean(_paper_value(paper, "doi", ""))
    if paper_id:
        values.append(paper_id)
    if doi and doi.casefold() not in {value.casefold() for value in values}:
        values.append(f"doi:{doi}")
    return "; ".join(values)


def _gold_identifier(document: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("pmcid", "pmid", "doi"):
        value = _clean(document.get(key, ""))
        if value:
            values.append(f"{key}:{value}")
    return "; ".join(values)


def _result_row(
    example: dict[str, Any],
    trace: RetrievalTrace,
    result: RetrievalMRRResult,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ordered = sorted(
        trace.papers,
        key=lambda paper: int(_paper_value(paper, "retrieval_rank", 0) or 0),
    )

    retrieved_titles: list[str] = []
    retrieved_ids: list[str] = []
    for fallback_rank, paper in enumerate(ordered, start=1):
        rank = int(_paper_value(paper, "retrieval_rank", fallback_rank) or fallback_rank)
        title = _clean(_paper_value(paper, "title", ""))
        identifier = _paper_identifier(paper)
        if title:
            retrieved_titles.append(f"{rank}: {title}")
        if identifier:
            retrieved_ids.append(f"{rank}: {identifier}")

    gold_documents = [
        document
        for document in example["gold_documents"]
        if isinstance(document, dict)
    ]
    gold_titles = [
        _clean(document.get("title", ""))
        for document in gold_documents
        if _clean(document.get("title", ""))
    ]
    gold_ids = [
        identifier
        for document in gold_documents
        if (identifier := _gold_identifier(document))
    ]

    rank = result.rank
    return {
        "question_id": _clean(example["id"]),
        "category": _clean(example.get("category", "")),
        "question": _clean(example["question"]),
        "status": "success",
        "error": "",
        "query_strategy": args.query_strategy,
        "llm_model": args.llm_model if args.query_strategy != "raw" else "",
        "llm_provider": (
            args.llm_provider if args.query_strategy != "raw" else ""
        ),
        "llm_base_url": (
            args.llm_base_url if args.query_strategy != "raw" else ""
        ),
        "paperclip_ranking": args.paperclip_ranking,
        "paperclip_source": args.paperclip_source,
        "paperclip_full_corpus": int(args.paperclip_full_corpus),
        "retrieval_limit": args.retrieval_limit,
        "query_sent": _clean(trace.query.search_query),
        "expansion_terms": " | ".join(
            _clean(term) for term in getattr(trace.query, "expanded_terms", [])
        ),
        "retrieved_count": len(ordered),
        "retrieved_documents": " | ".join(retrieved_titles),
        "retrieved_document_ids": " | ".join(retrieved_ids),
        "gold_documents": " | ".join(gold_titles),
        "gold_document_ids": " | ".join(gold_ids),
        "first_relevant_rank": "" if rank is None else rank,
        "hit_at_1": int(rank is not None and rank <= 1),
        "hit_at_3": int(rank is not None and rank <= 3),
        "hit_at_5": int(rank is not None and rank <= 5),
        "hit_at_10": int(rank is not None and rank <= 10),
        "reciprocal_rank": round(float(result.score), 6),
        "matched_document": _clean(result.matched_title),
        "match_type": _clean(result.match_type),
        "matched_identifier": _clean(result.matched_identifier),
        "paperclip_result_id": _clean(trace.result_id),
    }


def _output_file(args: argparse.Namespace) -> Path:
    folder = (
        Path(args.output_dir)
        / f"{args.query_strategy}_{args.paperclip_ranking}_retrieval_mrr"
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "results.csv"


def _write_results(rows: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(RESULT_FIELDS))
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in RESULT_FIELDS}
            for row in rows
        )
    temporary.replace(output_file)


def _load_existing_rows(
    output_file: Path,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if not output_file.exists() or not args.resume:
        return {}

    with output_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [dict(row) for row in csv.DictReader(file)]
    if not rows:
        return {}

    expected = {
        "query_strategy": str(args.query_strategy),
        "paperclip_ranking": str(args.paperclip_ranking),
        "paperclip_source": str(args.paperclip_source),
        "paperclip_full_corpus": str(int(args.paperclip_full_corpus)),
        "retrieval_limit": str(args.retrieval_limit),
        "llm_model": args.llm_model if args.query_strategy != "raw" else "",
        "llm_provider": (
            args.llm_provider if args.query_strategy != "raw" else ""
        ),
        "llm_base_url": (
            args.llm_base_url if args.query_strategy != "raw" else ""
        ),
    }
    first = rows[0]
    mismatched = [
        key for key, value in expected.items()
        if str(first.get(key, "")) != value
    ]
    if mismatched:
        raise ValueError(
            f"Existing output {output_file} uses different settings: "
            f"{', '.join(mismatched)}. Delete it or run with --no-resume."
        )

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = _clean(row.get("question_id", ""))
        if not question_id:
            continue
        if not row.get("status"):
            row["status"] = "error" if row.get("error") else "success"
        result[question_id] = row
    return result


def _gold_values(example: dict[str, Any]) -> tuple[str, str]:
    gold_documents = [
        document for document in example["gold_documents"]
        if isinstance(document, dict)
    ]
    titles = " | ".join(
        title for document in gold_documents
        if (title := _clean(document.get("title", "")))
    )
    identifiers = " | ".join(
        identifier for document in gold_documents
        if (identifier := _gold_identifier(document))
    )
    return titles, identifiers


def _error_row(
    example: dict[str, Any],
    args: argparse.Namespace,
    error: Exception,
) -> dict[str, Any]:
    query = getattr(error, "query", None)
    gold_titles, gold_ids = _gold_values(example)
    return {
        "question_id": _clean(example["id"]),
        "category": _clean(example.get("category", "")),
        "question": _clean(example["question"]),
        "status": "error",
        "error": _clean(error),
        "query_strategy": args.query_strategy,
        "llm_model": args.llm_model if args.query_strategy != "raw" else "",
        "llm_provider": (
            args.llm_provider if args.query_strategy != "raw" else ""
        ),
        "llm_base_url": (
            args.llm_base_url if args.query_strategy != "raw" else ""
        ),
        "paperclip_ranking": args.paperclip_ranking,
        "paperclip_source": args.paperclip_source,
        "paperclip_full_corpus": int(args.paperclip_full_corpus),
        "retrieval_limit": args.retrieval_limit,
        "query_sent": _clean(getattr(query, "search_query", "")),
        "expansion_terms": " | ".join(
            _clean(term) for term in getattr(query, "expanded_terms", [])
        ),
        "retrieved_count": 0,
        "retrieved_documents": "",
        "retrieved_document_ids": "",
        "gold_documents": gold_titles,
        "gold_document_ids": gold_ids,
        "first_relevant_rank": "",
        "hit_at_1": 0,
        "hit_at_3": 0,
        "hit_at_5": 0,
        "hit_at_10": 0,
        "reciprocal_rank": 0.0,
        "matched_document": "",
        "match_type": "",
        "matched_identifier": "",
        "paperclip_result_id": "",
    }


def _ordered_rows(
    selected: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        rows_by_id[str(example["id"])]
        for example in selected
        if str(example["id"]) in rows_by_id
    ]


def _print_summary(rows: list[dict[str, Any]], retrieval_limit: int) -> None:
    if not rows:
        print("\nNo results were produced.")
        return

    errors = sum(str(row.get("status", "")) == "error" for row in rows)
    print("\nRetrieval summary")
    print(f"Questions: {len(rows)}")
    print(f"Errors: {errors}")
    for cutoff in (1, 3, 5, 10):
        if cutoff <= retrieval_limit:
            recall = sum(int(row[f"hit_at_{cutoff}"]) for row in rows) / len(rows)
            print(f"Recall@{cutoff}: {recall:.4f}")
    mrr = sum(float(row["reciprocal_rank"]) for row in rows) / len(rows)
    print(f"MRR@{retrieval_limit}: {mrr:.4f}")


def run_evaluation(
    args: argparse.Namespace,
    *,
    retrieval_fn: Callable[
        [str, argparse.Namespace, Callable[[str], str] | None], RetrievalTrace
    ] = _retrieve,
) -> Path:
    examples = load_benchmark(args.benchmark)
    selected = select_questions(examples, args.num_questions, seed=args.seed)
    generator = _prepare_llm_generator(args)
    output_file = _output_file(args)
    rows_by_id = _load_existing_rows(output_file, args)

    for index, example in enumerate(selected, start=1):
        question_id = str(example["id"])
        existing = rows_by_id.get(question_id)
        if existing is not None:
            is_error = str(existing.get("status", "")) == "error"
            if not is_error or not args.retry_errors:
                print(f"[{index}/{len(selected)}] {question_id} | skipped (saved)")
                continue

        print(f"[{index}/{len(selected)}] {question_id}")
        try:
            trace = retrieval_fn(str(example["question"]), args, generator)
            result = evaluate_retrieval_mrr_detailed(
                list(example["gold_documents"]), trace.papers
            )
            row = _result_row(example, trace, result, args)
        except Exception as exc:
            row = _error_row(example, args, exc)
            print(f"  ERROR: {_clean(exc)}")

        rows_by_id[question_id] = row
        # Save atomically after every question so an interruption does not lose work.
        _write_results(_ordered_rows(selected, rows_by_id), output_file)

    rows = _ordered_rows(selected, rows_by_id)
    _print_summary(rows, args.retrieval_limit)
    _write_results(rows, output_file)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw, HyDE, and LLM-expanded queries with Paperclip "
            "BM25, vector, or hybrid document retrieval."
        )
    )
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--num-questions", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument(
        "--query-strategy",
        choices=QUERY_STRATEGIES,
        default="raw",
    )
    parser.add_argument(
        "--paperclip-ranking",
        choices=PAPERCLIP_RANKINGS,
        default="hybrid",
    )
    parser.add_argument("--paperclip-source", default=settings.paperclip_source)
    parser.add_argument(
        "--paperclip-full-corpus",
        action=argparse.BooleanOptionalAction,
        default=settings.paperclip_full_corpus,
    )
    parser.add_argument(
        "--paperclip-timeout",
        type=float,
        default=settings.paperclip_timeout,
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=settings.retrieval_limit,
    )
    parser.add_argument(
        "--empty-result-retries",
        type=int,
        default=DEFAULT_EMPTY_RESULT_RETRIES,
        help=(
            "Retry the same Paperclip request when it returns zero papers. "
            "Default: 2 retries (3 total attempts)."
        ),
    )

    parser.add_argument(
        "--llm-provider",
        choices=("ollama", "openai"),
        default=settings.llm_provider,
        help="Use 'openai' for Interweb and other OpenAI-compatible APIs.",
    )
    parser.add_argument("--llm-model", default=settings.llm_model)
    parser.add_argument("--llm-base-url", default=settings.llm_base_url)
    parser.add_argument("--llm-api-key-env", default=settings.llm_api_key_env)
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=settings.hyde_temperature,
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=max(settings.hyde_timeout, settings.expansion_timeout),
    )
    parser.add_argument("--llm-cache", default=None)
    parser.add_argument(
        "--hyde-max-tokens",
        type=int,
        default=settings.hyde_max_tokens,
    )
    parser.add_argument(
        "--expansion-max-tokens",
        type=int,
        default=settings.expansion_max_tokens,
    )
    parser.add_argument(
        "--expansion-max-terms",
        type=int,
        default=settings.expansion_max_terms,
    )
    parser.add_argument(
        "--expansion-max-query-chars",
        type=int,
        default=settings.expansion_max_query_chars,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a compatible partial results.csv. Enabled by default.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry rows previously saved with status=error.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.retrieval_limit < 10:
        raise ValueError("retrieval_limit must be at least 10 for Recall@10")
    if args.empty_result_retries < 0:
        raise ValueError("empty_result_retries must be 0 or greater")
    if args.paperclip_timeout <= 0 or args.llm_timeout <= 0:
        raise ValueError("timeouts must be greater than 0")
    if not 0 <= args.llm_temperature <= 2:
        raise ValueError("llm_temperature must be between 0 and 2")
    if args.hyde_max_tokens <= 0 or args.expansion_max_tokens <= 0:
        raise ValueError("LLM token limits must be greater than 0")
    if args.expansion_max_terms <= 0:
        raise ValueError("expansion_max_terms must be greater than 0")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    output_file = run_evaluation(args)
    print(f"Saved: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
