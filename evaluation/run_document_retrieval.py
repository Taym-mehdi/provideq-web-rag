from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from web_rag.config import get_settings
from web_rag.europepmc_retriever import retrieve_papers_europepmc
from web_rag.multi_source_retriever import reciprocal_rank_fusion
from web_rag.paperclip_retriever import retrieve_papers as retrieve_papers_paperclip
from web_rag.query_reformulation import (
    QueryGenerationError,
    build_raw_query,
    reformulate_query,
)

from .retrieval_evaluation import RetrievalMRRResult, evaluate_retrieval_mrr_detailed
from .run_evaluation import (
    QUERY_STRATEGIES,
    PAPERCLIP_RANKINGS,
    _prepare_llm_generator,
    load_benchmark,
    select_questions,
)


DEFAULT_BENCHMARK = Path("benchmark/provideq_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("outputs/document_retrieval")
RETRIEVERS = ("paperclip", "europepmc", "fusion")
EUROPEPMC_MODES = ("direct", "multi")


@dataclass(frozen=True)
class RetrievalTrace:
    papers: list[Any]
    paperclip_query: str
    europepmc_query: str
    europepmc_query_variants: list[str]
    paperclip_result_id: str = ""
    warnings: tuple[str, ...] = ()


RESULT_FIELDS = (
    "configuration",
    "question_id",
    "category",
    "question",
    "status",
    "error",
    "warnings",
    "retriever",
    "query_strategy",
    "paperclip_ranking",
    "paperclip_source",
    "paperclip_candidate_limit",
    "paperclip_query_fusion",
    "query_fusion_rrf_k",
    "reformulated_query_weight",
    "europepmc_mode",
    "europepmc_synonym",
    "europepmc_candidate_limit",
    "rrf_k",
    "retrieval_limit",
    "paperclip_query",
    "europepmc_query",
    "europepmc_query_variants",
    "retrieved_count",
    "retrieved_documents",
    "retrieved_document_ids",
    "retrieved_sources",
    "gold_documents",
    "gold_document_ids",
    "first_relevant_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "hit_at_20",
    "reciprocal_rank",
    "matched_document",
    "match_type",
    "matched_identifier",
    "paperclip_result_id",
)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _paper_value(paper: Any, field: str, default: Any = "") -> Any:
    if isinstance(paper, dict):
        return paper.get(field, default)
    return getattr(paper, field, default)


def _metadata_value(paper: Any, field: str) -> str:
    metadata = _paper_value(paper, "metadata", {})
    if isinstance(metadata, dict):
        return _clean(metadata.get(field, ""))
    return ""


def _paper_identifier(paper: Any) -> str:
    values: list[str] = []
    for label, value in (
        ("paper_id", _paper_value(paper, "paper_id", "")),
        ("pmcid", _metadata_value(paper, "pmcid")),
        ("pmid", _metadata_value(paper, "pmid")),
        ("doi", _paper_value(paper, "doi", "")),
    ):
        cleaned = _clean(value)
        if not cleaned:
            continue
        rendered = cleaned if label == "paper_id" else f"{label}:{cleaned}"
        if rendered.casefold() not in {item.casefold() for item in values}:
            values.append(rendered)
    return "; ".join(values)


def _gold_identifier(document: dict[str, Any]) -> str:
    return "; ".join(
        f"{key}:{_clean(document.get(key, ''))}"
        for key in ("pmcid", "pmid", "doi")
        if _clean(document.get(key, ""))
    )


def _make_query(
    question: str,
    args: argparse.Namespace,
    generator: Callable[[str], str] | None,
):
    return reformulate_query(
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
        expansion_generator=generator if args.query_strategy == "llmexpand" else None,
    )


def _retrieve(
    question: str,
    args: argparse.Namespace,
    generator: Callable[[str], str] | None,
) -> RetrievalTrace:
    warnings: list[str] = []
    try:
        query_bundle = _make_query(question, args, generator)
    except QueryGenerationError as exc:
        if not args.paperclip_query_fusion:
            raise
        # Query fusion is a safe augmentation of raw retrieval. If the generator
        # is temporarily unavailable, keep the question alive by degrading to raw.
        query_bundle = build_raw_query(question)
        warnings.append(f"query reformulation failed; used raw query: {_clean(exc)}")

    # HyDE is useful for Paperclip's vector/hybrid search, but Europe PMC is a
    # lexical/fielded search engine. By default Europe PMC receives the original
    # question and builds its own progressively relaxed title/abstract queries.
    raw_query = _clean(query_bundle.normalized_question)
    paperclip_query = _clean(query_bundle.search_query)
    europepmc_query = (
        paperclip_query if args.europepmc_use_reformulated_query else _clean(question)
    )

    paperclip_papers: list[Any] = []
    paperclip_result_ids: list[str] = []
    paperclip_query_log = paperclip_query
    epmc_result = None

    if args.retriever in {"paperclip", "fusion"}:
        paperclip_limit = (
            args.paperclip_candidate_limit
            if args.retriever == "fusion"
            else args.retrieval_limit
        )

        def search_paperclip(query: str):
            return retrieve_papers_paperclip(
                query,
                limit=paperclip_limit,
                source=args.paperclip_source,
                ranking=args.paperclip_ranking,
                full_corpus=args.paperclip_full_corpus,
                load_full_text=False,
                timeout=args.paperclip_timeout,
            )

        try:
            if args.paperclip_query_fusion and paperclip_query != raw_query:
                raw_result = search_paperclip(raw_query)
                reformulated_result = search_paperclip(paperclip_query)
                reformulated_name = f"paperclip_{args.query_strategy}"
                query_fusion = reciprocal_rank_fusion(
                    {
                        "paperclip_raw": list(raw_result.papers),
                        reformulated_name: list(reformulated_result.papers),
                    },
                    limit=paperclip_limit,
                    rrf_k=args.query_fusion_rrf_k,
                    source_weights={
                        "paperclip_raw": 1.0,
                        reformulated_name: args.reformulated_query_weight,
                    },
                )
                paperclip_papers = query_fusion.papers
                paperclip_query_log = (
                    f"raw => {raw_query} || {args.query_strategy} => {paperclip_query}"
                )
                for label, result in (
                    ("raw", raw_result),
                    (args.query_strategy, reformulated_result),
                ):
                    result_id = _clean(getattr(result, "result_id", ""))
                    if result_id:
                        paperclip_result_ids.append(f"{label}:{result_id}")
            else:
                paperclip_result = search_paperclip(paperclip_query)
                paperclip_papers = list(paperclip_result.papers)
                result_id = _clean(getattr(paperclip_result, "result_id", ""))
                if result_id:
                    paperclip_result_ids.append(result_id)
        except Exception as exc:
            if args.retriever != "fusion":
                raise
            warnings.append(f"Paperclip failed; used Europe PMC only: {_clean(exc)}")

    if args.retriever in {"europepmc", "fusion"}:
        epmc_limit = (
            args.europepmc_candidate_limit
            if args.retriever == "fusion"
            else args.retrieval_limit
        )
        try:
            epmc_result = retrieve_papers_europepmc(
                europepmc_query,
                limit=epmc_limit,
                candidate_limit=args.europepmc_candidate_limit,
                synonym=args.europepmc_synonym,
                mode=args.europepmc_mode,
                rrf_k=args.rrf_k,
                timeout=args.europepmc_timeout,
                cache_dir=args.europepmc_cache_dir,
            )
        except Exception as exc:
            if args.retriever != "fusion":
                raise
            warnings.append(f"Europe PMC failed; used Paperclip only: {_clean(exc)}")

    if args.retriever == "fusion" and not paperclip_papers and epmc_result is None:
        detail = " | ".join(warnings) or "both retrieval sources returned no result"
        raise RuntimeError(f"Fusion retrieval failed: {detail}")

    if args.retriever == "paperclip":
        papers = list(paperclip_papers)
    elif args.retriever == "europepmc":
        papers = list(epmc_result.papers if epmc_result else [])
    else:
        fused = reciprocal_rank_fusion(
            {
                "paperclip": list(paperclip_papers),
                "europepmc": list(epmc_result.papers if epmc_result else []),
            },
            limit=args.retrieval_limit,
            rrf_k=args.rrf_k,
        )
        papers = fused.papers

    # Standalone retrievers already have their requested final limit. Fusion is
    # explicitly cut to retrieval_limit above.
    papers = papers[: args.retrieval_limit]
    for rank, paper in enumerate(papers, start=1):
        paper.retrieval_rank = rank

    return RetrievalTrace(
        papers=papers,
        paperclip_query=(
            paperclip_query_log
            if args.retriever in {"paperclip", "fusion"}
            else ""
        ),
        europepmc_query=europepmc_query if epmc_result is not None else "",
        europepmc_query_variants=(
            list(epmc_result.query_variants) if epmc_result is not None else []
        ),
        paperclip_result_id="; ".join(paperclip_result_ids),
        warnings=tuple(warnings),
    )


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
    retrieved_sources: list[str] = []
    for fallback_rank, paper in enumerate(ordered, start=1):
        rank = int(_paper_value(paper, "retrieval_rank", fallback_rank) or fallback_rank)
        title = _clean(_paper_value(paper, "title", ""))
        if title:
            retrieved_titles.append(f"{rank}: {title}")
        identifier = _paper_identifier(paper)
        if identifier:
            retrieved_ids.append(f"{rank}: {identifier}")
        source = _clean(_paper_value(paper, "source", ""))
        if source:
            retrieved_sources.append(f"{rank}: {source}")

    gold_documents = [
        document for document in example["gold_documents"] if isinstance(document, dict)
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
        "configuration": _config_name(args),
        "question_id": _clean(example["id"]),
        "category": _clean(example.get("category", "")),
        "question": _clean(example["question"]),
        "status": "success",
        "error": "",
        "warnings": " | ".join(trace.warnings),
        "retriever": args.retriever,
        "query_strategy": args.query_strategy,
        "paperclip_ranking": args.paperclip_ranking if args.retriever != "europepmc" else "",
        "paperclip_source": args.paperclip_source if args.retriever != "europepmc" else "",
        "paperclip_candidate_limit": args.paperclip_candidate_limit if args.retriever == "fusion" else "",
        "paperclip_query_fusion": int(args.paperclip_query_fusion),
        "query_fusion_rrf_k": (
            args.query_fusion_rrf_k if args.paperclip_query_fusion else ""
        ),
        "reformulated_query_weight": (
            args.reformulated_query_weight if args.paperclip_query_fusion else ""
        ),
        "europepmc_mode": args.europepmc_mode if args.retriever != "paperclip" else "",
        "europepmc_synonym": int(args.europepmc_synonym) if args.retriever != "paperclip" else "",
        "europepmc_candidate_limit": args.europepmc_candidate_limit if args.retriever != "paperclip" else "",
        "rrf_k": args.rrf_k if args.retriever != "paperclip" else "",
        "retrieval_limit": args.retrieval_limit,
        "paperclip_query": trace.paperclip_query,
        "europepmc_query": trace.europepmc_query,
        "europepmc_query_variants": " | ".join(trace.europepmc_query_variants),
        "retrieved_count": len(ordered),
        "retrieved_documents": " | ".join(retrieved_titles),
        "retrieved_document_ids": " | ".join(retrieved_ids),
        "retrieved_sources": " | ".join(retrieved_sources),
        "gold_documents": " | ".join(gold_titles),
        "gold_document_ids": " | ".join(gold_ids),
        "first_relevant_rank": "" if rank is None else rank,
        "hit_at_1": int(rank is not None and rank <= 1),
        "hit_at_3": int(rank is not None and rank <= 3),
        "hit_at_5": int(rank is not None and rank <= 5),
        "hit_at_10": int(rank is not None and rank <= 10),
        "hit_at_20": int(rank is not None and rank <= 20),
        "reciprocal_rank": round(float(result.score), 6),
        "matched_document": _clean(result.matched_title),
        "match_type": _clean(result.match_type),
        "matched_identifier": _clean(result.matched_identifier),
        "paperclip_result_id": trace.paperclip_result_id,
    }


def _error_row(example: dict[str, Any], args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    gold_documents = [
        document for document in example["gold_documents"] if isinstance(document, dict)
    ]
    gold_titles = " | ".join(
        _clean(document.get("title", "")) for document in gold_documents
    )
    gold_ids = " | ".join(
        identifier
        for document in gold_documents
        if (identifier := _gold_identifier(document))
    )
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "configuration": _config_name(args),
            "question_id": _clean(example["id"]),
            "category": _clean(example.get("category", "")),
            "question": _clean(example["question"]),
            "status": "error",
            "error": _clean(exc),
            "retriever": args.retriever,
            "query_strategy": args.query_strategy,
            "paperclip_ranking": args.paperclip_ranking,
            "paperclip_source": args.paperclip_source,
            "paperclip_candidate_limit": args.paperclip_candidate_limit,
            "paperclip_query_fusion": int(args.paperclip_query_fusion),
            "query_fusion_rrf_k": (
                args.query_fusion_rrf_k if args.paperclip_query_fusion else ""
            ),
            "reformulated_query_weight": (
                args.reformulated_query_weight if args.paperclip_query_fusion else ""
            ),
            "europepmc_mode": args.europepmc_mode,
            "europepmc_synonym": int(args.europepmc_synonym),
            "europepmc_candidate_limit": args.europepmc_candidate_limit,
            "rrf_k": args.rrf_k,
            "retrieval_limit": args.retrieval_limit,
            "gold_documents": gold_titles,
            "gold_document_ids": gold_ids,
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_5": 0,
            "hit_at_10": 0,
            "hit_at_20": 0,
            "reciprocal_rank": 0.0,
        }
    )
    return row


def _config_name(args: argparse.Namespace) -> str:
    run_name = _clean(getattr(args, "run_name", ""))
    if run_name:
        return run_name

    parts = [args.retriever, args.query_strategy]
    if args.paperclip_query_fusion and args.query_strategy != "raw":
        weight = format(args.reformulated_query_weight, "g").replace(".", "p")
        parts.append(f"qfk{args.query_fusion_rrf_k}_w{weight}")
    if args.retriever in {"paperclip", "fusion"}:
        parts.append(args.paperclip_ranking)
    if args.retriever in {"europepmc", "fusion"}:
        parts.append(args.europepmc_mode)
        parts.append("syn" if args.europepmc_synonym else "nosyn")
    return "_".join(parts)


def _output_file(args: argparse.Namespace) -> Path:
    folder = Path(args.output_dir) / _config_name(args)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "results.csv"


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_FIELDS))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in RESULT_FIELDS} for row in rows)
    temporary.replace(path)


def _load_existing(path: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if not args.resume or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return {
        _clean(row.get("question_id", "")): row
        for row in rows
        if _clean(row.get("question_id", ""))
    }


def _ordered(selected: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [by_id[str(example["id"])] for example in selected if str(example["id"]) in by_id]


def _print_summary(
    rows: list[dict[str, Any]],
    label: str,
    retrieval_limit: int,
) -> None:
    print(f"\n=== {label} ===")
    if not rows:
        print("No results")
        return
    errors = sum(_clean(row.get("status", "")) == "error" for row in rows)
    warnings = sum(bool(_clean(row.get("warnings", ""))) for row in rows)
    print(f"Questions: {len(rows)}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
    for cutoff in (1, 3, 5, 10, 20):
        if cutoff > retrieval_limit:
            continue
        recall = sum(int(float(row.get(f"hit_at_{cutoff}", 0) or 0)) for row in rows) / len(rows)
        print(f"Recall@{cutoff}: {recall:.4f}")
    mrr = sum(float(row.get("reciprocal_rank", 0) or 0) for row in rows) / len(rows)
    print(f"MRR@{retrieval_limit}: {mrr:.4f}")


def run(args: argparse.Namespace) -> Path:
    examples = load_benchmark(args.benchmark)
    selected = select_questions(examples, args.num_questions, seed=args.seed)
    generator = _prepare_llm_generator(args)
    output = _output_file(args)
    rows_by_id = _load_existing(output, args)

    for index, example in enumerate(selected, start=1):
        qid = str(example["id"])
        existing = rows_by_id.get(qid)
        if existing is not None:
            is_error = _clean(existing.get("status", "")) == "error"
            has_warning = bool(_clean(existing.get("warnings", "")))
            should_retry = (is_error and args.retry_errors) or (
                has_warning and args.retry_warnings
            )
            if not should_retry:
                print(f"[{index}/{len(selected)}] {qid} | skipped (saved)")
                continue

        print(f"[{index}/{len(selected)}] {qid}")
        try:
            trace = _retrieve(str(example["question"]), args, generator)
            metric = evaluate_retrieval_mrr_detailed(
                list(example["gold_documents"]), trace.papers
            )
            row = _result_row(example, trace, metric, args)
        except Exception as exc:
            row = _error_row(example, args, exc)
            print(f"  ERROR: {_clean(exc)}")

        rows_by_id[qid] = row
        _write_rows(_ordered(selected, rows_by_id), output)

    rows = _ordered(selected, rows_by_id)
    _write_rows(rows, output)
    _print_summary(rows, _config_name(args), args.retrieval_limit)
    return output


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate document retrieval with Paperclip, Europe PMC, or RRF fusion. "
            "This stops before chunking/reranking."
        )
    )
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--num-questions", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--run-name",
        default="",
        help=(
            "Optional lowercase underscore-separated folder name. When omitted, "
            "a name is generated from the retrieval settings."
        ),
    )
    parser.add_argument("--retriever", choices=RETRIEVERS, required=True)
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=settings.retrieval_limit,
    )

    parser.add_argument("--query-strategy", choices=QUERY_STRATEGIES, default="raw")
    parser.add_argument("--llm-provider", choices=("ollama", "openai"), default=settings.llm_provider)
    parser.add_argument("--llm-model", default=settings.llm_model)
    parser.add_argument("--llm-base-url", default=settings.llm_base_url)
    parser.add_argument("--llm-api-key-env", default=settings.llm_api_key_env)
    parser.add_argument("--llm-temperature", type=float, default=settings.hyde_temperature)
    parser.add_argument("--llm-timeout", type=float, default=max(settings.hyde_timeout, settings.expansion_timeout))
    parser.add_argument("--llm-cache", default=None)
    parser.add_argument("--hyde-max-tokens", type=int, default=settings.hyde_max_tokens)
    parser.add_argument("--expansion-max-tokens", type=int, default=settings.expansion_max_tokens)
    parser.add_argument("--expansion-max-terms", type=int, default=settings.expansion_max_terms)
    parser.add_argument("--expansion-max-query-chars", type=int, default=settings.expansion_max_query_chars)

    parser.add_argument("--paperclip-ranking", choices=PAPERCLIP_RANKINGS, default="hybrid")
    parser.add_argument("--paperclip-source", default=settings.paperclip_source)
    parser.add_argument(
        "--paperclip-full-corpus",
        action=argparse.BooleanOptionalAction,
        default=settings.paperclip_full_corpus,
    )
    parser.add_argument("--paperclip-timeout", type=float, default=settings.paperclip_timeout)
    parser.add_argument("--paperclip-candidate-limit", type=int, default=30)
    parser.add_argument(
        "--paperclip-query-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retrieve Paperclip with both the raw and reformulated query, then "
            "combine the rankings with weighted RRF."
        ),
    )
    parser.add_argument(
        "--query-fusion-rrf-k",
        type=int,
        default=10,
        help="RRF smoothing constant for raw/reformulated Paperclip query fusion.",
    )
    parser.add_argument(
        "--reformulated-query-weight",
        type=float,
        default=0.5,
        help="Weight assigned to the HyDE/LLM-expanded Paperclip ranking.",
    )

    parser.add_argument("--europepmc-mode", choices=EUROPEPMC_MODES, default="multi")
    parser.add_argument(
        "--europepmc-synonym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Europe PMC MeSH/UniProt synonym expansion (default: on).",
    )
    parser.add_argument("--europepmc-timeout", type=float, default=45.0)
    parser.add_argument("--europepmc-candidate-limit", type=int, default=30)
    parser.add_argument("--europepmc-cache-dir", default="outputs/europepmc_cache")
    parser.add_argument(
        "--europepmc-use-reformulated-query",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pass HyDE/LLM-expanded text to Europe PMC. Disabled by default because "
            "Europe PMC is lexical and works better with source-specific keyword queries."
        ),
    )
    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--retry-warnings",
        action="store_true",
        help="When resuming, rerun rows that used an LLM or source fallback.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.run_name and not re.fullmatch(
        r"[a-z0-9]+(?:_[a-z0-9]+)*",
        args.run_name,
    ):
        raise ValueError(
            "run_name must contain only lowercase letters, numbers, and single underscores"
        )
    if args.retrieval_limit < 10:
        raise ValueError("retrieval_limit must be at least 10")
    for name in ("paperclip_candidate_limit", "europepmc_candidate_limit"):
        if getattr(args, name) < args.retrieval_limit:
            raise ValueError(f"{name} must be >= retrieval_limit")
    if args.rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")
    if args.query_fusion_rrf_k <= 0:
        raise ValueError("query_fusion_rrf_k must be greater than 0")
    if args.reformulated_query_weight <= 0:
        raise ValueError("reformulated_query_weight must be greater than 0")
    if args.paperclip_query_fusion and args.query_strategy == "raw":
        raise ValueError("paperclip_query_fusion requires hyde or llmexpand")
    if args.paperclip_query_fusion and args.retriever == "europepmc":
        raise ValueError("paperclip_query_fusion requires paperclip or fusion retrieval")
    if args.europepmc_timeout <= 0 or args.paperclip_timeout <= 0:
        raise ValueError("timeouts must be greater than 0")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    output = run(args)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
