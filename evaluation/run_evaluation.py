from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from web_rag import run_pipeline
from web_rag.paperclip_retriever import retrieve_papers
from web_rag.query_reformulation import reformulate_query

from .lexical_evaluation import evaluate_lexical
from .llm_judge_evaluation import LLMJudgeEvaluator
from .retrieval_evaluation import RetrievalMRRResult, evaluate_retrieval_mrr_detailed
from .semantic_evaluation import DEFAULT_MODEL, SemanticEvaluator


DEFAULT_BENCHMARK = Path("benchmark/provideq_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class MRRRetrievalTrace:
    query: Any
    papers: list[Any]
    result_id: str = ""


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

    return examples


def select_questions(
    examples: list[dict[str, Any]],
    number: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if number < 1 or number > len(examples):
        raise ValueError(
            f"num_questions must be between 1 and {len(examples)} "
            f"for this benchmark"
        )

    selected = random.Random(seed).sample(examples, number)
    return sorted(selected, key=lambda example: str(example["id"]))


def _evidence_texts(result: Any) -> list[str]:
    return [
        str(record.evidence_text).strip()
        for record in result.records
        if str(record.evidence_text).strip()
    ]


def _write_results(
    output_dir: Path,
    folder_name: str,
    rows: list[dict[str, Any]],
) -> Path:
    run_dir = output_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "results.csv"

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["question_id", "question", "answers", "gold_answer", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_file



def _write_mrr_results(
    output_dir: Path,
    folder_name: str,
    rows: list[dict[str, Any]],
) -> Path:
    run_dir = output_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "results.csv"

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "category",
                "question",
                "query_strategy",
                "paperclip_ranking",
                "paperclip_source",
                "query_sent",
                "query_keywords",
                "query_expanded_terms",
                "retrieved_documents",
                "retrieved_document_ids",
                "gold_documents",
                "gold_document_ids",
                "gold_in_selected_corpus",
                "matched_document",
                "matched_identifier",
                "match_type",
                "first_relevant_rank",
                "hit",
                "score",
                "paperclip_result_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_file


def _retrieve_for_mrr(question: str, args: argparse.Namespace) -> MRRRetrievalTrace:
    query = reformulate_query(
        question,
        args.query_strategy,
        hyde_model=args.hyde_model,
        hyde_base_url=args.hyde_base_url,
        hyde_temperature=args.hyde_temperature,
        hyde_max_tokens=args.hyde_max_tokens,
        hyde_seed=args.hyde_seed,
        hyde_timeout=args.hyde_timeout,
        expansion_model=args.expansion_model,
        expansion_base_url=args.expansion_base_url,
        expansion_temperature=args.expansion_temperature,
        expansion_max_tokens=args.expansion_max_tokens,
        expansion_seed=args.expansion_seed,
        expansion_timeout=args.expansion_timeout,
        expansion_max_terms=args.expansion_max_terms,
        expansion_max_query_chars=args.expansion_max_query_chars,
    )
    retrieval = retrieve_papers(
        query.search_query,
        limit=args.retrieval_limit,
        source=args.paperclip_source,
        ranking=args.paperclip_ranking,
    )
    return MRRRetrievalTrace(
        query=query,
        papers=list(retrieval.papers),
        result_id=str(getattr(retrieval, "result_id", "") or ""),
    )

def _paperclip_sources(value: str) -> set[str]:
    return {
        item.strip().casefold()
        for item in str(value).split(",")
        if item.strip()
    }


def _gold_document_in_selected_corpus(
    gold_document: dict[str, Any],
    paperclip_source: str,
) -> bool:
    """Return whether the gold paper is identifiable in the selected corpus.

    The current benchmark provides PMCIDs for papers known to be in PMC. For the
    current Paperclip configuration, PMCID presence is therefore the reliable
    eligibility signal. DOI/PMID alone do not prove that full text exists in PMC.
    """
    sources = _paperclip_sources(paperclip_source)
    if "all" in sources or "abstracts_only" in sources:
        return True
    if "pmc" in sources and str(gold_document.get("pmcid", "")).strip():
        return True
    return False


def _example_in_selected_corpus(
    example: dict[str, Any],
    paperclip_source: str,
) -> bool:
    return any(
        _gold_document_in_selected_corpus(document, paperclip_source)
        for document in example.get("gold_documents", [])
        if isinstance(document, dict)
    )


def run_evaluation(
    args: argparse.Namespace,
    *,
    pipeline_fn: Callable[..., Any] = run_pipeline,
    retrieval_fn: Callable[[str, argparse.Namespace], Any] = _retrieve_for_mrr,
) -> list[Path]:
    examples = load_benchmark(args.benchmark)
    if bool(getattr(args, "corpus_eligible_only", False)):
        eligible_examples = [
            example
            for example in examples
            if _example_in_selected_corpus(example, args.paperclip_source)
        ]
        print(
            f"Corpus-eligible benchmark questions: {len(eligible_examples)}/"
            f"{len(examples)} for source '{args.paperclip_source}'"
        )
        examples = eligible_examples

    selected = select_questions(examples, args.num_questions, seed=args.seed)

    layers = (
        ["mrr", "lexical", "semantic", "judge"]
        if args.evaluation == "all"
        else [args.evaluation]
    )
    semantic_evaluator = None
    judge_evaluator = None

    if "semantic" in layers:
        semantic_evaluator = SemanticEvaluator(
            args.semantic_model,
            device=args.semantic_device,
            batch_size=args.semantic_batch_size,
        )
    if "judge" in layers:
        judge_evaluator = LLMJudgeEvaluator(
            provider=args.judge_provider,
            model=args.judge_model,
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
            retries=args.judge_retries,
        )

    scores: dict[str, list[dict[str, Any]]] = {layer: [] for layer in layers}
    no_save = bool(getattr(args, "no_save", False))

    for index, example in enumerate(selected, start=1):
        question = str(example["question"])
        if not no_save:
            print(f"[{index}/{len(selected)}] {example['id']}")

        result = None
        query_bundle: Any = None
        paperclip_result_id = ""
        evidence: list[str] = []
        retrieved_papers: list[Any] = []

        if layers == ["mrr"]:
            trace = retrieval_fn(question, args)
            if isinstance(trace, MRRRetrievalTrace):
                query_bundle = trace.query
                retrieved_papers = list(trace.papers)
                paperclip_result_id = trace.result_id
            else:
                # Backward compatibility for custom tests that return only papers.
                retrieved_papers = list(trace)
        else:
            result = pipeline_fn(
                question,
                retrieval_limit=args.retrieval_limit,
                query_strategy=args.query_strategy,
                hyde_model=args.hyde_model,
                hyde_base_url=args.hyde_base_url,
                hyde_temperature=args.hyde_temperature,
                hyde_max_tokens=args.hyde_max_tokens,
                hyde_seed=args.hyde_seed,
                hyde_timeout=args.hyde_timeout,
                expansion_model=args.expansion_model,
                expansion_base_url=args.expansion_base_url,
                expansion_temperature=args.expansion_temperature,
                expansion_max_tokens=args.expansion_max_tokens,
                expansion_seed=args.expansion_seed,
                expansion_timeout=args.expansion_timeout,
                expansion_max_terms=args.expansion_max_terms,
                expansion_max_query_chars=args.expansion_max_query_chars,
                paperclip_source=args.paperclip_source,
                paperclip_ranking=args.paperclip_ranking,
                chunk_window_size=args.chunk_window_size,
                chunk_stride=args.chunk_stride,
                reranker=args.reranker,
                top_k=args.top_k,
                max_chunks_per_paper=args.max_chunks_per_paper,
                near_duplicate_threshold=args.near_duplicate_threshold,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
                medcpt_device=args.medcpt_device,
                hybrid_lexical_weight=args.hybrid_lexical_weight,
                hybrid_medcpt_weight=args.hybrid_medcpt_weight,
            )
            evidence = _evidence_texts(result)
            query_bundle = getattr(result, "query", None)
            pipeline_info = getattr(result, "pipeline", None)
            paperclip_result_id = str(
                getattr(pipeline_info, "paperclip_result_id", "") or ""
            )
            retrieved_papers = list(getattr(result, "retrieved_papers", []))

        gold_answers = [str(answer) for answer in example["gold_answers"]]
        answerable = bool(example["answerable"])

        if "mrr" in layers:
            mrr_result = evaluate_retrieval_mrr_detailed(
                list(example["gold_documents"]), retrieved_papers
            )
            scores["mrr"].append(
                _mrr_result_row(
                    example,
                    retrieved_papers,
                    query_bundle=query_bundle,
                    query_strategy=args.query_strategy,
                    paperclip_ranking=args.paperclip_ranking,
                    paperclip_source=args.paperclip_source,
                    paperclip_result_id=paperclip_result_id,
                    result=mrr_result,
                )
            )

        if "lexical" in layers:
            score, best_evidence = evaluate_lexical(
                gold_answers, evidence, answerable=answerable
            )
            scores["lexical"].append(_result_row(example, best_evidence, score))

        if "semantic" in layers and semantic_evaluator is not None:
            score, best_evidence = semantic_evaluator.score(
                gold_answers, evidence, answerable=answerable
            )
            scores["semantic"].append(_result_row(example, best_evidence, score))

        if "judge" in layers and judge_evaluator is not None:
            score, best_evidence = judge_evaluator.score(
                question,
                gold_answers,
                evidence,
                answerable=answerable,
            )
            scores["judge"].append(_result_row(example, best_evidence, score))

        if no_save:
            _print_question_scores(str(example["id"]), layers, scores)

    if no_save:
        _print_average_scores(layers, scores)
        return []

    saved: list[Path] = []
    for layer in layers:
        if layer == "mrr":
            folder_name = (
                f"{args.query_strategy}_{args.paperclip_ranking}_retrieval_mrr"
            )
            saved.append(
                _write_mrr_results(Path(args.output_dir), folder_name, scores[layer])
            )
        else:
            folder_name = (
                f"{args.query_strategy}_{args.paperclip_ranking}_{args.reranker}_{layer}"
            )
            saved.append(_write_results(Path(args.output_dir), folder_name, scores[layer]))

    return saved


def _clean_csv_text(value: str) -> str:
    return " ".join(value.split())


def _display_score(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    return f"{float(value):.2f}"


def _print_question_scores(
    question_id: str,
    layers: list[str],
    scores: dict[str, list[dict[str, Any]]],
) -> None:
    if len(layers) == 1:
        score = scores[layers[0]][-1]["score"]
        print(f"{question_id} | score: {_display_score(score)}")
        return

    layer_scores = " | ".join(
        f"{layer}: {_display_score(scores[layer][-1]['score'])}" for layer in layers
    )
    print(f"{question_id} | {layer_scores}")


def _print_average_scores(
    layers: list[str],
    scores: dict[str, list[dict[str, Any]]],
) -> None:
    print()
    for layer in layers:
        numeric_scores = [
            float(row["score"])
            for row in scores[layer]
            if row["score"] not in (None, "")
        ]
        average = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
        label = "Average score" if len(layers) == 1 else f"Average {layer} score"
        print(f"{label}: {_display_score(average)}")



def _paper_value(paper: Any, field: str, default: Any = "") -> Any:
    if isinstance(paper, dict):
        return paper.get(field, default)
    return getattr(paper, field, default)


def _paper_identifier_text(paper: Any) -> str:
    values: list[str] = []
    paper_id = _clean_csv_text(str(_paper_value(paper, "paper_id", "")))
    doi = _clean_csv_text(str(_paper_value(paper, "doi", "")))
    if paper_id:
        values.append(paper_id)
    if doi and doi.casefold() not in {value.casefold() for value in values}:
        values.append(f"doi:{doi}")
    return "; ".join(values)


def _gold_identifier_text(document: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("pmcid", "pmid", "doi"):
        value = _clean_csv_text(str(document.get(key, "")))
        if value:
            values.append(f"{key}:{value}")
    return "; ".join(values)


def _query_list_text(query_bundle: Any, field: str) -> str:
    values = getattr(query_bundle, field, []) if query_bundle is not None else []
    if not isinstance(values, list):
        return ""
    return " | ".join(_clean_csv_text(str(value)) for value in values if str(value).strip())


def _mrr_result_row(
    example: dict[str, Any],
    retrieved_papers: list[Any],
    *,
    query_bundle: Any,
    query_strategy: str,
    paperclip_ranking: str,
    paperclip_source: str,
    paperclip_result_id: str,
    result: RetrievalMRRResult,
) -> dict[str, Any]:
    ordered = sorted(
        retrieved_papers,
        key=lambda paper: int(_paper_value(paper, "retrieval_rank", 0) or 0),
    )
    retrieved_titles: list[str] = []
    retrieved_ids: list[str] = []
    for fallback_rank, paper in enumerate(ordered, start=1):
        retrieval_rank = int(
            _paper_value(paper, "retrieval_rank", fallback_rank) or fallback_rank
        )
        title = _clean_csv_text(str(_paper_value(paper, "title", "")))
        if title:
            retrieved_titles.append(f"{retrieval_rank}: {title}")
        identifier = _paper_identifier_text(paper)
        if identifier:
            retrieved_ids.append(f"{retrieval_rank}: {identifier}")

    gold_documents = [
        document
        for document in example["gold_documents"]
        if isinstance(document, dict)
    ]
    gold_titles = [
        _clean_csv_text(str(document.get("title", "")))
        for document in gold_documents
        if str(document.get("title", "")).strip()
    ]
    gold_ids = [
        identifier
        for document in gold_documents
        if (identifier := _gold_identifier_text(document))
    ]

    query_sent = ""
    if query_bundle is not None:
        query_sent = _clean_csv_text(str(getattr(query_bundle, "search_query", "")))

    return {
        "question_id": str(example["id"]),
        "category": _clean_csv_text(str(example.get("category", ""))),
        "question": _clean_csv_text(str(example["question"])),
        "query_strategy": _clean_csv_text(query_strategy),
        "paperclip_ranking": _clean_csv_text(paperclip_ranking),
        "paperclip_source": _clean_csv_text(paperclip_source),
        "query_sent": query_sent,
        "query_keywords": _query_list_text(query_bundle, "keywords"),
        "query_expanded_terms": _query_list_text(query_bundle, "expanded_terms"),
        "retrieved_documents": " | ".join(retrieved_titles),
        "retrieved_document_ids": " | ".join(retrieved_ids),
        "gold_documents": " | ".join(gold_titles),
        "gold_document_ids": " | ".join(gold_ids),
        "gold_in_selected_corpus": int(
            _example_in_selected_corpus(example, paperclip_source)
        ),
        "matched_document": _clean_csv_text(result.matched_title),
        "matched_identifier": _clean_csv_text(result.matched_identifier),
        "match_type": _clean_csv_text(result.match_type),
        "first_relevant_rank": "" if result.rank is None else result.rank,
        "hit": int(result.rank is not None),
        "score": round(float(result.score), 4),
        "paperclip_result_id": _clean_csv_text(paperclip_result_id),
    }

def _result_row(
    example: dict[str, Any],
    best_evidence: str,
    score: float | None,
) -> dict[str, Any]:
    return {
        "question_id": str(example["id"]),
        "question": _clean_csv_text(str(example["question"])),
        "answers": _clean_csv_text(best_evidence),
        "gold_answer": " | ".join(
            _clean_csv_text(str(answer)) for answer in example["gold_answers"]
        ),
        "score": "" if score is None else round(float(score), 4),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the ProvideQ Paperclip Web RAG pipeline")

    parser.add_argument("--evaluation", choices=("mrr", "lexical", "semantic", "judge", "all"), default="lexical")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print per-question and average scores without writing results.csv files.",
    )

    parser.add_argument("--paperclip-source", default="pmc,biorxiv,medrxiv,arxiv,abstracts_only")
    parser.add_argument(
        "--corpus-eligible-only",
        action="store_true",
        help=(
            "Evaluate only questions whose gold paper is identifiable in the "
            "selected Paperclip corpus. For source=pmc, this requires a PMCID."
        ),
    )
    parser.add_argument("--paperclip-ranking", choices=("bm25", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--retrieval-limit", type=int, default=10)
    parser.add_argument("--query-strategy", choices=("raw", "synonym", "hyde", "llmexpand"), default="hyde")
    parser.add_argument("--hyde-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--hyde-base-url", default="http://localhost:11434")
    parser.add_argument("--hyde-temperature", type=float, default=0.0)
    parser.add_argument("--hyde-max-tokens", type=int, default=256)
    parser.add_argument("--hyde-seed", type=int, default=42)
    parser.add_argument("--hyde-timeout", type=float, default=180.0)

    parser.add_argument("--expansion-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--expansion-base-url", default="http://localhost:11434")
    parser.add_argument("--expansion-temperature", type=float, default=0.0)
    parser.add_argument("--expansion-max-tokens", type=int, default=420)
    parser.add_argument("--expansion-seed", type=int, default=42)
    parser.add_argument("--expansion-timeout", type=float, default=180.0)
    parser.add_argument("--expansion-max-terms", type=int, default=32)
    parser.add_argument("--expansion-max-query-chars", type=int, default=1200)

    parser.add_argument("--chunk-window-size", type=int, default=3)
    parser.add_argument("--chunk-stride", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chunks-per-paper", type=int, default=2)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.80)

    parser.add_argument("--reranker", choices=("lexical", "medcpt", "hybrid"), default="hybrid")
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--medcpt-device", default="auto")
    parser.add_argument("--hybrid-lexical-weight", type=float, default=0.30)
    parser.add_argument("--hybrid-medcpt-weight", type=float, default=0.70)

    parser.add_argument("--semantic-model", default=DEFAULT_MODEL)
    parser.add_argument("--semantic-device", default="auto")
    parser.add_argument("--semantic-batch-size", type=int, default=8)

    parser.add_argument("--judge-provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-retries", type=int, default=2)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = run_evaluation(args)
    for path in paths:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
