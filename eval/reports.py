from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from eval.metrics.lexical import (
    QuestionLexicalMetrics,
    evaluate_question_lexical,
    extract_retrieved_snippet_texts,
)


def load_raw_results_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Load raw retrieval results from JSONL.
    """
    raw_path = Path(path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw results JSONL not found: {raw_path}")

    records: list[dict[str, Any]] = []

    with raw_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {raw_path}"
                ) from error

    return records


def compute_lexical_metrics_for_run(
    raw_results_path: str | Path,
    k_values: list[int],
) -> list[QuestionLexicalMetrics]:
    """
    Compute lexical metrics for all questions and all k values.
    """
    raw_records = load_raw_results_jsonl(raw_results_path)
    all_metrics: list[QuestionLexicalMetrics] = []

    for raw_record in raw_records:
        result = raw_record.get("result", {})

        if result.get("status") == "error":
            continue

        question_id = raw_record.get("question_id", "")
        question = raw_record.get("question", "")
        gold_nuggets = raw_record.get("gold_nuggets", [])
        retrieved_snippets = extract_retrieved_snippet_texts(raw_record)

        for k in k_values:
            metrics = evaluate_question_lexical(
                question_id=question_id,
                question=question,
                gold_nuggets=gold_nuggets,
                retrieved_snippets=retrieved_snippets,
                k=k,
            )

            all_metrics.append(metrics)

    return all_metrics


def aggregate_lexical_metrics(
    question_metrics: list[QuestionLexicalMetrics],
) -> list[dict[str, Any]]:
    """
    Aggregate per-question metrics into one row per k.
    """
    grouped: dict[int, list[QuestionLexicalMetrics]] = {}

    for metric in question_metrics:
        grouped.setdefault(metric.k, []).append(metric)

    aggregate_rows: list[dict[str, Any]] = []

    for k, metrics_for_k in sorted(grouped.items()):
        question_count = len(metrics_for_k)

        if question_count == 0:
            continue

        aggregate_rows.append(
            {
                "k": k,
                "question_count": question_count,
                "ROUGE1_Nugget@k": round(
                    sum(m.rouge1_nugget_at_k for m in metrics_for_k) / question_count,
                    4,
                ),
                "ROUGEL_Nugget@k": round(
                    sum(m.rougel_nugget_at_k for m in metrics_for_k) / question_count,
                    4,
                ),
                "ROUGE_Nugget@k": round(
                    sum(m.rouge_nugget_at_k for m in metrics_for_k) / question_count,
                    4,
                ),
                "BM25_Nugget@k": round(
                    sum(m.bm25_nugget_at_k for m in metrics_for_k) / question_count,
                    4,
                ),
                "avg_nuggets_per_question": round(
                    sum(m.nugget_count for m in metrics_for_k) / question_count,
                    2,
                ),
                "avg_retrieved_snippets_used": round(
                    sum(m.retrieved_snippet_count for m in metrics_for_k) / question_count,
                    2,
                ),
            }
        )

    return aggregate_rows


def save_lexical_question_metrics(
    question_metrics: list[QuestionLexicalMetrics],
    output_path: str | Path,
) -> None:
    """
    Save per-question lexical metrics.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question_id",
        "question",
        "k",
        "ROUGE1_Nugget@k",
        "ROUGEL_Nugget@k",
        "ROUGE_Nugget@k",
        "BM25_Nugget@k",
        "nugget_count",
        "retrieved_snippet_count",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for metric in question_metrics:
            writer.writerow(
                {
                    "question_id": metric.question_id,
                    "question": metric.question,
                    "k": metric.k,
                    "ROUGE1_Nugget@k": metric.rouge1_nugget_at_k,
                    "ROUGEL_Nugget@k": metric.rougel_nugget_at_k,
                    "ROUGE_Nugget@k": metric.rouge_nugget_at_k,
                    "BM25_Nugget@k": metric.bm25_nugget_at_k,
                    "nugget_count": metric.nugget_count,
                    "retrieved_snippet_count": metric.retrieved_snippet_count,
                }
            )


def save_lexical_nugget_details(
    question_metrics: list[QuestionLexicalMetrics],
    output_path: str | Path,
) -> None:
    """
    Save per-nugget detailed scores.

    This is useful for error analysis because it shows which gold nuggets were
    covered and which were missed.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question_id",
        "question",
        "k",
        "nugget",
        "best_rouge1_f1",
        "best_rougel_f1",
        "best_rouge_combined",
        "best_bm25_normalized",
        "best_rouge1_rank",
        "best_rougel_rank",
        "best_bm25_rank",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for metric in question_metrics:
            for nugget_result in metric.nugget_results:
                writer.writerow(
                    {
                        "question_id": metric.question_id,
                        "question": metric.question,
                        "k": metric.k,
                        "nugget": nugget_result.nugget,
                        "best_rouge1_f1": nugget_result.best_rouge1_f1,
                        "best_rougel_f1": nugget_result.best_rougel_f1,
                        "best_rouge_combined": nugget_result.best_rouge_combined,
                        "best_bm25_normalized": nugget_result.best_bm25_normalized,
                        "best_rouge1_rank": nugget_result.best_rouge1_rank,
                        "best_rougel_rank": nugget_result.best_rougel_rank,
                        "best_bm25_rank": nugget_result.best_bm25_rank,
                    }
                )


def save_lexical_aggregate_metrics(
    aggregate_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Save aggregate lexical metrics.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "k",
        "question_count",
        "ROUGE1_Nugget@k",
        "ROUGEL_Nugget@k",
        "ROUGE_Nugget@k",
        "BM25_Nugget@k",
        "avg_nuggets_per_question",
        "avg_retrieved_snippets_used",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in aggregate_rows:
            writer.writerow(row)


def run_lexical_report(
    raw_results_path: str | Path,
    output_dir: str | Path,
    k_values: list[int],
) -> dict[str, Path]:
    """
    Compute lexical metrics and save report files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    question_metrics = compute_lexical_metrics_for_run(
        raw_results_path=raw_results_path,
        k_values=k_values,
    )

    aggregate_rows = aggregate_lexical_metrics(question_metrics)

    aggregate_path = output_path / "lexical_aggregate_metrics.csv"
    question_path = output_path / "lexical_question_metrics.csv"
    nugget_path = output_path / "lexical_nugget_details.csv"

    save_lexical_aggregate_metrics(
        aggregate_rows=aggregate_rows,
        output_path=aggregate_path,
    )

    save_lexical_question_metrics(
        question_metrics=question_metrics,
        output_path=question_path,
    )

    save_lexical_nugget_details(
        question_metrics=question_metrics,
        output_path=nugget_path,
    )

    return {
        "aggregate": aggregate_path,
        "question": question_path,
        "nugget": nugget_path,
    }