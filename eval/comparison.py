from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEXICAL_AGGREGATE_FILE = "lexical_aggregate_metrics.csv"
SEMANTIC_AGGREGATE_FILE = "semantic_aggregate_metrics.csv"
LEXICAL_QUESTION_FILE = "lexical_question_metrics.csv"
SEMANTIC_QUESTION_FILE = "semantic_question_metrics.csv"

LEXICAL_METRIC_COLUMNS = [
    "ROUGE1_Nugget@k",
    "ROUGEL_Nugget@k",
    "ROUGE_Nugget@k",
    "BM25_Nugget@k",
]

SEMANTIC_METRIC_COLUMNS = [
    "SemanticNuggetMatch@k",
    "SemanticAnswerMatch@k",
]

ALL_METRIC_COLUMNS = LEXICAL_METRIC_COLUMNS + SEMANTIC_METRIC_COLUMNS


@dataclass
class RunComparisonInput:
    """
    One evaluation run to include in a comparison report.
    """

    label: str
    run_dir: Path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """
    Read CSV rows as dictionaries.

    Missing files are allowed at a higher level, because a run may have lexical
    metrics but not semantic metrics yet.
    """
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write rows to CSV with automatically collected field names.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def parse_float(value: Any) -> float | None:
    """
    Convert metric values to floats when possible.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def load_aggregate_metrics_for_run(
    run_input: RunComparisonInput,
) -> list[dict[str, Any]]:
    """
    Load aggregate lexical and semantic metric rows for one run.

    Output:
    - one row per k
    - contains all available aggregate metric columns
    """
    run_dir = run_input.run_dir

    lexical_rows = read_csv_rows(run_dir / LEXICAL_AGGREGATE_FILE)
    semantic_rows = read_csv_rows(run_dir / SEMANTIC_AGGREGATE_FILE)

    rows_by_k: dict[str, dict[str, Any]] = {}

    for row in lexical_rows:
        k = row.get("k", "")

        if not k:
            continue

        rows_by_k.setdefault(
            k,
            {
                "run_label": run_input.label,
                "run_dir": str(run_dir),
                "k": k,
            },
        )

        rows_by_k[k]["question_count"] = row.get("question_count", "")

        for metric in LEXICAL_METRIC_COLUMNS:
            rows_by_k[k][metric] = row.get(metric, "")

        rows_by_k[k]["avg_nuggets_per_question"] = row.get(
            "avg_nuggets_per_question",
            "",
        )

        rows_by_k[k]["avg_retrieved_snippets_used"] = row.get(
            "avg_retrieved_snippets_used",
            "",
        )

    for row in semantic_rows:
        k = row.get("k", "")

        if not k:
            continue

        rows_by_k.setdefault(
            k,
            {
                "run_label": run_input.label,
                "run_dir": str(run_dir),
                "k": k,
            },
        )

        rows_by_k[k]["question_count"] = row.get(
            "question_count",
            rows_by_k[k].get("question_count", ""),
        )

        rows_by_k[k]["embedding_model"] = row.get("embedding_model", "")

        for metric in SEMANTIC_METRIC_COLUMNS:
            rows_by_k[k][metric] = row.get(metric, "")

        rows_by_k[k]["avg_nuggets_per_question"] = row.get(
            "avg_nuggets_per_question",
            rows_by_k[k].get("avg_nuggets_per_question", ""),
        )

        rows_by_k[k]["avg_retrieved_snippets_used"] = row.get(
            "avg_retrieved_snippets_used",
            rows_by_k[k].get("avg_retrieved_snippets_used", ""),
        )

    return [
        rows_by_k[k]
        for k in sorted(rows_by_k.keys(), key=lambda value: int(value))
    ]


def aggregate_wide_to_long(
    wide_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert wide aggregate rows into long format.

    Long format is easier for plotting:

        run_label, k, metric_name, metric_value
    """
    long_rows: list[dict[str, Any]] = []

    for row in wide_rows:
        for metric_name in ALL_METRIC_COLUMNS:
            metric_value = row.get(metric_name, "")

            if metric_value == "":
                continue

            long_rows.append(
                {
                    "run_label": row.get("run_label", ""),
                    "run_dir": row.get("run_dir", ""),
                    "k": row.get("k", ""),
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "question_count": row.get("question_count", ""),
                    "embedding_model": row.get("embedding_model", ""),
                }
            )

    return long_rows


def compute_aggregate_deltas(
    wide_rows: list[dict[str, Any]],
    baseline_label: str,
) -> list[dict[str, Any]]:
    """
    Compute metric deltas against a baseline run.

    Delta = current run metric - baseline run metric

    This helps quickly answer:
    - Did the new ranker improve?
    - On which metric?
    - At which k?
    """
    baseline_by_k: dict[str, dict[str, Any]] = {}

    for row in wide_rows:
        if row.get("run_label") == baseline_label:
            baseline_by_k[str(row.get("k", ""))] = row

    delta_rows: list[dict[str, Any]] = []

    for row in wide_rows:
        run_label = row.get("run_label", "")

        if run_label == baseline_label:
            continue

        k = str(row.get("k", ""))
        baseline_row = baseline_by_k.get(k)

        if baseline_row is None:
            continue

        for metric_name in ALL_METRIC_COLUMNS:
            current_value = parse_float(row.get(metric_name))
            baseline_value = parse_float(baseline_row.get(metric_name))

            if current_value is None or baseline_value is None:
                continue

            delta = current_value - baseline_value

            delta_rows.append(
                {
                    "baseline_label": baseline_label,
                    "run_label": run_label,
                    "k": k,
                    "metric_name": metric_name,
                    "baseline_value": round(baseline_value, 4),
                    "run_value": round(current_value, 4),
                    "delta": round(delta, 4),
                    "relative_change_percent": round(
                        (delta / baseline_value) * 100,
                        2,
                    )
                    if baseline_value != 0
                    else "",
                }
            )

    return delta_rows


def load_question_metrics_for_run(
    run_input: RunComparisonInput,
) -> list[dict[str, Any]]:
    """
    Load per-question lexical and semantic metrics for one run.

    Output:
    - one row per question_id and k
    - includes all available metrics
    """
    run_dir = run_input.run_dir

    lexical_rows = read_csv_rows(run_dir / LEXICAL_QUESTION_FILE)
    semantic_rows = read_csv_rows(run_dir / SEMANTIC_QUESTION_FILE)

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in lexical_rows:
        question_id = row.get("question_id", "")
        k = row.get("k", "")

        if not question_id or not k:
            continue

        key = (question_id, k)

        rows_by_key.setdefault(
            key,
            {
                "run_label": run_input.label,
                "run_dir": str(run_dir),
                "question_id": question_id,
                "question": row.get("question", ""),
                "k": k,
            },
        )

        for metric in LEXICAL_METRIC_COLUMNS:
            rows_by_key[key][metric] = row.get(metric, "")

        rows_by_key[key]["nugget_count"] = row.get("nugget_count", "")
        rows_by_key[key]["retrieved_snippet_count"] = row.get(
            "retrieved_snippet_count",
            "",
        )

    for row in semantic_rows:
        question_id = row.get("question_id", "")
        k = row.get("k", "")

        if not question_id or not k:
            continue

        key = (question_id, k)

        rows_by_key.setdefault(
            key,
            {
                "run_label": run_input.label,
                "run_dir": str(run_dir),
                "question_id": question_id,
                "question": row.get("question", ""),
                "k": k,
            },
        )

        rows_by_key[key]["embedding_model"] = row.get("embedding_model", "")

        for metric in SEMANTIC_METRIC_COLUMNS:
            rows_by_key[key][metric] = row.get(metric, "")

        rows_by_key[key]["answer_best_rank"] = row.get("answer_best_rank", "")

        if not rows_by_key[key].get("nugget_count"):
            rows_by_key[key]["nugget_count"] = row.get("nugget_count", "")

        if not rows_by_key[key].get("retrieved_snippet_count"):
            rows_by_key[key]["retrieved_snippet_count"] = row.get(
                "retrieved_snippet_count",
                "",
            )

    return sorted(
        rows_by_key.values(),
        key=lambda item: (item.get("question_id", ""), int(item.get("k", 0))),
    )


def create_comparison_inputs(
    run_dirs: list[str],
    run_labels: list[str] | None = None,
) -> list[RunComparisonInput]:
    """
    Build comparison inputs from CLI arguments.

    If labels are not given, folder names are used.
    """
    if run_labels and len(run_labels) != len(run_dirs):
        raise ValueError(
            "Number of --run-labels must match number of --run-dirs."
        )

    inputs: list[RunComparisonInput] = []

    for index, run_dir_value in enumerate(run_dirs):
        run_dir = Path(run_dir_value)

        if run_labels:
            label = run_labels[index]
        else:
            label = run_dir.name

        inputs.append(
            RunComparisonInput(
                label=label,
                run_dir=run_dir,
            )
        )

    return inputs


def run_comparison_report(
    run_dirs: list[str],
    output_dir: str | Path,
    run_labels: list[str] | None = None,
    baseline_label: str | None = None,
) -> dict[str, Path]:
    """
    Create comparison reports for multiple evaluation runs.

    Created files:
    - comparison_aggregate_wide.csv
    - comparison_aggregate_long.csv
    - comparison_aggregate_deltas.csv
    - comparison_question_wide.csv
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_inputs = create_comparison_inputs(
        run_dirs=run_dirs,
        run_labels=run_labels,
    )

    if not comparison_inputs:
        raise ValueError("At least one run directory is required.")

    effective_baseline_label = baseline_label or comparison_inputs[0].label

    aggregate_wide_rows: list[dict[str, Any]] = []
    question_wide_rows: list[dict[str, Any]] = []

    for run_input in comparison_inputs:
        aggregate_wide_rows.extend(
            load_aggregate_metrics_for_run(run_input)
        )

        question_wide_rows.extend(
            load_question_metrics_for_run(run_input)
        )

    aggregate_long_rows = aggregate_wide_to_long(aggregate_wide_rows)

    delta_rows = compute_aggregate_deltas(
        wide_rows=aggregate_wide_rows,
        baseline_label=effective_baseline_label,
    )

    aggregate_wide_path = output_path / "comparison_aggregate_wide.csv"
    aggregate_long_path = output_path / "comparison_aggregate_long.csv"
    delta_path = output_path / "comparison_aggregate_deltas.csv"
    question_wide_path = output_path / "comparison_question_wide.csv"

    write_csv_rows(aggregate_wide_path, aggregate_wide_rows)
    write_csv_rows(aggregate_long_path, aggregate_long_rows)
    write_csv_rows(delta_path, delta_rows)
    write_csv_rows(question_wide_path, question_wide_rows)

    return {
        "aggregate_wide": aggregate_wide_path,
        "aggregate_long": aggregate_long_path,
        "deltas": delta_path,
        "question_wide": question_wide_path,
    }