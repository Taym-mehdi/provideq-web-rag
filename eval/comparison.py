from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEXICAL_AGGREGATE_FILE = "lexical_aggregate_metrics.csv"
SEMANTIC_AGGREGATE_FILE = "semantic_aggregate_metrics.csv"

METRIC_COLUMNS = {
    "ROUGE1_Nugget@k",
    "ROUGEL_Nugget@k",
    "ROUGE_Nugget@k",
    "BM25_Nugget@k",
    "SemanticNuggetMatch@k",
    "SemanticAnswerMatch@k",
}


@dataclass
class RunSpec:
    """
    One retrieval/evaluation run to compare.

    Example:
        label = "lexical"
        run_dir = outputs/evaluation/provideq20_lexical
    """

    label: str
    run_dir: Path


@dataclass
class MetricValue:
    """
    One metric value for one run and one k.
    """

    run_label: str
    metric_group: str
    metric_name: str
    k: int
    value: float


def parse_run_spec(raw_spec: str) -> RunSpec:
    """
    Parse run specification from CLI.

    Expected format:
        label=path

    Example:
        lexical=outputs/evaluation/provideq20_lexical
    """
    if "=" not in raw_spec:
        raise ValueError(
            "Run specifications must use the format label=path. "
            f"Invalid value: {raw_spec}"
        )

    label, path = raw_spec.split("=", 1)

    label = label.strip()
    path = path.strip()

    if not label:
        raise ValueError(f"Run label is empty in: {raw_spec}")

    if not path:
        raise ValueError(f"Run path is empty in: {raw_spec}")

    return RunSpec(
        label=label,
        run_dir=Path(path),
    )


def parse_run_specs(raw_specs: list[str]) -> list[RunSpec]:
    """
    Parse multiple run specifications.
    """
    if len(raw_specs) < 2:
        raise ValueError("At least two runs are required for comparison.")

    run_specs = [
        parse_run_spec(raw_spec)
        for raw_spec in raw_specs
    ]

    labels = [run.label for run in run_specs]

    if len(labels) != len(set(labels)):
        raise ValueError("Run labels must be unique.")

    return run_specs


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """
    Read CSV rows as dictionaries.
    """
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def safe_float(value: Any) -> float | None:
    """
    Convert a value to float if possible.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    """
    Convert a value to int if possible.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_aggregate_metrics_for_run(run_spec: RunSpec) -> list[MetricValue]:
    """
    Load lexical and semantic aggregate metrics from one run directory.
    """
    metric_values: list[MetricValue] = []

    aggregate_files = [
        ("lexical", run_spec.run_dir / LEXICAL_AGGREGATE_FILE),
        ("semantic", run_spec.run_dir / SEMANTIC_AGGREGATE_FILE),
    ]

    for metric_group, path in aggregate_files:
        rows = read_csv_rows(path)

        for row in rows:
            k = safe_int(row.get("k"))

            if k is None:
                continue

            for column_name, raw_value in row.items():
                if column_name not in METRIC_COLUMNS:
                    continue

                value = safe_float(raw_value)

                if value is None:
                    continue

                metric_values.append(
                    MetricValue(
                        run_label=run_spec.label,
                        metric_group=metric_group,
                        metric_name=column_name,
                        k=k,
                        value=value,
                    )
                )

    return metric_values


def load_all_metrics(run_specs: list[RunSpec]) -> list[MetricValue]:
    """
    Load aggregate metric values for all runs.
    """
    all_values: list[MetricValue] = []

    for run_spec in run_specs:
        run_values = load_aggregate_metrics_for_run(run_spec)
        all_values.extend(run_values)

    return all_values


def build_metric_lookup(
    metric_values: list[MetricValue],
) -> dict[tuple[str, str, int], MetricValue]:
    """
    Build lookup by:
        run_label, metric_name, k
    """
    lookup: dict[tuple[str, str, int], MetricValue] = {}

    for metric_value in metric_values:
        key = (
            metric_value.run_label,
            metric_value.metric_name,
            metric_value.k,
        )
        lookup[key] = metric_value

    return lookup


def relative_delta_percent(
    baseline_value: float,
    compared_value: float,
) -> float | str:
    """
    Compute relative improvement percentage.

    If baseline is zero, return an empty string to avoid misleading division.
    """
    if baseline_value == 0:
        return ""

    return round(((compared_value - baseline_value) / baseline_value) * 100, 2)


def compare_runs(
    run_specs: list[RunSpec],
    baseline_label: str,
) -> list[dict[str, Any]]:
    """
    Compare all runs against one baseline.

    Output format:
    one row per compared run, metric, and k.
    """
    labels = {run_spec.label for run_spec in run_specs}

    if baseline_label not in labels:
        raise ValueError(
            f"Baseline label '{baseline_label}' not found in runs: "
            + ", ".join(sorted(labels))
        )

    metric_values = load_all_metrics(run_specs)
    lookup = build_metric_lookup(metric_values)

    available_metric_keys = sorted(
        {
            (metric_value.metric_group, metric_value.metric_name, metric_value.k)
            for metric_value in metric_values
        }
    )

    comparison_rows: list[dict[str, Any]] = []

    compared_labels = [
        run_spec.label
        for run_spec in run_specs
        if run_spec.label != baseline_label
    ]

    for metric_group, metric_name, k in available_metric_keys:
        baseline_metric = lookup.get((baseline_label, metric_name, k))

        if baseline_metric is None:
            continue

        for compared_label in compared_labels:
            compared_metric = lookup.get((compared_label, metric_name, k))

            if compared_metric is None:
                continue

            delta = round(compared_metric.value - baseline_metric.value, 4)

            comparison_rows.append(
                {
                    "metric_group": metric_group,
                    "metric_name": metric_name,
                    "k": k,
                    "baseline_run": baseline_label,
                    "baseline_score": baseline_metric.value,
                    "compared_run": compared_label,
                    "compared_score": compared_metric.value,
                    "delta": delta,
                    "relative_delta_percent": relative_delta_percent(
                        baseline_value=baseline_metric.value,
                        compared_value=compared_metric.value,
                    ),
                }
            )

    return comparison_rows


def save_comparison_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Save comparison rows to one CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "metric_group",
        "metric_name",
        "k",
        "baseline_run",
        "baseline_score",
        "compared_run",
        "compared_score",
        "delta",
        "relative_delta_percent",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def run_comparison(
    raw_run_specs: list[str],
    baseline_label: str,
    output_path: str | Path,
) -> Path:
    """
    Run comparison and save one CSV output.
    """
    run_specs = parse_run_specs(raw_run_specs)

    rows = compare_runs(
        run_specs=run_specs,
        baseline_label=baseline_label,
    )

    save_comparison_rows(
        rows=rows,
        output_path=output_path,
    )

    return Path(output_path)