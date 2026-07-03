from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EVALUATION_FIELDS = [
    "timestamp",
    "layer",
    "evaluation_name",
    "metric",
    "ranker",
    "run_name",
    "k",
    "score",
    "questions_evaluated",
    "total_gold_nuggets",
    "parameters_json",
    "retrieval_results_path",
    "benchmark_path",
]


def _ranker_from_run_name(run_name: str) -> str:
    name = (run_name or "").strip()
    if name.startswith("ranker_"):
        return name.removeprefix("ranker_")
    return name


def append_evaluation_rows(
    rows: list[dict[str, Any]],
    *,
    evaluation_root: str | Path,
    run_name: str | None = None,
    layer: str | None = None,
) -> Path:
    if not rows:
        raise ValueError("No evaluation rows to write.")

    selected_layer = layer or str(rows[0].get("layer") or "evaluation").strip().lower()
    output_dir = Path(evaluation_root) / selected_layer
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_results.csv"
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    timestamp = datetime.now().isoformat(timespec="seconds")
    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVALUATION_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field, "") for field in EVALUATION_FIELDS}
            normalized["timestamp"] = normalized["timestamp"] or timestamp
            normalized["layer"] = normalized["layer"] or selected_layer
            normalized["run_name"] = normalized["run_name"] or run_name or ""
            normalized["ranker"] = normalized["ranker"] or _ranker_from_run_name(str(normalized["run_name"]))
            if not isinstance(normalized.get("parameters_json"), str):
                normalized["parameters_json"] = json.dumps(normalized["parameters_json"], ensure_ascii=False)
            writer.writerow(normalized)

    return output_path
