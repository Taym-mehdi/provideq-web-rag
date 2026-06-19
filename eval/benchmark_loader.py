from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


NUGGET_SEPARATOR = "||"


@dataclass(frozen=True)
class BenchmarkExample:
    question_id: str
    question: str
    gold_answer: str
    gold_nuggets: list[str]
    notes: str = ""


def split_gold_nuggets(raw_value: str) -> list[str]:
    return [part.strip() for part in (raw_value or "").split(NUGGET_SEPARATOR) if part.strip()]


def load_benchmark(path: str | Path) -> list[BenchmarkExample]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    examples: list[BenchmarkExample] = []
    with benchmark_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"question_id", "question", "gold_answer", "gold_nuggets"}
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Benchmark is missing columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            example = BenchmarkExample(
                question_id=(row.get("question_id") or "").strip(),
                question=(row.get("question") or "").strip(),
                gold_answer=(row.get("gold_answer") or "").strip(),
                gold_nuggets=split_gold_nuggets(row.get("gold_nuggets") or ""),
                notes=(row.get("notes") or "").strip(),
            )
            if not example.question_id:
                raise ValueError(f"Benchmark row {row_number} is missing question_id.")
            if not example.question:
                raise ValueError(f"Benchmark row {row_number} is missing question.")
            if not example.gold_answer:
                raise ValueError(f"Benchmark row {row_number} is missing gold_answer.")
            if not example.gold_nuggets:
                raise ValueError(f"Benchmark row {row_number} is missing gold_nuggets.")
            examples.append(example)

    return examples


def benchmark_summary(examples: list[BenchmarkExample]) -> dict[str, float | int]:
    questions = len(examples)
    nuggets = sum(len(example.gold_nuggets) for example in examples)
    return {
        "questions": questions,
        "gold_nuggets": nuggets,
        "avg_nuggets_per_question": round(nuggets / questions, 2) if questions else 0,
    }
