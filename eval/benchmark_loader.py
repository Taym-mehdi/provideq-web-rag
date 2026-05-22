from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


NUGGET_SEPARATOR = "||"


@dataclass
class BenchmarkExample:
    """
    One benchmark question for Web RAG evaluation.

    Each example contains:
    - question_id: stable ID for reports
    - question: input question to run through the Web RAG pipeline
    - gold_answer: reference answer used for semantic answer matching and LLM judging
    - gold_nuggets: important evidence units that retrieved snippets should cover
    - notes: optional comments for manual tracking
    """

    question_id: str
    question: str
    gold_answer: str
    gold_nuggets: list[str]
    notes: str = ""


def split_gold_nuggets(raw_nuggets: str) -> list[str]:
    """
    Split the gold_nuggets CSV field into a list of nuggets.

    The CSV uses '||' as separator because nuggets may contain normal commas.
    """
    if not raw_nuggets:
        return []

    nuggets = [
        nugget.strip()
        for nugget in raw_nuggets.split(NUGGET_SEPARATOR)
        if nugget.strip()
    ]

    return nuggets


def validate_example(example: BenchmarkExample) -> None:
    """
    Validate that a benchmark example has the required fields.

    We fail early here because evaluation results become meaningless if the
    benchmark is malformed.
    """
    if not example.question_id:
        raise ValueError("Benchmark row is missing question_id.")

    if not example.question:
        raise ValueError(f"Benchmark row {example.question_id} is missing question.")

    if not example.gold_answer:
        raise ValueError(f"Benchmark row {example.question_id} is missing gold_answer.")

    if not example.gold_nuggets:
        raise ValueError(f"Benchmark row {example.question_id} has no gold_nuggets.")


def load_benchmark(csv_path: str | Path) -> list[BenchmarkExample]:
    """
    Load benchmark examples from CSV.

    Required CSV columns:
    - question_id
    - question
    - gold_answer
    - gold_nuggets

    Optional CSV columns:
    - notes
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")

    examples: list[BenchmarkExample] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "question_id",
            "question",
            "gold_answer",
            "gold_nuggets",
        }

        if reader.fieldnames is None:
            raise ValueError("Benchmark CSV has no header row.")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Benchmark CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            example = BenchmarkExample(
                question_id=(row.get("question_id") or "").strip(),
                question=(row.get("question") or "").strip(),
                gold_answer=(row.get("gold_answer") or "").strip(),
                gold_nuggets=split_gold_nuggets(row.get("gold_nuggets") or ""),
                notes=(row.get("notes") or "").strip(),
            )

            validate_example(example)
            examples.append(example)

    return examples


def summarize_benchmark(examples: list[BenchmarkExample]) -> dict[str, int]:
    """
    Return simple summary statistics for the benchmark.
    """
    total_questions = len(examples)
    total_nuggets = sum(len(example.gold_nuggets) for example in examples)

    if total_questions == 0:
        average_nuggets_per_question = 0
    else:
        average_nuggets_per_question = round(total_nuggets / total_questions, 2)

    return {
        "questions": total_questions,
        "gold_nuggets": total_nuggets,
        "avg_nuggets_per_question": average_nuggets_per_question,
    }