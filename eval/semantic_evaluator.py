from __future__ import annotations

# Based on:
# Reimers and Gurevych (2019), "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks".
# Zhang et al. (2020), "BERTScore: Evaluating Text Generation with BERT".

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.benchmark_loader import load_benchmark
from eval.evaluation_writer import append_evaluation_rows


SUPPORTED_SEMANTIC_METRICS = ("nugget", "answer")
TEXT_KEYS = ("evidence_text", "text", "snippet_text", "content")
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _safe_json_loads(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _record_text(record: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = record.get(key)
        if value:
            return str(value).strip()
    return ""


def _load_retrieval_results(path: str | Path) -> dict[str, dict[str, Any]]:
    results_path = Path(path)
    if not results_path.exists():
        raise FileNotFoundError(f"Retrieval results CSV not found: {results_path}")

    output: dict[str, dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            question_id = (row.get("question_id") or "").strip()
            if not question_id:
                continue
            records = _safe_json_loads(row.get("records_json") or "", [])
            if not isinstance(records, list):
                records = []
            row["records"] = [record for record in records if isinstance(record, dict)]
            output[question_id] = row
    return output


def _parse_metrics(metric_string: str) -> list[str]:
    if not metric_string or metric_string.lower() == "all":
        return list(SUPPORTED_SEMANTIC_METRICS)
    requested = [item.strip().lower() for item in metric_string.split(",") if item.strip()]
    aliases = {
        "semanticnuggetmatch": "nugget",
        "semantic_nugget": "nugget",
        "semantic_nugget_match": "nugget",
        "nugget": "nugget",
        "semanticanswermatch": "answer",
        "semantic_answer": "answer",
        "semantic_answer_match": "answer",
        "answer": "answer",
    }
    normalized = [aliases.get(metric, metric) for metric in requested]
    invalid = [metric for metric in normalized if metric not in SUPPORTED_SEMANTIC_METRICS]
    if invalid:
        raise ValueError(f"Unsupported semantic metrics: {', '.join(invalid)}")
    return normalized


def _load_sentence_transformer(model_name: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for semantic evaluation. "
            "Install it with: pip install sentence-transformers"
        ) from error

    if device:
        return SentenceTransformer(model_name, device=device)
    return SentenceTransformer(model_name)


def _encode(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.matmul(left, right.T)


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _semantic_nugget_score(model: Any, nuggets: list[str], snippets: list[str], batch_size: int) -> list[float]:
    if not nuggets:
        return []
    if not snippets:
        return [0.0 for _ in nuggets]

    nugget_embeddings = _encode(model, nuggets, batch_size)
    snippet_embeddings = _encode(model, snippets, batch_size)
    similarities = _cosine_matrix(nugget_embeddings, snippet_embeddings)
    if similarities.size == 0:
        return [0.0 for _ in nuggets]
    return [_clip_score(score) for score in similarities.max(axis=1).tolist()]


def _semantic_answer_score(model: Any, gold_answer: str, snippets: list[str], batch_size: int) -> float:
    context = " ".join(snippet.strip() for snippet in snippets if snippet.strip())
    if not gold_answer.strip() or not context.strip():
        return 0.0
    embeddings = _encode(model, [gold_answer, context], batch_size)
    if embeddings.shape[0] != 2:
        return 0.0
    return _clip_score(np.matmul(embeddings[0], embeddings[1].T))


def evaluate_semantic(
    *,
    benchmark_path: str | Path,
    retrieval_results_path: str | Path,
    run_name: str,
    evaluation_root: str | Path,
    metrics: str,
    k: int,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> Path:
    examples = load_benchmark(benchmark_path)
    retrieval_results = _load_retrieval_results(retrieval_results_path)
    selected_metrics = _parse_metrics(metrics)
    model = _load_sentence_transformer(model_name, device)

    rows: list[dict[str, Any]] = []
    total_nuggets = sum(len(example.gold_nuggets) for example in examples)

    for metric in selected_metrics:
        scores: list[float] = []
        questions_with_evidence = 0

        for example in examples:
            result_row = retrieval_results.get(example.question_id, {})
            records = result_row.get("records", []) if isinstance(result_row, dict) else []
            snippets = [_record_text(record) for record in records[:k]]
            snippets = [snippet for snippet in snippets if snippet]
            if snippets:
                questions_with_evidence += 1

            if metric == "nugget":
                scores.extend(_semantic_nugget_score(model, example.gold_nuggets, snippets, batch_size))
            elif metric == "answer":
                scores.append(_semantic_answer_score(model, example.gold_answer, snippets, batch_size))
            else:
                raise ValueError(f"Unsupported semantic metric: {metric}")

        metric_display = {
            "nugget": "SemanticNuggetMatch",
            "answer": "SemanticAnswerMatch",
        }[metric]
        evaluation_name = f"semantic_{metric_display}@{k}"
        score = sum(scores) / len(scores) if scores else 0.0
        rows.append(
            {
                "run_name": run_name,
                "ranker": run_name.removeprefix("ranker_"),
                "evaluation_name": evaluation_name,
                "layer": "semantic",
                "metric": metric_display,
                "k": k,
                "score": round(score, 6),
                "questions_evaluated": len(examples),
                "total_gold_nuggets": total_nuggets,
                "parameters_json": {
                    "metric": metric,
                    "k": k,
                    "model_name": model_name,
                    "device": device,
                    "batch_size": batch_size,
                    "score_range": "cosine similarity clipped to [0, 1]",
                    "questions_with_evidence": questions_with_evidence,
                    "aggregation": (
                        "nugget: mean over all gold nuggets using best matching top-k snippet; "
                        "answer: mean over questions using gold answer vs concatenated top-k snippets"
                    ),
                },
                "retrieval_results_path": str(retrieval_results_path),
                "benchmark_path": str(benchmark_path),
            }
        )

    return append_evaluation_rows(rows, evaluation_root=evaluation_root, run_name=run_name, layer="semantic")
