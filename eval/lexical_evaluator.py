from __future__ import annotations

# Paper basis:
# Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
# Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eval.benchmark_loader import load_benchmark
from eval.evaluation_writer import append_evaluation_rows


SUPPORTED_LEXICAL_METRICS = ("rouge1", "rougel", "rouge", "bm25")
TEXT_KEYS = ("evidence_text", "text", "snippet", "snippet_text", "content", "passage")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge1_recall(reference: str, candidate: str) -> float:
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum(min(count, cand_counts[token]) for token, count in ref_counts.items())
    return overlap / len(ref_tokens)


def rougel_recall(reference: str, candidate: str) -> float:
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    return _lcs_length(ref_tokens, cand_tokens) / len(ref_tokens)


def _bm25_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n_docs = len(corpus_tokens)
    doc_freq: Counter[str] = Counter()
    for tokens in corpus_tokens:
        doc_freq.update(set(tokens))
    return {
        term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], corpus_tokens: list[list[str]], *, k1: float, b: float) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    idf = _bm25_idf(corpus_tokens)
    frequencies = Counter(doc_tokens)
    avg_doc_len = sum(len(tokens) for tokens in corpus_tokens) / max(len(corpus_tokens), 1)
    doc_len = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        tf = frequencies.get(term, 0)
        if tf == 0:
            continue
        denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1e-9))
        score += idf.get(term, 0.0) * (tf * (k1 + 1)) / denominator
    return score


def bm25_coverage(reference: str, candidates: list[str], *, k1: float, b: float) -> float:
    query_tokens = _tokenize(reference)
    candidate_tokens = [_tokenize(candidate) for candidate in candidates]
    candidate_tokens = [tokens for tokens in candidate_tokens if tokens]
    if not query_tokens or not candidate_tokens:
        return 0.0

    ideal_doc_tokens = query_tokens
    corpus_tokens = candidate_tokens + [ideal_doc_tokens]
    best_candidate = max(
        _bm25_score(query_tokens, doc_tokens, corpus_tokens, k1=k1, b=b)
        for doc_tokens in candidate_tokens
    )
    ideal_score = _bm25_score(query_tokens, ideal_doc_tokens, corpus_tokens, k1=k1, b=b)
    if ideal_score <= 0:
        return 0.0
    return min(best_candidate / ideal_score, 1.0)


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
        return list(SUPPORTED_LEXICAL_METRICS)
    requested = [item.strip().lower() for item in metric_string.split(",") if item.strip()]
    invalid = [metric for metric in requested if metric not in SUPPORTED_LEXICAL_METRICS]
    if invalid:
        raise ValueError(f"Unsupported lexical metrics: {', '.join(invalid)}")
    return requested


def _score_nugget(metric: str, nugget: str, snippets: list[str], *, bm25_k1: float, bm25_b: float) -> float:
    if not snippets:
        return 0.0
    if metric == "rouge1":
        return max(rouge1_recall(nugget, snippet) for snippet in snippets)
    if metric == "rougel":
        return max(rougel_recall(nugget, snippet) for snippet in snippets)
    if metric == "rouge":
        rouge1 = max(rouge1_recall(nugget, snippet) for snippet in snippets)
        rougel = max(rougel_recall(nugget, snippet) for snippet in snippets)
        return (rouge1 + rougel) / 2
    if metric == "bm25":
        return bm25_coverage(nugget, snippets, k1=bm25_k1, b=bm25_b)
    raise ValueError(f"Unsupported metric: {metric}")


def evaluate_lexical(
    *,
    benchmark_path: str | Path,
    retrieval_results_path: str | Path,
    run_name: str,
    evaluation_root: str | Path,
    metrics: str,
    k: int,
    bm25_k1: float,
    bm25_b: float,
) -> Path:
    examples = load_benchmark(benchmark_path)
    retrieval_results = _load_retrieval_results(retrieval_results_path)
    selected_metrics = _parse_metrics(metrics)

    rows: list[dict[str, Any]] = []
    total_nuggets = sum(len(example.gold_nuggets) for example in examples)

    for metric in selected_metrics:
        nugget_scores: list[float] = []
        questions_with_evidence = 0

        for example in examples:
            result_row = retrieval_results.get(example.question_id, {})
            records = result_row.get("records", []) if isinstance(result_row, dict) else []
            snippets = [_record_text(record) for record in records[:k]]
            snippets = [snippet for snippet in snippets if snippet]
            if snippets:
                questions_with_evidence += 1

            for nugget in example.gold_nuggets:
                score = _score_nugget(metric, nugget, snippets, bm25_k1=bm25_k1, bm25_b=bm25_b)
                nugget_scores.append(score)

        metric_display = {
            "rouge1": "ROUGE1_Nugget",
            "rougel": "ROUGEL_Nugget",
            "rouge": "ROUGE_Nugget",
            "bm25": "BM25_Nugget",
        }[metric]
        evaluation_name = f"lexical_{metric_display}@{k}"
        score = sum(nugget_scores) / len(nugget_scores) if nugget_scores else 0.0
        rows.append(
            {
                "run_name": run_name,
                "ranker": run_name.removeprefix("ranker_"),
                "evaluation_name": evaluation_name,
                "layer": "lexical",
                "metric": metric_display,
                "k": k,
                "score": round(score, 6),
                "questions_evaluated": len(examples),
                "total_gold_nuggets": total_nuggets,
                "parameters_json": {
                    "metric": metric,
                    "k": k,
                    "bm25_k1": bm25_k1,
                    "bm25_b": bm25_b,
                    "questions_with_evidence": questions_with_evidence,
                    "aggregation": "mean over all gold nuggets; each nugget uses best matching top-k snippet",
                },
                "retrieval_results_path": str(retrieval_results_path),
                "benchmark_path": str(benchmark_path),
            }
        )

    return append_evaluation_rows(rows, evaluation_root=evaluation_root, run_name=run_name, layer="lexical")
