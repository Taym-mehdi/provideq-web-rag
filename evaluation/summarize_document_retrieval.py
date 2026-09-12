from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = (1, 3, 5, 10, 20)


def summarize_file(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows in {path}")

    n = len(rows)
    first = rows[0]
    summary: dict[str, object] = {
        "configuration": path.parent.name,
        "retriever": first.get("retriever", ""),
        "query_strategy": first.get("query_strategy", ""),
        "paperclip_ranking": first.get("paperclip_ranking", ""),
        "europepmc_mode": first.get("europepmc_mode", ""),
        "europepmc_synonym": first.get("europepmc_synonym", ""),
        "questions": n,
        "errors": sum(row.get("status") == "error" for row in rows),
    }
    for cutoff in METRICS:
        summary[f"recall_at_{cutoff}"] = round(
            sum(int(float(row.get(f"hit_at_{cutoff}", 0) or 0)) for row in rows) / n,
            6,
        )
    for cutoff in (10, 20):
        summary[f"mrr_at_{cutoff}"] = round(
            sum(
                1.0 / rank
                for row in rows
                if (rank := int(float(row.get("first_relevant_rank", 0) or 0)))
                and rank <= cutoff
            ) / n,
            6,
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize document retrieval runs.")
    parser.add_argument("--input-dir", default="outputs/document_retrieval")
    parser.add_argument(
        "--output",
        default="outputs/document_retrieval/summary.csv",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    files = sorted(root.glob("*/results.csv"))
    if not files:
        raise FileNotFoundError(f"No retrieval result files found under {root}")

    summaries = [summarize_file(path) for path in files]
    summaries.sort(
        key=lambda row: (
            -float(row["recall_at_20"]),
            -float(row["mrr_at_20"]),
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nDocument retrieval comparison")
    print(
        f"{'configuration':42s} {'R@1':>7s} {'R@3':>7s} "
        f"{'R@5':>7s} {'R@10':>7s} {'R@20':>7s} "
        f"{'MRR@10':>7s} {'MRR@20':>7s}"
    )
    for row in summaries:
        print(
            f"{str(row['configuration']):42.42s} "
            f"{float(row['recall_at_1']):7.4f} "
            f"{float(row['recall_at_3']):7.4f} "
            f"{float(row['recall_at_5']):7.4f} "
            f"{float(row['recall_at_10']):7.4f} "
            f"{float(row['recall_at_20']):7.4f} "
            f"{float(row['mrr_at_10']):7.4f} "
            f"{float(row['mrr_at_20']):7.4f}"
        )
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
