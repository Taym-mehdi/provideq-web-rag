from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .europepmc_retriever import paper_keys
from .models import Paper
from .text_utils import clean_text


@dataclass(frozen=True)
class MultiSourceRetrieval:
    papers: list[Paper]
    source_counts: dict[str, int]


def _fallback_keys(paper: Paper) -> list[str]:
    keys = paper_keys(paper)
    paper_id = clean_text(paper.paper_id)
    pmcid_match = re.search(r"\bPMC\d+\b", paper_id, flags=re.I)
    if pmcid_match:
        key = "pmcid:" + pmcid_match.group(0).upper()
        if key not in keys:
            keys.append(key)
    if paper_id:
        keys.append("paper_id:" + paper_id.casefold())
    return keys


def reciprocal_rank_fusion(
    rankings: dict[str, list[Paper]],
    *,
    limit: int = 10,
    rrf_k: int = 60,
    source_weights: dict[str, float] | None = None,
) -> MultiSourceRetrieval:
    if not rankings:
        return MultiSourceRetrieval(papers=[], source_counts={})
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    weights = source_weights or {}
    invalid_weights = {
        name: value
        for name, value in weights.items()
        if not isinstance(value, (int, float)) or value <= 0
    }
    if invalid_weights:
        raise ValueError("source weights must be positive numbers")

    entries: list[dict[str, Any]] = []
    key_to_index: dict[str, int] = {}

    for source_name, papers in rankings.items():
        source_weight = float(weights.get(source_name, 1.0))
        for fallback_rank, paper in enumerate(papers, start=1):
            rank = int(paper.retrieval_rank or fallback_rank)
            keys = _fallback_keys(paper)
            existing_index = next(
                (key_to_index[key] for key in keys if key in key_to_index),
                None,
            )
            if existing_index is None:
                existing_index = len(entries)
                entries.append(
                    {
                        "paper": paper,
                        "rrf": 0.0,
                        "best_rank": rank,
                        "source_ranks": {},
                        "sources": [],
                    }
                )
            entry = entries[existing_index]
            entry["rrf"] += source_weight / (rrf_k + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
            entry["source_ranks"][source_name] = rank
            entry.setdefault("source_weights", {})[source_name] = source_weight
            if source_name not in entry["sources"]:
                entry["sources"].append(source_name)

            # Prefer the richer representation when a duplicate is encountered.
            current: Paper = entry["paper"]
            current_richness = int(bool(current.abstract)) + int(bool(current.doi)) + int(bool(current.text))
            candidate_richness = int(bool(paper.abstract)) + int(bool(paper.doi)) + int(bool(paper.text))
            if candidate_richness > current_richness:
                entry["paper"] = paper

            for key in keys:
                key_to_index[key] = existing_index

    entries.sort(
        key=lambda entry: (
            -float(entry["rrf"]),
            -len(entry["sources"]),
            int(entry["best_rank"]),
            clean_text(entry["paper"].title).casefold(),
        )
    )

    fused: list[Paper] = []
    for rank, entry in enumerate(entries[:limit], start=1):
        paper: Paper = entry["paper"]
        paper.retrieval_rank = rank
        paper.source = "+".join(entry["sources"])
        paper.metadata["fusion_rrf_score"] = float(entry["rrf"])
        paper.metadata["fusion_sources"] = list(entry["sources"])
        paper.metadata["fusion_source_ranks"] = dict(entry["source_ranks"])
        paper.metadata["fusion_source_weights"] = dict(entry["source_weights"])
        fused.append(paper)

    return MultiSourceRetrieval(
        papers=fused,
        source_counts={name: len(papers) for name, papers in rankings.items()},
    )
