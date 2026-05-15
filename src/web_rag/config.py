from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """
    Central configuration for the project.
    """

    europe_pmc_search_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    user_agent: str = "ProvideQ-WebRAG/0.1"

    default_page_size: int = 8
    default_top_k: int = 5
    snippet_window: int = 2

    default_ranker: str = "medcpt-hybrid"

    medcpt_model_name: str = "ncbi/MedCPT-Cross-Encoder"
    medcpt_batch_size: int = 8
    medcpt_max_length: int = 512

    hybrid_weight_medcpt: float = 0.65
    hybrid_weight_lexical: float = 0.20
    hybrid_weight_slots: float = 0.15


def get_settings() -> Settings:
    return Settings()