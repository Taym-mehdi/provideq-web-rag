from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:

    europe_pmc_search_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    user_agent: str = "ProvideQ-WebRAG/0.1"
    default_page_size: int = 8
    default_top_k: int = 5
    snippet_window: int = 2


def get_settings() -> Settings:
    
    return Settings()