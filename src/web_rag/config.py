"""Configuration for the ProvideQ Web RAG baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Tuple


EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_RESULT_TYPE = "core"
EUROPE_PMC_FORMAT = "json"

DEFAULT_PAGE_SIZE = 8
DEFAULT_TOP_K = 10
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_USER_AGENT = "ProvideQ-Web-RAG/0.1"

DEFAULT_MAX_QUERY_TERMS = 12
DEFAULT_MIN_KEYWORD_LENGTH = 3

DEFAULT_SNIPPET_WINDOW_SIZE = 3
DEFAULT_SNIPPET_STRIDE = 1
DEFAULT_MIN_SNIPPET_CHARS = 40
DEFAULT_MAX_SNIPPET_CHARS = 1200
DEFAULT_MIN_SNIPPET_WORD_COUNT = 10

DEFAULT_RANKER = "lexical"
SUPPORTED_RANKERS: Tuple[str, ...] = (
    "lexical",
    "bm25",
    "medcpt",
    "embedding",
    "dense",
    "medcpt_embedding",
    "hybrid",
    "medcpt_hybrid",
)

BM25_K1 = 1.5
BM25_B = 0.75

MEDCPT_QUERY_ENCODER_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_ENCODER_MODEL = "ncbi/MedCPT-Article-Encoder"
MEDCPT_CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"
MEDCPT_BATCH_SIZE = 8

HYBRID_LEXICAL_WEIGHT = 0.45
HYBRID_MEDCPT_WEIGHT = 0.55


@dataclass(frozen=True)
class Settings:
    europe_pmc_search_url: str = EUROPE_PMC_SEARCH_URL
    europe_pmc_result_type: str = EUROPE_PMC_RESULT_TYPE
    europe_pmc_format: str = EUROPE_PMC_FORMAT
    page_size: int = DEFAULT_PAGE_SIZE
    top_k: int = DEFAULT_TOP_K
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    user_agent: str = DEFAULT_USER_AGENT

    max_query_terms: int = DEFAULT_MAX_QUERY_TERMS
    min_keyword_length: int = DEFAULT_MIN_KEYWORD_LENGTH

    snippet_window_size: int = DEFAULT_SNIPPET_WINDOW_SIZE
    snippet_stride: int = DEFAULT_SNIPPET_STRIDE
    min_snippet_chars: int = DEFAULT_MIN_SNIPPET_CHARS
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS
    min_snippet_word_count: int = DEFAULT_MIN_SNIPPET_WORD_COUNT

    default_ranker: str = DEFAULT_RANKER
    supported_rankers: Tuple[str, ...] = SUPPORTED_RANKERS
    bm25_k1: float = BM25_K1
    bm25_b: float = BM25_B

    medcpt_query_encoder_model: str = MEDCPT_QUERY_ENCODER_MODEL
    medcpt_article_encoder_model: str = MEDCPT_ARTICLE_ENCODER_MODEL
    medcpt_cross_encoder_model: str = MEDCPT_CROSS_ENCODER_MODEL
    medcpt_batch_size: int = MEDCPT_BATCH_SIZE
    hybrid_lexical_weight: float = HYBRID_LEXICAL_WEIGHT
    hybrid_medcpt_weight: float = HYBRID_MEDCPT_WEIGHT

    @property
    def europe_pmc_url(self) -> str:
        return self.europe_pmc_search_url

    @property
    def europe_pmc_endpoint(self) -> str:
        return self.europe_pmc_search_url

    @property
    def timeout(self) -> int:
        return self.request_timeout

    @property
    def default_page_size(self) -> int:
        return self.page_size

    @property
    def default_top_k(self) -> int:
        return self.top_k

    @property
    def snippet_window(self) -> int:
        return self.snippet_window_size

    @property
    def window_size(self) -> int:
        return self.snippet_window_size

    @property
    def window_stride(self) -> int:
        return self.snippet_stride


WebRAGConfig = Settings
WebRAGSettings = Settings

_SETTINGS: Settings | None = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_settings(**overrides) -> Settings:
    global _SETTINGS

    if _SETTINGS is None:
        _SETTINGS = Settings(
            europe_pmc_search_url=os.getenv("EUROPE_PMC_SEARCH_URL", EUROPE_PMC_SEARCH_URL),
            page_size=_env_int("WEB_RAG_PAGE_SIZE", DEFAULT_PAGE_SIZE),
            top_k=_env_int("WEB_RAG_TOP_K", DEFAULT_TOP_K),
            request_timeout=_env_int("WEB_RAG_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
            user_agent=os.getenv("WEB_RAG_USER_AGENT", DEFAULT_USER_AGENT),
            max_query_terms=_env_int("WEB_RAG_MAX_QUERY_TERMS", DEFAULT_MAX_QUERY_TERMS),
            min_keyword_length=_env_int("WEB_RAG_MIN_KEYWORD_LENGTH", DEFAULT_MIN_KEYWORD_LENGTH),
            snippet_window_size=_env_int("WEB_RAG_SNIPPET_WINDOW_SIZE", DEFAULT_SNIPPET_WINDOW_SIZE),
            snippet_stride=_env_int("WEB_RAG_SNIPPET_STRIDE", DEFAULT_SNIPPET_STRIDE),
            min_snippet_chars=_env_int("WEB_RAG_MIN_SNIPPET_CHARS", DEFAULT_MIN_SNIPPET_CHARS),
            max_snippet_chars=_env_int("WEB_RAG_MAX_SNIPPET_CHARS", DEFAULT_MAX_SNIPPET_CHARS),
            min_snippet_word_count=_env_int("WEB_RAG_MIN_SNIPPET_WORD_COUNT", DEFAULT_MIN_SNIPPET_WORD_COUNT),
            bm25_k1=_env_float("WEB_RAG_BM25_K1", BM25_K1),
            bm25_b=_env_float("WEB_RAG_BM25_B", BM25_B),
            medcpt_batch_size=_env_int("WEB_RAG_MEDCPT_BATCH_SIZE", MEDCPT_BATCH_SIZE),
            hybrid_lexical_weight=_env_float("WEB_RAG_HYBRID_LEXICAL_WEIGHT", HYBRID_LEXICAL_WEIGHT),
            hybrid_medcpt_weight=_env_float("WEB_RAG_HYBRID_MEDCPT_WEIGHT", HYBRID_MEDCPT_WEIGHT),
        )

    if overrides:
        return replace(_SETTINGS, **overrides)
    return _SETTINGS


DEFAULT_SETTINGS = get_settings()


__all__ = [
    "Settings",
    "WebRAGConfig",
    "WebRAGSettings",
    "get_settings",
    "DEFAULT_SETTINGS",
    "EUROPE_PMC_SEARCH_URL",
    "EUROPE_PMC_RESULT_TYPE",
    "EUROPE_PMC_FORMAT",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_TOP_K",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_USER_AGENT",
    "DEFAULT_MAX_QUERY_TERMS",
    "DEFAULT_MIN_KEYWORD_LENGTH",
    "DEFAULT_SNIPPET_WINDOW_SIZE",
    "DEFAULT_SNIPPET_STRIDE",
    "DEFAULT_MIN_SNIPPET_CHARS",
    "DEFAULT_MAX_SNIPPET_CHARS",
    "DEFAULT_MIN_SNIPPET_WORD_COUNT",
    "DEFAULT_RANKER",
    "SUPPORTED_RANKERS",
    "BM25_K1",
    "BM25_B",
    "MEDCPT_QUERY_ENCODER_MODEL",
    "MEDCPT_ARTICLE_ENCODER_MODEL",
    "MEDCPT_CROSS_ENCODER_MODEL",
    "MEDCPT_BATCH_SIZE",
    "HYBRID_LEXICAL_WEIGHT",
    "HYBRID_MEDCPT_WEIGHT",
]
