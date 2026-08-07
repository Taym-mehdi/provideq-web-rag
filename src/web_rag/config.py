from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


# Load project-local configuration once. Existing shell variables keep priority.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _dotenv_path in (Path.cwd() / ".env", _PROJECT_ROOT / ".env"):
    if _dotenv_path.is_file():
        load_dotenv(dotenv_path=_dotenv_path, override=False)
        break


PAPERCLIP_RANKINGS = ("bm25", "vector", "hybrid")
PAPERCLIP_ACADEMIC_SOURCES = (
    "all",
    "pmc",
    "biorxiv",
    "medrxiv",
    "arxiv",
    "abstracts_only",
)
PAPERCLIP_MODES = ("any", "all", "50%", "75%", "phrase")
QUERY_STRATEGIES = ("raw", "hyde", "llmexpand")
LLM_PROVIDERS = ("ollama", "openai")
CHUNKING_METHODS = ("sentence_window",)
RERANKERS = ("lexical", "medcpt", "hybrid")


@dataclass(frozen=True)
class Settings:
    retrieval_limit: int = 10
    query_strategy: str = "raw"

    # Shared LLM settings. Interweb is OpenAI-compatible.
    llm_provider: str = "openai"
    llm_model: str = "agents-a1:35b-a3b"
    llm_base_url: str = "https://interweb.l3s.uni-hannover.de"
    llm_api_key_env: str = "INTERWEB_APIKEY"

    # Strategy-specific overrides. By default both use the shared model/URL.
    hyde_model: str = "agents-a1:35b-a3b"
    hyde_base_url: str = "https://interweb.l3s.uni-hannover.de"
    hyde_temperature: float = 0.0
    hyde_max_tokens: int = 180
    hyde_seed: int = 42
    hyde_timeout: float = 300.0

    expansion_model: str = "agents-a1:35b-a3b"
    expansion_base_url: str = "https://interweb.l3s.uni-hannover.de"
    expansion_temperature: float = 0.0
    expansion_max_tokens: int = 160
    expansion_seed: int = 42
    expansion_timeout: float = 300.0
    expansion_max_terms: int = 8
    expansion_max_query_chars: int = 600

    paperclip_source: str = "pmc,biorxiv,medrxiv,arxiv,abstracts_only"
    paperclip_ranking: str = "hybrid"
    paperclip_max_full_text_lines: int = 5000
    paperclip_timeout: float = 120.0
    paperclip_full_corpus: bool = True

    chunking_method: str = "sentence_window"
    chunk_window_size: int = 3
    chunk_stride: int = 1
    min_chunk_chars: int = 60
    max_chunk_chars: int = 1200
    min_chunk_words: int = 10
    context_backoff: bool = True

    reranker: str = "hybrid"
    top_k: int = 5
    max_chunks_per_paper: int = 2
    near_duplicate_threshold: float = 0.80

    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    medcpt_query_model: str = "ncbi/MedCPT-Query-Encoder"
    medcpt_article_model: str = "ncbi/MedCPT-Article-Encoder"
    medcpt_batch_size: int = 8
    medcpt_device: str = "auto"

    hybrid_lexical_weight: float = 0.30
    hybrid_medcpt_weight: float = 0.70


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    llm_model = os.getenv("WEB_RAG_LLM_MODEL", "agents-a1:35b-a3b")
    llm_base_url = os.getenv(
        "WEB_RAG_LLM_BASE_URL",
        "https://interweb.l3s.uni-hannover.de",
    )
    llm_timeout = _env_float("WEB_RAG_LLM_TIMEOUT", 300.0)
    llm_temperature = _env_float("WEB_RAG_LLM_TEMPERATURE", 0.0)

    return Settings(
        retrieval_limit=_env_int("WEB_RAG_RETRIEVAL_LIMIT", 10),
        query_strategy=os.getenv("WEB_RAG_QUERY_STRATEGY", "raw"),
        llm_provider=os.getenv("WEB_RAG_LLM_PROVIDER", "openai"),
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key_env=os.getenv(
            "WEB_RAG_LLM_API_KEY_ENV",
            "INTERWEB_APIKEY",
        ),
        hyde_model=os.getenv("WEB_RAG_HYDE_MODEL", llm_model),
        hyde_base_url=os.getenv("WEB_RAG_HYDE_BASE_URL", llm_base_url),
        hyde_temperature=_env_float(
            "WEB_RAG_HYDE_TEMPERATURE",
            llm_temperature,
        ),
        hyde_max_tokens=_env_int("WEB_RAG_HYDE_MAX_TOKENS", 180),
        hyde_seed=_env_int("WEB_RAG_HYDE_SEED", 42),
        hyde_timeout=_env_float("WEB_RAG_HYDE_TIMEOUT", llm_timeout),
        expansion_model=os.getenv("WEB_RAG_EXPANSION_MODEL", llm_model),
        expansion_base_url=os.getenv(
            "WEB_RAG_EXPANSION_BASE_URL",
            llm_base_url,
        ),
        expansion_temperature=_env_float(
            "WEB_RAG_EXPANSION_TEMPERATURE",
            llm_temperature,
        ),
        expansion_max_tokens=_env_int("WEB_RAG_EXPANSION_MAX_TOKENS", 160),
        expansion_seed=_env_int("WEB_RAG_EXPANSION_SEED", 42),
        expansion_timeout=_env_float(
            "WEB_RAG_EXPANSION_TIMEOUT",
            llm_timeout,
        ),
        expansion_max_terms=_env_int("WEB_RAG_EXPANSION_MAX_TERMS", 8),
        expansion_max_query_chars=_env_int(
            "WEB_RAG_EXPANSION_MAX_QUERY_CHARS",
            600,
        ),
        paperclip_source=os.getenv(
            "WEB_RAG_PAPERCLIP_SOURCE",
            "pmc,biorxiv,medrxiv,arxiv,abstracts_only",
        ),
        paperclip_ranking=os.getenv("WEB_RAG_PAPERCLIP_RANKING", "hybrid"),
        paperclip_max_full_text_lines=_env_int(
            "WEB_RAG_PAPERCLIP_MAX_LINES",
            5000,
        ),
        paperclip_timeout=_env_float("WEB_RAG_PAPERCLIP_TIMEOUT", 120.0),
        paperclip_full_corpus=_env_bool(
            "WEB_RAG_PAPERCLIP_FULL_CORPUS",
            True,
        ),
        chunking_method=os.getenv(
            "WEB_RAG_CHUNKING_METHOD",
            "sentence_window",
        ),
        chunk_window_size=_env_int("WEB_RAG_CHUNK_WINDOW_SIZE", 3),
        chunk_stride=_env_int("WEB_RAG_CHUNK_STRIDE", 1),
        min_chunk_chars=_env_int("WEB_RAG_MIN_CHUNK_CHARS", 60),
        max_chunk_chars=_env_int("WEB_RAG_MAX_CHUNK_CHARS", 1200),
        min_chunk_words=_env_int("WEB_RAG_MIN_CHUNK_WORDS", 10),
        context_backoff=_env_bool("WEB_RAG_CONTEXT_BACKOFF", True),
        reranker=os.getenv("WEB_RAG_RERANKER", "hybrid"),
        top_k=_env_int("WEB_RAG_TOP_K", 5),
        max_chunks_per_paper=_env_int("WEB_RAG_MAX_CHUNKS_PER_PAPER", 2),
        near_duplicate_threshold=_env_float(
            "WEB_RAG_NEAR_DUPLICATE_THRESHOLD",
            0.80,
        ),
        bm25_k1=_env_float("WEB_RAG_BM25_K1", 1.5),
        bm25_b=_env_float("WEB_RAG_BM25_B", 0.75),
        medcpt_query_model=os.getenv(
            "WEB_RAG_MEDCPT_QUERY_MODEL",
            "ncbi/MedCPT-Query-Encoder",
        ),
        medcpt_article_model=os.getenv(
            "WEB_RAG_MEDCPT_ARTICLE_MODEL",
            "ncbi/MedCPT-Article-Encoder",
        ),
        medcpt_batch_size=_env_int("WEB_RAG_MEDCPT_BATCH_SIZE", 8),
        medcpt_device=os.getenv("WEB_RAG_MEDCPT_DEVICE", "auto"),
        hybrid_lexical_weight=_env_float(
            "WEB_RAG_HYBRID_LEXICAL_WEIGHT",
            0.30,
        ),
        hybrid_medcpt_weight=_env_float(
            "WEB_RAG_HYBRID_MEDCPT_WEIGHT",
            0.70,
        ),
    )


def validate_paperclip_source(value: str) -> None:
    sources = [item.strip().casefold() for item in value.split(",") if item.strip()]
    if not sources:
        raise ValueError("paperclip_source must contain at least one source")
    unsupported = [item for item in sources if item not in PAPERCLIP_ACADEMIC_SOURCES]
    if unsupported:
        choices = ", ".join(PAPERCLIP_ACADEMIC_SOURCES)
        raise ValueError(
            f"Unsupported Paperclip source: {', '.join(unsupported)}. "
            f"Choose from: {choices}"
        )


def validate_settings(settings: Settings) -> None:
    validate_paperclip_source(settings.paperclip_source)
    if settings.paperclip_ranking not in PAPERCLIP_RANKINGS:
        raise ValueError(
            f"paperclip_ranking must be one of: {', '.join(PAPERCLIP_RANKINGS)}"
        )
    if settings.query_strategy not in QUERY_STRATEGIES:
        raise ValueError(
            f"query_strategy must be one of: {', '.join(QUERY_STRATEGIES)}"
        )
    if settings.llm_provider not in LLM_PROVIDERS:
        raise ValueError(
            f"llm_provider must be one of: {', '.join(LLM_PROVIDERS)}"
        )
    if not settings.llm_model.strip():
        raise ValueError("llm_model must not be empty")
    if not settings.llm_base_url.strip():
        raise ValueError("llm_base_url must not be empty")
    if not settings.llm_api_key_env.strip():
        raise ValueError("llm_api_key_env must not be empty")
    if not settings.hyde_model.strip():
        raise ValueError("hyde_model must not be empty")
    if not settings.hyde_base_url.strip():
        raise ValueError("hyde_base_url must not be empty")
    if not 0 <= settings.hyde_temperature <= 2:
        raise ValueError("hyde_temperature must be between 0 and 2")
    if settings.hyde_max_tokens <= 0:
        raise ValueError("hyde_max_tokens must be greater than 0")
    if settings.hyde_timeout <= 0:
        raise ValueError("hyde_timeout must be greater than 0")
    if not settings.expansion_model.strip():
        raise ValueError("expansion_model must not be empty")
    if not settings.expansion_base_url.strip():
        raise ValueError("expansion_base_url must not be empty")
    if not 0 <= settings.expansion_temperature <= 2:
        raise ValueError("expansion_temperature must be between 0 and 2")
    if settings.expansion_max_tokens <= 0:
        raise ValueError("expansion_max_tokens must be greater than 0")
    if settings.expansion_timeout <= 0:
        raise ValueError("expansion_timeout must be greater than 0")
    if settings.expansion_max_terms <= 0:
        raise ValueError("expansion_max_terms must be greater than 0")
    if settings.expansion_max_query_chars < 100:
        raise ValueError("expansion_max_query_chars must be at least 100")
    if settings.chunking_method not in CHUNKING_METHODS:
        raise ValueError(
            f"chunking_method must be one of: {', '.join(CHUNKING_METHODS)}"
        )
    if settings.reranker not in RERANKERS:
        raise ValueError(f"reranker must be one of: {', '.join(RERANKERS)}")
    if settings.retrieval_limit <= 0 or settings.retrieval_limit > 1000:
        raise ValueError("retrieval_limit must be between 1 and 1000")
    if settings.paperclip_max_full_text_lines <= 0:
        raise ValueError("paperclip_max_full_text_lines must be greater than 0")
    if settings.paperclip_timeout <= 0:
        raise ValueError("paperclip_timeout must be greater than 0")
    if settings.chunk_window_size <= 0 or settings.chunk_stride <= 0:
        raise ValueError("chunk window and stride must be greater than 0")
    if settings.min_chunk_chars < 0 or settings.min_chunk_words < 0:
        raise ValueError("minimum chunk thresholds must be non-negative")
    if settings.max_chunk_chars < settings.min_chunk_chars:
        raise ValueError(
            "max_chunk_chars must be greater than or equal to min_chunk_chars"
        )
    if settings.top_k <= 0 or settings.max_chunks_per_paper <= 0:
        raise ValueError("top_k and max_chunks_per_paper must be greater than 0")
    if not 0 <= settings.near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    if settings.bm25_k1 <= 0 or not 0 <= settings.bm25_b <= 1:
        raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")
    if settings.medcpt_batch_size <= 0:
        raise ValueError("medcpt_batch_size must be greater than 0")
    if settings.hybrid_lexical_weight < 0 or settings.hybrid_medcpt_weight < 0:
        raise ValueError("hybrid weights must be non-negative")
    if settings.hybrid_lexical_weight + settings.hybrid_medcpt_weight <= 0:
        raise ValueError("at least one hybrid weight must be greater than 0")
