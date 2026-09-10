from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False


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
CHUNKING_METHODS = ("token_aware",)
RERANKERS = ("medcpt", "lexical", "hybrid")

DEFAULT_MEDCPT_MODEL = "ncbi/MedCPT-Cross-Encoder"


@dataclass(frozen=True)
class Settings:
    # Best current document-retrieval setting from the completed comparison.
    retrieval_limit: int = 10
    query_strategy: str = "raw"

    # Optional HyDE and LLM-expansion experiments.
    llm_provider: str = "openai"
    llm_model: str = "agents-a1:35b-a3b"
    llm_base_url: str = "https://interweb.l3s.uni-hannover.de"
    llm_api_key_env: str = "INTERWEB_APIKEY"

    hyde_model: str = "agents-a1:35b-a3b"
    hyde_base_url: str = "https://interweb.l3s.uni-hannover.de"
    hyde_temperature: float = 0.0
    hyde_max_tokens: int = 100
    hyde_seed: int = 42
    hyde_timeout: float = 300.0

    expansion_model: str = "agents-a1:35b-a3b"
    expansion_base_url: str = "https://interweb.l3s.uni-hannover.de"
    expansion_temperature: float = 0.0
    expansion_max_tokens: int = 120
    expansion_seed: int = 42
    expansion_timeout: float = 300.0
    expansion_max_terms: int = 4
    expansion_max_query_chars: int = 400

    # Paperclip performs first-stage Web retrieval.
    paperclip_source: str = "pmc,biorxiv,medrxiv,arxiv,abstracts_only"
    paperclip_ranking: str = "hybrid"
    paperclip_max_full_text_lines: int = 5000
    paperclip_timeout: float = 120.0
    paperclip_full_corpus: bool = True

    # Mirrors Aryan's local token-aware chunking settings for plain Paperclip text.
    chunking_method: str = "token_aware"
    chunk_tokenizer_model: str = DEFAULT_MEDCPT_MODEL
    chunk_max_tokens: int = 512
    chunk_overlap_fraction: float = 0.20
    chunk_max_overlap_sentences: int = 5
    min_chunk_words: int = 5

    # MedCPT cross-encoder is the default final chunk reranker.
    reranker: str = "medcpt"
    top_k: int = 20
    max_chunks_per_paper: int = 4
    near_duplicate_threshold: float = 0.90

    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    medcpt_model: str = DEFAULT_MEDCPT_MODEL
    medcpt_max_length: int = 512
    medcpt_batch_size: int = 8
    medcpt_device: str = "auto"

    # Kept as an optional comparison, not the default.
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
    defaults = Settings()
    llm_model = os.getenv("WEB_RAG_LLM_MODEL", defaults.llm_model)
    llm_base_url = os.getenv("WEB_RAG_LLM_BASE_URL", defaults.llm_base_url)
    llm_timeout = _env_float("WEB_RAG_LLM_TIMEOUT", defaults.hyde_timeout)
    llm_temperature = _env_float(
        "WEB_RAG_LLM_TEMPERATURE",
        defaults.hyde_temperature,
    )

    return Settings(
        retrieval_limit=_env_int(
            "WEB_RAG_RETRIEVAL_LIMIT",
            defaults.retrieval_limit,
        ),
        query_strategy=os.getenv(
            "WEB_RAG_QUERY_STRATEGY",
            defaults.query_strategy,
        ),
        llm_provider=os.getenv("WEB_RAG_LLM_PROVIDER", defaults.llm_provider),
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key_env=os.getenv(
            "WEB_RAG_LLM_API_KEY_ENV",
            defaults.llm_api_key_env,
        ),
        hyde_model=os.getenv("WEB_RAG_HYDE_MODEL", llm_model),
        hyde_base_url=os.getenv("WEB_RAG_HYDE_BASE_URL", llm_base_url),
        hyde_temperature=_env_float(
            "WEB_RAG_HYDE_TEMPERATURE",
            llm_temperature,
        ),
        hyde_max_tokens=_env_int(
            "WEB_RAG_HYDE_MAX_TOKENS",
            defaults.hyde_max_tokens,
        ),
        hyde_seed=_env_int("WEB_RAG_HYDE_SEED", defaults.hyde_seed),
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
        expansion_max_tokens=_env_int(
            "WEB_RAG_EXPANSION_MAX_TOKENS",
            defaults.expansion_max_tokens,
        ),
        expansion_seed=_env_int(
            "WEB_RAG_EXPANSION_SEED",
            defaults.expansion_seed,
        ),
        expansion_timeout=_env_float(
            "WEB_RAG_EXPANSION_TIMEOUT",
            llm_timeout,
        ),
        expansion_max_terms=_env_int(
            "WEB_RAG_EXPANSION_MAX_TERMS",
            defaults.expansion_max_terms,
        ),
        expansion_max_query_chars=_env_int(
            "WEB_RAG_EXPANSION_MAX_QUERY_CHARS",
            defaults.expansion_max_query_chars,
        ),
        paperclip_source=os.getenv(
            "WEB_RAG_PAPERCLIP_SOURCE",
            defaults.paperclip_source,
        ),
        paperclip_ranking=os.getenv(
            "WEB_RAG_PAPERCLIP_RANKING",
            defaults.paperclip_ranking,
        ),
        paperclip_max_full_text_lines=_env_int(
            "WEB_RAG_PAPERCLIP_MAX_LINES",
            defaults.paperclip_max_full_text_lines,
        ),
        paperclip_timeout=_env_float(
            "WEB_RAG_PAPERCLIP_TIMEOUT",
            defaults.paperclip_timeout,
        ),
        paperclip_full_corpus=_env_bool(
            "WEB_RAG_PAPERCLIP_FULL_CORPUS",
            defaults.paperclip_full_corpus,
        ),
        chunking_method=os.getenv(
            "WEB_RAG_CHUNKING_METHOD",
            defaults.chunking_method,
        ),
        chunk_tokenizer_model=os.getenv(
            "WEB_RAG_CHUNK_TOKENIZER_MODEL",
            defaults.chunk_tokenizer_model,
        ),
        chunk_max_tokens=_env_int(
            "WEB_RAG_CHUNK_MAX_TOKENS",
            defaults.chunk_max_tokens,
        ),
        chunk_overlap_fraction=_env_float(
            "WEB_RAG_CHUNK_OVERLAP",
            defaults.chunk_overlap_fraction,
        ),
        chunk_max_overlap_sentences=_env_int(
            "WEB_RAG_CHUNK_MAX_OVERLAP_SENTENCES",
            defaults.chunk_max_overlap_sentences,
        ),
        min_chunk_words=_env_int(
            "WEB_RAG_MIN_CHUNK_WORDS",
            defaults.min_chunk_words,
        ),
        reranker=os.getenv("WEB_RAG_RERANKER", defaults.reranker),
        top_k=_env_int("WEB_RAG_TOP_K", defaults.top_k),
        max_chunks_per_paper=_env_int(
            "WEB_RAG_MAX_CHUNKS_PER_PAPER",
            defaults.max_chunks_per_paper,
        ),
        near_duplicate_threshold=_env_float(
            "WEB_RAG_NEAR_DUPLICATE_THRESHOLD",
            defaults.near_duplicate_threshold,
        ),
        bm25_k1=_env_float("WEB_RAG_BM25_K1", defaults.bm25_k1),
        bm25_b=_env_float("WEB_RAG_BM25_B", defaults.bm25_b),
        medcpt_model=os.getenv(
            "WEB_RAG_MEDCPT_MODEL",
            defaults.medcpt_model,
        ),
        medcpt_max_length=_env_int(
            "WEB_RAG_MEDCPT_MAX_LENGTH",
            defaults.medcpt_max_length,
        ),
        medcpt_batch_size=_env_int(
            "WEB_RAG_MEDCPT_BATCH_SIZE",
            defaults.medcpt_batch_size,
        ),
        medcpt_device=os.getenv(
            "WEB_RAG_MEDCPT_DEVICE",
            defaults.medcpt_device,
        ),
        hybrid_lexical_weight=_env_float(
            "WEB_RAG_HYBRID_LEXICAL_WEIGHT",
            defaults.hybrid_lexical_weight,
        ),
        hybrid_medcpt_weight=_env_float(
            "WEB_RAG_HYBRID_MEDCPT_WEIGHT",
            defaults.hybrid_medcpt_weight,
        ),
    )


def validate_paperclip_source(value: str) -> None:
    sources = [
        item.strip().casefold()
        for item in value.split(",")
        if item.strip()
    ]
    if not sources:
        raise ValueError("paperclip_source must contain at least one source")
    unsupported = [
        item for item in sources
        if item not in PAPERCLIP_ACADEMIC_SOURCES
    ]
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
    if settings.chunking_method not in CHUNKING_METHODS:
        raise ValueError(
            f"chunking_method must be one of: {', '.join(CHUNKING_METHODS)}"
        )
    if settings.reranker not in RERANKERS:
        raise ValueError(
            f"reranker must be one of: {', '.join(RERANKERS)}"
        )
    if not 1 <= settings.retrieval_limit <= 1000:
        raise ValueError("retrieval_limit must be between 1 and 1000")
    if settings.paperclip_max_full_text_lines <= 0:
        raise ValueError("paperclip_max_full_text_lines must be greater than 0")
    if settings.paperclip_timeout <= 0:
        raise ValueError("paperclip_timeout must be greater than 0")
    if not settings.chunk_tokenizer_model.strip():
        raise ValueError("chunk_tokenizer_model must not be empty")
    if settings.chunk_max_tokens < 64:
        raise ValueError("chunk_max_tokens must be at least 64")
    if not 0 <= settings.chunk_overlap_fraction < 1:
        raise ValueError("chunk_overlap_fraction must be in [0, 1)")
    if settings.chunk_max_overlap_sentences < 0:
        raise ValueError("chunk_max_overlap_sentences must be non-negative")
    if settings.min_chunk_words < 0:
        raise ValueError("min_chunk_words must be non-negative")
    if settings.top_k <= 0 or settings.max_chunks_per_paper <= 0:
        raise ValueError("top_k and max_chunks_per_paper must be greater than 0")
    if not 0 <= settings.near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    if settings.bm25_k1 <= 0 or not 0 <= settings.bm25_b <= 1:
        raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")
    if not settings.medcpt_model.strip():
        raise ValueError("medcpt_model must not be empty")
    if settings.medcpt_max_length < 64:
        raise ValueError("medcpt_max_length must be at least 64")
    if settings.medcpt_batch_size <= 0:
        raise ValueError("medcpt_batch_size must be greater than 0")
    if settings.hybrid_lexical_weight < 0:
        raise ValueError("hybrid_lexical_weight must be non-negative")
    if settings.hybrid_medcpt_weight < 0:
        raise ValueError("hybrid_medcpt_weight must be non-negative")
    if settings.hybrid_lexical_weight + settings.hybrid_medcpt_weight <= 0:
        raise ValueError("at least one hybrid weight must be greater than 0")

    for name, value in (
        ("llm_model", settings.llm_model),
        ("llm_base_url", settings.llm_base_url),
        ("llm_api_key_env", settings.llm_api_key_env),
        ("hyde_model", settings.hyde_model),
        ("hyde_base_url", settings.hyde_base_url),
        ("expansion_model", settings.expansion_model),
        ("expansion_base_url", settings.expansion_base_url),
    ):
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
    for name, value in (
        ("hyde_timeout", settings.hyde_timeout),
        ("expansion_timeout", settings.expansion_timeout),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")
    for name, value in (
        ("hyde_temperature", settings.hyde_temperature),
        ("expansion_temperature", settings.expansion_temperature),
    ):
        if not 0 <= value <= 2:
            raise ValueError(f"{name} must be between 0 and 2")
    if settings.hyde_max_tokens <= 0:
        raise ValueError("hyde_max_tokens must be greater than 0")
    if settings.expansion_max_tokens <= 0:
        raise ValueError("expansion_max_tokens must be greater than 0")
    if settings.expansion_max_terms <= 0:
        raise ValueError("expansion_max_terms must be greater than 0")
    if settings.expansion_max_query_chars < 100:
        raise ValueError("expansion_max_query_chars must be at least 100")
