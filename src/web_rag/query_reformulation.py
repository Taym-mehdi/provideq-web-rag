# Research basis:
# - Gao et al. (ACL 2023), HyDE.
# - Wang et al. (EMNLP 2023), Query2doc.
# - Zhang et al. (Findings of EMNLP 2024), MuGI.

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import QueryBundle
from .text_utils import clean_text, deduplicate_text, extract_keywords, extract_numeric_terms


BIOMEDICAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "serum": ("blood serum",),
    "plasma": ("blood plasma",),
    "blood": ("whole blood",),
    "gel": ("serum separator", "SST"),
    "tube": ("collection tube",),
    "tubes": ("collection tube",),
    "centrifugation": ("pre-centrifugation",),
    "centrifuge": ("centrifugation", "pre-centrifugation"),
    "delay": ("processing delay",),
    "delayed": ("delay", "processing delay"),
    "storage": ("sample storage",),
    "temperature": ("storage temperature",),
    "stability": ("stable", "unstable"),
    "stable": ("stability", "unstable"),
    "potassium": ("K+",),
    "sodium": ("Na+",),
    "phosphate": ("phosphorus",),
    "crp": ("C-reactive protein",),
}

PHRASES = (
    "serum separator tubes",
    "serum separator tube",
    "serum gel tubes",
    "serum gel tube",
    "delayed centrifugation",
    "pre-centrifugation",
    "processing delay",
    "sample storage",
    "storage temperature",
    "room temperature",
    "whole blood",
    "c-reactive protein",
)

_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

_EXPANSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "core_concepts": {"type": "array", "items": {"type": "string"}},
        "synonyms": {"type": "array", "items": {"type": "string"}},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "title_query": {"type": "string"},
        "abstract_query": {"type": "string"},
    },
    "required": [
        "core_concepts",
        "synonyms",
        "conditions",
        "title_query",
        "abstract_query",
    ],
}


class QueryGenerationError(RuntimeError):
    pass


class HyDEGenerationError(QueryGenerationError):
    pass


class LLMExpansionError(QueryGenerationError):
    pass


def build_raw_query(question: str) -> QueryBundle:
    normalized = clean_text(question)
    if not normalized:
        raise ValueError("question must not be empty")
    return QueryBundle(
        original_question=question,
        normalized_question=normalized,
        strategy="raw",
        search_query=normalized,
        keywords=extract_keywords(normalized),
    )


def build_synonym_query(question: str, *, max_terms: int = 24) -> QueryBundle:
    normalized = clean_text(question)
    if not normalized:
        raise ValueError("question must not be empty")
    if max_terms <= 0:
        raise ValueError("max_terms must be greater than 0")

    keywords = extract_keywords(normalized)
    lowered = normalized.casefold()
    expanded: list[str] = [*keywords, *extract_numeric_terms(normalized)]
    expanded.extend(phrase for phrase in PHRASES if phrase in lowered)
    for keyword in keywords:
        expanded.extend(BIOMEDICAL_EXPANSIONS.get(keyword, ()))

    expanded_terms = deduplicate_text(expanded)[:max_terms]
    return QueryBundle(
        original_question=question,
        normalized_question=normalized,
        strategy="synonym",
        search_query=" ".join(expanded_terms) or normalized,
        keywords=keywords,
        expanded_terms=expanded_terms,
    )


def build_hyde_query(
    question: str,
    *,
    model: str = "qwen2.5:7b-instruct",
    base_url: str = _OLLAMA_CHAT_URL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int = 42,
    timeout: float = 180.0,
    generator: Callable[[str], str] | None = None,
) -> QueryBundle:
    normalized = clean_text(question)
    if not normalized:
        raise ValueError("question must not be empty")

    prompt = _build_hyde_prompt(normalized)
    generated = generator(prompt) if generator is not None else _call_ollama(
        prompt,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        timeout=timeout,
        task_name="HyDE generation",
    )
    hypothetical_document = _clean_hypothetical_document(generated)
    if len(hypothetical_document) < 60:
        raise HyDEGenerationError("HyDE returned an empty or unusually short hypothetical document")

    return QueryBundle(
        original_question=question,
        normalized_question=normalized,
        strategy="hyde",
        search_query=hypothetical_document,
        keywords=extract_keywords(normalized),
        hypothetical_document=hypothetical_document,
    )


def build_llm_expansion_query(
    question: str,
    *,
    model: str = "qwen2.5:7b-instruct",
    base_url: str = _OLLAMA_CHAT_URL,
    temperature: float = 0.0,
    max_tokens: int = 420,
    seed: int = 42,
    timeout: float = 180.0,
    max_terms: int = 32,
    max_query_chars: int = 1200,
    generator: Callable[[str], str] | None = None,
) -> QueryBundle:
    """Create a conservative multi-representation biomedical retrieval query.

    The LLM does not answer the question. It produces core concepts, established
    synonyms/abbreviations, explicit conditions, and two complementary query
    formulations. The original question is retained in the final query to reduce
    topic drift.
    """

    normalized = clean_text(question)
    if not normalized:
        raise ValueError("question must not be empty")
    if max_terms <= 0:
        raise ValueError("max_terms must be greater than 0")
    if max_query_chars < 100:
        raise ValueError("max_query_chars must be at least 100")

    prompt = _build_llm_expansion_prompt(normalized)
    generated = generator(prompt) if generator is not None else _call_ollama(
        prompt,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        timeout=timeout,
        task_name="LLM query expansion",
        response_format=_EXPANSION_SCHEMA,
    )
    payload = _parse_expansion_payload(generated)

    core_concepts = _clean_string_list(payload.get("core_concepts"), limit=8)
    synonyms = _clean_string_list(payload.get("synonyms"), limit=12)
    conditions = _clean_string_list(payload.get("conditions"), limit=8)
    title_query = clean_text(str(payload.get("title_query", "")))
    abstract_query = clean_text(str(payload.get("abstract_query", "")))
    query_variants = deduplicate_text(
        [value for value in (title_query, abstract_query) if value]
    )[:2]

    if not (core_concepts or synonyms or conditions or query_variants):
        raise LLMExpansionError("LLM query expansion returned no usable expansion fields")

    keywords = extract_keywords(normalized)
    deterministic_expansions: list[str] = []
    for keyword in keywords:
        deterministic_expansions.extend(BIOMEDICAL_EXPANSIONS.get(keyword, ()))

    expanded_terms = deduplicate_text(
        [
            *core_concepts,
            *synonyms,
            *conditions,
            *extract_numeric_terms(normalized),
            *deterministic_expansions,
        ]
    )[:max_terms]

    # Keep the original question first, then add two complementary natural-language
    # representations for vector retrieval and compact controlled terms for hybrid retrieval.
    narrative_parts = [normalized, *query_variants]
    search_query = clean_text(
        ". ".join(part.rstrip(" .") for part in narrative_parts if part)
        + (f". {'; '.join(expanded_terms)}" if expanded_terms else "")
    )
    search_query = _truncate_at_word_boundary(search_query, max_query_chars)

    return QueryBundle(
        original_question=question,
        normalized_question=normalized,
        strategy="llmexpand",
        search_query=search_query,
        keywords=keywords,
        expanded_terms=expanded_terms,
        expansion_details={
            "core_concepts": core_concepts,
            "synonyms": synonyms,
            "conditions": conditions,
            "query_variants": query_variants,
        },
    )


def reformulate_query(
    question: str,
    strategy: str = "hyde",
    *,
    hyde_model: str = "qwen2.5:7b-instruct",
    hyde_base_url: str = _OLLAMA_CHAT_URL,
    hyde_temperature: float = 0.0,
    hyde_max_tokens: int = 256,
    hyde_seed: int = 42,
    hyde_timeout: float = 180.0,
    hyde_generator: Callable[[str], str] | None = None,
    expansion_model: str = "qwen2.5:7b-instruct",
    expansion_base_url: str = _OLLAMA_CHAT_URL,
    expansion_temperature: float = 0.0,
    expansion_max_tokens: int = 420,
    expansion_seed: int = 42,
    expansion_timeout: float = 180.0,
    expansion_max_terms: int = 32,
    expansion_max_query_chars: int = 1200,
    expansion_generator: Callable[[str], str] | None = None,
) -> QueryBundle:
    selected = strategy.strip().casefold().replace("-", "_")
    if selected == "raw":
        return build_raw_query(question)
    if selected == "synonym":
        return build_synonym_query(question)
    if selected == "hyde":
        return build_hyde_query(
            question,
            model=hyde_model,
            base_url=hyde_base_url,
            temperature=hyde_temperature,
            max_tokens=hyde_max_tokens,
            seed=hyde_seed,
            timeout=hyde_timeout,
            generator=hyde_generator,
        )
    if selected in {"llmexpand", "llm_expansion", "structured_expansion"}:
        return build_llm_expansion_query(
            question,
            model=expansion_model,
            base_url=expansion_base_url,
            temperature=expansion_temperature,
            max_tokens=expansion_max_tokens,
            seed=expansion_seed,
            timeout=expansion_timeout,
            max_terms=expansion_max_terms,
            max_query_chars=expansion_max_query_chars,
            generator=expansion_generator,
        )
    raise ValueError("Unknown query strategy. Choose from: raw, synonym, hyde, llmexpand")


def _build_hyde_prompt(question: str) -> str:
    return f"""
Write one concise hypothetical passage that could plausibly appear in the methods,
results, or discussion section of a peer-reviewed biomedical paper and that directly
addresses the retrieval question below.

Use the scientific terminology and all important conditions from the question, such as
the analyte, specimen type, collection tube, processing delay, temperature, duration,
and stability outcome when they are present. Preserve numbers stated in the question,
but do not invent additional exact measurements, citations, authors, article titles, or
DOIs. Write only the passage, without a heading or explanation. Aim for 100 to 180 words.

Retrieval question:
{question}
""".strip()


def _build_llm_expansion_prompt(question: str) -> str:
    return f"""
Transform the biomedical question below into a high-recall literature retrieval
representation. Do not answer the question and do not predict the study result.

Return JSON only, using exactly this schema:
{{
  "core_concepts": ["..."],
  "synonyms": ["..."],
  "conditions": ["..."],
  "title_query": "...",
  "abstract_query": "..."
}}

Rules:
- Preserve every explicit analyte, specimen type, collection tube, processing step,
  comparison, negation, population, temperature, duration, and number.
- Use only established biomedical synonyms, abbreviations, spelling variants, and
  closely equivalent terminology. Do not add loosely related diseases or analytes.
- core_concepts: 3 to 8 concise concepts.
- synonyms: at most 12 concise synonym or abbreviation phrases.
- conditions: at most 8 explicit experimental or pre-analytical constraints.
- title_query: 8 to 25 words, phrased like a likely scientific article title.
- abstract_query: 15 to 45 words, phrased like a methods/abstract retrieval sentence.
- Avoid generic filler such as "study", "research", "paper", and "literature" unless
  it is part of a necessary biomedical phrase.
- Do not invent authors, citations, DOIs, exact outcomes, or values that are absent
  from the question. The title_query is only a synthetic search formulation.

Question:
{question}
""".strip()


def _call_ollama(
    prompt: str,
    *,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    timeout: float,
    task_name: str,
    response_format: str | dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You create faithful biomedical information-retrieval query "
                    "representations. Preserve the user's constraints and never answer "
                    "the question unless explicitly asked to generate a hypothetical passage."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": seed,
        },
    }
    if response_format is not None:
        payload["format"] = response_format

    request = Request(
        _normalize_ollama_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise QueryGenerationError(f"Ollama returned HTTP {exc.code} during {task_name}: {body}") from exc
    except URLError as exc:
        raise QueryGenerationError(
            f"Could not connect to Ollama for {task_name}. Start Ollama and install the selected model."
        ) from exc

    content = response_data.get("message", {}).get("content", "")
    if not content:
        raise QueryGenerationError(f"Ollama returned an empty response during {task_name}")
    return str(content)


def _normalize_ollama_url(base_url: str) -> str:
    value = (base_url or _OLLAMA_CHAT_URL).strip().rstrip("/")
    if value.endswith("/api/chat"):
        return value
    if value.endswith("/api"):
        return f"{value}/chat"
    return f"{value}/api/chat"


def _clean_hypothetical_document(text: str) -> str:
    value = clean_text(text)
    prefixes = (
        "hypothetical passage:",
        "hypothetical document:",
        "passage:",
    )
    lowered = value.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            value = value[len(prefix) :].strip()
            break
    return value.strip('"').strip()


def _parse_expansion_payload(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()

    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise LLMExpansionError("LLM query expansion did not return a JSON object")

    try:
        payload = json.loads(value[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMExpansionError(f"LLM query expansion returned invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise LLMExpansionError("LLM query expansion JSON must be an object")
    return payload


def _clean_string_list(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, (str, int, float))]
    else:
        candidates = []

    cleaned = [clean_text(str(item)).strip(" -;,.\t") for item in candidates]
    return deduplicate_text([item for item in cleaned if item])[:limit]


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:.-")
    return shortened or text[:max_chars]
