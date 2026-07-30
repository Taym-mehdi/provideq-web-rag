# Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., ACL 2023).

from __future__ import annotations

import json
from collections.abc import Callable
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


class HyDEGenerationError(RuntimeError):
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
    raise ValueError("Unknown query strategy. Choose from: raw, synonym, hyde")


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


def _call_ollama(
    prompt: str,
    *,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate hypothetical scientific passages for biomedical literature "
                    "retrieval. The passage is a retrieval representation, not a final answer."
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
        raise HyDEGenerationError(f"Ollama returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise HyDEGenerationError(
            "Could not connect to Ollama for HyDE generation. Start Ollama and install the selected model."
        ) from exc

    content = response_data.get("message", {}).get("content", "")
    if not content:
        raise HyDEGenerationError("Ollama returned an empty HyDE response")
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
