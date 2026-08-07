from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import QueryBundle
from .text_utils import (
    STOPWORDS,
    clean_text,
    deduplicate_text,
    extract_keywords,
    extract_numeric_terms,
    normalize_for_deduplication,
    tokenize,
)


_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
_LLM_PROVIDERS = ("ollama", "openai")

# Bump this whenever either query-generation prompt changes. Evaluation caches
# include this value so old generations cannot silently contaminate a new run.
QUERY_PROMPT_VERSION = "2026-08-biomedical-ir-v4"

_HYDE_SYSTEM_PROMPT = (
    "You are a biomedical information-retrieval specialist. Generate a neutral, "
    "retrieval-oriented pseudo-abstract that represents the kind of paper that "
    "would answer the question. Preserve the user's intent and constraints. Do "
    "not solve the question or invent experimental findings."
)

_EXPANSION_SYSTEM_PROMPT = (
    "You are a biomedical information-retrieval specialist. Produce controlled "
    "query-expansion terminology for scientific article retrieval. Preserve the "
    "original intent, avoid answer leakage, and return only the requested JSON."
)

_GENERIC_EXPANSION_TERMS = {
    "article",
    "biomedical article",
    "biomedical literature",
    "biomarker stability",
    "literature",
    "paper",
    "pre-analytical factors",
    "pre-analytical variables",
    "preanalytical factors",
    "preanalytical variables",
    "quality control",
    "research",
    "research paper",
    "result",
    "results",
    "sample handling",
    "sample stability",
    "scientific article",
    "scientific literature",
    "specimen handling",
    "specimen stability",
    "study",
}

_EXPANSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "added_terms": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 12,
        }
    },
    "required": ["added_terms"],
    "additionalProperties": False,
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


def build_hyde_query(
    question: str,
    *,
    model: str = "qwen2.5:7b-instruct",
    base_url: str = _OLLAMA_CHAT_URL,
    temperature: float = 0.0,
    max_tokens: int = 180,
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
    if len(hypothetical_document.split()) < 25:
        raise HyDEGenerationError("HyDE returned an unusually short hypothetical passage")

    # Anchor the pseudo-document with the original question. This retains exact
    # entities and constraints while still adding document-style retrieval context,
    # reducing the query drift observed with a standalone hypothetical passage.
    search_query = f"{normalized} {hypothetical_document}"

    return QueryBundle(
        original_question=question,
        normalized_question=normalized,
        strategy="hyde",
        search_query=search_query,
        keywords=extract_keywords(normalized),
        hypothetical_document=hypothetical_document,
    )


def build_llm_expansion_query(
    question: str,
    *,
    model: str = "qwen2.5:7b-instruct",
    base_url: str = _OLLAMA_CHAT_URL,
    temperature: float = 0.0,
    max_tokens: int = 160,
    seed: int = 42,
    timeout: float = 180.0,
    max_terms: int = 8,
    max_query_chars: int = 600,
    generator: Callable[[str], str] | None = None,
) -> QueryBundle:
    """Append a small set of faithful biomedical expansion terms to the raw question."""

    normalized = clean_text(question)
    if not normalized:
        raise ValueError("question must not be empty")
    if max_terms <= 0:
        raise ValueError("max_terms must be greater than 0")
    if max_query_chars < 100:
        raise ValueError("max_query_chars must be at least 100")

    prompt = _build_llm_expansion_prompt(normalized, max_terms=max_terms)
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
    expanded_terms = _validated_expansion_terms(
        payload.get("added_terms"),
        question=normalized,
        limit=max_terms,
    )

    search_query = normalized
    if expanded_terms:
        search_query = f"{normalized} {'; '.join(expanded_terms)}"
    search_query = _truncate_at_word_boundary(search_query, max_query_chars)

    return QueryBundle(
        original_question=question,
        normalized_question=normalized,
        strategy="llmexpand",
        search_query=search_query,
        keywords=extract_keywords(normalized),
        expanded_terms=expanded_terms,
        expansion_details={"added_terms": expanded_terms},
    )


def reformulate_query(
    question: str,
    strategy: str = "raw",
    *,
    hyde_model: str = "qwen2.5:7b-instruct",
    hyde_base_url: str = _OLLAMA_CHAT_URL,
    hyde_temperature: float = 0.0,
    hyde_max_tokens: int = 180,
    hyde_seed: int = 42,
    hyde_timeout: float = 180.0,
    hyde_generator: Callable[[str], str] | None = None,
    expansion_model: str = "qwen2.5:7b-instruct",
    expansion_base_url: str = _OLLAMA_CHAT_URL,
    expansion_temperature: float = 0.0,
    expansion_max_tokens: int = 160,
    expansion_seed: int = 42,
    expansion_timeout: float = 180.0,
    expansion_max_terms: int = 8,
    expansion_max_query_chars: int = 600,
    expansion_generator: Callable[[str], str] | None = None,
) -> QueryBundle:
    selected = strategy.strip().casefold().replace("-", "_")

    if selected == "raw":
        return build_raw_query(question)
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
    if selected in {"llmexpand", "llm_expansion"}:
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

    raise ValueError("Unknown query strategy. Choose from: raw, hyde, llmexpand")


def make_llm_generator(
    provider: str,
    *,
    model: str,
    base_url: str,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 180,
    seed: int = 42,
    timeout: float = 180.0,
    json_output: bool = False,
) -> Callable[[str], str]:
    """Create a reusable generator for Ollama or an OpenAI-compatible API."""

    selected = provider.strip().casefold()
    if selected not in _LLM_PROVIDERS:
        raise ValueError(f"provider must be one of: {', '.join(_LLM_PROVIDERS)}")
    if not model.strip():
        raise ValueError("model must not be empty")
    if not base_url.strip():
        raise ValueError("base_url must not be empty")

    if selected == "ollama":
        response_format = _EXPANSION_SCHEMA if json_output else None

        def generate_ollama(prompt: str) -> str:
            return _call_ollama(
                prompt,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                timeout=timeout,
                task_name="query generation",
                response_format=response_format,
            )

        return generate_ollama

    if not api_key:
        raise ValueError("An API key is required for the OpenAI-compatible provider")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise QueryGenerationError(
            "The openai package is required for an OpenAI-compatible API"
        ) from exc

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)

    system_prompt = (
        _EXPANSION_SYSTEM_PROMPT if json_output else _HYDE_SYSTEM_PROMPT
    )

    def generate_openai(prompt: str) -> str:
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Query reformulation needs concise final text/JSON, not hidden
            # reasoning. Qwen reasoning models can otherwise spend the entire
            # token budget in reasoning_content and return message.content=None.
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        }
        if json_output:
            request["response_format"] = {"type": "json_object"}

        try:
            completion = client.chat.completions.create(**request)
        except Exception as first_error:
            # Some OpenAI-compatible servers do not expose response_format even
            # when the selected model can still follow a JSON-only prompt.
            if json_output and "response_format" in request:
                request.pop("response_format", None)
                try:
                    completion = client.chat.completions.create(**request)
                except Exception as second_error:
                    raise QueryGenerationError(
                        "OpenAI-compatible query generation failed: "
                        f"{second_error}"
                    ) from second_error
            else:
                raise QueryGenerationError(
                    f"OpenAI-compatible query generation failed: {first_error}"
                ) from first_error

        content = completion.choices[0].message.content if completion.choices else ""
        if not content:
            raise QueryGenerationError("The LLM returned an empty response")
        return str(content)

    return generate_openai


def _build_hyde_prompt(question: str) -> str:
    return f"""
Create one neutral hypothetical abstract-style passage for dense scientific-paper
retrieval. The passage should describe a paper that investigates the question, not
pretend to know its answer.

Output requirements:
- Write 2 or 3 connected sentences, about 45 to 80 words total.
- State the study topic, biospecimen or biological material, measured outcome, and
  the processing, storage, or comparison variables explicitly present in the question.
- Use standard title-and-abstract terminology, including safe acronym expansions,
  spelling variants, or a broader class logically entailed by a named product.
- Preserve every explicit analyte, specimen, tube, method, comparison, temperature,
  duration, number, unit, and negation.
- For questions asking which, how many, how much, whether, why, or for a
  recommendation, describe what the paper evaluates or reports; do not supply a
  candidate answer, quantity, effect direction, cause, or recommended action.
- Do not introduce new analytes, products, diseases, populations, temperatures,
  durations, numerical findings, effect directions, thresholds, or conclusions.
- Do not include a heading, citation, author, journal, article title, PMID, or DOI.

Question:
{question}
""".strip()


def _build_llm_expansion_prompt(question: str, *, max_terms: int) -> str:
    return f"""
Generate a controlled, high-precision biomedical query expansion for scientific-paper
retrieval. The original question will remain unchanged at the beginning of the search
query. Return only additional terminology that is likely to occur in relevant article
titles, abstracts, indexing terms, or methods sections.

Return JSON only:
{{"added_terms": ["term 1", "term 2"]}}

Use this selection process:
1. Identify the question's explicit anchors: named product, specimen, analyte, assay,
   measurement concept, processing step, storage condition, comparison, and outcome.
2. Generate candidates only from these high-value classes, in priority order:
   a. Standard abbreviation/full-form pairs used in biomedical papers, for example
      total allowable error <-> TEa or RNA integrity number <-> RIN.
   b. Formal commercial or standardized names for a product already named in the
      question, plus its directly entailed specimen or device class, for example
      PAXgene Blood RNA Tube or RNA-stabilizing blood collection tube.
   c. Canonical title/abstract phrases for the same concept, such as delayed
      centrifugation, freeze-thaw stability, preanalytical robustness, or
      case-control biospecimen matching, but only when directly implied.
   d. Precise synonyms, spelling variants, hyphenation variants, MeSH-like phrases,
      and assay terminology for entities already present or logically entailed.
3. Rank candidates by specificity and likelihood of appearing in a relevant title or
   abstract. Keep only the strongest terms.

Rules:
- Prefer 3 to 6 high-information terms and never exceed {max_terms}.
- Each term must contain at most 5 words.
- Preserve named entities and technical meaning; do not rewrite the question.
- Do not return generic standalone phrases such as sample handling, sample stability,
  preanalytical variables, quality control, study, paper, research, result, cohort,
  precision, or reproducibility.
- Do not add a specimen, disease, population, product, temperature, duration,
  intervention, or assay that is not stated or directly entailed by the question.
- Do not guess the answer. Apply these leakage checks:
  * For a "which" question, never add candidate instances of the requested item.
  * For "how many" or "how much", never add quantities or ranges.
  * For "whether" or "did", never add a positive or negative conclusion.
  * For "why", never add possible causes.
  * For recommendations, never add the recommended action.
- Do not repeat wording already present unless the output is a recognized abbreviation,
  full-form counterpart, formal standardized name, or materially different synonym.
- Return an empty list if no safe and specific expansion exists.

Examples showing the desired level of specificity:
Question: Did any statistically changed serum biomarkers exceed their total allowable
error limits after one freeze-thaw cycle?
Output: {{"added_terms": ["TEa", "allowable total error", "serum analyte stability",
"single freeze-thaw cycle"]}}

Question: Which pre-analytical variations most affected RNA integrity in PAXgene blood
collection tubes?
Output: {{"added_terms": ["PAXgene Blood RNA Tube", "RNA-stabilizing blood collection tube",
"whole-blood RNA stabilization", "preanalytical robustness", "RNA integrity number RIN"]}}

Question: Which sample collection and handling conditions should be matched between
cases and controls to reduce biospecimen bias?
Output: {{"added_terms": ["case-control biospecimen matching", "preanalytical standardization",
"biorepository SOP", "specimen handling protocol"]}}

Do not copy an example term unless it is genuinely applicable to the input question.

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
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    _EXPANSION_SYSTEM_PROMPT
                    if response_format is not None
                    else _HYDE_SYSTEM_PROMPT
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
        raise QueryGenerationError(
            f"Ollama returned HTTP {exc.code} during {task_name}: {body}"
        ) from exc
    except URLError as exc:
        raise QueryGenerationError(
            f"Could not connect to Ollama during {task_name}"
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
    for prefix in (
        "hypothetical passage:",
        "hypothetical document:",
        "passage:",
    ):
        if value.casefold().startswith(prefix):
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
        raise LLMExpansionError("LLM expansion did not return a JSON object")

    try:
        payload = json.loads(value[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMExpansionError(f"LLM expansion returned invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise LLMExpansionError("LLM expansion JSON must be an object")
    return payload


def _validated_expansion_terms(
    value: Any,
    *,
    question: str,
    limit: int,
) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, (str, int, float))]
    else:
        raise LLMExpansionError("LLM expansion must contain an 'added_terms' list")

    question_lower = question.casefold()
    question_tokens = {
        token for token in tokenize(question)
        if token not in STOPWORDS
    }
    allowed_numeric_terms = set(extract_numeric_terms(question))
    valid: list[str] = []
    seen_normalized: set[str] = set()

    for candidate in candidates:
        term = clean_text(str(candidate)).strip(" -;,.")
        if not term or len(term.split()) > 5:
            continue

        normalized = normalize_for_deduplication(term)
        if not normalized or normalized in seen_normalized:
            continue
        if normalized in _GENERIC_EXPANSION_TERMS:
            continue
        if term.casefold() in question_lower:
            continue

        term_tokens = {
            token for token in tokenize(term)
            if token not in STOPWORDS
        }
        # Reject a phrase that contributes no new lexical signal. Standardized
        # spelling variants remain possible because their normalized token differs.
        if term_tokens and term_tokens.issubset(question_tokens):
            continue

        introduced_numeric_terms = set(extract_numeric_terms(term)) - allowed_numeric_terms
        if introduced_numeric_terms:
            continue

        seen_normalized.add(normalized)
        valid.append(term)

    return deduplicate_text(valid)[:limit]


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:.-")
    return shortened or text[:max_chars]
