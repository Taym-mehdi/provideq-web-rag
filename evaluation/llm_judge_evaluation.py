# Papers: "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems" (Saad-Falcon et al., NAACL 2024) and "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (Liu et al., EMNLP 2023).

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_ALLOWED_SCORES = {-1.0, 0.0, 0.5, 1.0}
_OLLAMA_DEFAULT_URL = "http://localhost:11434/api/chat"


class LLMJudgeEvaluator:
    def __init__(
        self,
        *,
        provider: str = "ollama",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be non-negative")

        normalized_provider = provider.strip().lower()
        if normalized_provider not in {"ollama", "openai"}:
            raise ValueError("judge provider must be 'ollama' or 'openai'")

        self.provider = normalized_provider
        self.model = model or (
            "qwen2.5:7b-instruct" if self.provider == "ollama" else "gpt-5-mini"
        )
        self.base_url = base_url
        self.retries = retries
        self.client = client

        if self.provider == "openai" and self.client is None:
            self.client = self._create_openai_client(api_key, base_url)

    @staticmethod
    def _create_openai_client(api_key: str | None, base_url: str | None) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI judge evaluation requires the openai package.") from exc

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("Set OPENAI_API_KEY or pass --judge-api-key.")

        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def score(
        self,
        question: str,
        gold_answers: list[str],
        evidence_texts: list[str],
        *,
        answerable: bool,
    ) -> tuple[float, str]:
        """Judge each snippet separately and return the highest-scoring snippet."""
        if not evidence_texts:
            return (1.0 if not answerable else 0.0), ""

        best_score = float("-inf")
        best_evidence = ""
        for evidence in evidence_texts:
            score = self._score_single(
                question,
                gold_answers,
                evidence,
                answerable=answerable,
            )
            if score > best_score:
                best_score = score
                best_evidence = evidence

        return best_score, best_evidence

    def _score_single(
        self,
        question: str,
        gold_answers: list[str],
        evidence_text: str,
        *,
        answerable: bool,
    ) -> float:
        prompt = self._build_prompt(question, gold_answers, evidence_text, answerable)
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                if self.provider == "ollama":
                    result = self._call_ollama(prompt)
                else:
                    result = self._call_openai(prompt)

                score = float(result["score"])
                if score not in _ALLOWED_SCORES:
                    raise ValueError(f"Unexpected judge score: {score}")
                return score
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)

        raise RuntimeError(f"LLM judge failed after {self.retries + 1} attempts") from last_error

    def _call_ollama(self, prompt: str) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "enum": [-1, 0, 0.5, 1]},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator of biomedical retrieval evidence. "
                        "Use only the supplied benchmark truth and retrieved evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
        }

        request = Request(
            self.base_url or _OLLAMA_DEFAULT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Start Ollama and make sure the model is installed."
            ) from exc

        content = response_data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Ollama returned an empty response")
        return json.loads(content)

    def _call_openai(self, prompt: str) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator of biomedical retrieval evidence. "
                        "Use only the supplied benchmark truth and retrieved evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "evidence_judgment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "enum": [-1, 0, 0.5, 1],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["score", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return json.loads(response.output_text)

    @staticmethod
    def _build_prompt(
        question: str,
        gold_answers: list[str],
        evidence_text: str,
        answerable: bool,
    ) -> str:
        gold = "\n".join(f"- {answer}" for answer in gold_answers) or "- No supported answer"

        return f"""
Evaluate one retrieved evidence snippet for the benchmark question.

Question:
{question}

Benchmark answerable:
{answerable}

Benchmark truth:
{gold}

Retrieved evidence snippet:
{evidence_text}

Assign exactly one score:
1   = the snippet fully supports all essential parts of the benchmark truth.
0.5 = the snippet supports part of the benchmark truth and contains no direct contradiction.
0   = the snippet is irrelevant, insufficient, or does not establish the benchmark truth.
-1  = the snippet directly contradicts the benchmark truth or supports the opposite conclusion.

For an unanswerable benchmark item, use 1 only when the snippet correctly establishes that no supported answer can be given, 0 when it is irrelevant or inconclusive, and -1 when it misleadingly supports a definite answer to the unsupported premise.
Return only the required JSON object.
""".strip()
