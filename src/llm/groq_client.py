"""A Groq-backed LLMClient implementation. Groq is free and fast, but -- confirmed
directly against its live API, not assumed -- it rejects `logprobs` on every model tried
with "`logprobs` is not supported with this model" (see docs/phases/05-llm-provider-choice.md
for the equivalent finding on other providers). So true_false_confidence() here asks the
model to report its own confidence as text instead of reading internal token
probabilities -- a real, weaker method than OpenAIClient's, kept only as a free stand-in
for iterating without spending API credit. Do not use this client for the numbers that go
into the final reproduction/evaluation -- use OpenAIClient for those.

Groq exposes an OpenAI-compatible endpoint, so this reuses the `openai` package pointed at
a different base_url instead of adding a second SDK dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class GroqClient:
    model: str = "qwen/qwen3.8-27b"
    api_key: str | None = None

    def _client(self) -> Any:
        import openai

        api_key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Groq API key found -- pass api_key= explicitly or set the "
                "GROQ_API_KEY environment variable."
            )
        return openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    def _retrying(self) -> Retrying:
        import openai

        return Retrying(
            retry=retry_if_exception_type(
                (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
            ),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            stop=stop_after_attempt(5),
            reraise=True,
        )

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        client = self._client()
        response = self._retrying()(
            client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content: str | None = response.choices[0].message.content
        return content or ""

    def complete_samples(
        self, prompt: str, n: int, *, max_tokens: int = 512, temperature: float = 0.7
    ) -> list[str]:
        return [
            self.complete(prompt, max_tokens=max_tokens, temperature=temperature) for _ in range(n)
        ]

    def true_false_confidence(self, statement: str) -> float:
        """Text-based fallback: Groq cannot return real logprobs (verified above), so we
        ask the model to type out its own confidence number instead of reading internal
        probabilities. This is a documented simplification, not the base paper's method.
        """
        prompt = (
            "Rate how likely the following statement is to be factually true, on a "
            "scale from 0.0 (definitely false) to 1.0 (definitely true). Respond with "
            "only the number, nothing else.\n\n"
            f"Statement: {statement}"
        )
        client = self._client()
        response = self._retrying()(
            client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        try:
            score = float(content.strip())
        except ValueError:
            # The model didn't answer with a bare number -- fail toward "unsure" rather
            # than guessing or crashing downstream.
            return 0.5
        return max(0.0, min(1.0, score))
