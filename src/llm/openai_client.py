"""A real, API-backed LLMClient implementation. Read docs/phases/05-llm-provider-choice.md
before picking a different provider -- some providers silently ignore the `logprobs`
parameter this class depends on, without erroring.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.prompts import true_false_confidence_prompt


@dataclass
class OpenAIClient:
    """Implements the LLMClient protocol against the OpenAI chat completions API.

    model: e.g. "gpt-3.5-turbo-0125" -- one of the two backbones BTProp itself evaluates
        against, for directly comparable numbers.
    api_key: if None, falls back to the OPENAI_API_KEY environment variable -- never
        hard-code a key in this file or anywhere in the repo.
    """

    model: str = "gpt-3.5-turbo-0125"
    api_key: str | None = None

    def _client(self) -> Any:
        # Imported locally so this module doesn't hard-require the `openai` package just
        # to be imported -- only actually calling a method does.
        import openai

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No OpenAI API key found -- pass api_key= explicitly or set the "
                "OPENAI_API_KEY environment variable."
            )
        return openai.OpenAI(api_key=api_key)

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

    def true_false_confidence(self, statement: str) -> float:
        client = self._client()
        prompt = true_false_confidence_prompt(statement)

        response = self._retrying()(
            client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )

        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

        p_true = 0.0
        p_false = 0.0
        for entry in top_logprobs:
            token = entry.token.strip().lower()
            probability = math.exp(entry.logprob)
            if token == "true":
                p_true += probability
            elif token == "false":
                p_false += probability

        total = p_true + p_false
        if total == 0.0:
            # Neither "True" nor "False" showed up among the top-5 candidate tokens at
            # all -- a deliberate, documented fallback rather than an accidental NaN.
            return 0.5
        return p_true / total
