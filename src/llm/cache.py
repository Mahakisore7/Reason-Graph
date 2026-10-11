"""Disk-based response caching -- wraps any LLMClient, adds nothing else. Composition, not
inheritance: this wraps an object satisfying LLMClient and delegates to it, so caching can
be added to ANY implementation without that implementation knowing caching exists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm.base import LLMClient


def _hash_key(*parts: Any) -> str:
    """Turns `parts` into a stable, unique, filesystem-safe key.

    Two calls with the SAME arguments must produce the SAME key. Two calls with even
    slightly DIFFERENT arguments (e.g. a different temperature) must produce a DIFFERENT
    key -- that second property is the one that's easy to get subtly wrong.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CachingLLMClient:
    """Wraps `inner` and caches every call to disk under `cache_dir`, keyed by a hash of
    its exact arguments. Each cached file stores the inputs alongside the result, so the
    cache directory is debuggable just by reading a file.
    """

    inner: LLMClient
    cache_dir: Path
    model_tag: str = "default"

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        key = _hash_key(self.model_tag, "complete", prompt, max_tokens, temperature)
        path = self.cache_dir / f"{key}.json"

        if path.exists():
            cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return str(cached["result"])

        result = self.inner.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        path.write_text(
            json.dumps(
                {
                    "method": "complete",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "result": result,
                }
            ),
            encoding="utf-8",
        )
        return result

    def complete_samples(
        self, prompt: str, n: int, *, max_tokens: int = 512, temperature: float = 0.7
    ) -> list[str]:
        key = _hash_key(self.model_tag, "complete_samples", prompt, n, max_tokens, temperature)
        path = self.cache_dir / f"{key}.json"

        if path.exists():
            cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return [str(s) for s in cached["result"]]

        result = self.inner.complete_samples(
            prompt, n, max_tokens=max_tokens, temperature=temperature
        )
        path.write_text(
            json.dumps(
                {
                    "method": "complete_samples",
                    "prompt": prompt,
                    "n": n,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "result": result,
                }
            ),
            encoding="utf-8",
        )
        return result

    def true_false_confidence(self, statement: str) -> float:
        key = _hash_key(self.model_tag, "true_false_confidence", statement)
        path = self.cache_dir / f"{key}.json"

        if path.exists():
            cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return float(cached["result"])

        result = self.inner.true_false_confidence(statement)
        path.write_text(
            json.dumps(
                {
                    "method": "true_false_confidence",
                    "statement": statement,
                    "result": result,
                }
            ),
            encoding="utf-8",
        )
        return result
