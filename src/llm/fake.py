"""A deterministic, canned LLMClient for tests -- never makes a network call. This is what
makes src.tree (Phase 2) unit-testable with zero API key and zero flakiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeLLMClient:
    """Canned responses keyed by exact prompt/statement text, plus call counters for
    assertions.

    Example:
        client = FakeLLMClient(
            completions={"decompose: The sky is blue and grass is green":
                         "The sky is blue.\\nGrass is green."},
            confidences={"The sky is blue.": 0.95, "Grass is green.": 0.9},
        )
    """

    completions: dict[str, str] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)
    default_confidence: float = 0.5

    complete_call_count: int = field(default=0, init=False)
    samples_call_count: int = field(default=0, init=False)
    confidence_call_count: int = field(default=0, init=False)

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.complete_call_count += 1
        if prompt not in self.completions:
            raise KeyError(f"FakeLLMClient has no canned completion for prompt: {prompt!r}")
        return self.completions[prompt]

    def complete_samples(
        self, prompt: str, n: int, *, max_tokens: int = 512, temperature: float = 0.7
    ) -> list[str]:
        self.samples_call_count += 1
        if prompt not in self.samples:
            raise KeyError(f"FakeLLMClient has no canned samples for prompt: {prompt!r}")
        return self.samples[prompt][:n]

    def true_false_confidence(self, statement: str) -> float:
        self.confidence_call_count += 1
        return self.confidences.get(statement, self.default_confidence)
