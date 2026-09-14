"""The LLMClient protocol every backbone implementation must satisfy.

Kept deliberately small: two methods cover everything the rest of the pipeline needs.
src.inference never imports this module (module-boundary rule, docs/phases/01-architecture.md)
-- only src.tree and src.surrogate depend on it.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Anything that can generate text and score a statement's truthfulness."""

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Free-form text completion. Used by the three child-generation strategies
        (Decomposition, Premises, Correction) -- docs/phases/04-research-notes-btprop.md,
        Step 1."""
        ...

    def true_false_confidence(self, statement: str) -> float:
        """Confidence in [0, 1] that `statement` is true. Matches BTProp's own scoring
        method: prompt "True or False? {statement}" and normalise the model's own
        next-token probabilities for "True" vs "False"
        (docs/phases/04-research-notes-btprop.md, Step 3). A first-class method rather
        than something built on `complete` because it fundamentally needs token-level
        probabilities, not just a text string back."""
        ...
