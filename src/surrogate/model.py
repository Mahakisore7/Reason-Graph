"""The surrogate confidence scorer -- Our Novelty, Extension 2.

The paper's authors name cost as BTProp's main limitation: every node of every tree needs LLM
calls. This model predicts the LLM's own true_false_confidence() from the claim text alone
(hashed words -> ridge regression), so a chosen fraction of nodes can skip the LLM call. It is
trained on the LLM's confidences, never on the human truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge

from src.surrogate.embedder import Embedder, HashingEmbedder


@dataclass
class SurrogateConfidenceModel:
    embedder: Embedder = field(default_factory=HashingEmbedder)
    alpha: float = 1.0
    _regressor: Ridge = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._regressor = Ridge(alpha=self.alpha)

    def fit(self, statements: list[str], confidences: list[float]) -> None:
        if len(statements) != len(confidences):
            raise ValueError(
                f"{len(statements)} statements but {len(confidences)} confidences -- "
                "they must pair up one to one"
            )
        if len(statements) < 2:
            # With one example the regression can only learn a constant, never a pattern.
            raise ValueError("need at least 2 examples to fit the surrogate")
        self._regressor.fit(self.embedder.embed(statements), np.asarray(confidences))
        self._fitted = True

    def predict_many(self, statements: list[str]) -> list[float]:
        if not self._fitted:
            raise RuntimeError("SurrogateConfidenceModel.predict called before fit()")
        if not statements:
            return []
        predictions = self._regressor.predict(self.embedder.embed(statements))
        # Ridge has no output-range guarantee; a confidence must be a probability.
        return [float(p) for p in np.clip(predictions, 0.0, 1.0)]

    def predict(self, statement: str) -> float:
        return self.predict_many([statement])[0]


def mean_absolute_error_against_llm(
    model: SurrogateConfidenceModel, statements: list[str], true_confidences: list[float]
) -> float:
    """How far the surrogate is from the LLM's own confidence on held-out statements."""
    predictions = np.asarray(model.predict_many(statements))
    return float(np.mean(np.abs(predictions - np.asarray(true_confidences))))
