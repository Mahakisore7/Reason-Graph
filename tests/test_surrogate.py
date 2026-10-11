"""Tests for the surrogate confidence scorer (Phase 4)."""

from __future__ import annotations

import pytest

from src.surrogate.embedder import HashingEmbedder
from src.surrogate.model import SurrogateConfidenceModel, mean_absolute_error_against_llm

# Absolute claims ("always", "all", "never") -> low confidence; hedged ones -> high.
TRAIN = [
    ("Cats always land on their feet.", 0.05),
    ("Cats sometimes land on their feet.", 0.90),
    ("All swans are white.", 0.10),
    ("Most swans are white.", 0.90),
    ("Dogs never bite people.", 0.05),
    ("Dogs usually avoid biting people.", 0.85),
    ("All birds can always fly.", 0.10),
    ("Many birds can fly.", 0.90),
    ("Every medicine always works for all patients.", 0.05),
    ("Some medicines often work for many patients.", 0.85),
]


def _fitted(alpha: float = 0.1) -> SurrogateConfidenceModel:
    model = SurrogateConfidenceModel(alpha=alpha)
    model.fit([s for s, _ in TRAIN], [c for _, c in TRAIN])
    return model


def test_learns_a_real_pattern_on_unseen_sentences() -> None:
    model = _fitted()
    absolute = model.predict("All politicians always tell the complete truth.")
    hedged = model.predict("Many politicians try to be honest most of the time.")
    assert absolute < hedged - 0.2


def test_predictions_are_clipped_to_probabilities() -> None:
    model = _fitted(alpha=1e-6)
    for p in model.predict_many(["always always always never all", "usually most many some"]):
        assert 0.0 <= p <= 1.0


def test_mean_absolute_error_is_small_on_training_data() -> None:
    model = _fitted()
    assert (
        mean_absolute_error_against_llm(model, [s for s, _ in TRAIN], [c for _, c in TRAIN]) < 0.1
    )


def test_predict_before_fit_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        SurrogateConfidenceModel().predict("Anything.")


def test_fit_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="pair up"):
        SurrogateConfidenceModel().fit(["a", "b"], [0.5])
    with pytest.raises(ValueError, match="at least 2"):
        SurrogateConfidenceModel().fit(["a"], [0.5])


def test_embedding_shape_and_unit_length() -> None:
    vectors = HashingEmbedder(n_features=64).embed(["One short sentence.", "Another one here."])
    assert vectors.shape == (2, 64)
    assert (vectors >= 0).all()
    assert (vectors**2).sum(axis=1) == pytest.approx(1.0)
