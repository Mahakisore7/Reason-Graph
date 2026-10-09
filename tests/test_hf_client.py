"""Tests for the probability arithmetic in HuggingFaceClient -- no model download."""

from __future__ import annotations

import pytest

from src.llm.hf_client import true_probability


def test_normalises_over_true_and_false_only() -> None:
    # vocab of 4: ids 0,1 mean "True", id 2 means "False", id 3 is something else
    probs = [0.3, 0.1, 0.2, 0.4]
    assert true_probability(probs, {0, 1}, {2}) == pytest.approx(0.4 / 0.6)


def test_no_mass_on_either_word_falls_back_to_half() -> None:
    assert true_probability([0.0, 0.0, 1.0], {0}, {1}) == 0.5


def test_all_mass_on_false() -> None:
    assert true_probability([0.0, 0.9, 0.1], {0}, {1}) == 0.0
