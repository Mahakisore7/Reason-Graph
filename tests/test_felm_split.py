"""Tests for the fit/test split -- pure function, no dataset download."""

from __future__ import annotations

from scripts.fit_emission_model import split_by_response


def _responses() -> list[list[tuple[str, bool]]]:
    with_errors = [[(f"err{i}-a", True), (f"err{i}-b", False)] for i in range(10)]
    error_free = [[(f"ok{i}-a", True), (f"ok{i}-b", True)] for i in range(10)]
    return with_errors + error_free


def test_no_answer_is_split_across_halves() -> None:
    fit, test = split_by_response(_responses(), fit_fraction=0.5, seed=0)
    fit_answers = {s.split("-")[0] for s, _ in fit}
    test_answers = {s.split("-")[0] for s, _ in test}
    assert not fit_answers & test_answers
    assert len(fit) + len(test) == 40


def test_false_examples_are_shared_evenly() -> None:
    fit, test = split_by_response(_responses(), fit_fraction=0.5, seed=0)
    assert sum(1 for _, label in fit if not label) == 5
    assert sum(1 for _, label in test if not label) == 5


def test_same_seed_gives_same_split() -> None:
    assert split_by_response(_responses(), 0.5, seed=7) == split_by_response(
        _responses(), 0.5, seed=7
    )
