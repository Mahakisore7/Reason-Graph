"""Tests for the evaluation metrics."""

from __future__ import annotations

import pytest

from src.eval.metrics import auc_pr_false, best_threshold, cross_fitted_f1, f1_false

LABELS = [True, True, False, True, False, True]
PERFECT = [0.9, 0.8, 0.1, 0.7, 0.2, 0.95]  # every false statement scores lowest


def test_perfect_scores_give_perfect_metrics() -> None:
    assert auc_pr_false(LABELS, PERFECT) == pytest.approx(1.0)
    t = best_threshold(LABELS, PERFECT)
    assert f1_false(LABELS, PERFECT, t) == pytest.approx(1.0)


def test_auc_pr_treats_false_as_positive() -> None:
    reversed_scores = [1 - p for p in PERFECT]  # false statements now look most true
    assert auc_pr_false(LABELS, reversed_scores) < 0.5


def test_cross_fitted_threshold_never_sees_its_own_group() -> None:
    # Groups 0-3: true 0.5, false 0.2. Group 4: true 0.9, false 0.7 -- only a cut-off chosen
    # ON group 4 would catch that 0.7; the other groups' cut-off (between 0.2 and 0.5) misses it.
    labels = [True, False] * 5
    scores = [0.5, 0.2] * 4 + [0.9, 0.7]
    groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    f1 = cross_fitted_f1(labels, scores, groups, num_folds=5)
    assert f1 == pytest.approx(2 * 4 / (2 * 4 + 1))  # 4 caught, 1 missed, 0 false alarms
