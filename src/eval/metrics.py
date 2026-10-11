"""Evaluation metrics, as in the paper: AUROC, AUC-PR and F1 for detecting false statements.

Positive class = FALSE (a hallucination), since the task is to catch them; a statement's
"hallucination score" is 1 - P(true). AUROC is the same either way round; AUC-PR and F1 are not.

F1 needs a cut-off ("call it false below this P(true)"). Picking the cut-off on the very
statements being scored would flatter the number, so it is chosen by cross-fitting: answers are
split into folds, and each fold is classified with the cut-off that maximises F1 on the OTHER
folds. No statement ever helps choose its own cut-off.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def auroc(labels: Sequence[bool], p_true: Sequence[float]) -> float:
    return float(roc_auc_score(labels, p_true))


def auc_pr_false(labels: Sequence[bool], p_true: Sequence[float]) -> float:
    """Average precision for finding the false statements."""
    return float(average_precision_score([not y for y in labels], [1 - p for p in p_true]))


def best_threshold(labels: Sequence[bool], p_true: Sequence[float]) -> float:
    """The cut-off t maximising F1 when every statement with P(true) < t is called false."""
    candidates = sorted(set(p_true)) + [float("inf")]
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = f1_false(labels, p_true, t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def f1_false(labels: Sequence[bool], p_true: Sequence[float], threshold: float) -> float:
    predicted_false = [p < threshold for p in p_true]
    return float(f1_score([not y for y in labels], predicted_false, zero_division=0.0))


def cross_fitted_f1(
    labels: Sequence[bool],
    p_true: Sequence[float],
    groups: Sequence[int],
    *,
    num_folds: int = 5,
    seed: int = 42,
) -> float:
    """F1 for false statements, each group (answer) classified with a cut-off chosen on the
    other folds only. All predictions are pooled before computing F1."""
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    fold_of = {g: i % num_folds for i, g in enumerate(unique)}
    predicted_false = [False] * len(labels)
    for f in range(num_folds):
        train = [i for i, g in enumerate(groups) if fold_of[g] != f]
        test = [i for i, g in enumerate(groups) if fold_of[g] == f]
        if not test:
            continue
        t = best_threshold([labels[i] for i in train], [p_true[i] for i in train])
        for i in test:
            predicted_false[i] = p_true[i] < t
    return float(f1_score([not y for y in labels], predicted_false, zero_division=0.0))


def all_metrics(
    labels: Sequence[bool], p_true: Sequence[float], groups: Sequence[int]
) -> dict[str, float]:
    return {
        "auroc": auroc(labels, p_true),
        "auc_pr_false": auc_pr_false(labels, p_true),
        "f1_false": cross_fitted_f1(labels, p_true, groups),
        "false_rate": sum(not y for y in labels) / len(labels),  # AUC-PR of random guessing
    }
