"""Tests for the Phase 4 experiment helpers."""

from __future__ import annotations

from scripts.evaluate_surrogate import (
    ScoredNode,
    cross_fitted_predictions,
    scored_nodes,
    with_confidences,
)
from src.tree.builder import RawBeliefTree, RawNode
from src.tree.relationships import Relationship


def _tree() -> RawBeliefTree:
    root = RawNode("n0", "Root claim here.", 0.7, None, None, "root", ["n1", "n2"])
    premise = RawNode("n1", "A premise.", 0.4, "n0", Relationship.REVERSE_ENTAILMENT, "premise")
    fix = RawNode("n2", "A correction.", 1.0, "n0", Relationship.CONTRADICTION, "correction")
    return RawBeliefTree({"n0": root, "n1": premise, "n2": fix}, "n0")


def test_corrections_are_not_llm_scored_nodes() -> None:
    assert [(n.node_id, n.is_root) for n in scored_nodes([_tree()])] == [
        ("n0", True),
        ("n1", False),
    ]


def test_with_confidences_copies_instead_of_mutating() -> None:
    tree = _tree()
    changed = with_confidences(tree, {"n1": 0.9})
    assert changed.nodes["n1"].confidence == 0.9
    assert tree.nodes["n1"].confidence == 0.4


def test_a_node_is_never_predicted_by_a_model_that_saw_its_answer() -> None:
    # Answer 0's text appears nowhere else; if its own nodes leaked into training, ridge would
    # reproduce their unusual target (0.0) almost exactly.
    nodes = [ScoredNode(0, "n0", True, "zebra quartz xylophone", 0.0)] + [
        ScoredNode(k, "n0", True, f"ordinary sentence number {k}", 0.8) for k in range(1, 10)
    ]
    predictions = cross_fitted_predictions(nodes, list(range(10)), 10, extra_train=[])
    assert predictions[0] > 0.5
