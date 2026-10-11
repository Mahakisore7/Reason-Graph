"""Tests for the answer-level graph (Phase 3): trees joined by NLI cross links."""

from __future__ import annotations

import pytest

from src.inference.hmt import LogisticEmission, posterior_root_true
from src.inference.loopy import run_loopy_bp
from src.pipeline import Calibration, CrossLink, build_answer_graph, label_cross_links
from src.pipeline import build_hidden_markov_tree as raw_tree_to_hmt
from src.tree.builder import RawBeliefTree, RawNode
from src.tree.relationships import FakeRelationshipLabeler, Relationship

CURVE = Calibration("curve", LogisticEmission(1.0, 0.0, base_rate_true=0.5), lambda c: c)


def _tree(
    statement: str, confidence: float, premise_confidence: float | None = None
) -> RawBeliefTree:
    nodes = {"n0": RawNode("n0", statement, confidence, None, None, "root")}
    if premise_confidence is not None:
        nodes["n0"].children.append("n1")
        nodes["n1"] = RawNode(
            "n1", "premise", premise_confidence, "n0", Relationship.REVERSE_ENTAILMENT, "premise"
        )
    return RawBeliefTree(nodes, "n0")


def _marginals(trees: list[RawBeliefTree], links: list[CrossLink]) -> list[float]:
    answer = build_answer_graph(trees, links, CURVE)
    result = run_loopy_bp(answer.graph, convergence_tol=1e-12, damping=0.5)
    assert result.converged
    return [result.marginals[r] for r in answer.root_names]


def test_without_links_each_sentence_keeps_its_tree_posterior() -> None:
    trees = [_tree("A", 0.8, premise_confidence=0.3), _tree("B", 0.4)]
    expected = [posterior_root_true(raw_tree_to_hmt(t, CURVE)) for t in trees]
    assert _marginals(trees, []) == pytest.approx(expected, abs=1e-9)


def test_contradiction_lowers_both_and_equivalence_pulls_together() -> None:
    trees = [_tree("A", 0.8), _tree("B", 0.7)]
    alone = _marginals(trees, [])
    contradicted = _marginals(trees, [CrossLink(1, 0, Relationship.CONTRADICTION)])
    assert contradicted[0] < alone[0] and contradicted[1] < alone[1]

    weak, strong = _tree("A", 0.3), _tree("B", 0.9)
    linked = _marginals([weak, strong], [CrossLink(1, 0, Relationship.EQUIVALENCE)])
    assert linked[0] > _marginals([weak], [])[0]


def test_three_mutually_linked_sentences_form_a_loop() -> None:
    trees = [_tree(s, 0.7) for s in "ABC"]
    links = [
        CrossLink(1, 0, Relationship.ENTAILMENT),
        CrossLink(2, 0, Relationship.ENTAILMENT),
        CrossLink(2, 1, Relationship.EQUIVALENCE),
    ]
    assert build_answer_graph(trees, links, CURVE).has_cycle
    assert not build_answer_graph(trees, links[:2], CURVE).has_cycle


def test_label_cross_links_keeps_only_related_pairs() -> None:
    labeler = FakeRelationshipLabeler(labels={("C", "A"): Relationship.CONTRADICTION})
    assert label_cross_links(["A", "B", "C"], labeler) == [
        CrossLink(child=2, parent=0, relationship=Relationship.CONTRADICTION)
    ]
    assert labeler.call_count == 3  # every unordered pair once


def test_link_rules_filter_contradictions() -> None:
    from src.pipeline import RuleLabeler
    from src.tree.nli_labeler import FakeDirectionalNLI

    one_sided = FakeDirectionalNLI(labels={("B", "A"): "contradiction"})
    both = FakeDirectionalNLI(labels={("B", "A"): "contradiction", ("A", "B"): "contradiction"})
    entails = FakeDirectionalNLI(labels={("B", "A"): "entailment"})
    assert RuleLabeler(one_sided, "any").label("B", "A") is Relationship.CONTRADICTION
    assert RuleLabeler(one_sided, "strict").label("B", "A") is Relationship.NEUTRAL
    assert RuleLabeler(both, "strict").label("B", "A") is Relationship.CONTRADICTION
    assert RuleLabeler(both, "no_contradiction").label("B", "A") is Relationship.NEUTRAL
    assert RuleLabeler(entails, "no_contradiction").label("B", "A") is Relationship.ENTAILMENT
