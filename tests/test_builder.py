"""Tests for build_belief_tree -- all against FakeLLMClient + FakeRelationshipLabeler, zero
network access needed."""

from __future__ import annotations

from src.llm.fake import FakeLLMClient
from src.llm.prompts import (
    correction_answer_prompt,
    correction_question_prompt,
    correction_statement_prompt,
    decomposition_prompt,
    premises_prompt,
)
from src.tree.builder import build_belief_tree
from src.tree.relationships import FakeRelationshipLabeler, Relationship

ROOT = "The Eiffel Tower, built in 1889, is located in Paris."


def test_decomposition_child_gets_equivalence_and_correct_strategy_label() -> None:
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(ROOT): (
                "1. The Eiffel Tower was built in 1889.\n2. The Eiffel Tower is in Paris."
            ),
        },
        confidences={ROOT: 0.9},
        default_confidence=0.7,
    )
    labeler = FakeRelationshipLabeler()

    # max_depth=1: root is processed (splits into 2 children), but the children -- now at
    # depth 1 -- are never processed themselves, so no further canned completions are needed.
    tree = build_belief_tree(llm, labeler, ROOT, max_depth=1, correction_samples=1)

    root = tree.nodes[tree.root_id]
    assert len(root.children) == 2
    for child_id in root.children:
        child = tree.nodes[child_id]
        assert child.strategy == "decomposition"
        assert child.relationship_to_parent is Relationship.EQUIVALENCE
        assert child.parent_id == tree.root_id


def test_neutral_labelled_premise_is_dropped() -> None:
    supported_text = "Gustave Eiffel's company designed the tower."
    contradicted_text = "The tower was designed by a different engineer."
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(ROOT): "NONE",
            premises_prompt(ROOT): f"SUPPORT: {supported_text}\nCONTRADICT: {contradicted_text}",
            correction_question_prompt(ROOT): "q",
            correction_answer_prompt("q"): "a",
            correction_statement_prompt(ROOT, "q", "a"): "corrected",
        },
        confidences={ROOT: 0.9, supported_text: 0.8, contradicted_text: 0.2},
        default_confidence=0.5,
    )
    labeler = FakeRelationshipLabeler(
        labels={
            (supported_text, ROOT): Relationship.ENTAILMENT,
            (contradicted_text, ROOT): Relationship.NEUTRAL,
        }
    )

    tree = build_belief_tree(llm, labeler, ROOT, max_depth=1, correction_samples=1)

    root = tree.nodes[tree.root_id]
    statements = {tree.nodes[cid].statement for cid in root.children}
    assert supported_text in statements
    assert contradicted_text not in statements


def test_correction_child_confidence_is_fixed_not_reprobed() -> None:
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(ROOT): "NONE",
            premises_prompt(ROOT): "",
            correction_question_prompt(ROOT): "q",
            correction_answer_prompt("q"): "a",
            correction_statement_prompt(ROOT, "q", "a"): "corrected statement",
        },
        confidences={ROOT: 0.9},
        default_confidence=0.5,
    )
    labeler = FakeRelationshipLabeler()

    tree = build_belief_tree(llm, labeler, ROOT, max_depth=1, correction_samples=1)

    root = tree.nodes[tree.root_id]
    correction_children = [
        tree.nodes[cid] for cid in root.children if tree.nodes[cid].strategy == "correction"
    ]
    assert len(correction_children) == 1
    assert correction_children[0].confidence == 1.0
    assert correction_children[0].statement == "corrected statement"
    # Confidence probed only for the root -- never for the fixed-confidence correction child.
    assert llm.confidence_call_count == 1


def test_tree_stops_growing_exactly_at_max_depth() -> None:
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(ROOT): "NONE",
            premises_prompt(ROOT): "",
            correction_question_prompt(ROOT): "q0",
            correction_answer_prompt("q0"): "a0",
            correction_statement_prompt(ROOT, "q0", "a0"): "level1 statement",
        },
        confidences={ROOT: 0.9},
        default_confidence=0.5,
    )
    labeler = FakeRelationshipLabeler()

    # If depth control were broken, this would try to expand "level1 statement" too and
    # crash with KeyError (no canned completion for it) -- passing proves the stop is exact.
    tree = build_belief_tree(llm, labeler, ROOT, max_depth=1, correction_samples=1)

    root = tree.nodes[tree.root_id]
    assert len(root.children) == 1
    leaf = tree.nodes[root.children[0]]
    assert leaf.children == []
