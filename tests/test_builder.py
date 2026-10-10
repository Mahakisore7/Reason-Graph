"""Tests for build_belief_tree (official BTProp flow) -- FakeLLMClient + FakeRelationshipLabeler."""

from __future__ import annotations

from src.llm.fake import FakeLLMClient
from src.llm.prompts import (
    decomposition_prompt,
    perturb_fillin_prompt,
    perturb_question_prompt,
    select_strategy_prompt,
    support_premises_prompt,
)
from src.tree.builder import build_belief_tree
from src.tree.relationships import FakeRelationshipLabeler, Relationship

ROOT = "The Eiffel Tower, built in 1889, is in Berlin."
ONE_CLAIM = f"Claim 1: {ROOT}"


def _premise_llm(premises: list[str]) -> FakeLLMClient:
    lines = "\n".join(f"Premise {i + 1}: {p}" for i, p in enumerate(premises))
    return FakeLLMClient(
        completions={
            decomposition_prompt(ROOT): ONE_CLAIM,
            select_strategy_prompt(ROOT): "Output: Logical Relation",
        },
        samples={support_premises_prompt(ROOT): [f"Judgement: True\n{lines}", "Judgement: False"]},
        confidences={ROOT: 0.9},
    )


def test_decomposition_children_are_nli_filtered() -> None:
    claims = ["The Eiffel Tower was built in 1889.", "The Eiffel Tower is in Berlin.", "Cats purr."]
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(ROOT): "\n".join(
                f"Claim {i + 1}: {c}" for i, c in enumerate(claims)
            )
        }
    )
    labeler = FakeRelationshipLabeler(
        labels={
            (claims[0], ROOT): Relationship.REVERSE_ENTAILMENT,
            (claims[1], ROOT): Relationship.REVERSE_ENTAILMENT,
        }
    )
    tree = build_belief_tree(llm, labeler, ROOT)
    kids = [tree.nodes[c] for c in tree.nodes[tree.root_id].children]
    assert [k.statement for k in kids] == claims[:2]  # "Cats purr." is unrelated -> dropped
    assert all(k.strategy == "decomposition" for k in kids)


def test_nothing_checkable_means_no_children() -> None:
    llm = FakeLLMClient(
        completions={decomposition_prompt(ROOT): "Claim 1: No check-worthy claims available."}
    )
    tree = build_belief_tree(llm, FakeRelationshipLabeler(), ROOT)
    assert tree.nodes[tree.root_id].children == []


def test_premises_drop_unrelated_and_keep_at_most_three() -> None:
    premises = ["P1.", "P2.", "Unrelated.", "P3.", "P4."]
    related = {(p, ROOT): Relationship.REVERSE_ENTAILMENT for p in premises if p != "Unrelated."}
    llm = _premise_llm(premises)
    tree = build_belief_tree(llm, FakeRelationshipLabeler(labels=related), ROOT)
    kids = [tree.nodes[c] for c in tree.nodes[tree.root_id].children]
    assert [k.statement for k in kids] == ["P1.", "P2.", "P3."]
    assert all(k.strategy == "premise" for k in kids)
    assert llm.confidence_call_count == 1 + 3  # root + each kept premise


def test_corrections_skip_copies_of_parent_and_have_fixed_confidence() -> None:
    question = "Where is the Eiffel Tower?"
    fixed = "The Eiffel Tower, built in 1889, is in Paris."
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(ROOT): ONE_CLAIM,
            select_strategy_prompt(ROOT): "Output: Statement Perturbation",
            perturb_question_prompt(ROOT): f"Question: {question}",
            perturb_fillin_prompt("Paris", ROOT): f"Revised Answer: {fixed}",
            perturb_fillin_prompt("Berlin", ROOT): f"Revised Answer: {ROOT}",
        },
        samples={question: ["Paris", "Berlin"]},
        confidences={ROOT: 0.9},
    )
    labeler = FakeRelationshipLabeler(labels={(fixed, ROOT): Relationship.CONTRADICTION})
    tree = build_belief_tree(llm, labeler, ROOT, max_depth=2)
    kids = [tree.nodes[c] for c in tree.nodes[tree.root_id].children]
    assert [k.statement for k in kids] == [fixed]  # the unchanged copy is skipped
    assert kids[0].confidence == 1.0
    assert kids[0].relationship_to_parent is Relationship.CONTRADICTION
    assert kids[0].children == []  # corrections are leaves even below max_depth
    assert llm.confidence_call_count == 1  # only the root was scored


def test_tree_stops_at_max_depth() -> None:
    llm = _premise_llm(["P1."])
    labeler = FakeRelationshipLabeler(labels={("P1.", ROOT): Relationship.REVERSE_ENTAILMENT})
    # If depth control were broken this would try to expand "P1." and hit a missing canned
    # completion (KeyError).
    tree = build_belief_tree(llm, labeler, ROOT, max_depth=1)
    premise = tree.nodes[tree.nodes[tree.root_id].children[0]]
    assert premise.children == []
