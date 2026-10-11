"""Phase 5: end-to-end checks of verify_statement / verify_answer against independently
computed answers."""

from __future__ import annotations

import itertools

import pytest

from src.inference.hmt import (
    DecompositionTransition,
    HiddenMarkovTree,
    LogisticEmission,
    TreeNode,
    posterior_root_true,
)
from src.llm.fake import FakeLLMClient
from src.llm.prompts import (
    decomposition_prompt,
    perturb_fillin_prompt,
    perturb_question_prompt,
    select_strategy_prompt,
    support_premises_prompt,
)
from src.pipeline import (
    CORRECTION_EMISSION,
    Calibration,
    RuleLabeler,
    verify_answer,
    verify_statement,
)
from src.tree.nli_labeler import FakeDirectionalNLI
from src.tree.relationships import (
    DEFAULT_TRANSITION_PARAMS,
    FakeRelationshipLabeler,
    Relationship,
)

CURVE = Calibration("curve", LogisticEmission(1.2, 0.3, base_rate_true=0.8), lambda c: c)
ROOT = "The Eiffel Tower, built in 1889, is in Berlin."


def test_decomposition_matches_phase0_tree_built_by_hand() -> None:
    claims = ["The Eiffel Tower was built in 1889.", "The Eiffel Tower is in Berlin."]
    llm = FakeLLMClient(
        completions={decomposition_prompt(ROOT): f"Claim 1: {claims[0]}\nClaim 2: {claims[1]}"},
        confidences={ROOT: 0.6, claims[0]: 0.9, claims[1]: 0.2},
    )
    labeler = FakeRelationshipLabeler(
        labels={(c, ROOT): Relationship.REVERSE_ENTAILMENT for c in claims}
    )
    result = verify_statement(llm, labeler, CURVE, ROOT)

    by_hand = HiddenMarkovTree(
        nodes={
            "r": TreeNode("r", 0.6, ("a", "b")),
            "a": TreeNode("a", 0.9),
            "b": TreeNode("b", 0.2),
        },
        root_id="r",
        emission=CURVE.emission,
        transition=DecompositionTransition(),
    )
    assert result.posterior_true == pytest.approx(posterior_root_true(by_hand), abs=1e-12)
    assert result.tree_size == 3


def test_premise_and_correction_each_use_their_own_relationship() -> None:
    premise, fixed, question = "Gustave Eiffel built it.", "It is in Paris.", "Where is it?"
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(ROOT): f"Claim 1: {ROOT}",
            select_strategy_prompt(ROOT): "Output: both",
            perturb_question_prompt(ROOT): f"Question: {question}",
            perturb_fillin_prompt("Paris", ROOT): f"Revised Answer: {fixed}",
        },
        samples={
            support_premises_prompt(ROOT): [f"Premise 1: {premise}", "Judgement: True"],
            question: ["Paris"],
        },
        confidences={ROOT: 0.7, premise: 0.3},
    )
    labeler = FakeRelationshipLabeler(
        labels={
            (premise, ROOT): Relationship.ENTAILMENT,
            (fixed, ROOT): Relationship.CONTRADICTION,
        }
    )
    result = verify_statement(llm, labeler, CURVE, ROOT)

    # Brute force over the root and both children, each child with its OWN parameters.
    e = CURVE.emission.prob
    unnorm = {True: 0.0, False: 0.0}
    for z_root, z_premise, z_fix in itertools.product((True, False), repeat=3):
        p_premise = DEFAULT_TRANSITION_PARAMS[Relationship.ENTAILMENT][0 if z_root else 1]
        p_fix = DEFAULT_TRANSITION_PARAMS[Relationship.CONTRADICTION][0 if z_root else 1]
        unnorm[z_root] += (
            0.5
            * e(0.7, z_root)
            * (p_premise if z_premise else 1 - p_premise)
            * e(0.3, z_premise)
            * (p_fix if z_fix else 1 - p_fix)
            * CORRECTION_EMISSION[z_fix]
        )
    expected = unnorm[True] / (unnorm[True] + unnorm[False])
    assert result.posterior_true == pytest.approx(expected, abs=1e-12)


def test_threshold_changes_the_verdict_not_the_probability() -> None:
    llm = FakeLLMClient(
        completions={decomposition_prompt(ROOT): "Claim 1: No check-worthy claims available."},
        confidences={ROOT: 0.6},
    )
    low = verify_statement(llm, FakeRelationshipLabeler(), CURVE, ROOT, trust_threshold=0.1)
    high = verify_statement(llm, FakeRelationshipLabeler(), CURVE, ROOT, trust_threshold=0.99)
    assert low.posterior_true == high.posterior_true
    assert low.is_trustworthy and not high.is_trustworthy


def test_root_only_tree_is_the_calibrated_confidence() -> None:
    llm = FakeLLMClient(confidences={ROOT: 0.6})
    result = verify_statement(llm, FakeRelationshipLabeler(), CURVE, ROOT, max_depth=0)
    # Prior 0.5 and emission p/base vs (1-p)/(1-base): Bayes by hand.
    p = CURVE.emission.p_true_given_score(0.6)  # type: ignore[attr-defined]
    expected = (p / 0.8) / (p / 0.8 + (1 - p) / 0.2)
    assert result.posterior_true == pytest.approx(expected)


def test_verify_answer_links_related_sentences_and_reports_convergence() -> None:
    a, b = "Paris is the capital of France.", "France's capital is Paris."
    llm = FakeLLMClient(confidences={a: 0.9, b: 0.4})
    nli = FakeDirectionalNLI(labels={(b, a): "entailment", (a, b): "entailment"})
    result = verify_answer(
        llm, FakeRelationshipLabeler(), RuleLabeler(nli), CURVE, [a, b], max_depth=0
    )
    assert [x.relationship for x in result.links] == [Relationship.EQUIVALENCE]
    assert result.converged
    assert result.posterior_true[1] > result.tree_only[1]  # the confident twin lifts it
