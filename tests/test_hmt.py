"""Hand-computed checkpoints for the exact Hidden Markov Tree inference (Phase 0).

Every expected value here was derived independently on paper before being written down
(see docs/phases/phase-0-setup-and-first-algorithm.md) -- if a test fails, the bug is in
src/inference/hmt.py, not in these expected numbers.
"""

import pytest

from src.inference.hmt import (
    DecompositionTransition,
    HiddenMarkovTree,
    IndependentChildTransition,
    LogisticEmission,
    LookupEmission,
    TreeNode,
    posterior_root_true,
)

EMISSION_TABLE = {
    0.9: {True: 0.8, False: 0.1},
    0.2: {True: 0.2, False: 0.7},
    0.6: {True: 0.6, False: 0.4},
}


def test_checkpoint_1_single_child() -> None:
    tree = HiddenMarkovTree(
        nodes={
            "root": TreeNode("root", score=0.9, children=("child",)),
            "child": TreeNode("child", score=0.2),
        },
        root_id="root",
        emission=LookupEmission(EMISSION_TABLE),
        transition=IndependentChildTransition(
            p_true_given_parent_true=0.9, p_true_given_parent_false=0.3
        ),
        prior_true=0.5,
    )
    assert posterior_root_true(tree) == pytest.approx(0.7843137254901961)


def test_checkpoint_2_two_independent_children() -> None:
    tree = HiddenMarkovTree(
        nodes={
            "root": TreeNode("root", score=0.9, children=("child1", "child2")),
            "child1": TreeNode("child1", score=0.2),
            "child2": TreeNode("child2", score=0.6),
        },
        root_id="root",
        emission=LookupEmission(EMISSION_TABLE),
        transition=IndependentChildTransition(
            p_true_given_parent_true=0.9, p_true_given_parent_false=0.3
        ),
        prior_true=0.5,
    )
    assert posterior_root_true(tree) == pytest.approx(0.8209483368719038)


def test_checkpoint_3_leaf_only_tree() -> None:
    tree = HiddenMarkovTree(
        nodes={"root": TreeNode("root", score=0.9)},
        root_id="root",
        emission=LookupEmission(EMISSION_TABLE),
        transition=IndependentChildTransition(
            p_true_given_parent_true=0.9, p_true_given_parent_false=0.3
        ),
        prior_true=0.5,
    )
    assert posterior_root_true(tree) == pytest.approx(0.8888888888888888)


def test_decomposition_transition_parent_true() -> None:
    transition = DecompositionTransition()
    assert transition.joint_prob("root", True, (True, True)) == pytest.approx(1.0)
    assert transition.joint_prob("root", True, (True, False)) == pytest.approx(0.0)
    assert transition.joint_prob("root", True, (False, True)) == pytest.approx(0.0)
    assert transition.joint_prob("root", True, (False, False)) == pytest.approx(0.0)


def test_decomposition_transition_parent_false() -> None:
    transition = DecompositionTransition()
    assert transition.joint_prob("root", False, (True, True)) == pytest.approx(0.0)
    for config in [(True, False), (False, True), (False, False)]:
        assert transition.joint_prob("root", False, config) == pytest.approx(1 / 3)


def test_malformed_tree_bad_root_raises() -> None:
    with pytest.raises(ValueError, match="root_id"):
        HiddenMarkovTree(
            nodes={"a": TreeNode("a", score=0.5)},
            root_id="does-not-exist",
            emission=LookupEmission(EMISSION_TABLE),
            transition=IndependentChildTransition(0.9, 0.3),
        )


def test_malformed_tree_dangling_child_raises() -> None:
    with pytest.raises(ValueError, match="child"):
        HiddenMarkovTree(
            nodes={"root": TreeNode("root", score=0.9, children=("missing",))},
            root_id="root",
            emission=LookupEmission(EMISSION_TABLE),
            transition=IndependentChildTransition(0.9, 0.3),
        )


def test_lookup_emission_unknown_score_raises() -> None:
    with pytest.raises(KeyError):
        LookupEmission(EMISSION_TABLE).prob(0.42, True)


def test_logistic_emission_is_monotone_in_score() -> None:
    emission = LogisticEmission(slope=1.5, intercept=0.3, base_rate_true=0.8)
    ratios = [emission.prob(s, True) / emission.prob(s, False) for s in (0.0, 0.2, 0.5, 0.8, 1.0)]
    assert ratios == sorted(ratios)


def test_logistic_emission_single_node_posterior_recovers_calibrated_probability() -> None:
    # With the prior set to the base rate, a one-node tree's posterior must equal the fitted
    # P(true | score) -- checks the p(S) cancellation argument in LogisticEmission's docstring.
    emission = LogisticEmission(slope=1.2, intercept=-0.4, base_rate_true=0.7)
    tree = HiddenMarkovTree(
        nodes={"root": TreeNode(node_id="root", score=0.65)},
        root_id="root",
        emission=emission,
        transition=IndependentChildTransition(0.9, 0.1),
        prior_true=0.7,
    )
    assert posterior_root_true(tree) == pytest.approx(emission.p_true_given_score(0.65))
