"""Exact inference on a Hidden Markov Tree via the upward-downward (belief propagation)
algorithm. This is the probabilistic model behind BTProp (Hou et al., NAACL 2025). See
docs/phases/04-research-notes-btprop.md for the full derivation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class TreeNode:
    """One claim in the belief tree.

    node_id: unique identifier for this node.
    score: the observed confidence score S_u (e.g. from prompting an LLM with
        "True or False? {statement}" and normalising its next-token probabilities).
    children: IDs of this node's children, in a fixed order (order matters: it must match
        the order your TransitionModel expects for its `children_truth` tuples).
    """

    node_id: str
    score: float
    children: tuple[str, ...] = field(default_factory=tuple)


class EmissionModel(Protocol):
    """p(S_u | Z_u = truth) -- how reliable the LLM's confidence score tends to be."""

    def prob(self, score: float, truth: bool) -> float: ...


class TransitionModel(Protocol):
    """p(Z_children = children_truth | Z_parent = parent_truth), jointly over all children.

    children_truth is a tuple of bools in the same order as TreeNode.children.
    """

    def joint_prob(
        self, parent_id: str, parent_truth: bool, children_truth: tuple[bool, ...]
    ) -> float: ...


@dataclass(frozen=True)
class HiddenMarkovTree:
    """A full belief tree plus the probability models needed to run inference on it.

    nodes: every node, keyed by node_id.
    root_id: the node whose posterior truthfulness you ultimately want.
    emission: shared emission model, p(S_u | Z_u).
    transition: shared transition model, p(Z_children | Z_parent).
    prior_true: p(Z_u = True) before observing anything.
    """

    nodes: dict[str, TreeNode]
    root_id: str
    emission: EmissionModel
    transition: TransitionModel
    prior_true: float = 0.5

    def __post_init__(self) -> None:
        if self.root_id not in self.nodes:
            raise ValueError(f"root_id {self.root_id!r} is not in nodes")
        for node in self.nodes.values():
            for child_id in node.children:
                if child_id not in self.nodes:
                    raise ValueError(
                        f"node {node.node_id!r} references child {child_id!r}, "
                        "which is not in nodes"
                    )


@dataclass(frozen=True)
class LookupEmission:
    """An emission model backed by an explicit {score: {True: p, False: p}} table.

    In the base paper this table is fit empirically by binning [0, 1] into ranges and
    counting how often each occurs among true vs. false statements
    (docs/phases/04-research-notes-btprop.md, Step 4) -- a Phase 2 upgrade, once real
    labelled data exists. For now, an unrecognised score fails loudly rather than guessing.
    """

    table: dict[float, dict[bool, float]]

    def prob(self, score: float, truth: bool) -> float:
        if score not in self.table:
            raise KeyError(
                f"no emission probability for score={score!r}; "
                f"known scores are {sorted(self.table)}"
            )
        return self.table[score][truth]


@dataclass(frozen=True)
class LogisticEmission:
    """A smooth emission model: P(true | score) = sigmoid(slope * logit(score) + intercept),
    fit on labelled data (Platt scaling). With positive slope, a higher score can never count
    as weaker evidence of truth -- unlike a binned table fit on few examples, which can flip.

    HMT inference needs p(S | Z), not P(Z | S). Bayes gives p(S | Z) = P(Z | S) p(S) / P(Z);
    p(S) is the same for Z=True and Z=False, so it cancels in the posterior, and dividing by
    the base rate P(Z) of the data the curve was fit on is enough.
    """

    slope: float
    intercept: float
    base_rate_true: float
    eps: float = 1e-6  # keeps logit finite for scores of exactly 0 or 1

    def p_true_given_score(self, score: float) -> float:
        s = min(max(score, self.eps), 1 - self.eps)
        z = self.slope * math.log(s / (1 - s)) + self.intercept
        return 1 / (1 + math.exp(-z))

    def prob(self, score: float, truth: bool) -> float:
        p = self.p_true_given_score(score)
        return p / self.base_rate_true if truth else (1 - p) / (1 - self.base_rate_true)


@dataclass(frozen=True)
class IndependentChildTransition:
    """Transition model for the Premises / Correction strategies. Each child is
    conditionally independent of its siblings given the parent
    (docs/phases/04-research-notes-btprop.md, Step 4).
    """

    p_true_given_parent_true: float
    p_true_given_parent_false: float

    def joint_prob(
        self, parent_id: str, parent_truth: bool, children_truth: tuple[bool, ...]
    ) -> float:
        p_true = self.p_true_given_parent_true if parent_truth else self.p_true_given_parent_false
        joint = 1.0
        for child_truth in children_truth:
            joint *= p_true if child_truth else (1.0 - p_true)
        return joint


@dataclass(frozen=True)
class DecompositionTransition:
    """Transition model for the Statement Decomposition strategy. Children are jointly
    equivalent to the parent: if the parent is True, every child must be True; if the
    parent is False, at least one child must be False, with the remaining probability
    spread uniformly over every other joint configuration
    (docs/phases/04-research-notes-btprop.md, Step 4).
    """

    def joint_prob(
        self, parent_id: str, parent_truth: bool, children_truth: tuple[bool, ...]
    ) -> float:
        all_children_true = all(children_truth)
        if parent_truth:
            return 1.0 if all_children_true else 0.0
        if all_children_true:
            return 0.0
        num_children = len(children_truth)
        # int ** int types as Any in typeshed (a negative exponent would return float),
        # so shift instead: unambiguously an int, no cast needed.
        num_configurations = (1 << num_children) - 1
        return 1.0 / num_configurations


def compute_beta(tree: HiddenMarkovTree, node_id: str) -> dict[bool, float]:
    """Recursively compute beta(z, node_id) for z in {True, False}.

    beta(z, u) = p(every observed score in u's subtree | Z_u = z)

    Leaves: beta(z, u) = p(S_u | Z_u = z)                                    [emission only]
    Internal nodes: beta(z, u) = p(S_u | Z_u = z) * sum over joint child truth
                     assignments of [ p(children | Z_u = z) * product of children's beta ]

    See docs/phases/04-research-notes-btprop.md, Step 5, for the full derivation.
    """
    node = tree.nodes[node_id]

    if not node.children:
        return {
            True: tree.emission.prob(node.score, True),
            False: tree.emission.prob(node.score, False),
        }

    children_beta = [compute_beta(tree, child_id) for child_id in node.children]

    beta: dict[bool, float] = {}
    for parent_truth in (True, False):
        total = 0.0
        for children_truth in itertools.product([True, False], repeat=len(node.children)):
            term = tree.transition.joint_prob(node_id, parent_truth, children_truth)
            for child_beta, child_truth in zip(children_beta, children_truth, strict=True):
                term *= child_beta[child_truth]
            total += term
        beta[parent_truth] = tree.emission.prob(node.score, parent_truth) * total

    return beta


def posterior_root_true(tree: HiddenMarkovTree) -> float:
    """P(Z_root = True | every observed score in the tree) -- the final, corrected trust
    score. Combines beta(True, root) and beta(False, root) with the prior via Bayes' rule.
    """
    beta = compute_beta(tree, tree.root_id)
    unnorm_true = beta[True] * tree.prior_true
    unnorm_false = beta[False] * (1 - tree.prior_true)

    denominator = unnorm_true + unnorm_false
    if denominator == 0:
        raise ValueError(
            "both Z_root=True and Z_root=False have zero likelihood given the observed "
            "scores -- the emission/transition model is inconsistent with this tree's data"
        )
    return unnorm_true / denominator
