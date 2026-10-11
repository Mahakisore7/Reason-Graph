"""End-to-end wiring (Phase 5): statement -> belief tree -> probabilities -> verdict.

    verify_statement(...)  the paper's method on one statement: build its belief tree, turn it
                           into a Hidden Markov Tree, compute P(statement is true).
    verify_answer(...)     our extension (D33, D35): every sentence of an answer gets its tree,
                           related sentences are linked, loopy BP scores them all together.

Both return their working (trees, links) so a verdict can be inspected, not just read off.
The experiment scripts use these same functions, so the numbers in docs/project-log come from
exactly this code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.inference.hmt import (
    DecompositionTransition,
    EmissionModel,
    HiddenMarkovTree,
    TreeNode,
    posterior_root_true,
)
from src.inference.loopy import FactorGraph, LoopyBPResult, add_tree, pairwise_factor, run_loopy_bp
from src.llm.base import LLMClient
from src.tree.builder import RawBeliefTree, build_belief_tree
from src.tree.nli_labeler import DirectionalNLI, NLIRelationshipLabeler
from src.tree.relationships import DEFAULT_TRANSITION_PARAMS, Relationship, RelationshipLabeler

# Corrections have confidence fixed at 1.0 and their own emission in the official code
# (configs/hmm.yaml, perturb_emission_probs, top bucket): 0.8 if true, 0.2 if false. They're
# marked with a score outside [0, 1] so one emission model can serve the whole tree.
CORRECTION_SCORE = -1.0
CORRECTION_EMISSION = {True: 0.8, False: 0.2}

# Cross links between sentences (D33): probabilities kept inside [LINK_EPS, 1 - LINK_EPS] so an
# NLI mistake makes two sentences unlikely together, never impossible. Fixed in advance.
LINK_EPS = 0.05
MAX_ITERATIONS = 200
CONVERGENCE_TOL = 1e-6
DAMPING = 0.5  # fixed in advance; only affects speed of settling, not the fixed point
LINK_RULES = ("any", "strict", "no_contradiction")
DEFAULT_LINK_RULE = "no_contradiction"  # chosen on the fit half (D35)


@dataclass(frozen=True)
class WithCorrectionEmission:
    base: EmissionModel

    def prob(self, score: float, truth: bool) -> float:
        if score == CORRECTION_SCORE:
            return CORRECTION_EMISSION[truth]
        return self.base.prob(score, truth)


@dataclass(frozen=True)
class Calibration:
    """One way of turning an LLM confidence into evidence: an emission model, plus how a raw
    confidence becomes the score that model expects (a bucket index for the table, the
    confidence itself for the smooth curve)."""

    name: str
    emission: EmissionModel
    to_score: Callable[[float], float]


@dataclass
class DispatchingTransition:
    """One TransitionModel for a whole tree whose parents need different rules. Phase 0's
    joint_prob receives parent_id on every call, so each parent's rule is looked up directly --
    no state carried between calls:
        decomposition children -> DecompositionTransition (parent true iff all children true);
        premise / correction   -> each child independent, with its own relationship's
                                  (P(child true | parent true), P(child true | parent false)).
    """

    raw_tree: RawBeliefTree
    relationship_params: dict[Relationship, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_TRANSITION_PARAMS)
    )
    _decomposition: DecompositionTransition = field(
        default_factory=DecompositionTransition, init=False, repr=False
    )

    def joint_prob(
        self, parent_id: str, parent_truth: bool, children_truth: tuple[bool, ...]
    ) -> float:
        child_ids = self.raw_tree.nodes[parent_id].children
        if not child_ids:
            return 1.0  # nothing to distribute: the empty configuration is certain
        first_child = self.raw_tree.nodes[child_ids[0]]
        if first_child.strategy == "decomposition":
            return self._decomposition.joint_prob(parent_id, parent_truth, children_truth)

        probability = 1.0
        for child_id, child_truth in zip(child_ids, children_truth, strict=True):
            relationship = self.raw_tree.nodes[child_id].relationship_to_parent
            assert relationship is not None, "every non-root node is NLI-labelled"
            p_true_given_true, p_true_given_false = self.relationship_params[relationship]
            p_true = p_true_given_true if parent_truth else p_true_given_false
            probability *= p_true if child_truth else (1.0 - p_true)
        return probability


def build_hidden_markov_tree(raw: RawBeliefTree, calibration: Calibration) -> HiddenMarkovTree:
    nodes = {
        node_id: TreeNode(
            node_id=node_id,
            score=(
                CORRECTION_SCORE
                if raw_node.strategy == "correction"
                else calibration.to_score(raw_node.confidence)
            ),
            children=tuple(raw_node.children),
        )
        for node_id, raw_node in raw.nodes.items()
    }
    return HiddenMarkovTree(
        nodes=nodes,
        root_id=raw.root_id,
        emission=WithCorrectionEmission(calibration.emission),
        transition=DispatchingTransition(raw),
    )


@dataclass(frozen=True)
class VerificationResult:
    statement: str
    posterior_true: float
    is_trustworthy: bool
    tree_size: int
    raw_tree: RawBeliefTree


def verify_statement(
    llm: LLMClient,
    labeler: RelationshipLabeler,
    calibration: Calibration,
    statement: str,
    *,
    max_depth: int = 1,
    trust_threshold: float = 0.5,
) -> VerificationResult:
    """The paper's method on one statement. The threshold is a decision rule applied after the
    posterior is fixed; it never changes the posterior."""
    raw = build_belief_tree(llm, labeler, statement, max_depth=max_depth)
    posterior = posterior_root_true(build_hidden_markov_tree(raw, calibration))
    return VerificationResult(
        statement=statement,
        posterior_true=posterior,
        is_trustworthy=posterior >= trust_threshold,
        tree_size=len(raw.nodes),
        raw_tree=raw,
    )


# ---- Our extension: one graph per answer ----------------------------------------------------


@dataclass(frozen=True)
class CrossLink:
    child: int  # index of the sentence used as NLI premise
    parent: int  # index of the sentence used as NLI hypothesis
    relationship: Relationship


def soften(p: float) -> float:
    return min(max(p, LINK_EPS), 1 - LINK_EPS)


@dataclass
class RuleLabeler:
    """Which NLI answers count as a cross link (D35). Tree links always use the paper's
    labeller; only links BETWEEN sentences go through a rule:
        any               -- the paper's labeller: contradiction if either direction says so;
        strict            -- contradiction only if BOTH directions say so;
        no_contradiction  -- contradiction links dropped, entailment-type links kept.
    """

    directional: DirectionalNLI
    rule: str = DEFAULT_LINK_RULE

    def label(self, premise: str, hypothesis: str) -> Relationship:
        base = NLIRelationshipLabeler(self.directional).label(premise, hypothesis)
        if base is not Relationship.CONTRADICTION or self.rule == "any":
            return base
        if self.rule == "no_contradiction":
            return Relationship.NEUTRAL
        forward = self.directional.label_direction(premise, hypothesis)
        backward = self.directional.label_direction(hypothesis, premise)
        both = forward == "contradiction" and backward == "contradiction"
        return Relationship.CONTRADICTION if both else Relationship.NEUTRAL


def label_cross_links(statements: list[str], labeler: RelationshipLabeler) -> list[CrossLink]:
    """One label per unordered pair (the labeler already checks both directions). The later
    sentence is the 'child' only as a naming convention -- the relationship carries the
    direction."""
    links = []
    for parent in range(len(statements)):
        for child in range(parent + 1, len(statements)):
            relationship = labeler.label(statements[child], statements[parent])
            if relationship is not Relationship.NEUTRAL:
                links.append(CrossLink(child, parent, relationship))
    return links


@dataclass
class AnswerGraph:
    graph: FactorGraph
    root_names: list[str]
    has_cycle: bool = field(init=False)

    def __post_init__(self) -> None:
        self.has_cycle = self.graph.has_cycle()


def build_answer_graph(
    trees: list[RawBeliefTree], links: list[CrossLink], calibration: Calibration
) -> AnswerGraph:
    graph = FactorGraph()
    root_names = []
    for k, raw in enumerate(trees):
        prefix = f"s{k}/"
        add_tree(graph, build_hidden_markov_tree(raw, calibration), prefix)
        root_names.append(prefix + raw.root_id)
    for link in links:
        p_tt, p_tf = DEFAULT_TRANSITION_PARAMS[link.relationship]
        graph.add_factor(
            pairwise_factor(
                root_names[link.parent], root_names[link.child], soften(p_tt), soften(p_tf)
            )
        )
    return AnswerGraph(graph, root_names)


def run_answer_graph(answer_graph: AnswerGraph) -> LoopyBPResult:
    return run_loopy_bp(
        answer_graph.graph,
        max_iterations=MAX_ITERATIONS,
        convergence_tol=CONVERGENCE_TOL,
        damping=DAMPING,
    )


@dataclass(frozen=True)
class AnswerVerification:
    sentences: list[str]
    posterior_true: list[float]  # per sentence, from the answer graph
    tree_only: list[float]  # per sentence, the paper's method alone, for comparison
    links: list[CrossLink]
    converged: bool  # convergence is reported, never assumed
    trees: list[RawBeliefTree]


def verify_answer(
    llm: LLMClient,
    labeler: RelationshipLabeler,
    link_labeler: RelationshipLabeler,
    calibration: Calibration,
    sentences: list[str],
    *,
    max_depth: int = 1,
) -> AnswerVerification:
    trees = [build_belief_tree(llm, labeler, s, max_depth=max_depth) for s in sentences]
    links = label_cross_links(sentences, link_labeler)
    answer_graph = build_answer_graph(trees, links, calibration)
    result = run_answer_graph(answer_graph)
    return AnswerVerification(
        sentences=sentences,
        posterior_true=[result.marginals[r] for r in answer_graph.root_names],
        tree_only=[posterior_root_true(build_hidden_markov_tree(t, calibration)) for t in trees],
        links=links,
        converged=result.converged,
        trees=trees,
    )
