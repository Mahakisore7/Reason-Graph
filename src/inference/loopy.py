"""Loopy belief propagation on a factor graph -- Our Novelty, Extension 1.

The paper's Hidden Markov Tree gives every claim exactly one parent, so the exact upward
recursion in hmt.py works. Once a claim is linked to more than one other claim (e.g. a sentence
that supports two other sentences of the same answer), the graph has loops and that recursion no
longer applies. Loopy BP runs the same message-passing idea on any graph, repeated until the
messages stop changing. On a tree it is exact; on a graph with loops it is an approximation with
no convergence guarantee, so convergence is always measured and reported, never assumed.

Representation: every claim is a binary variable (True / False). Every piece of knowledge about
the claims is a *factor* -- a non-negative table over the truth values of the claims it touches:
    - evidence:   a one-claim factor, p(S_u | Z_u) (times the prior for a root);
    - a tree link: a factor over a parent and ALL its children, p(children | parent). Joint
      factors keep the paper's decomposition rule (parent true only if every child is true)
      exact -- a pairwise-only graph could not express it (D32);
    - a cross link: a two-claim factor between claims that NLI says are related.

Message updates are sum-product, synchronous ("flooding"): every factor->variable message is
computed from last round's variable->factor messages, then every variable->factor message from
the new factor->variable ones. Optional damping mixes in the previous message to help loops
settle. Both are documented choices, not the only options.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.inference.hmt import HiddenMarkovTree

Message = tuple[float, float]  # (value at False, value at True), normalised to sum to 1
_UNIFORM: Message = (0.5, 0.5)


@dataclass(frozen=True)
class Factor:
    """A non-negative table over the joint truth values of `variables`, keyed by tuples of
    bools in the same order as `variables`."""

    variables: tuple[str, ...]
    table: dict[tuple[bool, ...], float]

    @classmethod
    def from_function(
        cls, variables: Sequence[str], fn: Callable[[tuple[bool, ...]], float]
    ) -> Factor:
        variables = tuple(variables)
        if len(set(variables)) != len(variables):
            raise ValueError(f"a factor cannot repeat a variable: {variables}")
        table = {a: fn(a) for a in itertools.product((False, True), repeat=len(variables))}
        if any(v < 0 for v in table.values()):
            raise ValueError("factor values must be non-negative")
        return cls(variables, table)


def unary_factor(variable: str, p_false: float, p_true: float) -> Factor:
    return Factor((variable,), {(False,): p_false, (True,): p_true})


def pairwise_factor(
    parent: str, child: str, p_child_true_given_parent_true: float, p_child_true_given_false: float
) -> Factor:
    """psi(Z_parent, Z_child) = p(Z_child | Z_parent), the same shape as one child's term in
    IndependentChildTransition."""

    def fn(a: tuple[bool, ...]) -> float:
        parent_truth, child_truth = a
        p = p_child_true_given_parent_true if parent_truth else p_child_true_given_false
        return p if child_truth else 1.0 - p

    return Factor.from_function((parent, child), fn)


@dataclass
class FactorGraph:
    variables: list[str] = field(default_factory=list)
    factors: list[Factor] = field(default_factory=list)

    def add_variable(self, name: str) -> None:
        if name in self.variables:
            raise ValueError(f"variable {name!r} already exists")
        self.variables.append(name)

    def add_factor(self, factor: Factor) -> None:
        missing = [v for v in factor.variables if v not in self.variables]
        if missing:
            raise ValueError(f"factor references unknown variables {missing}")
        self.factors.append(factor)

    def has_cycle(self) -> bool:
        """True if the variable-factor graph has a loop -- i.e. loopy BP is approximate here.
        A connected graph is a tree iff edges = nodes - 1; counted per connected component."""
        parent = {v: v for v in self.variables}

        def find(v: str) -> str:
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        # Union-find over the bipartite graph: each factor is a node joined to its variables.
        for i, factor in enumerate(self.factors):
            parent[f"#factor{i}"] = f"#factor{i}"
            for v in factor.variables:
                a, b = find(f"#factor{i}"), find(v)
                if a == b:
                    return True
                parent[a] = b
        return False


@dataclass(frozen=True)
class LoopyBPResult:
    """Convergence is reported, never assumed: `converged` is True only if a round of updates
    actually changed no message by more than the tolerance."""

    marginals: dict[str, float]  # variable -> P(Z = True)
    converged: bool
    iterations_used: int
    max_message_delta_at_stop: float


def _normalise(false_value: float, true_value: float) -> Message:
    total = false_value + true_value
    if total <= 0:
        raise ValueError(
            "a message is zero for both truth values -- the factors contradict each other"
        )
    return (false_value / total, true_value / total)


def run_loopy_bp(
    graph: FactorGraph,
    *,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    damping: float = 0.0,
) -> LoopyBPResult:
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")

    # Edges of the bipartite graph, as (factor index, variable).
    edges = [(i, v) for i, f in enumerate(graph.factors) for v in f.variables]
    factors_of: dict[str, list[int]] = {v: [] for v in graph.variables}
    for i, v in edges:
        factors_of[v].append(i)

    var_to_factor: dict[tuple[int, str], Message] = dict.fromkeys(edges, _UNIFORM)
    factor_to_var: dict[tuple[int, str], Message] = dict.fromkeys(edges, _UNIFORM)

    converged = False
    delta = float("inf")
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        new_f2v: dict[tuple[int, str], Message] = {}
        for i, factor in enumerate(graph.factors):
            incoming = [var_to_factor[(i, v)] for v in factor.variables]
            for pos, target in enumerate(factor.variables):
                sums = [0.0, 0.0]
                for assignment, value in factor.table.items():
                    if value == 0.0:
                        continue
                    term = value
                    for other, truth in enumerate(assignment):
                        if other != pos:
                            term *= incoming[other][truth]
                    sums[assignment[pos]] += term
                new_f2v[(i, target)] = _normalise(sums[0], sums[1])

        new_v2f: dict[tuple[int, str], Message] = {}
        for i, v in edges:
            false_value, true_value = 1.0, 1.0
            for j in factors_of[v]:
                if j != i:
                    false_value *= new_f2v[(j, v)][0]
                    true_value *= new_f2v[(j, v)][1]
            new_v2f[(i, v)] = _normalise(false_value, true_value)

        if damping:
            for table, old in ((new_f2v, factor_to_var), (new_v2f, var_to_factor)):
                for key, m in table.items():
                    table[key] = _normalise(
                        (1 - damping) * m[0] + damping * old[key][0],
                        (1 - damping) * m[1] + damping * old[key][1],
                    )

        delta = max(
            (
                abs(new[key][1] - old[key][1])
                for new, old in ((new_f2v, factor_to_var), (new_v2f, var_to_factor))
                for key in new
            ),
            default=0.0,
        )
        factor_to_var, var_to_factor = new_f2v, new_v2f
        if delta < convergence_tol:
            converged = True
            break

    marginals: dict[str, float] = {}
    for v in graph.variables:
        false_value, true_value = 1.0, 1.0
        for j in factors_of[v]:
            false_value *= factor_to_var[(j, v)][0]
            true_value *= factor_to_var[(j, v)][1]
        marginals[v] = _normalise(false_value, true_value)[1]

    return LoopyBPResult(marginals, converged, iterations, delta)


def factor_graph_from_tree(tree: HiddenMarkovTree, prefix: str = "") -> FactorGraph:
    """The same model as a HiddenMarkovTree, written as factors: the prior on the root, one
    evidence factor per node, one joint transition factor per parent. Loopy BP on this graph
    must reproduce posterior_root_true exactly -- the check that the two implementations agree.
    `prefix` keeps variable names unique when several trees go into one graph."""
    graph = FactorGraph()
    add_tree(graph, tree, prefix)
    return graph


def add_tree(graph: FactorGraph, tree: HiddenMarkovTree, prefix: str = "") -> None:
    for node_id in tree.nodes:
        graph.add_variable(prefix + node_id)
    for node_id, node in tree.nodes.items():
        p_true = tree.emission.prob(node.score, True)
        p_false = tree.emission.prob(node.score, False)
        if node_id == tree.root_id:
            p_true *= tree.prior_true
            p_false *= 1 - tree.prior_true
        graph.add_factor(unary_factor(prefix + node_id, p_false, p_true))
        if node.children:

            def fn(a: tuple[bool, ...], parent_id: str = node_id) -> float:
                return tree.transition.joint_prob(parent_id, a[0], a[1:])

            names = [prefix + node_id] + [prefix + c for c in node.children]
            graph.add_factor(Factor.from_function(names, fn))


__all__ = [
    "Factor",
    "FactorGraph",
    "LoopyBPResult",
    "add_tree",
    "factor_graph_from_tree",
    "pairwise_factor",
    "run_loopy_bp",
    "unary_factor",
]
