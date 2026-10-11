"""Checks for loopy BP (Phase 3): exact on trees (must match hmt.py), close to a brute-force
oracle on a looped graph, and honest about convergence."""

from __future__ import annotations

import itertools

import pytest

from src.inference.hmt import (
    DecompositionTransition,
    HiddenMarkovTree,
    IndependentChildTransition,
    LookupEmission,
    TreeNode,
    posterior_root_true,
)
from src.inference.loopy import (
    Factor,
    FactorGraph,
    factor_graph_from_tree,
    pairwise_factor,
    run_loopy_bp,
    unary_factor,
)

EMISSION_TABLE = {
    0.9: {True: 0.8, False: 0.1},
    0.2: {True: 0.2, False: 0.7},
    0.6: {True: 0.6, False: 0.4},
}


def brute_force_marginals(graph: FactorGraph) -> dict[str, float]:
    """Exact marginals by enumerating all 2^n joint assignments -- a test oracle only."""
    totals = dict.fromkeys(graph.variables, 0.0)
    z = 0.0
    for values in itertools.product((False, True), repeat=len(graph.variables)):
        assignment = dict(zip(graph.variables, values, strict=True))
        weight = 1.0
        for f in graph.factors:
            weight *= f.table[tuple(assignment[v] for v in f.variables)]
        z += weight
        for v, truth in assignment.items():
            if truth:
                totals[v] += weight
    return {v: t / z for v, t in totals.items()}


def _two_child_tree(
    transition: IndependentChildTransition | DecompositionTransition,
) -> HiddenMarkovTree:
    return HiddenMarkovTree(
        nodes={
            "root": TreeNode("root", score=0.9, children=("child1", "child2")),
            "child1": TreeNode("child1", score=0.2),
            "child2": TreeNode("child2", score=0.6),
        },
        root_id="root",
        emission=LookupEmission(EMISSION_TABLE),
        transition=transition,
    )


def test_matches_phase0_checkpoint_on_a_tree() -> None:
    graph = factor_graph_from_tree(_two_child_tree(IndependentChildTransition(0.9, 0.3)))
    assert not graph.has_cycle()
    result = run_loopy_bp(graph, convergence_tol=1e-12)
    assert result.converged
    assert result.iterations_used <= 5
    assert result.marginals["root"] == pytest.approx(0.8209483368719038, abs=1e-12)


def test_matches_exact_tree_with_the_joint_decomposition_rule() -> None:
    tree = _two_child_tree(DecompositionTransition())
    result = run_loopy_bp(factor_graph_from_tree(tree), convergence_tol=1e-12)
    assert result.converged
    assert result.marginals["root"] == pytest.approx(posterior_root_true(tree), abs=1e-12)


def _diamond() -> FactorGraph:
    # root -> A, root -> B, A -> C, B -> C: C has two parents, so the graph has a loop.
    graph = FactorGraph()
    for v in ("root", "A", "B", "C"):
        graph.add_variable(v)
    for v, p in (("root", 0.9), ("A", 0.7), ("B", 0.3), ("C", 0.6)):
        graph.add_factor(unary_factor(v, 1 - p, p))
    graph.add_factor(pairwise_factor("root", "A", 0.85, 0.2))
    graph.add_factor(pairwise_factor("root", "B", 0.8, 0.25))
    graph.add_factor(pairwise_factor("A", "C", 0.9, 0.15))
    graph.add_factor(pairwise_factor("B", "C", 0.8, 0.3))
    return graph


def test_close_to_brute_force_on_a_looped_graph_but_not_exact() -> None:
    graph = _diamond()
    assert graph.has_cycle()
    result = run_loopy_bp(graph, convergence_tol=1e-10)
    exact = brute_force_marginals(graph)
    assert result.converged
    for v in graph.variables:
        assert result.marginals[v] == pytest.approx(exact[v], abs=0.05)
    # On a real loop BP double-counts evidence; an exact match would be suspicious.
    assert max(abs(result.marginals[v] - exact[v]) for v in graph.variables) > 1e-6


def test_damping_reaches_the_same_fixed_point() -> None:
    plain = run_loopy_bp(_diamond(), convergence_tol=1e-10)
    damped = run_loopy_bp(_diamond(), convergence_tol=1e-10, damping=0.5)
    assert damped.converged
    for v, p in plain.marginals.items():
        assert damped.marginals[v] == pytest.approx(p, abs=1e-6)


def test_zero_iterations_never_claims_convergence() -> None:
    result = run_loopy_bp(_diamond(), max_iterations=0)
    assert not result.converged
    assert result.iterations_used == 0


def test_too_few_iterations_reports_not_converged() -> None:
    result = run_loopy_bp(_diamond(), max_iterations=1, convergence_tol=1e-12)
    assert not result.converged
    assert result.max_message_delta_at_stop >= 1e-12


def test_factor_with_unknown_variable_is_rejected() -> None:
    graph = FactorGraph()
    graph.add_variable("a")
    with pytest.raises(ValueError, match="unknown"):
        graph.add_factor(pairwise_factor("a", "b", 0.9, 0.1))


def test_factor_cannot_repeat_a_variable() -> None:
    with pytest.raises(ValueError, match="repeat"):
        Factor.from_function(("a", "a"), lambda _: 1.0)
