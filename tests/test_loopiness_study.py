"""Checks for the loopiness study's building blocks."""

from __future__ import annotations

import random

import pytest

from scripts.loopiness_study import MIXED, exact_marginals, random_graph
from src.inference.loopy import run_loopy_bp


def test_random_graph_has_the_requested_number_of_cycles() -> None:
    rng = random.Random(0)
    tree = random_graph(rng, 8, 0, MIXED)
    looped = random_graph(rng, 8, 3, MIXED)
    pairwise = lambda g: sum(1 for f in g.factors if len(f.variables) == 2)  # noqa: E731
    assert pairwise(tree) == 7 and not tree.has_cycle()
    assert pairwise(looped) == 7 + 3 and looped.has_cycle()


def test_loopy_bp_is_exact_on_trees_in_the_study_setup() -> None:
    rng = random.Random(1)
    for _ in range(5):
        graph = random_graph(rng, 7, 0, MIXED)
        result = run_loopy_bp(graph, convergence_tol=1e-12)
        exact = exact_marginals(graph)
        assert result.converged
        for v in graph.variables:
            assert result.marginals[v] == pytest.approx(exact[v], abs=1e-9)
