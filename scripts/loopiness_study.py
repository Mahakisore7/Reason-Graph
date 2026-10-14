"""Exact inference vs. loopy BP as the graph gets more loops (mentor-feedback study, Objective 2).

For random binary graphs of N nodes we add 0, 1, 2, ... extra edges to a random spanning tree
(the number of extra edges is the number of independent cycles, "loopiness"). For each graph we
compare loopy BP against the exact marginals (brute force over all 2^N assignments -- the exact
answer that variable elimination would also give, tractable only because N is small):

    - mean / max absolute error of the marginals P(Z_v = T),
    - how often BP converged and how many rounds it needed,
    - with and without damping (our real runs use damping 0.5).

Two coupling regimes: "mixed" draws each edge from the three softened relationship types the
real answer graphs use (equivalence, entailment, reverse entailment); "strong" makes every edge
a softened equivalence (0.95 / 0.05), the hard case for loopy BP (evidence is double-counted
around cycles). Evidence per node is random with P(T) in [0.15, 0.85].

Usage: python scripts/loopiness_study.py [--graphs 200] [--nodes 10]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from pathlib import Path

from src.inference.loopy import FactorGraph, pairwise_factor, run_loopy_bp, unary_factor

EXTRA_EDGES = (0, 1, 2, 3, 4, 6, 8, 10)
MIXED = ((0.95, 0.05), (0.5, 0.05), (0.95, 0.5))  # softened equivalence / entailment / reverse
STRONG = ((0.95, 0.05),)
SEED = 42


def random_graph(
    rng: random.Random, num_nodes: int, extra_edges: int, couplings: tuple[tuple[float, float], ...]
) -> FactorGraph:
    names = [f"v{i}" for i in range(num_nodes)]
    graph = FactorGraph()
    for name in names:
        graph.add_variable(name)
        p = rng.uniform(0.15, 0.85)
        graph.add_factor(unary_factor(name, 1 - p, p))
    edges: set[tuple[int, int]] = set()
    for i in range(1, num_nodes):  # random spanning tree: each node attaches to an earlier one
        edges.add((rng.randrange(i), i))
    while len(edges) < num_nodes - 1 + extra_edges:
        a, b = sorted(rng.sample(range(num_nodes), 2))
        edges.add((a, b))
    for a, b in sorted(edges):
        p_tt, p_tf = rng.choice(couplings)
        graph.add_factor(pairwise_factor(names[a], names[b], p_tt, p_tf))
    return graph


def exact_marginals(graph: FactorGraph) -> dict[str, float]:
    """Brute force over all 2^N joint assignments (the exact answer; test oracle only)."""
    totals = dict.fromkeys(graph.variables, 0.0)
    z = 0.0
    for values in itertools.product((False, True), repeat=len(graph.variables)):
        assignment = dict(zip(graph.variables, values, strict=True))
        weight = 1.0
        for factor in graph.factors:
            weight *= factor.table[tuple(assignment[v] for v in factor.variables)]
        z += weight
        for v, truth in assignment.items():
            if truth:
                totals[v] += weight
    return {v: t / z for v, t in totals.items()}


def study(num_graphs: int, num_nodes: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for regime, couplings in (("mixed", MIXED), ("strong", STRONG)):
        for extra in EXTRA_EDGES:
            rng = random.Random(f"{SEED}-{regime}-{extra}")
            graphs = [random_graph(rng, num_nodes, extra, couplings) for _ in range(num_graphs)]
            exact = [exact_marginals(g) for g in graphs]
            for damping in (0.0, 0.5):
                errors: list[float] = []
                worst: list[float] = []
                converged = 0
                rounds: list[int] = []
                for g, truth in zip(graphs, exact, strict=True):
                    result = run_loopy_bp(
                        g, max_iterations=200, convergence_tol=1e-6, damping=damping
                    )
                    diffs = [abs(result.marginals[v] - truth[v]) for v in g.variables]
                    errors.append(statistics.fmean(diffs))
                    worst.append(max(diffs))
                    converged += result.converged
                    rounds.append(result.iterations_used)
                rows.append(
                    {
                        "regime": regime,
                        "extra_edges": extra,
                        "damping": damping,
                        "mean_abs_error": statistics.fmean(errors),
                        "max_abs_error": max(worst),
                        "converged_share": converged / num_graphs,
                        "median_rounds": statistics.median(rounds),
                    }
                )
                print(
                    f"{regime:<6} cycles={extra:<2} damping={damping}: "
                    f"mean err {rows[-1]['mean_abs_error']:.4f}  max err "
                    f"{rows[-1]['max_abs_error']:.3f}  converged "
                    f"{rows[-1]['converged_share']:.0%}  median rounds {rows[-1]['median_rounds']}"
                )
    return {"graphs_per_setting": num_graphs, "nodes": num_nodes, "seed": SEED, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=10)
    args = parser.parse_args()
    out = Path("results/loopiness_study.json")
    out.write_text(json.dumps(study(args.graphs, args.nodes), indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
