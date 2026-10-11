"""Phase 3 experiment (Our Novelty, Extension 1): tree vs. answer-level graph on FELM-Science.

The paper checks each sentence of an answer on its own, in its own belief tree. But the
sentences of one answer talk about the same thing: one may support another ("Ground spoilers
are on the wings" / "Ground spoilers reduce lift after landing"), or contradict it. Here every
test answer becomes ONE graph:
    - each sentence keeps exactly the belief tree the paper's method builds for it
      (scripts/evaluate_felm.py, same LLM cache, same calibration);
    - every pair of sentence roots in the answer is NLI-labelled, and related pairs are joined
      by a cross link with the same relationship probabilities as the tree links.
A sentence linked to two others has two "parents", so the graph can have loops; loopy BP
(src/inference/loopy.py) replaces the exact tree recursion. Each sentence's score is then its
marginal in the answer graph, compared with its tree-only posterior on the same sentences.

Cross links are softened: an NLI mistake between two sentences must not make two claims
logically impossible together (a hard zero), only unlikely. LINK_EPS is fixed in advance, never
tuned on the test half (D33).

Run scripts/evaluate_felm.py first (same backend and flags): the trees come from its cache,
so with --offline this runs on a laptop with no GPU.

Usage: python scripts/evaluate_graph.py --backend hf --standalone [--offline] [--limit N]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import roc_auc_score

from scripts.backends import add_backend_args, make_backend
from scripts.evaluate_felm import MAX_DEPTH, Calibration, load_calibrations, raw_tree_to_hmt
from scripts.fit_emission_model import (
    add_standalone_arg,
    load_felm_science_test_answers,
    load_standalone,
    result_name,
)
from src.inference.hmt import posterior_root_true
from src.inference.loopy import FactorGraph, add_tree, pairwise_factor, run_loopy_bp
from src.tree.builder import RawBeliefTree, build_belief_tree
from src.tree.nli_labeler import CrossEncoderDirectionalNLI, NLIRelationshipLabeler
from src.tree.relationships import DEFAULT_TRANSITION_PARAMS, Relationship, RelationshipLabeler

LINK_EPS = 0.05  # cross-link probabilities are kept inside [LINK_EPS, 1 - LINK_EPS]
MAX_ITERATIONS = 200
CONVERGENCE_TOL = 1e-6
DAMPING = 0.5  # fixed in advance; only affects speed of settling, not the fixed point


@dataclass(frozen=True)
class CrossLink:
    child: int  # index of the sentence used as NLI premise
    parent: int  # index of the sentence used as NLI hypothesis
    relationship: Relationship


def soften(p: float) -> float:
    return min(max(p, LINK_EPS), 1 - LINK_EPS)


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
        add_tree(graph, raw_tree_to_hmt(raw, calibration), prefix)
        root_names.append(prefix + raw.root_id)
    for link in links:
        p_tt, p_tf = DEFAULT_TRANSITION_PARAMS[link.relationship]
        graph.add_factor(
            pairwise_factor(
                root_names[link.parent], root_names[link.child], soften(p_tt), soften(p_tf)
            )
        )
    return AnswerGraph(graph, root_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    add_standalone_arg(parser)
    parser.add_argument("--limit", type=int, default=None, help="only the first N test answers")
    args = parser.parse_args()
    backend = make_backend(args)
    print(f"LLM: {backend.model} ({backend.confidence_method})")

    name = result_name(args, backend.run_name)
    calibrations = load_calibrations(name)
    answers = load_felm_science_test_answers(load_standalone(args, backend.run_name))
    answers = answers[: args.limit]
    print(f"{len(answers)} test answers, {sum(len(a) for a in answers)} sentences.")

    llm = backend.cached("felm-eval")
    labeler = NLIRelationshipLabeler(directional=CrossEncoderDirectionalNLI())

    variants = [f"{c.name}_{kind}" for c in calibrations for kind in ("tree", "graph")]
    y_true: list[bool] = []
    scores: dict[str, list[float]] = {v: [] for v in variants}
    per_answer: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []  # one per (answer, calibration) loopy-BP run
    num_links = 0
    looped_converged: list[bool] = []  # convergence of each run on a graph with a loop
    all_converged: list[bool] = []
    max_iterations_used = 0

    for a, answer in enumerate(answers):
        statements = [s for s, _ in answer]
        trees = [build_belief_tree(llm, labeler, s, max_depth=MAX_DEPTH) for s in statements]
        links = label_cross_links(statements, labeler)
        row: dict[str, list[float]] = {}
        try:
            for c in calibrations:
                row[f"{c.name}_tree"] = [posterior_root_true(raw_tree_to_hmt(t, c)) for t in trees]
                answer_graph = build_answer_graph(trees, links, c)
                result = run_loopy_bp(
                    answer_graph.graph,
                    max_iterations=MAX_ITERATIONS,
                    convergence_tol=CONVERGENCE_TOL,
                    damping=DAMPING,
                )
                row[f"{c.name}_graph"] = [result.marginals[r] for r in answer_graph.root_names]
                all_converged.append(result.converged)
                if answer_graph.has_cycle:
                    looped_converged.append(result.converged)
                max_iterations_used = max(max_iterations_used, result.iterations_used)
                runs.append(
                    {
                        "answer": a,
                        "calibration": c.name,
                        "has_cycle": answer_graph.has_cycle,
                        "converged": result.converged,
                        "iterations": result.iterations_used,
                        "max_delta": result.max_message_delta_at_stop,
                    }
                )
        except ValueError as error:
            # Degenerate evidence (zero likelihood for both truth values) -- skip the answer.
            print(f"  answer {a}: skipped ({error})")
            continue

        y_true += [label for _, label in answer]
        num_links += len(links)
        for v in variants:
            scores[v] += row[v]
        per_answer.append(
            {
                "sentences": [
                    {"statement": s, "label": label, **{v: row[v][k] for v in variants}}
                    for k, (s, label) in enumerate(answer)
                ],
                "links": [
                    {"child": x.child, "parent": x.parent, "relationship": x.relationship.value}
                    for x in links
                ],
            }
        )
        print(f"  answer {a + 1}/{len(answers)}: {len(statements)} sentences, {len(links)} links")

    auroc: dict[str, float | None] = dict.fromkeys(variants)
    if len(set(y_true)) == 2:
        for v in variants:
            auroc[v] = float(roc_auc_score(y_true, scores[v]))
            print(f"AUROC {v:<12} {auroc[v]:.4f}")
    else:
        print("AUROC not computed: need both true and false sentences.")

    convergence = {
        "runs": len(all_converged),
        "converged": sum(all_converged),
        "runs_on_looped_graphs": len(looped_converged),
        "converged_on_looped_graphs": sum(looped_converged),
        "max_iterations_used": max_iterations_used,
    }
    print(f"{num_links} cross links; convergence: {convergence}")

    out_path = Path(f"results/felm_science_graph_{name}.json")
    out_path.write_text(
        json.dumps(
            {
                "llm": backend.model,
                "num_answers": len(per_answer),
                "num_sentences": len(y_true),
                "num_cross_links": num_links,
                "settings": {
                    "link_eps": LINK_EPS,
                    "max_iterations": MAX_ITERATIONS,
                    "convergence_tol": CONVERGENCE_TOL,
                    "damping": DAMPING,
                },
                "auroc": auroc,
                "convergence": convergence,
                "runs": runs,
                "answers": per_answer,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
