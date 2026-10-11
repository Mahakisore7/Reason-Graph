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
from pathlib import Path

from sklearn.metrics import roc_auc_score

from scripts.backends import add_backend_args, make_backend
from scripts.evaluate_felm import MAX_DEPTH, load_calibrations, raw_tree_to_hmt
from scripts.fit_emission_model import (
    add_standalone_arg,
    load_felm_science_test_answers,
    load_standalone,
    result_name,
)
from src.inference.hmt import posterior_root_true
from src.inference.loopy import run_loopy_bp
from src.pipeline import (
    CONVERGENCE_TOL,
    DAMPING,
    LINK_EPS,
    LINK_RULES,
    MAX_ITERATIONS,
    RuleLabeler,
    build_answer_graph,
    label_cross_links,
)
from src.tree.builder import build_belief_tree
from src.tree.nli_labeler import (
    CrossEncoderDirectionalNLI,
    NLIRelationshipLabeler,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    add_standalone_arg(parser)
    parser.add_argument("--limit", type=int, default=None, help="only the first N test answers")
    parser.add_argument("--link-rule", choices=LINK_RULES, default="any")
    args = parser.parse_args()
    backend = make_backend(args)
    print(f"LLM: {backend.model} ({backend.confidence_method})")

    name = result_name(args, backend.run_name)
    calibrations = load_calibrations(name)
    answers = load_felm_science_test_answers(load_standalone(args, backend.run_name))
    answers = answers[: args.limit]
    print(f"{len(answers)} test answers, {sum(len(a) for a in answers)} sentences.")

    llm = backend.cached("felm-eval")
    nli = CrossEncoderDirectionalNLI()
    labeler = NLIRelationshipLabeler(directional=nli)
    link_labeler = RuleLabeler(nli, args.link_rule)
    print(f"cross-link rule: {args.link_rule}")

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
        links = label_cross_links(statements, link_labeler)
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

    suffix = "" if args.link_rule == "any" else f"-{args.link_rule}"
    out_path = Path(f"results/felm_science_graph_{name}{suffix}.json")
    out_path.write_text(
        json.dumps(
            {
                "llm": backend.model,
                "num_answers": len(per_answer),
                "num_sentences": len(y_true),
                "num_cross_links": num_links,
                "settings": {
                    "link_rule": args.link_rule,
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
