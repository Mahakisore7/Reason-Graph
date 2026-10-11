"""Chooses the cross-link rule (D35) on the FIT half, so the test half never influences it.

For every fit-half answer: one node per sentence carrying its Llama confidence (already
computed when the calibration was fitted -- no new LLM calls), joined by cross links under each
rule, scored with loopy BP. The rule with the highest fit-half AUROC (smooth-curve calibration)
is the one reported on the test half; the others are reported too, for transparency.

No trees here: the fit half has none (building them costs Kaggle GPU time). The links are what
is being chosen, and they are the same kind of object with or without trees.

Usage: python scripts/choose_link_rule.py --backend hf --standalone --offline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.backends import add_backend_args, make_backend
from scripts.evaluate_felm import load_calibrations
from scripts.fit_emission_model import (
    FIT_FRACTION,
    SPLIT_SEED,
    add_standalone_arg,
    load_felm_science_responses,
    load_standalone,
    result_name,
    split_answers,
)
from src.eval.metrics import all_metrics
from src.inference.loopy import run_loopy_bp
from src.pipeline import (
    CONVERGENCE_TOL,
    DAMPING,
    LINK_RULES,
    MAX_ITERATIONS,
    RuleLabeler,
    build_answer_graph,
    label_cross_links,
)
from src.tree.builder import RawBeliefTree, RawNode
from src.tree.nli_labeler import CrossEncoderDirectionalNLI


def single_node_tree(statement: str, confidence: float) -> RawBeliefTree:
    return RawBeliefTree({"n0": RawNode("n0", statement, confidence, None, None, "root")}, "n0")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    add_standalone_arg(parser)
    args = parser.parse_args()
    backend = make_backend(args)
    name = result_name(args, backend.run_name)
    curve = next(c for c in load_calibrations(name) if c.name == "curve")

    standalone = load_standalone(args, backend.run_name) or {}
    fit, _ = split_answers(load_felm_science_responses(), FIT_FRACTION, SPLIT_SEED)
    answers = [[(standalone.get(s, s), label) for s, label in r] for r in fit]
    llm = backend.cached("emission-fit")
    nli = CrossEncoderDirectionalNLI()

    labels = [label for answer in answers for _, label in answer]
    groups = [a for a, answer in enumerate(answers) for _ in answer]
    report: dict[str, dict[str, float]] = {}
    for rule in ("none", *LINK_RULES):
        scores: list[float] = []
        num_links = 0
        for answer in answers:
            statements = [s for s, _ in answer]
            trees = [single_node_tree(s, llm.true_false_confidence(s)) for s in statements]
            links = [] if rule == "none" else label_cross_links(statements, RuleLabeler(nli, rule))
            num_links += len(links)
            graph = build_answer_graph(trees, links, curve)
            result = run_loopy_bp(
                graph.graph,
                max_iterations=MAX_ITERATIONS,
                convergence_tol=CONVERGENCE_TOL,
                damping=DAMPING,
            )
            scores += [result.marginals[r] for r in graph.root_names]
        report[rule] = {**all_metrics(labels, scores, groups), "num_links": num_links}
        m = report[rule]
        print(
            f"{rule:<17} links={num_links:<4} AUROC={m['auroc']:.3f} "
            f"AUC-PR={m['auc_pr_false']:.3f} F1={m['f1_false']:.3f}"
        )

    chosen = max(LINK_RULES, key=lambda r: report[r]["auroc"])
    print(f"Chosen on the fit half: {chosen}")
    out = Path(f"results/link_rule_choice_{name}.json")
    out.write_text(
        json.dumps(
            {"fit_sentences": len(labels), "criterion": "auroc", "chosen": chosen, **report},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
