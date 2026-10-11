"""Phase 4 experiment (Our Novelty, Extension 2): how much accuracy do we lose when a cheap
surrogate replaces the LLM's confidence for a fraction of the tree nodes?

1. Rebuild the paper's tree for every test sentence (same LLM cache as scripts/evaluate_felm.py).
2. Every node that the LLM scored (root, premise, decomposition claim) is a (text, confidence)
   example. The surrogate learns the LLM's confidence from text -- never the human labels.
3. No leakage: test answers are split into 5 groups; each group's nodes are predicted by a
   surrogate trained on the other 4 groups plus the fit-half sentences. Nodes of one answer are
   never predicted by a model that saw that answer.
4. For delegation fractions 0, 25, 50, 75, 100%, that share of nodes (chosen at random, 5
   repeats) gets the surrogate's score instead of the LLM's, and the tree AUROC is recomputed.
   Two schemes: any node, or only the child nodes (the root keeps its real LLM score).

Reported next to it: the surrogate's error against the LLM, and the error of always predicting
the average confidence -- the surrogate has to beat that to have learned anything (D34).

Usage: python scripts/evaluate_surrogate.py --backend hf --standalone [--offline]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

from sklearn.metrics import roc_auc_score

from scripts.backends import add_backend_args, make_backend
from scripts.evaluate_felm import MAX_DEPTH, load_calibrations, raw_tree_to_hmt
from scripts.fit_emission_model import (
    add_standalone_arg,
    load_felm_science_split,
    load_felm_science_test_answers,
    load_standalone,
    result_name,
)
from src.inference.hmt import posterior_root_true
from src.surrogate.model import SurrogateConfidenceModel
from src.tree.builder import RawBeliefTree, build_belief_tree
from src.tree.nli_labeler import CrossEncoderDirectionalNLI, NLIRelationshipLabeler

NUM_FOLDS = 5
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
REPEATS = 5
SEED = 42


@dataclass(frozen=True)
class ScoredNode:
    sentence: int  # index into the flat list of test sentences
    node_id: str
    is_root: bool
    statement: str
    llm_confidence: float


def scored_nodes(trees: list[RawBeliefTree]) -> list[ScoredNode]:
    """Every node whose confidence came from an LLM call (corrections are fixed at 1.0)."""
    return [
        ScoredNode(k, node_id, node_id == tree.root_id, node.statement, node.confidence)
        for k, tree in enumerate(trees)
        for node_id, node in tree.nodes.items()
        if node.strategy != "correction"
    ]


def with_confidences(tree: RawBeliefTree, new: dict[str, float]) -> RawBeliefTree:
    nodes = {
        node_id: replace(node, confidence=new.get(node_id, node.confidence))
        for node_id, node in tree.nodes.items()
    }
    return RawBeliefTree(nodes, tree.root_id)


def cross_fitted_predictions(
    nodes: list[ScoredNode],
    answer_of_sentence: list[int],
    num_answers: int,
    extra_train: list[tuple[str, float]],
) -> list[float]:
    """Each node predicted by a surrogate that never saw its answer."""
    answers = list(range(num_answers))
    random.Random(SEED).shuffle(answers)
    fold_of_answer = {a: i % NUM_FOLDS for i, a in enumerate(answers)}
    fold = [fold_of_answer[answer_of_sentence[n.sentence]] for n in nodes]
    predictions = [0.0] * len(nodes)
    for f in range(NUM_FOLDS):
        train = extra_train + [
            (n.statement, n.llm_confidence) for n, nf in zip(nodes, fold, strict=True) if nf != f
        ]
        model = SurrogateConfidenceModel()
        model.fit([s for s, _ in train], [c for _, c in train])
        held_out = [i for i, nf in enumerate(fold) if nf == f]
        for i, p in zip(
            held_out, model.predict_many([nodes[i].statement for i in held_out]), strict=True
        ):
            predictions[i] = p
    return predictions


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
    standalone = load_standalone(args, backend.run_name)
    answers = load_felm_science_test_answers(standalone)[: args.limit]

    # The fit half's sentence confidences (already computed when fitting the calibration) are
    # always allowed as training data: no test answer is in them.
    fit_llm = backend.cached("emission-fit")
    fit_half, _ = load_felm_science_split(standalone)
    extra_train = [(s, fit_llm.true_false_confidence(s)) for s, _ in fit_half]

    llm = backend.cached("felm-eval")
    labeler = NLIRelationshipLabeler(directional=CrossEncoderDirectionalNLI())
    sentences = [(s, label) for answer in answers for s, label in answer]
    answer_of_sentence = [a for a, answer in enumerate(answers) for _ in answer]
    trees = []
    for k, (s, _) in enumerate(sentences):
        trees.append(build_belief_tree(llm, labeler, s, max_depth=MAX_DEPTH))
        if (k + 1) % 25 == 0 or k + 1 == len(sentences):
            print(f"  rebuilt {k + 1}/{len(sentences)} trees")
    labels = [label for _, label in sentences]

    nodes = scored_nodes(trees)
    predicted = cross_fitted_predictions(nodes, answer_of_sentence, len(answers), extra_train)
    llm_scores = [n.llm_confidence for n in nodes]
    mean_score = statistics.fmean(llm_scores)

    def mae(idx: list[int], values: list[float]) -> float:
        return statistics.fmean(abs(values[i] - llm_scores[i]) for i in idx)

    everything = list(range(len(nodes)))
    roots = [i for i, n in enumerate(nodes) if n.is_root]
    children = [i for i, n in enumerate(nodes) if not n.is_root]
    constant = [mean_score] * len(nodes)
    error = {
        part: {"surrogate": mae(idx, predicted), "always_average": mae(idx, constant)}
        for part, idx in (("all", everything), ("roots", roots), ("children", children))
        if idx
    }
    print(f"{len(nodes)} LLM-scored nodes ({len(roots)} roots, {len(children)} children)")
    print(f"mean absolute error vs LLM: {error}")

    curve: list[dict[str, object]] = []
    for scheme, pool in (("any_node", everything), ("children_only", children)):
        for fraction in FRACTIONS:
            aurocs: dict[str, list[float]] = {c.name: [] for c in calibrations}
            repeats = 1 if fraction in (0.0, 1.0) else REPEATS
            for r in range(repeats):
                chosen = random.Random(SEED + r).sample(pool, round(fraction * len(pool)))
                swap: dict[int, dict[str, float]] = {}
                for i in chosen:
                    swap.setdefault(nodes[i].sentence, {})[nodes[i].node_id] = predicted[i]
                changed = [with_confidences(t, swap.get(k, {})) for k, t in enumerate(trees)]
                for c in calibrations:
                    scores = [posterior_root_true(raw_tree_to_hmt(t, c)) for t in changed]
                    aurocs[c.name].append(float(roc_auc_score(labels, scores)))
            point = {
                "scheme": scheme,
                "fraction": fraction,
                "llm_confidence_calls_saved": round(fraction * len(pool)),
                "llm_confidence_calls_total": len(nodes),
                **{f"{name}_tree_auroc_mean": statistics.fmean(v) for name, v in aurocs.items()},
                **{f"{name}_tree_auroc_std": statistics.pstdev(v) for name, v in aurocs.items()},
            }
            curve.append(point)
            print(
                f"  {scheme:<13} {fraction:4.0%}: "
                + " ".join(
                    f"{name}={statistics.fmean(v):.4f}±{statistics.pstdev(v):.4f}"
                    for name, v in aurocs.items()
                )
            )

    out_path = Path(f"results/felm_science_surrogate_{name}.json")
    out_path.write_text(
        json.dumps(
            {
                "llm": backend.model,
                "num_answers": len(answers),
                "num_sentences": len(sentences),
                "num_scored_nodes": len(nodes),
                "num_fit_half_training_sentences": len(extra_train),
                "settings": {"folds": NUM_FOLDS, "repeats": REPEATS, "seed": SEED},
                "mean_absolute_error_vs_llm": error,
                "curve": curve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
