"""Reproduces the paper's own kind of experiment: build a belief tree for each labelled
statement, run Phase 0's exact HMT inference through it, and check whether the resulting
posterior separates true statements from false ones better than the LLM's raw, single-shot
confidence does -- measured as AUROC on FELM-Science, one of the paper's own three benchmarks.

This is a Phase 2 sanity-check script, not production src/ code: the RawBeliefTree ->
HiddenMarkovTree conversion below (AssembledTransition) is a minimal, direct version just to
get a number out. A cleaner, reusable version of this conversion belongs in Phase 5
("full assembly"), once the surrogate scorer and DAG extension exist too.

Backends (see scripts/backends.py): --backend groq gives a provisional number (typed-out
confidence); --backend hf runs an open-source model ourselves and reads real token
probabilities -- the paper's method, and the number that goes in the report.

Usage: uv run python scripts/evaluate_felm.py [--backend groq|hf] [--model NAME]
Run scripts/fit_emission_model.py with the same backend first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

from scripts.backends import add_backend_args, make_backend
from scripts.fit_emission_model import (
    add_standalone_arg,
    bin_index,
    load_felm_science_split,
    load_standalone,
    result_name,
)
from src.inference.hmt import (
    DecompositionTransition,
    HiddenMarkovTree,
    LogisticEmission,
    LookupEmission,
    TreeNode,
    posterior_root_true,
)
from src.pipeline import Calibration, build_hidden_markov_tree
from src.tree.builder import build_belief_tree
from src.tree.nli_labeler import CrossEncoderDirectionalNLI, NLIRelationshipLabeler

# One level of children (the paper expands two) -- D16: depth 2 multiplies the LLM calls per
# statement several times over, beyond one free Kaggle session.
MAX_DEPTH = 1


def load_calibrations(run_name: str) -> list[Calibration]:
    """The binned table (the paper's approach) and, if fitted, the smooth logistic curve."""
    path = Path(f"results/emission_model_felm_science_{run_name}.json")
    data = json.loads(path.read_text())
    table: dict[float, dict[bool, float]] = {}
    for bin_str, probs in data["lookup_table"].items():
        table[float(bin_str)] = {True: probs["true"], False: probs["false"]}
    calibrations = [
        Calibration("table", LookupEmission(table=table), lambda c: float(bin_index(c)))
    ]
    if "logistic" in data:
        calibrations.append(Calibration("curve", LogisticEmission(**data["logistic"]), lambda c: c))
    return calibrations


def root_only_posterior(confidence: float, calibration: Calibration) -> float:
    """The calibration on its own, no tree -- isolates what the calibration step does."""
    node = TreeNode(node_id="root", score=calibration.to_score(confidence))
    tree = HiddenMarkovTree(
        nodes={"root": node},
        root_id="root",
        emission=calibration.emission,
        transition=DecompositionTransition(),  # unused: a single node has no children
    )
    return posterior_root_true(tree)


raw_tree_to_hmt = build_hidden_markov_tree  # name used by the other experiment scripts


def main() -> None:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    add_standalone_arg(parser)
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N test segments (smoke test)"
    )
    args = parser.parse_args()
    backend = make_backend(args)
    print(f"LLM: {backend.model} ({backend.confidence_method})")

    name = result_name(args, backend.run_name)
    calibrations = load_calibrations(name)
    # Test half only -- calibration was fit on the other half (see
    # scripts/fit_emission_model.py, split_by_response), so nothing here was seen during fitting.
    _, sample = load_felm_science_split(load_standalone(args, backend.run_name))
    if args.limit is not None:
        sample = sample[: args.limit]
    num_false = sum(1 for _, is_true in sample if not is_true)
    print(f"Evaluating on the {len(sample)}-segment test half ({num_false} false).")

    llm = backend.cached("felm-eval")
    labeler = NLIRelationshipLabeler(directional=CrossEncoderDirectionalNLI())

    # Every variant is scored on the same statements and all are reported, so no variant is
    # quietly picked after looking at the test half.
    variants = ["raw"] + [f"{c.name}_{kind}" for c in calibrations for kind in ("only", "tree")]
    y_true: list[bool] = []
    scores: dict[str, list[float]] = {v: [] for v in variants}
    predictions: list[dict[str, object]] = []
    skipped = 0

    for i, (statement, is_true) in enumerate(sample):
        raw_tree = build_belief_tree(llm, labeler, statement, max_depth=MAX_DEPTH)
        root_confidence = raw_tree.nodes[raw_tree.root_id].confidence
        row = {"raw": root_confidence}
        try:
            for c in calibrations:
                row[f"{c.name}_only"] = root_only_posterior(root_confidence, c)
                row[f"{c.name}_tree"] = posterior_root_true(raw_tree_to_hmt(raw_tree, c))
        except ValueError:
            # Degenerate tree (both truth values had zero likelihood) -- skip rather than
            # crash the whole run over one statement.
            skipped += 1
            continue

        y_true.append(is_true)
        for v in variants:
            scores[v].append(row[v])
        predictions.append(
            {"statement": statement, "label": is_true, "tree_size": len(raw_tree.nodes), **row}
        )
        print(
            f"  {i + 1}/{len(sample)}: label={is_true} "
            + " ".join(f"{v}={row[v]:.3f}" for v in variants)
        )

    if skipped:
        print(f"Skipped {skipped} degenerate tree(s).")

    auroc: dict[str, float | None] = dict.fromkeys(variants)
    if len(set(y_true)) < 2:
        # AUROC compares true vs. false examples -- undefined if only one kind was scored
        # (happens on small --limit smoke tests).
        print("AUROC not computed: need both true and false segments in the sample.")
    else:
        for v in variants:
            auroc[v] = float(roc_auc_score(y_true, scores[v]))
            print(f"AUROC {v:<12} {auroc[v]:.4f}")

    out_path = Path(f"results/felm_science_auroc_{name}.json")
    out_path.write_text(
        json.dumps(
            {
                "llm": backend.model,
                "confidence_method": backend.confidence_method,
                "sample_size": len(y_true),
                "skipped": skipped,
                "max_depth": MAX_DEPTH,
                "standalone_rewrites": bool(args.standalone),
                # raw = LLM confidence; *_only = calibration alone; *_tree = calibration + tree
                "auroc": auroc,
                # Per-statement scores, for plots and error analysis later.
                "predictions": predictions,
            },
            indent=2,
        )
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
