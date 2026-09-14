"""Reproduces the paper's own kind of experiment: build a belief tree for each labelled
statement, run Phase 0's exact HMT inference through it, and check whether the resulting
posterior separates true statements from false ones better than the LLM's raw, single-shot
confidence does -- measured as AUROC on FELM-Science, one of the paper's own three benchmarks.

This is a Phase 2 sanity-check script, not production src/ code: the RawBeliefTree ->
HiddenMarkovTree conversion below (AssembledTransition) is a minimal, direct version just to
get a number out. A cleaner, reusable version of this conversion belongs in Phase 5
("full assembly"), once the surrogate scorer and DAG extension exist too.

Uses Groq (free, text-based self-reported confidence -- see src/llm/groq_client.py), so this
AUROC is a provisional number, not the paper's faithful reproduction. Re-run with OpenAIClient
once billing credit is available, for the number that actually goes in the report.

Usage: uv run python scripts/evaluate_felm.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import roc_auc_score

from scripts.fit_emission_model import bin_index, load_felm_science
from src.inference.hmt import (
    DecompositionTransition,
    HiddenMarkovTree,
    LookupEmission,
    TreeNode,
    posterior_root_true,
)
from src.llm.cache import CachingLLMClient
from src.llm.groq_client import GroqClient
from src.tree.builder import RawBeliefTree, build_belief_tree
from src.tree.nli_labeler import CrossEncoderDirectionalNLI, NLIRelationshipLabeler
from src.tree.relationships import DEFAULT_TRANSITION_PARAMS, Relationship

# Kept small deliberately: each example costs several real LLM calls on Groq's free tier.
# Scale up once this looks sane, or once OpenAI credit makes the "real" run affordable.
SAMPLE_SIZE = 60
# One level of children only -- at depth 2, correction alone (question + N*(answer+rewrite))
# multiplied across every depth-1 child would run into the thousands of calls per statement.
MAX_DEPTH = 1
CORRECTION_SAMPLES = 1


def load_emission_model() -> LookupEmission:
    path = Path("results/emission_model_felm_science_groq.json")
    data = json.loads(path.read_text())
    table: dict[float, dict[bool, float]] = {}
    for bin_str, probs in data["lookup_table"].items():
        table[float(bin_str)] = {True: probs["true"], False: probs["false"]}
    return LookupEmission(table=table)


@dataclass
class AssembledTransition:
    """Bridges RawBeliefTree's per-child Relationship labels onto Phase 0's single shared
    TransitionModel. Per parent, dispatches to whichever semantics its children actually need:
    decomposition children reuse DecompositionTransition unchanged (delegated, not
    re-derived); premise/correction children use DEFAULT_TRANSITION_PARAMS per relationship,
    combined assuming conditional independence across siblings (matching
    IndependentChildTransition's approach in Phase 0, but with per-child parameters instead of
    one global pair).
    """

    raw_tree: RawBeliefTree
    relationship_params: dict[Relationship, tuple[float, float]]
    _decomposition: DecompositionTransition = field(
        default_factory=DecompositionTransition, init=False, repr=False
    )

    def joint_prob(
        self, parent_id: str, parent_truth: bool, children_truth: tuple[bool, ...]
    ) -> float:
        child_ids = self.raw_tree.nodes[parent_id].children
        first_child = self.raw_tree.nodes[child_ids[0]]
        if first_child.strategy == "decomposition":
            return self._decomposition.joint_prob(parent_id, parent_truth, children_truth)

        probability = 1.0
        for child_id, child_truth in zip(child_ids, children_truth, strict=True):
            relationship = self.raw_tree.nodes[child_id].relationship_to_parent
            if relationship is None:
                # Correction children aren't NLI-labelled against the parent (see
                # src/tree/builder.py) -- treat as uninformative about the parent's truth.
                p_true = 0.5
            else:
                p_true_given_true, p_true_given_false = self.relationship_params[relationship]
                p_true = p_true_given_true if parent_truth else p_true_given_false
            probability *= p_true if child_truth else (1.0 - p_true)
        return probability


def raw_tree_to_hmt(raw: RawBeliefTree, emission: LookupEmission) -> HiddenMarkovTree:
    nodes = {
        node_id: TreeNode(
            node_id=node_id,
            score=float(bin_index(raw_node.confidence)),
            children=tuple(raw_node.children),
        )
        for node_id, raw_node in raw.nodes.items()
    }
    transition = AssembledTransition(raw_tree=raw, relationship_params=DEFAULT_TRANSITION_PARAMS)
    return HiddenMarkovTree(
        nodes=nodes, root_id=raw.root_id, emission=emission, transition=transition
    )


def main() -> None:
    emission = load_emission_model()
    examples = load_felm_science()
    random.Random(42).shuffle(examples)
    sample = examples[:SAMPLE_SIZE]
    print(f"Evaluating on {len(sample)} of {len(examples)} FELM-Science sentences.")

    llm = CachingLLMClient(
        inner=GroqClient(), cache_dir=Path("llm_cache"), model_tag="groq-felm-eval"
    )
    labeler = NLIRelationshipLabeler(directional=CrossEncoderDirectionalNLI())

    y_true: list[bool] = []
    tree_scores: list[float] = []
    raw_scores: list[float] = []
    skipped = 0

    for i, (statement, is_true) in enumerate(sample):
        raw_tree = build_belief_tree(
            llm, labeler, statement, max_depth=MAX_DEPTH, correction_samples=CORRECTION_SAMPLES
        )
        root_confidence = raw_tree.nodes[raw_tree.root_id].confidence
        try:
            posterior = posterior_root_true(raw_tree_to_hmt(raw_tree, emission))
        except ValueError:
            # Degenerate tree (both truth values had zero likelihood) -- skip rather than
            # crash the whole run over one statement.
            skipped += 1
            continue

        y_true.append(is_true)
        tree_scores.append(posterior)
        raw_scores.append(root_confidence)
        print(
            f"  {i + 1}/{len(sample)}: label={is_true} "
            f"raw={root_confidence:.2f} tree_posterior={posterior:.3f}"
        )

    if skipped:
        print(f"Skipped {skipped} degenerate tree(s).")

    raw_auroc = roc_auc_score(y_true, raw_scores)
    tree_auroc = roc_auc_score(y_true, tree_scores)
    print(f"Raw single-shot confidence AUROC: {raw_auroc:.4f}")
    print(f"Belief-tree posterior AUROC:      {tree_auroc:.4f}")

    Path("results/felm_science_auroc_groq.json").write_text(
        json.dumps(
            {
                "sample_size": len(y_true),
                "skipped": skipped,
                "max_depth": MAX_DEPTH,
                "raw_confidence_auroc": raw_auroc,
                "belief_tree_auroc": tree_auroc,
            },
            indent=2,
        )
    )
    print("Wrote results/felm_science_auroc_groq.json")


if __name__ == "__main__":
    main()
