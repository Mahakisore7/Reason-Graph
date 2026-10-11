"""Demo: check one statement, or several sentences of one answer, and show the working.

    python scripts/verify.py --backend groq "The Eiffel Tower is in Berlin."
    python scripts/verify.py --backend groq --answer "Sentence one." "Sentence two."

One statement -> the paper's method (belief tree). Several sentences with --answer -> our
answer-level graph as well (D33, D35). Prints every node of every tree: its strategy, its
relationship to the parent and the LLM's confidence, then P(true) and the verdict.

The calibration is the one fitted for the chosen backend (results/emission_model_...json): run
scripts/fit_emission_model.py with the same backend first. Note: Groq gives a typed-out
confidence, not token probabilities (D6) -- fine for a demo, not for the reported numbers,
which come from Llama-3-8B on Kaggle.
"""

from __future__ import annotations

import argparse

from scripts.backends import add_backend_args, make_backend
from scripts.evaluate_felm import MAX_DEPTH, load_calibrations
from scripts.fit_emission_model import add_standalone_arg, result_name
from src.pipeline import RuleLabeler, verify_answer, verify_statement
from src.tree.builder import RawBeliefTree
from src.tree.nli_labeler import CrossEncoderDirectionalNLI, NLIRelationshipLabeler


def show_tree(tree: RawBeliefTree, node_id: str | None = None, indent: int = 0) -> None:
    node = tree.nodes[node_id or tree.root_id]
    link = node.relationship_to_parent.value if node.relationship_to_parent else "-"
    print(
        f"{'    ' * indent}[{node.strategy}, {link}, confidence {node.confidence:.2f}] "
        f"{node.statement}"
    )
    for child in node.children:
        show_tree(tree, child, indent + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    add_standalone_arg(parser)
    parser.add_argument("--answer", action="store_true", help="sentences form one answer")
    parser.add_argument("sentences", nargs="+")
    args = parser.parse_args()
    backend = make_backend(args)
    # The smooth curve if it was fitted for this backend, otherwise the 5-bucket table.
    calibration = load_calibrations(result_name(args, backend.run_name))[-1]
    print(f"LLM: {backend.model}; calibration: {calibration.name}")
    llm = backend.cached("demo")
    nli = CrossEncoderDirectionalNLI()
    labeler = NLIRelationshipLabeler(directional=nli)

    if not args.answer:
        for statement in args.sentences:
            result = verify_statement(llm, labeler, calibration, statement, max_depth=MAX_DEPTH)
            show_tree(result.raw_tree)
            verdict = "looks true" if result.is_trustworthy else "LIKELY HALLUCINATION"
            print(f"P(true) = {result.posterior_true:.3f} -> {verdict}\n")
        return

    answer = verify_answer(
        llm, labeler, RuleLabeler(nli), calibration, args.sentences, max_depth=MAX_DEPTH
    )
    for tree in answer.trees:
        show_tree(tree)
    print("\nLinks between sentences:")
    for link in answer.links:
        print(f"  sentence {link.child + 1} -- {link.relationship.value} -- {link.parent + 1}")
    print(f"Loopy BP converged: {answer.converged}\n")
    for k, s in enumerate(args.sentences):
        print(f"{k + 1}. tree {answer.tree_only[k]:.3f}  graph {answer.posterior_true[k]:.3f}  {s}")


if __name__ == "__main__":
    main()
