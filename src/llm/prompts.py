"""Prompt templates for the LLM access layer. Plain functions returning strings, one per
prompt need (docs/phases/phase-1-environment-llm-access.md and
docs/phases/04-research-notes-btprop.md).
"""

from __future__ import annotations


def true_false_confidence_prompt(statement: str) -> str:
    """Elicits a single-word True/False answer, so reading next-token logprobs for
    exactly those two words is meaningful. The explicit instruction matters for chat models,
    which otherwise often open with "The statement is..." and bury the answer."""
    return (
        "Is the following statement true or false? Answer with exactly one word: True or "
        f"False.\n\nStatement: {statement}"
    )


def decomposition_prompt(statement: str) -> str:
    """Asks whether the statement bundles multiple separate facts; if so, split into a
    list such that the statement is true iff every fact is true. Signals an already-atomic
    statement with a literal NONE so the caller can detect it without guessing."""
    return (
        "Does the following statement bundle together more than one separate, "
        "independently checkable fact? If so, split it into a numbered list of atomic "
        "sub-statements such that the original statement is true if and only if every "
        "sub-statement is true. If the statement is already a single atomic fact, "
        "respond with exactly the word NONE and nothing else.\n\n"
        f"Statement: {statement}"
    )


def premises_prompt(statement: str) -> str:
    """Asks for premises that would SUPPORT the statement, and separately ones that would
    CONTRADICT it, in a format that can be reliably parsed back apart."""
    return (
        "List premises that would make the following statement more likely to be TRUE, "
        "then list premises that would make it more likely to be FALSE. Format each line "
        "as 'SUPPORT: <premise>' or 'CONTRADICT: <premise>', one premise per line, with "
        "no other text.\n\n"
        f"Statement: {statement}"
    )


def correction_question_prompt(statement: str) -> str:
    """Asks for the single key factual question this statement's truth depends on."""
    return (
        "What is the single key factual question that determines whether the following "
        "statement is true or false? Respond with only that question.\n\n"
        f"Statement: {statement}"
    )


def correction_answer_prompt(question: str) -> str:
    """Samples an answer to the key question produced by correction_question_prompt."""
    return question


def correction_statement_prompt(statement: str, question: str, answer: str) -> str:
    """Rewrites the original statement to be consistent with one sampled answer."""
    return (
        f"Original statement: {statement}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Rewrite the original statement so it is consistent with the answer above. If "
        "the original statement is already consistent with the answer, repeat it "
        "unchanged. Respond with only the rewritten statement."
    )


def nli_relationship_prompt(premise: str, hypothesis: str) -> str:
    """LLM-based NLI fallback, only needed if not using a dedicated NLI model (Phase 2
    discusses both options). Asks whether the premise entails, contradicts, or is
    unrelated to the hypothesis."""
    return (
        "Does the premise ENTAIL the hypothesis (the hypothesis must be true if the "
        "premise is true), CONTRADICT it (they cannot both be true), or is it NEUTRAL "
        "(unrelated)? Respond with exactly one word: ENTAIL, CONTRADICT, or NEUTRAL.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}"
    )
