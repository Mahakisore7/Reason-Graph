"""The belief-tree child-generation strategies, following the official BTProp code
(github.com/UCSB-NLP-Chang/BTProp, src/tree_gen.py) -- D29. Each is a pure function of
(LLMClient, statement) -> child statements, unit-testable with FakeLLMClient.

    decomposition_strategy  split into distinct factual claims ("Claim 1: ...")
    select_strategies       ask the LLM: premises, correction, or both
    premises_strategy       supportive premises, 2 samples ("Premise 1: ...")
    correction_strategy     key wh-question -> 5 sampled answers -> statement revised to match
"""

from __future__ import annotations

from src.llm.base import LLMClient
from src.llm.prompts import (
    decomposition_prompt,
    perturb_fillin_prompt,
    perturb_question_prompt,
    select_strategy_prompt,
    support_premises_prompt,
)

NO_CLAIMS = "No check-worthy claims"

# Official filter (tree_gen.py, unk_word_pool): drop generated statements that are really
# "I don't know"-style non-answers.
_UNKNOWN_PHRASES = [
    p.lower()
    for p in [
        "I don't", "Not specified", "cannot be determined", "No Answer", "No final answer",
        "I do not", "N/A", "No information", "I cannot", "I can't ", "unable to", "don't know",
        "No answer", "Nobody", "enough information", "specific information", "No specific",
        "not provided", "None", "No character", "No output", "Cannot answer", "Unavailable",
        "TBD", "To Be Determined", "I am asking about", "current year", "unclear", "confusion",
        "not aware of", "invalid", "additional context", "further context",
        "additional information", "Unable to determine", "further information",
    ]
]  # fmt: skip

PREMISE_SAMPLES = 2
CORRECTION_SAMPLES = 5


def filter_claims(statements: list[str]) -> list[str]:
    """Drops template leftovers ("[...]") and non-answers, as the official code does."""
    kept = []
    for s in statements:
        if "[" in s and "]" in s:
            continue
        if any(phrase in s.lower() for phrase in _UNKNOWN_PHRASES):
            continue
        kept.append(s)
    return kept


def _after_label(line: str) -> str:
    """'Claim 2: text' -> 'text'. The official code slices a fixed width instead, which breaks
    from 'Claim 10:' on; splitting at the first colon gives the same result below 10."""
    return line.split(":", 1)[1].strip() if ":" in line else ""


def is_no_claims(claims: list[str]) -> bool:
    return len(claims) == 1 and NO_CLAIMS in claims[0]


def decomposition_strategy(llm: LLMClient, statement: str) -> list[str]:
    """The claims in the statement. More than one: decompose. Exactly the NO_CLAIMS message:
    nothing checkable. Otherwise (one claim, or nothing parseable): use the other strategies."""
    response = llm.complete(decomposition_prompt(statement), max_tokens=300)
    claims = [
        _after_label(line) for line in response.strip().split("\n") if line.startswith("Claim")
    ]
    return filter_claims([c for c in claims if c])


def select_strategies(llm: LLMClient, statement: str) -> list[str]:
    """'premises', 'correction', or both -- checked in the official order (perturbation first);
    anything unrecognised falls back to both, as in the official code."""
    answer = llm.complete(select_strategy_prompt(statement), max_tokens=30).lower()
    if "perturb" in answer:
        return ["correction"]
    if "logic" in answer:
        return ["premises"]
    return ["premises", "correction"]


def premises_strategy(llm: LLMClient, statement: str) -> list[str]:
    """Supportive premises from 2 samples at temperature 0.7. The official code also has a
    contradicting-premise branch, but a for/else there always empties that list, so its trees
    contain supportive premises only -- we follow what the code does (D29). The caller drops
    NLI-unrelated premises and keeps the first 3."""
    premises = []
    for sample in llm.complete_samples(
        support_premises_prompt(statement), PREMISE_SAMPLES, max_tokens=512, temperature=0.7
    ):
        for raw_line in sample.split("\n"):
            line = raw_line.strip()
            if line.startswith("Premise"):
                text = _after_label(line)
                if text and "premises applicable" not in text:
                    premises.append(text)
    return premises


def correction_strategy(llm: LLMClient, statement: str) -> list[str]:
    """Statement correction ("perturbation"): ask for the key wh-question, sample 5 answers at
    temperature 0.7, and have the LLM revise the statement to agree with each answer. Corrected
    versions that contradict the original are evidence it's false."""
    response = llm.complete(perturb_question_prompt(statement), max_tokens=300)
    question = None
    for line in response.strip().split("\n"):
        if line.startswith("Question"):
            # Official: line.strip("Question: ") -- strips those *characters* from both ends,
            # which can eat the end of the question; take the text after the label instead.
            question = _after_label(line)
    if not question:
        return []

    revised = []
    for answer in llm.complete_samples(
        question, CORRECTION_SAMPLES, max_tokens=200, temperature=0.7
    ):
        output = llm.complete(perturb_fillin_prompt(answer, statement), max_tokens=100)
        output = output.strip().replace("*", "")
        for prefix in ("Revised Answer:", "Revised answer:", "Revised statement:"):
            if output.startswith(prefix):
                revised.append(output[len(prefix) :].strip())
                break
        else:
            if "Revised statement: " in output:
                revised.append(output.split("Revised statement: ")[-1].strip())
    return filter_claims([r for r in revised if r])
