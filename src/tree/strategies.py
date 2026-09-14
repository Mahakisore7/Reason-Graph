"""The three belief-tree child-generation strategies (docs/04-research-notes-btprop.md, Step 1).
Each is a pure function of (LLMClient, statement) -> structured children -- easy to unit test
independently with FakeLLMClient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.llm.base import LLMClient
from src.llm.prompts import (
    correction_answer_prompt,
    correction_question_prompt,
    correction_statement_prompt,
    decomposition_prompt,
    premises_prompt,
)


class PremiseStance(Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"


@dataclass(frozen=True)
class Premise:
    text: str
    stance: PremiseStance


_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s*(.+)$")
_PREMISE_LINE = re.compile(r"^\s*(SUPPORT|CONTRADICT)\s*:\s*(.+)$", re.IGNORECASE)


def decomposition_strategy(llm: LLMClient, statement: str) -> list[str]:
    """Strategy A. Returns the sub-facts if the statement decomposes, or [statement] unchanged
    if already atomic -- callers use len(result) > 1 to decide whether decomposition "took".
    """
    response = llm.complete(decomposition_prompt(statement))
    if response.strip().upper() == "NONE":
        return [statement]

    sub_facts: list[str] = []
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _NUMBERED_LINE.match(line)
        sub_facts.append(match.group(1).strip() if match else line)

    if len(sub_facts) < 2:
        # Only one usable line came back -- treat as "didn't really decompose".
        return [statement]
    return sub_facts


def premises_strategy(llm: LLMClient, statement: str) -> list[Premise]:
    """Strategy B. Parses the LLM's response into typed Premise objects."""
    response = llm.complete(premises_prompt(statement))

    premises: list[Premise] = []
    for raw_line in response.splitlines():
        match = _PREMISE_LINE.match(raw_line.strip())
        if not match:
            # Doesn't match "SUPPORT: ..." / "CONTRADICT: ..." -- skip, don't crash.
            continue
        text = match.group(2).strip()
        if not text:
            continue
        stance = (
            PremiseStance.SUPPORT
            if match.group(1).upper() == "SUPPORT"
            else PremiseStance.CONTRADICT
        )
        premises.append(Premise(text=text, stance=stance))
    return premises


def correction_strategy(llm: LLMClient, statement: str, num_samples: int = 2) -> list[str]:
    """Strategy C. Three LLM calls per sample: ask the key question, sample an answer, rewrite
    the statement to match. Nodes produced this way get confidence fixed at 1.0 by the CALLER
    (not this function).
    """
    question = llm.complete(correction_question_prompt(statement))

    corrected_statements: list[str] = []
    for _ in range(num_samples):
        # temperature=1.0: sampling the SAME answer every time would defeat the point of
        # num_samples > 1 -- we want different plausible answers, not one repeated N times.
        answer = llm.complete(correction_answer_prompt(question), temperature=1.0)
        corrected = llm.complete(correction_statement_prompt(statement, question, answer))
        corrected_statements.append(corrected)
    return corrected_statements
