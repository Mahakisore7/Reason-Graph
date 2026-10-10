"""Tests for the belief-tree strategies (official BTProp algorithm) -- FakeLLMClient only."""

from __future__ import annotations

from src.llm.fake import FakeLLMClient
from src.llm.prompts import (
    decomposition_prompt,
    perturb_fillin_prompt,
    perturb_question_prompt,
    select_strategy_prompt,
    support_premises_prompt,
)
from src.tree.strategies import (
    correction_strategy,
    decomposition_strategy,
    filter_claims,
    is_no_claims,
    premises_strategy,
    select_strategies,
)

S = "The Eiffel Tower, built in 1889, is in Berlin."


def test_decomposition_reads_claim_lines_only() -> None:
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(S): (
                "Here are the claims:\n"
                "Claim 1: The Eiffel Tower was built in 1889.\n"
                "Claim 2: The Eiffel Tower is in Berlin.\n"
                "Hope this helps."
            )
        }
    )
    assert decomposition_strategy(llm, S) == [
        "The Eiffel Tower was built in 1889.",
        "The Eiffel Tower is in Berlin.",
    ]


def test_decomposition_recognises_no_checkable_claims() -> None:
    statement = "I think pizza is the best food ever!"
    llm = FakeLLMClient(
        completions={decomposition_prompt(statement): "Claim 1: No check-worthy claims available."}
    )
    assert is_no_claims(decomposition_strategy(llm, statement))


def test_filter_claims_drops_non_answers_and_template_leftovers() -> None:
    assert filter_claims(
        ["I don't know the answer.", "[insert name] wrote it.", "Paris is big."]
    ) == ["Paris is big."]


def test_select_strategies_follows_official_order_and_fallback() -> None:
    def picked(answer: str) -> list[str]:
        return select_strategies(FakeLLMClient(completions={select_strategy_prompt(S): answer}), S)

    assert picked("Output: Statement Perturbation") == ["correction"]
    assert picked("Output: Logical Relation") == ["premises"]
    assert picked("Output: both") == ["premises", "correction"]
    assert picked("I am not sure") == ["premises", "correction"]


def test_premises_collects_from_both_samples_and_skips_none_applicable() -> None:
    llm = FakeLLMClient(
        samples={
            support_premises_prompt(S): [
                "Judgement: True\nPremise 1: A.\nPremise 2: B.",
                "Judgement: False\nPremise 1: No supportive premises applicable.",
            ]
        }
    )
    assert premises_strategy(llm, S) == ["A.", "B."]
    assert llm.samples_call_count == 1  # 2 samples in one call, not 2 cached calls


def test_correction_asks_question_samples_answers_and_parses_revisions() -> None:
    question = "Where is the Eiffel Tower located?"
    answers = ["Paris", "Paris, France", "I don't know", "Paris", "Lyon"]
    revised = {
        "Paris": "Revised Answer: The Eiffel Tower, built in 1889, is in Paris.",
        "Paris, France": "**Revised Answer:** The Eiffel Tower, built in 1889, is in Paris, France.",
        "I don't know": "Revised Answer: I don't know.",  # dropped by the non-answer filter
        "Lyon": "Something unparseable",
    }
    llm = FakeLLMClient(
        completions={
            perturb_question_prompt(S): f"Question: {question}",
            **{perturb_fillin_prompt(a, S): r for a, r in revised.items()},
        },
        samples={question: answers},
    )
    assert correction_strategy(llm, S) == [
        "The Eiffel Tower, built in 1889, is in Paris.",
        "The Eiffel Tower, built in 1889, is in Paris, France.",
        "The Eiffel Tower, built in 1889, is in Paris.",
    ]
    # 1 question + 5 fill-ins; the 5 answers come from one sampling call.
    assert llm.complete_call_count == 6
    assert llm.samples_call_count == 1


def test_correction_without_a_question_produces_nothing() -> None:
    llm = FakeLLMClient(completions={perturb_question_prompt(S): "No idea."})
    assert correction_strategy(llm, S) == []
