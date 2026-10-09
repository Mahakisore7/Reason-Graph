"""Tests for the three belief-tree child-generation strategies -- all against FakeLLMClient,
zero network access needed."""

from __future__ import annotations

from src.llm.fake import FakeLLMClient
from src.llm.prompts import (
    correction_answer_prompt,
    correction_question_prompt,
    correction_statement_prompt,
    decomposition_prompt,
    premises_prompt,
)
from src.tree.strategies import (
    Premise,
    PremiseStance,
    correction_strategy,
    decomposition_strategy,
    premises_strategy,
)


def test_decomposition_strategy_splits_bundled_statement() -> None:
    statement = "The sky is blue and grass is green."
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(statement): "1. The sky is blue.\n2. Grass is green.",
        }
    )
    assert decomposition_strategy(llm, statement) == ["The sky is blue.", "Grass is green."]


def test_decomposition_strategy_ignores_intro_and_closing_lines() -> None:
    statement = "The Eiffel Tower, built in 1889, is in Paris."
    llm = FakeLLMClient(
        completions={
            decomposition_prompt(statement): (
                "The statement can be considered as two separate facts:\n\n"
                "1. The Eiffel Tower was constructed in 1889.\n"
                "2. The Eiffel Tower is located in Paris.\n\n"
                "These two statements together make up the original."
            ),
        }
    )
    assert decomposition_strategy(llm, statement) == [
        "The Eiffel Tower was constructed in 1889.",
        "The Eiffel Tower is located in Paris.",
    ]


def test_decomposition_strategy_atomic_statement_returns_unchanged() -> None:
    statement = "The sky is blue."
    llm = FakeLLMClient(completions={decomposition_prompt(statement): "NONE"})
    assert decomposition_strategy(llm, statement) == [statement]


def test_decomposition_strategy_falls_back_on_single_usable_line() -> None:
    statement = "The sky is blue."
    llm = FakeLLMClient(completions={decomposition_prompt(statement): "1. The sky is blue."})
    assert decomposition_strategy(llm, statement) == [statement]


def test_premises_strategy_separates_support_and_contradict() -> None:
    statement = "Paris is the capital of France."
    llm = FakeLLMClient(
        completions={
            premises_prompt(statement): (
                "SUPPORT: Paris is France's seat of government.\n"
                "CONTRADICT: Some historical periods used a different capital.\n"
                "this line matches neither format and should be skipped"
            ),
        }
    )
    result = premises_strategy(llm, statement)
    assert result == [
        Premise(text="Paris is France's seat of government.", stance=PremiseStance.SUPPORT),
        Premise(
            text="Some historical periods used a different capital.",
            stance=PremiseStance.CONTRADICT,
        ),
    ]


def test_correction_strategy_makes_expected_number_of_calls() -> None:
    statement = "The Eiffel Tower is in Berlin."
    question = "Which city is the Eiffel Tower located in?"
    llm = FakeLLMClient(
        completions={
            correction_question_prompt(statement): question,
            correction_answer_prompt(question): "Paris",
            correction_statement_prompt(statement, question, "Paris"): (
                "The Eiffel Tower is in Paris."
            ),
        }
    )
    result = correction_strategy(llm, statement, num_samples=2)
    assert result == ["The Eiffel Tower is in Paris.", "The Eiffel Tower is in Paris."]
    # 1 question call + 2 samples * (1 answer call + 1 rewrite call) = 5
    assert llm.complete_call_count == 5
