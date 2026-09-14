"""Tests for the forward/backward -> 5-way Relationship combination logic in
NLIRelationshipLabeler. Uses FakeDirectionalNLI -- no model download needed."""

from __future__ import annotations

from src.tree.nli_labeler import FakeDirectionalNLI, NLIRelationshipLabeler
from src.tree.relationships import Relationship

CHILD = "The Eiffel Tower is in Paris."
PARENT = "The Eiffel Tower is in France."


def _labeler(forward: str, backward: str) -> NLIRelationshipLabeler:
    directional = FakeDirectionalNLI(
        labels={
            (CHILD, PARENT): forward,
            (PARENT, CHILD): backward,
        }
    )
    return NLIRelationshipLabeler(directional=directional)


def test_both_directions_entail_gives_equivalence() -> None:
    assert _labeler("entailment", "entailment").label(CHILD, PARENT) is Relationship.EQUIVALENCE


def test_only_forward_entails_gives_entailment() -> None:
    assert _labeler("entailment", "neutral").label(CHILD, PARENT) is Relationship.ENTAILMENT


def test_only_backward_entails_gives_reverse_entailment() -> None:
    assert _labeler("neutral", "entailment").label(CHILD, PARENT) is Relationship.REVERSE_ENTAILMENT


def test_either_direction_contradiction_gives_contradiction() -> None:
    assert _labeler("contradiction", "neutral").label(CHILD, PARENT) is Relationship.CONTRADICTION
    assert _labeler("neutral", "contradiction").label(CHILD, PARENT) is Relationship.CONTRADICTION


def test_both_neutral_gives_neutral() -> None:
    assert _labeler("neutral", "neutral").label(CHILD, PARENT) is Relationship.NEUTRAL


def test_contradiction_wins_even_if_other_direction_entails() -> None:
    # Contradiction is checked first: if either direction contradicts, that's decisive,
    # regardless of what the other direction says.
    assert (
        _labeler("contradiction", "entailment").label(CHILD, PARENT) is Relationship.CONTRADICTION
    )
