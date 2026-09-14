"""Determining the logical relationship between a parent claim and a child claim. Matches
docs/04-research-notes-btprop.md, Step 2: four relationships (equivalence, entailment, reverse
entailment, contradiction), normally decided by an NLI model run in both directions. "neutral"
means the pair is unrelated and the child should be dropped.

Direction convention used everywhere in this package: label(premise, hypothesis) is always
called as label(premise=<child candidate statement>, hypothesis=<parent statement>) -- i.e. we
ask "does the child entail/contradict the parent?", never the other way round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class Relationship(Enum):
    EQUIVALENCE = "equivalence"
    ENTAILMENT = "entailment"  # premise entails hypothesis
    REVERSE_ENTAILMENT = "reverse_entailment"  # hypothesis entails premise
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"  # unrelated -- caller should drop this child


class RelationshipLabeler(Protocol):
    def label(self, premise: str, hypothesis: str) -> Relationship: ...


@dataclass
class FakeRelationshipLabeler:
    """Canned relationship labels keyed by (premise, hypothesis), for tests."""

    labels: dict[tuple[str, str], Relationship] = field(default_factory=dict)
    default: Relationship = Relationship.NEUTRAL
    call_count: int = field(default=0, init=False)

    def label(self, premise: str, hypothesis: str) -> Relationship:
        self.call_count += 1
        return self.labels.get((premise, hypothesis), self.default)


# Starting (p_true_given_parent_true, p_true_given_parent_false) values, in the shape
# IndependentChildTransition (Phase 0) expects. Meant to be refit against real labelled data
# later -- these are just enough to get a first end-to-end run working.
#
# How each number was chosen (using the label(child, parent) convention above):
#   EQUIVALENCE:          child and parent share a truth value almost always.
#   ENTAILMENT:           child => parent. So if parent is FALSE, child MUST be false too
#                         (contrapositive) -- strong low number. If parent is true, that says
#                         little about the child on its own -- mild number, just above 0.5.
#   REVERSE_ENTAILMENT:   parent => child. So if parent is TRUE, child MUST be true too --
#                         strong high number. If parent is false, that says little about the
#                         child -- mild number, just below 0.5.
#   CONTRADICTION:        child and parent (almost) always have opposite truth values.
DEFAULT_TRANSITION_PARAMS: dict[Relationship, tuple[float, float]] = {
    Relationship.EQUIVALENCE: (0.95, 0.05),
    Relationship.ENTAILMENT: (0.65, 0.05),
    Relationship.REVERSE_ENTAILMENT: (0.95, 0.35),
    Relationship.CONTRADICTION: (0.05, 0.95),
}
