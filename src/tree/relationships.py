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


# (p_true_given_parent_true, p_true_given_parent_false) per relationship, in the shape
# IndependentChildTransition (Phase 0) expects. Taken from the official BTProp code
# (src/hmm.py, get_p_child_state_given_parent_true/false) -- D29. Strict logic, with 0.5
# wherever the relationship says nothing:
#   EQUIVALENCE:          same truth value.
#   ENTAILMENT:           child => parent. Parent false => child false; parent true says nothing.
#   REVERSE_ENTAILMENT:   parent => child. Parent true => child true; parent false says nothing.
#   CONTRADICTION:        parent true => child false; parent false says nothing (both can be
#                         false at once).
DEFAULT_TRANSITION_PARAMS: dict[Relationship, tuple[float, float]] = {
    Relationship.EQUIVALENCE: (1.0, 0.0),
    Relationship.ENTAILMENT: (0.5, 0.0),
    Relationship.REVERSE_ENTAILMENT: (1.0, 0.5),
    Relationship.CONTRADICTION: (0.0, 0.5),
}
