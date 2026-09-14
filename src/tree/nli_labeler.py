"""A real, model-backed RelationshipLabeler. Uses the same checkpoint as the official BTProp
repo (cross-encoder/nli-deberta-v3-large via sentence_transformers.CrossEncoder), run in BOTH
directions, so we can tell entailment from reverse-entailment -- a single-direction NLI model
can only ever say "entails / contradicts / neutral", it has no notion of which way round.

Split into two pieces, same shape as the LLM clients (Protocol + real + fake):
    DirectionalNLI    -- one call, one direction: "does premise entail/contradict/ignore
                         hypothesis?" CrossEncoderDirectionalNLI is the real implementation;
                         FakeDirectionalNLI is a canned stand-in for tests.
    NLIRelationshipLabeler -- calls a DirectionalNLI twice (both directions) and combines the
                         two answers into our 5-way Relationship. This combination logic is
                         plain Python, so it's tested with FakeDirectionalNLI, no model needed.

How the two one-directional answers combine:
    forward  = directional.label_direction(child, parent)   i.e. does child entail parent?
    backward = directional.label_direction(parent, child)   i.e. does parent entail child?

    forward=entailment, backward=entailment  -> EQUIVALENCE   (each implies the other)
    forward=entailment, backward!=entailment -> ENTAILMENT    (child => parent, only)
    backward=entailment, forward!=entailment -> REVERSE_ENTAILMENT  (parent => child, only)
    either direction says contradiction      -> CONTRADICTION
    otherwise (both neutral)                 -> NEUTRAL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol

from src.tree.relationships import Relationship


class DirectionalNLI(Protocol):
    def label_direction(self, premise: str, hypothesis: str) -> str:
        """Returns one of "entailment", "contradiction", "neutral"."""
        ...


@dataclass
class CrossEncoderDirectionalNLI:
    """Wraps sentence_transformers.CrossEncoder -- the same checkpoint the official BTProp
    repo uses for NLI labelling."""

    model_name: str = "cross-encoder/nli-deberta-v3-large"

    @cached_property
    def _model(self) -> Any:
        # Imported locally: this module shouldn't hard-require sentence_transformers just to
        # be imported -- only actually calling label_direction() does, same pattern as the
        # LLM clients importing `openai` locally.
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name)

    @cached_property
    def _id2label(self) -> dict[int, str]:
        # Read the label order from the model itself instead of hard-coding it -- checkpoints
        # are not guaranteed to agree on whether index 0 means "contradiction" or "entailment".
        return {k: v.lower() for k, v in self._model.model.config.id2label.items()}

    def label_direction(self, premise: str, hypothesis: str) -> str:
        scores = self._model.predict([(premise, hypothesis)])
        label_id = int(scores[0].argmax())
        return self._id2label[label_id]


@dataclass
class FakeDirectionalNLI:
    """Canned directional labels keyed by (premise, hypothesis), for testing the combination
    logic in NLIRelationshipLabeler without downloading a real model."""

    labels: dict[tuple[str, str], str] = field(default_factory=dict)
    default: str = "neutral"

    def label_direction(self, premise: str, hypothesis: str) -> str:
        return self.labels.get((premise, hypothesis), self.default)


@dataclass
class NLIRelationshipLabeler:
    directional: DirectionalNLI

    def label(self, premise: str, hypothesis: str) -> Relationship:
        forward = self.directional.label_direction(premise, hypothesis)
        backward = self.directional.label_direction(hypothesis, premise)

        if forward == "contradiction" or backward == "contradiction":
            return Relationship.CONTRADICTION
        if forward == "entailment" and backward == "entailment":
            return Relationship.EQUIVALENCE
        if forward == "entailment":
            return Relationship.ENTAILMENT
        if backward == "entailment":
            return Relationship.REVERSE_ENTAILMENT
        return Relationship.NEUTRAL
