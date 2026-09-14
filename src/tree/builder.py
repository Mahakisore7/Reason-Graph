"""Orchestrates the three strategies + relationship labelling + confidence scoring into one
tree.

Design decision (see docs/phases/phase-2-tree-construction.md, "A Design Decision That's Yours
to Make"): when decomposition doesn't split a statement, we run BOTH premises and correction --
option (b) from the doc. This is simpler than a dedicated "which strategy" prompt and strictly
more thorough (more LLM calls per node, but no extra strategy-selection logic to get wrong).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count

from src.llm.base import LLMClient
from src.tree.relationships import Relationship, RelationshipLabeler
from src.tree.strategies import (
    correction_strategy,
    decomposition_strategy,
    premises_strategy,
)


@dataclass
class RawNode:
    node_id: str
    statement: str
    confidence: float
    parent_id: str | None
    relationship_to_parent: Relationship | None
    strategy: str  # "root" | "decomposition" | "premise" | "correction"
    children: list[str] = field(default_factory=list)


@dataclass
class RawBeliefTree:
    """Intermediate representation -- statements, confidences, relationships, but not yet an
    inference-ready HiddenMarkovTree/FactorGraph. That conversion needs a *fitted* emission model
    (needs real labelled data) and belongs in Phase 5, not here."""

    nodes: dict[str, RawNode]
    root_id: str


def build_belief_tree(
    llm: LLMClient,
    labeler: RelationshipLabeler,
    root_statement: str,
    *,
    max_depth: int = 2,
    correction_samples: int = 2,
) -> RawBeliefTree:
    """Builds the tree breadth-first from root_statement, stopping at max_depth."""
    next_id = (f"n{i}" for i in count())

    root_id = next(next_id)
    nodes: dict[str, RawNode] = {
        root_id: RawNode(
            node_id=root_id,
            statement=root_statement,
            confidence=llm.true_false_confidence(root_statement),
            parent_id=None,
            relationship_to_parent=None,
            strategy="root",
        )
    }

    queue: deque[tuple[str, int]] = deque([(root_id, 0)])
    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        parent = nodes[node_id]

        def add_child(
            statement: str,
            *,
            confidence: float,
            relationship: Relationship | None,
            strategy: str,
            parent_id: str = node_id,
            depth: int = depth,
        ) -> None:
            child_id = next(next_id)
            nodes[child_id] = RawNode(
                node_id=child_id,
                statement=statement,
                confidence=confidence,
                parent_id=parent_id,
                relationship_to_parent=relationship,
                strategy=strategy,
            )
            nodes[parent_id].children.append(child_id)
            queue.append((child_id, depth + 1))

        sub_facts = decomposition_strategy(llm, parent.statement)
        if len(sub_facts) > 1:
            for fact in sub_facts:
                add_child(
                    fact,
                    confidence=llm.true_false_confidence(fact),
                    relationship=Relationship.EQUIVALENCE,
                    strategy="decomposition",
                )
            continue  # decomposition took -- don't also run premises/correction on this node

        for premise in premises_strategy(llm, parent.statement):
            relationship = labeler.label(premise.text, parent.statement)
            if relationship is Relationship.NEUTRAL:
                continue
            add_child(
                premise.text,
                confidence=llm.true_false_confidence(premise.text),
                relationship=relationship,
                strategy="premise",
            )

        for corrected_statement in correction_strategy(
            llm, parent.statement, num_samples=correction_samples
        ):
            add_child(
                corrected_statement,
                confidence=1.0,  # fixed, never re-probed -- see strategies.correction_strategy
                relationship=None,  # not NLI-labelled: it's a rewritten alternative, not a claim
                strategy="correction",
            )

    return RawBeliefTree(nodes=nodes, root_id=root_id)
