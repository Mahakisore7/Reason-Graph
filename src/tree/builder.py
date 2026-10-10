"""Builds one belief tree from a statement, following the official BTProp pipeline
(src/tree_gen.py, premise_generate_chatgpt; hmm_forward.py) -- D29:

1. Only the root may be decomposed. If the LLM extracts more than one claim, those claims are
   the children (parent true only if all are true). If it finds nothing checkable, the root
   gets no children.
2. Otherwise the LLM picks premises, correction, or both (D8 replaced by the paper's choice).
3. Every candidate child is NLI-labelled against its parent and dropped if unrelated. At most 3
   premises and 5 corrections are kept; a correction identical to its parent is skipped.
4. Decomposition and premise children get an LLM confidence and can be expanded further;
   corrections are leaves with confidence fixed at 1.0 (they carry a separate emission model).
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
    is_no_claims,
    premises_strategy,
    select_strategies,
)

MAX_PREMISES = 3
MAX_CORRECTIONS = 5


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
    """Statements, confidences and relationships -- not yet an inference-ready
    HiddenMarkovTree (that conversion needs a fitted emission model)."""

    nodes: dict[str, RawNode]
    root_id: str


def build_belief_tree(
    llm: LLMClient,
    labeler: RelationshipLabeler,
    root_statement: str,
    *,
    max_depth: int = 1,
) -> RawBeliefTree:
    """Builds the tree breadth-first from root_statement, expanding nodes above max_depth."""
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
        parent = nodes[node_id]
        if depth >= max_depth or parent.strategy == "correction":
            continue

        def add_children(
            statements: list[str],
            strategy: str,
            limit: int,
            *,
            parent: RawNode = parent,
            depth: int = depth,
        ) -> None:
            added = 0
            for statement in statements:
                if added == limit:
                    break
                if strategy == "correction" and statement == parent.statement:
                    continue
                relationship = labeler.label(statement, parent.statement)
                if relationship is Relationship.NEUTRAL:
                    continue
                child_id = next(next_id)
                nodes[child_id] = RawNode(
                    node_id=child_id,
                    statement=statement,
                    confidence=(
                        1.0 if strategy == "correction" else llm.true_false_confidence(statement)
                    ),
                    parent_id=parent.node_id,
                    relationship_to_parent=relationship,
                    strategy=strategy,
                )
                parent.children.append(child_id)
                queue.append((child_id, depth + 1))
                added += 1

        if parent.strategy == "root":
            claims = decomposition_strategy(llm, parent.statement)
            if is_no_claims(claims):
                continue
            if len(claims) > 1:
                add_children(claims, "decomposition", limit=len(claims))
                continue

        for strategy in select_strategies(llm, parent.statement):
            if strategy == "premises":
                add_children(premises_strategy(llm, parent.statement), "premise", MAX_PREMISES)
            else:
                add_children(
                    correction_strategy(llm, parent.statement), "correction", MAX_CORRECTIONS
                )

    return RawBeliefTree(nodes=nodes, root_id=root_id)
