"""
goal_tree.py -- goal tree GT = (N, A, tau), Def. 6, and the domain ontology
that Algorithm 1's decomposition step queries for prerequisites/alternatives.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GoalNode:
    """A node of GT. tau(n) in {and, or, leaf}. A leaf carries (cap, delta, mu)."""
    kind: str                       # "and" | "or" | "leaf"
    capability: Optional[str] = None
    target_complexity: Optional[int] = None
    modality: Optional[str] = None
    children: List["GoalNode"] = field(default_factory=list)

    def leaves(self) -> List["GoalNode"]:
        if self.kind == "leaf":
            return [self]
        out = []
        for c in self.children:
            out.extend(c.leaves())
        return out


class GoalTree:
    def __init__(self):
        self.root = GoalNode(kind="and")

    @property
    def leaves(self) -> List[GoalNode]:
        return self.root.leaves()

    def check_coverage(self, ontology: "Ontology", satisfied_predicates: set) -> List[str]:
        """CheckCoverage: capabilities in the tree whose prerequisites are still unmet."""
        missing = []
        for leaf in self.leaves:
            for pred in ontology.prerequisite_predicates(leaf.capability):
                if pred not in satisfied_predicates:
                    missing.append(leaf.capability)
        return missing


class Ontology:
    """
    A small, explicit domain ontology (smart-education context, sec:retrieval):
    for each objective, the prerequisites that must ALL be met (-> AND node)
    and the interchangeable alternatives (-> OR node), following a taxonomy
    of cognitive objectives in the spirit of Bloom's taxonomy.
    """

    def __init__(self, prerequisites: Dict[str, List[str]], alternatives: Dict[str, List[str]],
                 predicates_by_capability: Dict[str, List[str]]):
        self._prereq = prerequisites
        self._alt = alternatives
        self._predicates = predicates_by_capability

    def prerequisites(self, capability: str) -> List[str]:
        return self._prereq.get(capability, [])

    def alternatives(self, capability: str) -> List[str]:
        return self._alt.get(capability, [])

    def prerequisite_predicates(self, capability: str) -> List[str]:
        return self._predicates.get(capability, [])

    @classmethod
    def from_json(cls, path: str) -> "Ontology":
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(data.get("prerequisites", {}), data.get("alternatives", {}), data.get("predicates", {}))
