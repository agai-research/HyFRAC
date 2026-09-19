"""
trace.py -- the append-only causal trace T, and the causal explanation graph
CE built from it (Algorithm 4, lines 1-7).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List



@dataclass
class Event:
    kind: str
    payload: Dict[str, Any]
    timestamp: int
    summary: str = ""


class CausalTrace:
    """One request's full decision log; every agent writes here."""

    def __init__(self):
        self.events: List[Event] = []
        self._clock = 0

    def log(self, kind: str, payload: Dict[str, Any]) -> None:
        self._clock += 1
        summary = self._summarise(kind, payload)
        self.events.append(Event(kind=kind, payload=payload, timestamp=self._clock, summary=summary))

    @staticmethod
    def _summarise(kind: str, payload: Dict[str, Any]) -> str:
        if kind == "plan":
            return f"drafted an application with {len(payload.get('services', []))} service(s)"
        if kind == "critique":
            issues = payload.get("issues", [])
            return "no violation found" if not issues else f"{len(issues)} violation(s): {issues}"
        if kind == "revise":
            return f"substituted a fragment at repair iteration {payload.get('iteration')}"
        if kind == "deadlock":
            return "no substitution resolved the remaining violation(s)"
        if kind == "select":
            return f"selected the Pareto-optimal application, confidence={payload.get('confidence'):.2f}"
        return kind


@dataclass
class CausalNode:
    kind: str
    summary: str
    timestamp: int


class CausalGraph:
    """CE = (Ev, edges): the events that actually influenced W*, with causal edges."""

    def __init__(self, nodes: List[CausalNode], edges: List[tuple]):
        self.nodes = nodes
        self.edges = edges

    @classmethod
    def build(cls, trace: CausalTrace, relaxation_log: list) -> "CausalGraph":
        nodes = [CausalNode(kind=e.kind, summary=e.summary, timestamp=e.timestamp) for e in trace.events]
        for i, entry in enumerate(relaxation_log):
            nodes.append(CausalNode(kind="relaxation", summary=f"round {entry['round']}: added "
                                     f"{entry['added']}", timestamp=1000 + i))

        edges = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if nodes[i].timestamp < nodes[j].timestamp:
                    # a later event that consumed an earlier one's artefact: a "revise" or
                    # "select" step causally depends on the immediately preceding "critique"/"plan"
                    if nodes[j].kind in ("revise", "select", "critique") and nodes[i].kind in ("plan", "critique", "revise", "relaxation"):
                        if nodes[j].timestamp == nodes[i].timestamp + 1:
                            edges.append((i, j))
        return cls(nodes=nodes, edges=edges)

    def prune_unreachable(self, keep_kinds=("plan", "critique", "revise", "select", "relaxation")):
        """Keeps only events on the path that actually influenced the delivered application."""
        self.nodes = [n for n in self.nodes if n.kind in keep_kinds]
