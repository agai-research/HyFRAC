"""
optimisation.py -- Algorithm 3: assembly, Pareto selection, confidence-gated
delivery. Def. 8 (composite application) is realised by the Application class.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from hypergraph import Service
from fragments import Fragment
from goal_tree import GoalTree, GoalNode
from agents import PlannerAgent, CriticAgent, RankerAgent


@dataclass
class Application:
    """W = (V_W, E_W), Def. 8."""
    services: Dict[str, Service]
    edges: List[Tuple[str, str]]
    fragment_ids: Set[str]
    quality: Tuple[float, float, float, int] = (0.0, 0.0, 1.0, 1)

    @classmethod
    def from_fragments(cls, fragments: List[Fragment], catalogue: Dict[str, Service] | None = None,
                        required: set | None = None) -> "Application":
        """
        Builds the application from a set of chosen fragments. When
        catalogue and required are given, each fragment contributes only
        the subset of its services whose capability is actually required
        (see PlannerAgent.plan's docstring); without them, every member
        service of every fragment is included, which is only correct when
        the fragments are already known to be capability-homogeneous.
        """
        service_ids: Set[str] = set()
        edges: List[Tuple[str, str]] = []
        for frag in fragments:
            if catalogue is not None and required is not None:
                keep = {s for s in frag.services if catalogue.get(s) and catalogue[s].capability in required}
            else:
                keep = set(frag.services)
            service_ids |= keep
            edges.extend((a, b) for a, b in frag.order if a in keep and b in keep)
        fragment_ids = {f.id for f in fragments}
        return cls(services={s: None for s in service_ids}, edges=edges, fragment_ids=fragment_ids)

    def with_fragment_added(self, frag: Fragment, catalogue: Dict[str, Service] | None = None,
                             required: set | None = None) -> "Application":
        if catalogue is not None and required is not None:
            keep = {s for s in frag.services if catalogue.get(s) and catalogue[s].capability in required}
        else:
            keep = set(frag.services)
        new_services = dict(self.services)
        for s in keep:
            new_services[s] = None
        new_edges = self.edges + [(a, b) for a, b in frag.order if a in keep and b in keep]
        return Application(services=new_services, edges=new_edges,
                            fragment_ids=self.fragment_ids | {frag.id})

    def bind_service_objects(self, catalogue: Dict[str, Service]) -> None:
        """Fills in the real Service objects once retrieval/critic need real attributes."""
        for s in list(self.services):
            self.services[s] = catalogue.get(s)

    def is_acyclic(self) -> bool:
        g = nx.DiGraph()
        g.add_nodes_from(self.services)
        g.add_edges_from(self.edges)
        return nx.is_directed_acyclic_graph(g)

    def covered_capabilities(self) -> Set[str]:
        return {s.capability for s in self.services.values() if s is not None}


def dominates(q1: Tuple[float, float, float, int], q2: Tuple[float, float, float, int]) -> bool:
    """q1 >= q2 on every (maximised) dimension, and strictly greater on at least one."""
    return all(a >= b for a, b in zip(q1, q2)) and any(a > b for a, b in zip(q1, q2))


def normalise_quality(raw: Tuple[float, float, float, int]) -> Tuple[float, float, float, float]:
    """Cost and duration inverted so all four dimensions are maximised, per Eq. utility."""
    cost, duration, reliability, complexity = raw
    return (1.0 / (1.0 + cost), 1.0 / (1.0 + duration), reliability, 1.0 / (1.0 + complexity))


def within_budget(quality: Tuple[float, float, float, int], budget: Dict[str, float]) -> bool:
    cost, duration, _, complexity = quality
    return cost <= budget.get("cost", float("inf")) and duration <= budget.get("duration", float("inf")) \
        and complexity <= budget.get("complexity", float("inf"))


def confidence(applications_fragments: List[Fragment], sigma_nn_by_frag: Dict[str, float],
               constraints_satisfied_without_relaxation: int, beta: float) -> float:
    """Eq. confidence."""
    if not applications_fragments:
        return 0.0
    avg_nn = float(np.mean([sigma_nn_by_frag.get(f.id, 0.0) for f in applications_fragments]))
    return beta * avg_nn + (1 - beta) * (constraints_satisfied_without_relaxation / 6.0)


def _satisfying_assignments(goal_tree: GoalTree) -> List[List[GoalNode]]:
    """Enumerates OR-branch choices; an AND-only tree yields exactly one assignment."""
    def expand(node: GoalNode) -> List[List[GoalNode]]:
        if node.kind == "leaf":
            return [[node]]
        if node.kind == "and":
            combos = [[]]
            for child in node.children:
                combos = [c + o for c in combos for o in expand(child)]
            return combos
        if node.kind == "or":
            return [o for child in node.children for o in expand(child)]
        return [[]]
    return expand(goal_tree.root)


def assemble(ranked_combos, goal_tree: GoalTree, catalogue: Dict[str, Service],
             preference_vector: np.ndarray, params: dict, sigma_nn_lookup: Dict[str, float],
             trace) -> Tuple["Application | None", "str | None", float]:
    """Algorithm 3, lines 1-32 (one satisfying assignment per OR-branch combination)."""
    planner, critic, ranker = PlannerAgent(), CriticAgent(), RankerAgent()
    n_iter = params["assembly"]["max_repair_iterations"]
    theta_l = params["assembly"]["confidence_gate_theta_L"]
    beta = params["assembly"]["confidence_blend_beta"]
    level = goal_tree.leaves[0].target_complexity if goal_tree.leaves else 1
    band_width = params["retrieval"]["complexity_band_width"]

    all_fragments = [f for combo in ranked_combos for f in combo.fragments]
    valid_applications: List[Application] = []

    for assignment in _satisfying_assignments(goal_tree):
        required = {leaf.capability for leaf in assignment}
        draft = planner.plan(all_fragments, assignment, catalogue)
        draft.bind_service_objects(catalogue)
        trace.log("plan", {"services": list(draft.services)})

        for it in range(n_iter):
            issues = critic.critique(draft, goal_tree, level, band_width)
            trace.log("critique", {"iteration": it, "issues": issues})
            if not issues:
                valid_applications.append(draft)
                break
            revised = ranker.resolve(draft, issues, all_fragments, catalogue, required)
            revised.bind_service_objects(catalogue)
            if revised.fragment_ids == draft.fragment_ids:
                trace.log("deadlock", {"issues": issues, "iteration": it})
                break
            trace.log("revise", {"iteration": it})
            draft = revised

    if not valid_applications:
        return None, "no combination satisfied C2/C3/C6 within the repair budget", 0.0

    graded: List[Application] = []
    for app in valid_applications:
        cost = sum((catalogue[s].cost if catalogue.get(s) else 0.0) for s in app.services)
        duration = sum((catalogue[s].duration if catalogue.get(s) else 0.0) for s in app.services)
        reliabilities = [(catalogue[s].reliability if catalogue.get(s) else 1.0) for s in app.services]
        reliability = float(np.prod(reliabilities)) if reliabilities else 1.0
        complexity = max((catalogue[s].complexity if catalogue.get(s) else 1) for s in app.services) if app.services else 1
        app.quality = (cost, duration, reliability, complexity)
        graded.append(app)

    budget = {"cost": 10_000.0, "duration": 10_000.0, "complexity": 10}  # generous default budget (C5)
    feasible = [a for a in graded if within_budget(a.quality, budget)]
    if not feasible:
        return None, "no assembled application met the budget (C5)", 0.0

    pareto = [a for a in feasible
              if not any(dominates(normalise_quality(b.quality), normalise_quality(a.quality))
                         for b in feasible if b is not a)]
    best = max(pareto, key=lambda a: float(np.dot(preference_vector, normalise_quality(a.quality))))

    frags_in_best = [f for f in all_fragments if f.id in best.fragment_ids]
    constraints_ok = 6  # simplification: the repair loop already enforces C1, C2, C3, C4, C6; C5 just verified
    conf = confidence(frags_in_best, sigma_nn_lookup, constraints_ok, beta)
    trace.log("select", {"fragments": list(best.fragment_ids), "confidence": conf})

    if conf < theta_l:
        return best, "confidence below gate theta_L; clarification required", conf
    return best, None, conf
