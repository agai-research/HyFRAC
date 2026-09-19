"""
fragments.py -- composition fragments, Def. 4-5, and n-ary compatibility, Def. 7.

A fragment phi_e is derived once from a hyperedge e and stored in the
repository Phi (Def. 5). Fragments are retrieved, never rebuilt at request
time. Joint compatibility (Def. 7) is what the neuro-symbolic layer checks.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set, Tuple

import networkx as nx

from hypergraph import Hypergraph, Hyperedge


@dataclass
class Fragment:
    """phi_e = (V_phi, prec_phi, pre_phi, post_phi, R_phi, Q_phi, kappa_phi), Def. 4."""
    id: str
    source_edge: str
    services: Set[str]
    order: List[Tuple[str, str]]         # prec_phi as (before, after) pairs
    pre: Set[str]
    post: Set[str]
    resources: Set[str]
    quality: Tuple[float, float, float, int]   # (cost, duration, reliability, complexity), Eq. qosagg
    capabilities: Set[str]                     # kappa_phi
    well_formed: bool = True


def aggregate_quality(hg: Hypergraph, services: Set[str], order: List[Tuple[str, str]]) -> Tuple[float, float, float, int]:
    """Eq. qosagg: cost additive, duration = longest chain, reliability multiplicative, complexity = max."""
    dag = nx.DiGraph()
    dag.add_nodes_from(services)
    dag.add_edges_from(order)

    cost = sum(hg.services[s].cost for s in services)
    reliability = 1.0
    for s in services:
        reliability *= hg.services[s].reliability
    complexity = max((hg.services[s].complexity for s in services), default=1)

    if nx.is_directed_acyclic_graph(dag) and dag.number_of_nodes() > 0:
        # duration is additive along the longest chain; independent branches take the max
        duration_by_node = {s: hg.services[s].duration for s in services}
        topo = list(nx.topological_sort(dag))
        best = {n: duration_by_node[n] for n in topo}
        for n in topo:
            for succ in dag.successors(n):
                best[succ] = max(best[succ], best[n] + duration_by_node[succ])
        duration = max(best.values()) if best else 0.0
    else:
        duration = sum(hg.services[s].duration for s in services)

    return (cost, duration, reliability, complexity)


def derive_fragment(hg: Hypergraph, edge: Hyperedge) -> Fragment | None:
    """
    Builds phi_e from a hyperedge e (Def. 4). Only services participate in
    V_phi; resources and concepts of e contribute to R_phi and context only.
    """
    v_phi = {v for v in edge.members if v in hg.services}
    if not v_phi:
        return None

    order: List[Tuple[str, str]] = []
    if edge.edge_type == "prereq" and edge.tail and edge.head:
        for a in edge.tail:
            for b in edge.head:
                if a in v_phi and b in v_phi:
                    order.append((a, b))

    dag_check = nx.DiGraph()
    dag_check.add_nodes_from(v_phi)
    dag_check.add_edges_from(order)
    well_formed = nx.is_directed_acyclic_graph(dag_check)

    pre = set()
    minimal = [s for s in v_phi if all(s != b for _, b in order)]  # min_{prec_phi} V_phi
    for s in minimal:
        pre.update(hg.services[s].pre)
    post = set()
    for s in v_phi:
        post.update(hg.services[s].post)

    resources = set()
    for s in v_phi:
        resources.update(hg.services[s].resources)

    quality = aggregate_quality(hg, v_phi, order)
    capabilities = {hg.services[s].capability for s in v_phi}

    return Fragment(
        id=f"phi_{edge.id}", source_edge=edge.id, services=v_phi, order=order,
        pre=pre, post=post, resources=resources, quality=quality,
        capabilities=capabilities, well_formed=well_formed,
    )


class Repository:
    """Phi = { phi_e : e well-formed }, Def. 5. Materialised offline."""

    def __init__(self):
        self.fragments: Dict[str, Fragment] = {}

    def build(self, hg: Hypergraph) -> None:
        for edge in hg.hyperedges.values():
            frag = derive_fragment(hg, edge)
            if frag is not None and frag.well_formed:
                self.fragments[frag.id] = frag

    def by_capability(self, capability: str) -> List[Fragment]:
        return [f for f in self.fragments.values() if capability in f.capabilities]

    def bridging_candidates(self, missing_predicate: str) -> List[Fragment]:
        """Fragments whose post-conditions supply a missing predicate -- used by the mediator agent."""
        return [f for f in self.fragments.values() if missing_predicate in f.post]


def resource_pair_declared_jointly(hg: Hypergraph, fi: Fragment, fj: Fragment, shared: Set[str]) -> bool:
    """A shared resource is fine only if a co-exec hyperedge declares both fragments' source edges using it jointly."""
    for edge in hg.hyperedges.values():
        if edge.edge_type == "co-exec" and shared.issubset(edge.members):
            if fi.source_edge == edge.id or fj.source_edge == edge.id:
                return True
    return False


def pairwise_compatible(hg: Hypergraph, fi: Fragment, fj: Fragment) -> bool:
    """First two conditions of Def. 7 (resource sharing and no precedence conflict)."""
    shared = fi.resources & fj.resources
    if shared and not resource_pair_declared_jointly(hg, fi, fj, shared):
        return False
    # no precedence conflict: a service cannot be required to run both before and after another
    order = set(fi.order) | set(fj.order)
    reversed_pairs = {(b, a) for (a, b) in order}
    if order & reversed_pairs:
        return False
    return True


def jointly_compatible(hg: Hypergraph, fragments: List[Fragment]) -> Tuple[bool, str]:
    """
    Def. 7: every pair compatible AND the merged order prec_F is acyclic.
    Returns (ok, reason) so the neuro-symbolic layer can log H(F).
    """
    for fi, fj in combinations(fragments, 2):
        if not pairwise_compatible(hg, fi, fj):
            return False, f"resource or precedence conflict between {fi.id} and {fj.id}"

    merged = nx.DiGraph()
    for f in fragments:
        merged.add_edges_from(f.order)
    if not nx.is_directed_acyclic_graph(merged):
        return False, "cycle in the merged precedence order prec_F"
    return True, ""
