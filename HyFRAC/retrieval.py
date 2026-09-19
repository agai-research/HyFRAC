"""
retrieval.py -- Algorithm 1: request understanding and fragment retrieval.

Builds the goal tree from the request, then runs one community-pruned
incidence walk per leaf (Eq. bias), returning the top-k fragments per leaf.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

from hypergraph import Hypergraph
from fragments import Repository, Fragment
from goal_tree import GoalTree, Ontology
from agents import DecompositionAgent, ParsedRequest


def bias(hg: Hypergraph, vertex: str, capability: str, target_level: int, band_width: int,
         source_edge_confidence: float) -> float:
    """Eq. bias: b(v) = omega(e_v) * Cover(v, n) / (1 + max(0, delta_v - (delta_L + Delta)))."""
    if vertex in hg.services:
        cover = 1.0 if hg.services[vertex].capability == capability else 0.2
        delta_v = hg.services[vertex].complexity
    else:
        cover, delta_v = 0.5, target_level
    demotion = 1.0 + max(0.0, delta_v - (target_level + band_width))
    return source_edge_confidence * cover / demotion


@dataclass
class RetrievalResult:
    goal_tree: GoalTree
    candidate_fragments: Dict[str, List[Tuple[Fragment, float]]]  # leaf capability -> [(fragment, score)]
    parsed: ParsedRequest


def retrieve(request: str, ontology: Ontology, hg: Hypergraph, repository: Repository,
             decomposer: DecompositionAgent, params: dict,
             satisfied_predicates: set | None = None) -> RetrievalResult:
    """Algorithm 1, lines 1-26."""
    satisfied_predicates = satisfied_predicates or set()
    d_w = params["retrieval"]["walk_depth"]
    k = params["retrieval"]["breadth_k"]
    theta_com = params["retrieval"]["community_threshold"]
    theta_node = params["retrieval"]["node_threshold"]
    band_width = params["retrieval"]["complexity_band_width"]

    parsed = decomposer.parse(request)
    tree = decomposer.build_goal_tree(parsed)

    missing = tree.check_coverage(ontology, satisfied_predicates)
    if missing:
        tree = decomposer.repair(tree, missing)

    # community summaries are precomputed offline (sec:offline); build once and
    # reuse across every request rather than recomputing per call
    if not hg._community_summary:
        hg.build_community_summaries()
    candidates: Dict[str, List[Tuple[Fragment, float]]] = {}

    for leaf in tree.leaves:
        scored: Dict[str, Tuple[Fragment, float]] = {}
        anchors = {s for s, svc in hg.services.items() if svc.capability == leaf.capability}
        visited, frontier = set(), set(anchors)

        for _ in range(d_w):
            next_frontier = set()
            for vertex in frontier:
                for edge in hg.edges_containing(vertex):
                    if hg.community_relevance(edge.id, leaf.capability) < theta_com:
                        continue  # prune the whole community
                    for u in edge.members - visited:
                        score = bias(hg, u, leaf.capability, leaf.target_complexity or parsed.level,
                                     band_width, edge.confidence)
                        if score >= theta_node:
                            next_frontier.add(u)
                            visited.add(u)
                    frag = repository.fragments.get(f"phi_{edge.id}")
                    if frag is not None and leaf.capability in frag.capabilities:
                        best_bias = max((bias(hg, u, leaf.capability, leaf.target_complexity or parsed.level,
                                               band_width, edge.confidence) for u in edge.members), default=0.0)
                        combined = edge.confidence * best_bias
                        if frag.id not in scored or combined > scored[frag.id][1]:
                            scored[frag.id] = (frag, combined)
            frontier = next_frontier

        top_k = sorted(scored.values(), key=lambda pair: pair[1], reverse=True)[:k]
        candidates[leaf.capability] = top_k

    return RetrievalResult(goal_tree=tree, candidate_fragments=candidates, parsed=parsed)
