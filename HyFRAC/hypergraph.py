"""
hypergraph.py -- the typed service hypergraph H = (V, E, lambda, omega), Def. 2.

Vertices V = S (services) u R (resources) u C (concepts).
Hyperedges are typed: co-exec, res-share, prereq, cap-cover (Def. 2).
A prereq hyperedge is directed, split into a tail T(e) and a head H(e) (Def. 3).
Stored as a sparse incidence matrix (scipy.sparse), as specified in sec:prototype.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

HYPEREDGE_TYPES = ("co-exec", "res-share", "prereq", "cap-cover")


@dataclass
class Service:
    """A smart service, Def. 1: s = (id, cap, pre, post, R_s, Q_s, modality)."""
    id: str
    capability: str
    pre: List[str] = field(default_factory=list)   # predicates required before running
    post: List[str] = field(default_factory=list)  # predicates guaranteed after running
    resources: List[str] = field(default_factory=list)   # R_s
    cost: float = 0.0
    duration: float = 0.0
    reliability: float = 1.0
    complexity: int = 1
    modality: str = "any"


@dataclass
class Hyperedge:
    """One typed hyperedge e, with confidence omega(e). Directed only if prereq."""
    id: str
    members: Set[str]
    edge_type: str
    confidence: float = 1.0
    tail: Set[str] = field(default_factory=set)   # T(e), only used when edge_type == prereq
    head: Set[str] = field(default_factory=set)   # H(e)


class Hypergraph:
    """Stores V, E and builds the sparse incidence matrix H[v, e]."""

    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.resources: Set[str] = set()
        self.concepts: Set[str] = set()
        self.hyperedges: Dict[str, Hyperedge] = {}
        self._vertex_index: Dict[str, int] = {}
        self._edge_index: Dict[str, int] = {}
        self._incidence: csr_matrix | None = None
        self._community_of: Dict[str, int] = {}       # hyperedge id -> community id
        self._community_summary: Dict[int, Set[str]] = {}  # community id -> capabilities inside it
        self._vertex_to_edges: Dict[str, List[str]] = {}   # vertex -> hyperedge ids containing it

    # -- construction -----------------------------------------------------
    def add_service(self, service: Service) -> None:
        self.services[service.id] = service
        for r in service.resources:
            self.resources.add(r)

    def add_concept(self, concept_id: str) -> None:
        self.concepts.add(concept_id)

    def add_hyperedge(self, edge: Hyperedge) -> None:
        if edge.edge_type not in HYPEREDGE_TYPES:
            raise ValueError(f"unknown hyperedge type: {edge.edge_type}")
        self.hyperedges[edge.id] = edge
        for v in edge.members:
            self._vertex_to_edges.setdefault(v, []).append(edge.id)

    def build_incidence_matrix(self) -> csr_matrix:
        """H[v, e] = 1 iff v in e (Def. 2)."""
        vertices = sorted(self.services) + sorted(self.resources) + sorted(self.concepts)
        self._vertex_index = {v: i for i, v in enumerate(vertices)}
        edges = sorted(self.hyperedges)
        self._edge_index = {e: j for j, e in enumerate(edges)}

        mat = lil_matrix((len(vertices), len(edges)), dtype=np.float32)
        for e_id, edge in self.hyperedges.items():
            j = self._edge_index[e_id]
            for v in edge.members:
                if v in self._vertex_index:
                    mat[self._vertex_index[v], j] = 1.0
        self._incidence = mat.tocsr()
        return self._incidence

    @property
    def incidence(self) -> csr_matrix:
        if self._incidence is None:
            self.build_incidence_matrix()
        return self._incidence

    # -- structural helpers used by retrieval and neuro-symbolic modules ---
    def edges_containing(self, vertex: str) -> List[Hyperedge]:
        return [self.hyperedges[e_id] for e_id in self._vertex_to_edges.get(vertex, [])]

    def clique_expansion(self) -> Dict[Tuple[str, str], float]:
        """
        BG ablation (HyFRAC-BG): project H onto its clique expansion.
        A hyperedge of size z becomes C(z, 2) ordinary weighted edges; the
        weight of a pairwise edge is the max confidence among hyperedges that
        induced it (ties resolved by the more specific type: prereq > co-exec
        > res-share > cap-cover, matching how tightly the pair is bound).
        """
        rank = {"prereq": 3, "co-exec": 2, "res-share": 1, "cap-cover": 0}
        pair_weight: Dict[Tuple[str, str], Tuple[float, int]] = {}
        for edge in self.hyperedges.values():
            members = sorted(edge.members)
            for i in range(len(members)):
                for k in range(i + 1, len(members)):
                    pair = (members[i], members[k])
                    candidate = (edge.confidence, rank[edge.edge_type])
                    if pair not in pair_weight or candidate[1] > pair_weight[pair][1]:
                        pair_weight[pair] = candidate
        return {pair: w for pair, (w, _) in pair_weight.items()}

    # -- community summaries (Leiden if available, greedy-modularity fallback) --
    def build_community_summaries(self) -> None:
        """
        Groups hyperedges into communities and records the capabilities each
        community covers, so a traversal step can be pruned by relevance
        before it is expanded (sec:retrieval). Leiden (igraph/leidenalg) is
        the paper's stated choice; when unavailable in the environment this
        falls back to networkx's greedy-modularity communities on the
        hyperedge co-membership graph, which serves the same pruning role.
        """
        import networkx as nx
        try:
            import igraph as ig
            import leidenalg
            has_leiden = True
        except ImportError:
            has_leiden = False

        # build a graph over hyperedges: two hyperedges are linked if they share a vertex.
        # Using the vertex index turns this into O(sum of vertex degrees) instead of O(E^2).
        g = nx.Graph()
        g.add_nodes_from(self.hyperedges.keys())
        for edge_ids_sharing_vertex in self._vertex_to_edges.values():
            for i in range(len(edge_ids_sharing_vertex)):
                for k in range(i + 1, len(edge_ids_sharing_vertex)):
                    g.add_edge(edge_ids_sharing_vertex[i], edge_ids_sharing_vertex[k])

        if has_leiden and g.number_of_edges() > 0:
            ig_graph = ig.Graph.TupleList(g.edges(), vertices_from_edges=False)
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            communities = [set(ig_graph.vs[idx]["name"] for idx in part) for part in partition]
        else:
            # Louvain is near-linear and, unlike greedy-modularity, tractable at
            # corpus scale (thousands of hyperedges); still a legitimate
            # community-detection method when Leiden's dependencies are absent.
            communities = list(nx.algorithms.community.louvain_communities(g, seed=0)) if g.number_of_edges() else [
                {n} for n in g.nodes()
            ]

        self._community_of = {}
        self._community_summary = {}
        for cid, comm in enumerate(communities):
            caps = set()
            for e_id in comm:
                self._community_of[e_id] = cid
                for v in self.hyperedges[e_id].members:
                    if v in self.services:
                        caps.add(self.services[v].capability)
            self._community_summary[cid] = caps

    def community_relevance(self, edge_id: str, target_capability: str) -> float:
        """Relevance(summary, leaf) used by Algorithm 1's community pruning test."""
        if not self._community_summary:
            self.build_community_summaries()
        cid = self._community_of.get(edge_id)
        if cid is None:
            return 0.0
        caps = self._community_summary[cid]
        return 1.0 if target_capability in caps else (0.3 if caps else 0.0)

    # -- persistence --------------------------------------------------------
    def to_json(self, path: str) -> None:
        data = {
            "services": [s.__dict__ for s in self.services.values()],
            "resources": sorted(self.resources),
            "concepts": sorted(self.concepts),
            "hyperedges": [
                {**e.__dict__, "members": sorted(e.members), "tail": sorted(e.tail), "head": sorted(e.head)}
                for e in self.hyperedges.values()
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Hypergraph":
        with open(path) as f:
            data = json.load(f)
        h = cls()
        for s in data["services"]:
            h.add_service(Service(**s))
        for r in data.get("resources", []):
            h.resources.add(r)
        for c in data.get("concepts", []):
            h.add_concept(c)
        for e in data["hyperedges"]:
            h.add_hyperedge(Hyperedge(
                id=e["id"], members=set(e["members"]), edge_type=e["edge_type"],
                confidence=e["confidence"], tail=set(e.get("tail", [])), head=set(e.get("head", [])),
            ))
        h.build_incidence_matrix()
        return h
