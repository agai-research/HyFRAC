"""
neurosymbolic.py -- Algorithm 2: compatibility filtering and ranking.

The symbolic check (C1, C4) runs first and short-circuits a hard violation
before any neural call, exactly as the algorithm specifies. The neural
scorer is a small two-layer hypergraph neural network, matching the paper's
own equations, implemented here in plain NumPy rather than PyTorch: no
GPU-only dependency is required to run or verify the prototype, and the
architecture (Z1, Z2, sigmoid readout) is followed exactly.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from hypergraph import Hypergraph
from fragments import Fragment, Repository, jointly_compatible
from agents import MediatorAgent


# --------------------------------------------------------------------------
# Symbolic component: C1, C4, and the soft-penalty score, Eq. softpenalty
# --------------------------------------------------------------------------
def check_constraints(hg: Hypergraph, combo: List[Fragment], penalty_prereq: float,
                       penalty_co_exec: float) -> Tuple[float, str]:
    """Returns (sigma_sym, reason). sigma_sym = 0 on a hard violation (H(F) != empty)."""
    ok, reason = jointly_compatible(hg, combo)
    if not ok:
        return 0.0, reason

    # graded pass: hyperedges behind this combination whose confidence < 1
    penalty = 0.0
    for frag in combo:
        edge = hg.hyperedges.get(frag.source_edge)
        if edge is not None and edge.confidence < 1.0:
            weight = penalty_prereq if edge.edge_type == "prereq" else penalty_co_exec
            penalty += weight * (1.0 - edge.confidence)
    return float(np.exp(-penalty)), ""


# --------------------------------------------------------------------------
# Neural component: hypergraph neural network, Z1/Z2/sigma_nn equations
# --------------------------------------------------------------------------
class HypergraphScorer:
    """sigma_nn: two-layer HGNN over the incidence matrix, NumPy implementation."""

    def __init__(self, hidden_width: int, feature_width: int, cap_embed_width: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        in_width = feature_width  # feature_width already covers [Q_s || cap embedding], x_v in R^8
        self.theta1 = rng.normal(0, 0.1, size=(in_width, hidden_width))
        self.theta2 = rng.normal(0, 0.1, size=(hidden_width, hidden_width))
        self.w_out = rng.normal(0, 0.1, size=(hidden_width + 4,))  # + profile vector width
        self.cap_embeddings: Dict[str, np.ndarray] = {}
        self.cap_embed_width = cap_embed_width
        self.feature_width = feature_width
        self.hidden_width = hidden_width
        self.profile_width = 4

    def _cap_vector(self, capability: str) -> np.ndarray:
        if capability not in self.cap_embeddings:
            rng = np.random.default_rng(abs(hash(capability)) % (2**32))
            self.cap_embeddings[capability] = rng.normal(0, 0.3, size=self.cap_embed_width)
        return self.cap_embeddings[capability]

    def _node_features(self, hg: Hypergraph) -> Tuple[np.ndarray, List[str]]:
        vertices = sorted(hg.services) + sorted(hg.resources) + sorted(hg.concepts)
        rows = []
        for v in vertices:
            if v in hg.services:
                s = hg.services[v]
                q = np.array([s.cost, s.duration, s.reliability, s.complexity], dtype=np.float32)
                q = (q - q.min()) / (q.max() - q.min() + 1e-6)
                cap = self._cap_vector(s.capability)
            else:
                q = np.zeros(self.feature_width - self.cap_embed_width, dtype=np.float32)
                cap = np.zeros(self.cap_embed_width, dtype=np.float32)
            rows.append(np.concatenate([q, cap]))
        return np.stack(rows), vertices

    @staticmethod
    def _elu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, np.exp(np.clip(x, -30, 30)) - 1)

    def propagate(self, hg: Hypergraph) -> Tuple[np.ndarray, Dict[str, int]]:
        """Eq. Z1, Z2: two hypergraph-convolution layers."""
        incidence = hg.incidence.toarray()
        deg_v = incidence.sum(axis=1)
        deg_e = incidence.sum(axis=0)
        deg_v[deg_v == 0] = 1.0
        deg_e[deg_e == 0] = 1.0
        d_v_inv_sqrt = np.diag(1.0 / np.sqrt(deg_v))
        d_e_inv = np.diag(1.0 / deg_e)
        w_e = np.diag([hg.hyperedges[e].confidence for e in sorted(hg.hyperedges)])

        x, vertices = self._node_features(hg)
        propagation = d_v_inv_sqrt @ incidence @ w_e @ d_e_inv @ incidence.T @ d_v_inv_sqrt
        z1 = self._elu(propagation @ x @ self.theta1)
        z2 = self._elu(propagation @ z1 @ self.theta2)
        index = {v: i for i, v in enumerate(vertices)}
        return z2, index

    def score(self, z2: np.ndarray, index: Dict[str, int], combo: List[Fragment],
              preference_vector: np.ndarray) -> float:
        """sigma_nn(F, L): mean-pool member embeddings, concat profile, sigmoid readout.
        Takes a pre-computed (z2, index) pair -- see propagate() -- so the
        expensive graph propagation runs once per request, not once per
        candidate combination."""
        member_rows = []
        for frag in combo:
            for s in frag.services:
                if s in index:
                    member_rows.append(z2[index[s]])
        if not member_rows:
            return 0.0
        pooled = np.mean(member_rows, axis=0)
        vec = np.concatenate([pooled, preference_vector])
        logit = float(np.dot(vec, self.w_out))
        return 1.0 / (1.0 + np.exp(-logit))

    def pretrain(self, hg: Hypergraph, epochs: int = 300, lr: float = 0.3, batch_size: int = 128,
                 seed: int = 3) -> None:
        """
        Eq. pretrain: self-supervised link prediction over H. Positive pairs
        are services that co-occur in the same hyperedge; negative pairs are
        sampled service pairs that share no hyperedge. Without this step,
        w_out starts at its random initial value and sigma_nn is close to
        noise, which silently drags every downstream confidence estimate
        down. Only w_out is trained here (theta1, theta2 stay fixed random
        projections), consistent with the "partial freezing" already
        described for the online update in feedback.py -- pretraining fits
        the profile-facing readout to real structure, online learning then
        fine-tunes it further.
        """
        z2, index = self.propagate(hg)
        service_ids = list(hg.services.keys())
        edge_membership: Dict[str, set] = {s: set() for s in service_ids}
        positives = []
        for edge in hg.hyperedges.values():
            members = [m for m in edge.members if m in hg.services]
            for i in range(len(members)):
                for k in range(i + 1, len(members)):
                    positives.append((members[i], members[k]))
                    edge_membership[members[i]].add(members[k])
                    edge_membership[members[k]].add(members[i])
        if not positives:
            return

        rng = np.random.default_rng(seed)
        zero_profile = np.zeros(self.profile_width)  # no profile signal at pretrain time

        def sample_negative(u: str, tries: int = 8) -> str:
            for _ in range(tries):
                v = service_ids[rng.integers(len(service_ids))]
                if v != u and v not in edge_membership[u]:
                    return v
            return v

        for _ in range(epochs):
            batch_idx = rng.choice(len(positives), size=min(batch_size, len(positives)), replace=False)
            grad = np.zeros_like(self.w_out)
            for idx in batch_idx:
                u, v = positives[idx]
                grad += self._pretrain_gradient(z2, index, u, v, zero_profile, label=1.0)
                neg_v = sample_negative(u)
                grad += self._pretrain_gradient(z2, index, u, neg_v, zero_profile, label=0.0)
            self.w_out -= lr * grad / (2 * len(batch_idx))

    def _pretrain_gradient(self, z2: np.ndarray, index: Dict[str, int], u: str, v: str,
                            zero_profile: np.ndarray, label: float) -> np.ndarray:
        """One sigmoid+BCE gradient step of Eq. pretrain's link-prediction loss w.r.t. w_out."""
        pooled = np.mean([z2[index[u]], z2[index[v]]], axis=0)
        vec = np.concatenate([pooled, zero_profile])
        pred = 1.0 / (1.0 + np.exp(-float(vec @ self.w_out)))
        return (pred - label) * vec


# --------------------------------------------------------------------------
# Algorithm 2
# --------------------------------------------------------------------------
@dataclass
class RankedCombo:
    fragments: Tuple[Fragment, ...]
    score: float


def filter_and_rank(hg: Hypergraph, candidate_fragments: Dict[str, List[Tuple[Fragment, float]]],
                     preference_vector: np.ndarray, repository: Repository, scorer: HypergraphScorer,
                     params: dict) -> Tuple[List[RankedCombo], list]:
    """Algorithm 2, lines 1-21."""
    alpha = params["filtering"]["fusion_weight_alpha"]
    nu = params["filtering"]["max_fragments_per_combination"]
    l_max = params["filtering"]["max_relaxation_rounds"]
    pen_prereq = params["filtering"]["penalty_weight_prereq"]
    pen_coexec = params["filtering"]["penalty_weight_co_exec"]

    mediator = MediatorAgent()
    log: list = []
    ranked: List[RankedCombo] = []
    working = {leaf: [f for f, _ in items] for leaf, items in candidate_fragments.items()}
    round_idx = 0
    z2, index = scorer.propagate(hg)  # one dense propagation per request, reused below

    while True:
        violations = []
        for leaf, frags in working.items():
            for size in range(1, min(nu, len(frags)) + 1):
                for combo in combinations(frags, size):
                    sigma_sym, reason = check_constraints(hg, list(combo), pen_prereq, pen_coexec)
                    if sigma_sym == 0.0:
                        violations.append((combo, reason))
                        continue
                    sigma_nn = scorer.score(z2, index, list(combo), preference_vector)
                    sigma = alpha * sigma_nn + (1 - alpha) * sigma_sym
                    ranked.append(RankedCombo(fragments=combo, score=sigma))

        if ranked or round_idx >= l_max:
            break

        suggestions = mediator.mediate(violations, repository, None)
        if not suggestions:
            break
        for leaf in working:
            working[leaf] = list({f.id: f for f in working[leaf] + suggestions}.values())
        log.append({"round": round_idx, "added": [f.id for f in suggestions]})
        round_idx += 1
        z2, index = scorer.propagate(hg)  # hyperedge confidences may have shifted after mediation

    ranked.sort(key=lambda rc: rc.score, reverse=True)
    return ranked, log
