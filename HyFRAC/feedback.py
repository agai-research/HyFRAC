"""
feedback.py -- Algorithm 4: explanation verbalisation and
continuous improvement (online scorer update, episodic memory, profile and
hyperedge-weight revision).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np

from trace import CausalGraph
from agents import ExplanationAgent
from neurosymbolic import HypergraphScorer
from hypergraph import Hypergraph


@dataclass
class Episode:
    fragment_ids: List[str]
    reward: float
    profile_level: int


class EpisodicMemory:
    """M: past applications, retrievable as a warm-start template."""

    def __init__(self):
        self.episodes: List[Episode] = []

    def retrieve_similar(self, goal_capabilities: set) -> "Episode | None":
        if not self.episodes:
            return None
        return self.episodes[-1]  # most recent episode as a simple similarity stand-in

    def add_episode(self, fragment_ids: List[str], reward: float, level: int) -> None:
        self.episodes.append(Episode(fragment_ids=fragment_ids, reward=reward, profile_level=level))


def verbalise_explanation(causal_graph: CausalGraph, application, register: str,
                           min_coverage: float, max_retries: int) -> str:
    """Lines 8-15: retries verbalisation until citation coverage reaches mu, else falls back."""
    agent = ExplanationAgent()
    causal_graph.prune_unreachable()
    for _ in range(max_retries):
        text = agent.verbalise(causal_graph, application, register)
        cited = sum(1 for n in causal_graph.nodes if n.summary.split(":")[0] in text)
        coverage = cited / len(causal_graph.nodes) if causal_graph.nodes else 1.0
        if coverage >= min_coverage:
            return text
    return _template_render(causal_graph)  # deterministic fallback


def _template_render(causal_graph: CausalGraph) -> str:
    lines = ["Deterministic explanation (fallback rendering):"]
    for node in causal_graph.nodes:
        lines.append(f"- [{node.kind}] {node.summary}")
    return "\n".join(lines)


def update_scorer(scorer: HypergraphScorer, hg: Hypergraph, fragment_ids: List[str],
                   reward: float, params: dict, replay_batch: List[dict]) -> None:
    """
    Lines 16-20: one policy-gradient-style step, restricted per the paper's
    three safeguards -- theta1 stays frozen, a small learning rate, and a
    replay term computed on a batch of pretraining hyperedges.
    """
    eta = params["feedback"]["learning_rate_eta"]
    xi = params["feedback"]["gradient_clip_xi"]
    gamma_r = params["feedback"]["replay_weight_gamma_r"]

    z2, index = scorer.propagate(hg)  # one propagation reused across every finite-difference probe
    fake = _FakeFrag(fragment_ids, hg)
    pref = np.array([0.25, 0.25, 0.25, 0.25])

    # finite-difference gradient estimate on w_out only (theta1, theta2 stay frozen online,
    # matching "partial freezing" -- only the profile-facing readout adapts per request)
    epsilon = 1e-3
    base_score = scorer.score(z2, index, [fake], pref)
    grad = np.zeros_like(scorer.w_out)
    for i in range(len(scorer.w_out)):
        scorer.w_out[i] += epsilon
        bumped = scorer.score(z2, index, [fake], pref)
        scorer.w_out[i] -= epsilon
        grad[i] = (bumped - base_score) / epsilon

    replay_grad = np.zeros_like(grad) if not replay_batch else grad * 0.1  # small stand-in replay term
    step = eta * (reward * grad + gamma_r * replay_grad)
    norm = np.linalg.norm(step)
    if norm > xi:
        step = step * (xi / norm)
    scorer.w_out += step


class _FakeFrag:
    """Adapter so update_scorer can reuse HypergraphScorer.score without importing Fragment here."""
    def __init__(self, fragment_ids: List[str], hg: Hypergraph):
        self.id = "replay"
        self.services = set()
        for fid in fragment_ids:
            edge_id = fid.replace("phi_", "")
            edge = hg.hyperedges.get(edge_id)
            if edge:
                self.services |= (edge.members & set(hg.services))


def update_hyperedge_weights(hg: Hypergraph, fragment_ids: List[str], reward: float,
                              eta: float, min_weight: float) -> List[str]:
    """Lines 24-29: smooth omega(e) towards the observed reward; flag weak hyperedges."""
    flagged = []
    for fid in fragment_ids:
        edge_id = fid.replace("phi_", "")
        edge = hg.hyperedges.get(edge_id)
        if edge is None:
            continue
        edge.confidence = (1 - eta) * edge.confidence + eta * reward
        if edge.confidence < min_weight:
            flagged.append(edge_id)
    return flagged
