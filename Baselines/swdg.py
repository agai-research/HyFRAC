"""
swdg.py -- SWDG (Zhu et al.): generates a service workflow graph from a
single, fully-specified process description. Unlike every other compared
method, SWDG assumes no service catalogue and performs no selection among
alternatives -- the description itself names the activities; the task is
to recover the graph structure connecting them, not to choose services.
"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HyFRAC"))
from hypergraph import Hypergraph  # noqa: E402


@dataclass
class Node:
    id: str
    kind: str          # "activity" | "condition"
    text: str


def describe_process(objectives: List[str], hg: Hypergraph) -> str:
    """
    Builds a fully-specified process description from a profile's
    objectives, in the narrative style SWDG's own bank-lending example
    uses, since SWDG's input is a description text, not a short request.
    """
    sentences = []
    for i, cap in enumerate(objectives):
        pool = [s for s, svc in hg.services.items() if svc.capability == cap]
        if not pool:
            continue
        connector = "First," if i == 0 else ("Then," if i < len(objectives) - 1 else "Finally,")
        sentences.append(f"{connector} the system handles {cap} using the available resources. "
                          f"If the {cap} step requires more than one attempt, it must be repeated.")
    return " ".join(sentences)


def extract_nodes(description: str) -> List[Node]:
    """LLM-based extraction, stand-in: one activity node per sentence, one condition node per 'if' clause."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", description) if s.strip()]
    nodes: List[Node] = []
    for i, sent in enumerate(sentences):
        if sent.lower().startswith("if") or " if " in sent.lower():
            nodes.append(Node(id=f"c{i}", kind="condition", text=sent))
        else:
            nodes.append(Node(id=f"a{i}", kind="activity", text=sent))
    return nodes


def chain_connections(nodes: List[Node]) -> List[Tuple[str, str]]:
    """The order-of-appearance chain that guarantees initial-graph connectivity."""
    return [(nodes[i].id, nodes[i + 1].id) for i in range(len(nodes) - 1)]


def auxiliary_connections(nodes: List[Node]) -> List[Tuple[str, str]]:
    """
    LLM-proposed auxiliary edges, stand-in: a condition node is linked back
    to the nearest preceding activity it references (by shared keyword),
    matching how SWDG's own adjacency-table step links conditions to the
    activities they gate.
    """
    edges = []
    for i, node in enumerate(nodes):
        if node.kind != "condition":
            continue
        node_words = set(re.findall(r"[a-z]+", node.text.lower()))
        for j in range(i - 1, -1, -1):
            prior_words = set(re.findall(r"[a-z]+", nodes[j].text.lower()))
            if node_words & prior_words:
                edges.append((nodes[j].id, node.id))
                break
    return edges


class BiLSTMAggregator:
    """
    Eq. 1 of the SWDG paper: forward and backward LSTM passes over a node's
    neighbours, in ascending node-id order (the paper's own choice, since
    service workflows emphasise sequential order). A minimal, real LSTM
    cell -- not an averaging stand-in -- implemented in NumPy.
    """

    def __init__(self, input_width: int, hidden_width: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        gate_width = 4 * hidden_width
        self.w_f = rng.normal(0, 0.1, size=(input_width + hidden_width, gate_width))
        self.b_f = np.zeros(gate_width)
        self.w_b = rng.normal(0, 0.1, size=(input_width + hidden_width, gate_width))
        self.b_b = np.zeros(gate_width)
        self.hidden_width = hidden_width

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _run_direction(self, sequence: List[np.ndarray], w: np.ndarray, b: np.ndarray) -> np.ndarray:
        h = np.zeros(self.hidden_width)
        c = np.zeros(self.hidden_width)
        for x in sequence:
            gates = np.concatenate([x, h]) @ w + b
            i, f, g, o = np.split(gates, 4)
            i, f, o = self._sigmoid(i), self._sigmoid(f), self._sigmoid(o)
            g = np.tanh(g)
            c = f * c + i * g
            h = o * np.tanh(c)
        return h

    def aggregate(self, neighbour_features: List[np.ndarray]) -> np.ndarray:
        if not neighbour_features:
            return np.zeros(2 * self.hidden_width)
        forward = self._run_direction(neighbour_features, self.w_f, self.b_f)
        backward = self._run_direction(list(reversed(neighbour_features)), self.w_b, self.b_b)
        return np.concatenate([forward, backward])


class InductiveGNN:
    """Two-layer Bi-LSTM-aggregator GNN over the initial graph, sigmoid edge-existence readout."""

    def __init__(self, feature_width: int, hidden_width: int, seed: int = 0):
        self.layer1 = BiLSTMAggregator(feature_width, hidden_width, seed)
        self.layer2 = BiLSTMAggregator(2 * hidden_width, hidden_width, seed + 1)
        rng = np.random.default_rng(seed + 2)
        self.w_out = rng.normal(0, 0.1, size=4 * hidden_width)
        self.feature_width = feature_width

    @staticmethod
    def _leaky_relu(x, slope=0.01):
        return np.where(x > 0, x, slope * x)

    def _node_feature(self, node: Node) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(node.kind + node.text[:10])) % (2**32))
        vec = rng.normal(0, 0.3, size=self.feature_width)
        vec[0] = 1.0 if node.kind == "activity" else 0.0
        return vec

    def predict_edges(self, nodes: List[Node], existing_edges: List[Tuple[str, str]]
                       ) -> List[Tuple[str, str, float]]:
        """Runs the two aggregation layers, then scores every non-existing pair for a missing edge."""
        by_id = {n.id: n for n in nodes}
        adjacency: Dict[str, List[str]] = {n.id: [] for n in nodes}
        for a, b in existing_edges:
            adjacency[a].append(b)

        h0 = {n.id: self._node_feature(n) for n in nodes}
        ordered_ids = sorted(by_id.keys())  # ascending node-id order, per the paper

        h1 = {nid: self._leaky_relu(self.layer1.aggregate([h0[m] for m in adjacency[nid]]))
              for nid in ordered_ids}
        h2 = {nid: self._leaky_relu(self.layer2.aggregate([h1[m] for m in adjacency[nid]]))
              for nid in ordered_ids}

        predictions = []
        existing_set = set(existing_edges)
        for i, a in enumerate(ordered_ids):
            for b in ordered_ids[i + 1:]:
                if (a, b) in existing_set or (b, a) in existing_set:
                    continue
                concat = np.concatenate([h2[a], h2[b]])
                logit = float(concat @ self.w_out)
                prob = 1.0 / (1.0 + np.exp(-logit))
                predictions.append((a, b, prob))
        return predictions


def run_swdg(hypergraph_path: str, objectives: List[str], edge_threshold: float = 0.6) -> dict:
    hg = Hypergraph.from_json(hypergraph_path)
    description = describe_process(objectives, hg)
    if not description:
        return {"status": "clarification", "message": "no matching activity found for the objectives",
                "confidence": 0.0}

    nodes = extract_nodes(description)
    chain = chain_connections(nodes)
    auxiliary = auxiliary_connections(nodes)
    initial_edges = list(set(chain) | set(auxiliary))

    gnn = InductiveGNN(feature_width=8, hidden_width=8)
    predicted = gnn.predict_edges(nodes, initial_edges)
    completed_edges = initial_edges + [(a, b) for a, b, p in predicted if p >= edge_threshold]

    return {
        "status": "delivered",
        "process_description": description,
        "nodes": [{"id": n.id, "kind": n.kind, "text": n.text} for n in nodes],
        "initial_edges": initial_edges,
        "completed_edges": completed_edges,
        "num_predicted_edges_added": len(completed_edges) - len(initial_edges),
        "confidence": float(np.mean([p for _, _, p in predicted])) if predicted else 1.0,
    }
