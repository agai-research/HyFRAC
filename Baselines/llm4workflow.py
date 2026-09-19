"""
llm4workflow.py -- LLM4Workflow (Xu et al.), a Rewrite-Retrieve-Read pipeline.

Stage 1: one-time API knowledge base (documents + TF-IDF vector store).
Stage 2: task extraction and query rewriting (k+n optimised queries).
Stage 3: Maximum Marginal Relevance retrieval over the vector store.
Stage 4: one-shot workflow DAG generation matching each task to one API.

"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HyFRAC"))
from hypergraph import Hypergraph, Service  # noqa: E402

STOPWORDS = {"the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "this"}


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]


def _service_document(service: Service) -> str:
    """The service documentation an API knowledge base would hold (name, description, params, output)."""
    return (f"{service.id} provides {service.capability}. "
            f"parameters: resources {' '.join(service.resources)}. "
            f"preconditions {' '.join(service.pre)}. postconditions {' '.join(service.post)}. "
            f"cost {service.cost:.1f} duration {service.duration:.1f} complexity {service.complexity}.")


class VectorStore:
    """A TF-IDF vector store: real document retrieval, no external embedding API required."""

    def __init__(self, services: Dict[str, Service]):
        self.service_ids = list(services.keys())
        docs = [_tokenize(_service_document(services[s])) for s in self.service_ids]
        vocab = sorted({w for doc in docs for w in doc})
        self.vocab_index = {w: i for i, w in enumerate(vocab)}

        doc_freq = np.zeros(len(vocab))
        term_matrix = np.zeros((len(docs), len(vocab)))
        for row, doc in enumerate(docs):
            for w in doc:
                term_matrix[row, self.vocab_index[w]] += 1
            for w in set(doc):
                doc_freq[self.vocab_index[w]] += 1

        idf = np.log((len(docs) + 1) / (doc_freq + 1)) + 1
        self.matrix = term_matrix * idf  # TF-IDF
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.matrix = self.matrix / norms
        self.idf = idf

    def _embed_query(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab_index))
        for w in _tokenize(query):
            if w in self.vocab_index:
                vec[self.vocab_index[w]] += 1
        vec *= self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def mmr_search(self, query: str, top_k: int, lambda_diversity: float = 0.6) -> List[Tuple[str, float]]:
        """Maximum Marginal Relevance: balances relevance to the query against diversity among results."""
        q_vec = self._embed_query(query)
        relevance = self.matrix @ q_vec
        selected: List[int] = []
        candidates = list(np.argsort(-relevance)[: max(top_k * 4, top_k)])

        while candidates and len(selected) < top_k:
            best_idx, best_score = None, -np.inf
            for idx in candidates:
                diversity_penalty = max((self.matrix[idx] @ self.matrix[s] for s in selected), default=0.0)
                mmr_score = lambda_diversity * relevance[idx] - (1 - lambda_diversity) * diversity_penalty
                if mmr_score > best_score:
                    best_idx, best_score = idx, mmr_score
            selected.append(best_idx)
            candidates.remove(best_idx)

        return [(self.service_ids[i], float(relevance[i])) for i in selected]


@dataclass
class WorkflowNode:
    task: str
    service_id: str


def extract_tasks(query: str) -> List[str]:
    """Stage 2a: task extraction -- one atomic action per detected objective keyword."""
    from agents import CAPABILITY_KEYWORDS
    text = query.lower()
    tasks = [cap for cap, kws in CAPABILITY_KEYWORDS.items() if any(k in text for k in kws)]
    return tasks or [list(CAPABILITY_KEYWORDS.keys())[0]]


def rewrite_queries(tasks: List[str], n_expansions: int) -> List[str]:
    """Stage 2b: k+n optimised queries -- k direct task queries, n paraphrased expansions."""
    queries = list(tasks)
    for i in range(n_expansions):
        base = tasks[i % len(tasks)]
        queries.append(f"service that performs {base} reliably")
    return queries


def run_llm4workflow(hypergraph_path: str, query: str, k_expansions: int = 2,
                      top_k: int = 9) -> dict:
    """The full four-stage pipeline, on a matched service registry."""
    hg = Hypergraph.from_json(hypergraph_path)
    store = VectorStore(hg.services)  # Stage 1

    tasks = extract_tasks(query)  # Stage 2a
    queries = rewrite_queries(tasks, k_expansions)  # Stage 2b

    retrieved: Dict[str, List[Tuple[str, float]]] = {}  # Stage 3
    for q in queries:
        retrieved[q] = store.mmr_search(q, top_k=top_k)

    # Stage 4: match each task to its single best retrieved API, in task order
    workflow: List[WorkflowNode] = []
    used_services = set()
    for task in tasks:
        best_service, best_score = None, -np.inf
        for q, results in retrieved.items():
            for s_id, score in results:
                if s_id in used_services:
                    continue
                if hg.services[s_id].capability == task and score > best_score:
                    best_service, best_score = s_id, score
        if best_service is not None:
            workflow.append(WorkflowNode(task=task, service_id=best_service))
            used_services.add(best_service)

    if not workflow:
        return {"status": "clarification", "message": "no API matched any extracted task",
                "services": [], "confidence": 0.0}

    edges = [(workflow[i].service_id, workflow[i + 1].service_id) for i in range(len(workflow) - 1)]
    total_cost = sum(hg.services[n.service_id].cost for n in workflow)
    total_duration = sum(hg.services[n.service_id].duration for n in workflow)
    reliability = float(np.prod([hg.services[n.service_id].reliability for n in workflow]))
    complexity = max(hg.services[n.service_id].complexity for n in workflow)

    return {
        "status": "delivered",
        "services": [n.service_id for n in workflow],
        "tasks": [n.task for n in workflow],
        "dag_edges": edges,
        "quality": {"cost": total_cost, "duration": total_duration,
                    "reliability": reliability, "complexity": complexity},
        "confidence": reliability,  # LLM4Workflow reports no native confidence signal
    }
