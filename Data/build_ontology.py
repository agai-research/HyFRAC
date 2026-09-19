"""
build_ontology.py -- derives the domain ontology (prerequisites, alternatives,
predicates) directly from a constructed hypergraph's own prereq hyperedges,
so decomposition is exercised against relations that actually exist in the
corpus rather than a hand-authored, possibly mismatched, table.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HyFRAC"))
from hypergraph import Hypergraph  # noqa: E402


def build_ontology_from_corpus(hg: Hypergraph) -> Dict[str, List[str]]:
    """
    Reads prereq hyperedges into a prerequisite table. With 390 prereq
    hyperedges sampled over ordered pairs of 25 education capabilities
    (roughly two thirds of all 600 possible ordered pairs), promoting every
    sampled pair into a permanent capability-level rule produces a near-
    total prerequisite order: a single request can then transitively
    require dozens of capabilities, none of which the request actually
    asked for. A realistic course prerequisite structure is sparse -- a
    topic has at most one or two direct prerequisites, not most of the
    curriculum -- so only each head capability's single most-frequently
    sampled tail is kept, and only if doing so keeps the structure acyclic.
    """
    import networkx as nx
    from collections import Counter

    tail_counts: Dict[str, Counter] = {}
    for edge in hg.hyperedges.values():
        if edge.edge_type != "prereq":
            continue
        tail_caps = {hg.services[s].capability for s in edge.tail if s in hg.services}
        head_caps = {hg.services[s].capability for s in edge.head if s in hg.services}
        for head_cap in head_caps:
            counter = tail_counts.setdefault(head_cap, Counter())
            for tail_cap in tail_caps:
                if tail_cap != head_cap:
                    counter[tail_cap] += 1

    prerequisites: Dict[str, List[str]] = {}
    predicates: Dict[str, List[str]] = {}
    acyclic_check = nx.DiGraph()

    for head_cap, counter in tail_counts.items():
        for tail_cap, _ in counter.most_common():  # try the most-sampled tail first
            acyclic_check.add_edge(head_cap, tail_cap)
            if nx.is_directed_acyclic_graph(acyclic_check):
                prerequisites[head_cap] = [tail_cap]
                predicates[head_cap] = [f"{tail_cap}_known"]
                break
            acyclic_check.remove_edge(head_cap, tail_cap)

    return {
        "prerequisites": prerequisites,
        "alternatives": {},  # cap-cover already handles substitutability at the fragment level
        "predicates": predicates,
    }


def main() -> None:
    corpus_dir = Path(__file__).resolve().parent / "corpus"
    hg = Hypergraph.from_json(str(corpus_dir / "master_hypergraph.json"))
    ontology = build_ontology_from_corpus(hg)
    out_path = corpus_dir / "master_ontology.json"
    out_path.write_text(json.dumps(ontology, indent=2))
    print(f"built ontology with {len(ontology['prerequisites'])} prerequisite entries -> {out_path}")


if __name__ == "__main__":
    main()
