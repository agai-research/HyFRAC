"""
hyfrac_ablations.py -- HyFRAC-BG, HyFRAC-NS, HyFRAC-NF, HyFRAC-WfG.

Each ablation reuses HyFRAC's own modules and swaps out exactly one
component, matching sec:compared-methods' own definitions. None of them
duplicate HyFRAC's logic; they call into it directly so the rest of the
pipeline is provably unchanged.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HyFRAC"))

from hypergraph import Hypergraph  # noqa: E402
from fragments import Repository, Fragment  # noqa: E402
from goal_tree import Ontology, GoalTree  # noqa: E402
from agents import Backbone, DecompositionAgent  # noqa: E402
from retrieval import retrieve, bias  # noqa: E402
from neurosymbolic import HypergraphScorer, filter_and_rank  # noqa: E402
from optimisation import assemble  # noqa: E402
from trace import CausalTrace, CausalGraph  # noqa: E402
from feedback import verbalise_explanation  # noqa: E402


# --------------------------------------------------------------------------
# HyFRAC-NS: neural-only ranking, no pre-generation symbolic check
# --------------------------------------------------------------------------
def run_hyfrac_ns(hypergraph_path: str, ontology_path: str, params: dict, query: str) -> dict:
    """alpha=1.0, N=8 (per tab:baselines): identical pipeline, different fusion weight and repair budget."""
    ns_params = _deep_copy(params)
    ns_params["filtering"]["fusion_weight_alpha"] = 1.0
    ns_params["assembly"]["max_repair_iterations"] = 8
    return _run_full_pipeline(hypergraph_path, ontology_path, ns_params, query)


# --------------------------------------------------------------------------
# HyFRAC-WfG: workflow inference removed, an oracle goal tree is supplied
# --------------------------------------------------------------------------
def run_hyfrac_wfg(hypergraph_path: str, ontology_path: str, params: dict, query: str,
                    oracle_tree: GoalTree) -> dict:
    """Skips Algorithm 1's decomposition stage; everything downstream is unchanged."""
    hg = Hypergraph.from_json(hypergraph_path)
    repository = Repository()
    repository.build(hg)
    trace = CausalTrace()

    candidates: Dict[str, List[Tuple[Fragment, float]]] = {}
    hg.build_community_summaries()
    for leaf in oracle_tree.leaves:
        anchors = {s for s, svc in hg.services.items() if svc.capability == leaf.capability}
        scored = {}
        for vertex in anchors:
            for edge in hg.edges_containing(vertex):
                frag = repository.fragments.get(f"phi_{edge.id}")
                if frag is not None and leaf.capability in frag.capabilities:
                    score = edge.confidence * bias(hg, vertex, leaf.capability,
                                                    leaf.target_complexity or 1,
                                                    params["retrieval"]["complexity_band_width"],
                                                    edge.confidence)
                    if frag.id not in scored or score > scored[frag.id][1]:
                        scored[frag.id] = (frag, score)
        top_k = sorted(scored.values(), key=lambda p: p[1], reverse=True)[:params["retrieval"]["breadth_k"]]
        candidates[leaf.capability] = top_k

    preference_vector = np.array([0.25, 0.25, 0.25, 0.25])
    scorer = HypergraphScorer(params["hgnn"]["hidden_width"], params["hgnn"]["feature_width"],
                               params["hgnn"]["capability_embedding_width"])
    scorer.pretrain(hg, epochs=params["hgnn"]["pretrain_epochs"], lr=params["hgnn"]["pretrain_lr"])
    ranked, relaxation_log = filter_and_rank(hg, candidates, preference_vector, repository, scorer, params)
    return _finish_pipeline(hg, oracle_tree, ranked, relaxation_log, preference_vector, params, trace, 1, scorer)


# --------------------------------------------------------------------------
# HyFRAC-NF: fragment abstraction removed, retrieval and assembly operate
# directly on individual services
# --------------------------------------------------------------------------
def run_hyfrac_nf(hypergraph_path: str, ontology_path: str, params: dict, query: str) -> dict:
    """Compatibility is re-established at request time via singleton pseudo-fragments,
    never read from a stored, pre-validated unit."""
    hg = Hypergraph.from_json(hypergraph_path)
    ontology = Ontology.from_json(ontology_path)
    backbone = Backbone(params["backbone"])
    decomposer = DecompositionAgent(ontology, backbone)
    trace = CausalTrace()

    parsed = decomposer.parse(query)
    tree = decomposer.build_goal_tree(parsed)

    # a singleton, request-time "fragment" per matching service: no offline
    # validation, no stored joint-compatibility, rebuilt from scratch every call
    candidates: Dict[str, List[Tuple[Fragment, float]]] = {}
    for leaf in tree.leaves:
        scored = []
        for s_id, svc in hg.services.items():
            if svc.capability != leaf.capability:
                continue
            pseudo = Fragment(id=f"svc_{s_id}", source_edge="", services={s_id}, order=[],
                               pre=set(svc.pre), post=set(svc.post), resources=set(svc.resources),
                               quality=(svc.cost, svc.duration, svc.reliability, svc.complexity),
                               capabilities={svc.capability}, well_formed=True)
            score = bias(hg, s_id, leaf.capability, leaf.target_complexity or parsed.level,
                         params["retrieval"]["complexity_band_width"], 1.0)
            scored.append((pseudo, score))
        top_k = sorted(scored, key=lambda p: p[1], reverse=True)[:params["retrieval"]["breadth_k"]]
        candidates[leaf.capability] = top_k

    preference_vector = np.array(list(parsed.preferences.values()))
    scorer = HypergraphScorer(params["hgnn"]["hidden_width"], params["hgnn"]["feature_width"],
                               params["hgnn"]["capability_embedding_width"])
    scorer.pretrain(hg, epochs=params["hgnn"]["pretrain_epochs"], lr=params["hgnn"]["pretrain_lr"])
    repository = Repository()  # empty: NF never reads from a stored repository
    ranked, relaxation_log = filter_and_rank(hg, candidates, preference_vector, repository, scorer, params)
    return _finish_pipeline(hg, tree, ranked, relaxation_log, preference_vector, params, trace, parsed.level, scorer)


# --------------------------------------------------------------------------
# HyFRAC-BG: the hypergraph is replaced by its clique expansion before
# retrieval and filtering; the fragment catalogue itself is untouched
# --------------------------------------------------------------------------
def run_hyfrac_bg(hypergraph_path: str, ontology_path: str, params: dict, query: str) -> dict:
    """
    Retrieval walks the pairwise clique-expansion graph instead of the
    hypergraph (no hyperedge-community pruning, since a community is
    defined over hyperedge co-membership); the fragments a walk step
    surfaces are the same pre-built, n-ary-derived Fragment objects as
    HyFRAC's own repository -- only how they are found differs.
    """
    hg = Hypergraph.from_json(hypergraph_path)
    ontology = Ontology.from_json(ontology_path)
    repository = Repository()
    repository.build(hg)
    backbone = Backbone(params["backbone"])
    decomposer = DecompositionAgent(ontology, backbone)
    trace = CausalTrace()

    parsed = decomposer.parse(query)
    tree = decomposer.build_goal_tree(parsed)

    pairwise = hg.clique_expansion()
    graph = nx.Graph()
    for (u, v), w in pairwise.items():
        graph.add_edge(u, v, weight=w)

    candidates: Dict[str, List[Tuple[Fragment, float]]] = {}
    d_w, k = params["retrieval"]["walk_depth"], params["retrieval"]["breadth_k"]
    for leaf in tree.leaves:
        anchors = [s for s, svc in hg.services.items() if svc.capability == leaf.capability]
        visited = set(anchors)
        frontier = set(anchors)
        scored: Dict[str, Tuple[Fragment, float]] = {}
        for _ in range(d_w):
            next_frontier = set()
            for vertex in frontier:
                if vertex not in graph:
                    continue
                for neighbour, edata in graph[vertex].items():
                    if neighbour in visited:
                        continue
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
                    for edge in hg.edges_containing(neighbour):
                        frag = repository.fragments.get(f"phi_{edge.id}")
                        if frag is not None and leaf.capability in frag.capabilities:
                            score = edata["weight"]
                            if frag.id not in scored or score > scored[frag.id][1]:
                                scored[frag.id] = (frag, score)
            frontier = next_frontier
        top_k = sorted(scored.values(), key=lambda p: p[1], reverse=True)[:k]
        candidates[leaf.capability] = top_k

    preference_vector = np.array(list(parsed.preferences.values()))
    scorer = HypergraphScorer(params["hgnn"]["hidden_width"], params["hgnn"]["feature_width"],
                               params["hgnn"]["capability_embedding_width"])
    scorer.pretrain(hg, epochs=params["hgnn"]["pretrain_epochs"], lr=params["hgnn"]["pretrain_lr"])
    ranked, relaxation_log = filter_and_rank(hg, candidates, preference_vector, repository, scorer, params)
    return _finish_pipeline(hg, tree, ranked, relaxation_log, preference_vector, params, trace, parsed.level, scorer)


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------
def _deep_copy(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)


def _run_full_pipeline(hypergraph_path: str, ontology_path: str, params: dict, query: str) -> dict:
    """Runs HyFRAC's own unmodified retrieve -> filter_and_rank -> assemble chain."""
    hg = Hypergraph.from_json(hypergraph_path)
    ontology = Ontology.from_json(ontology_path)
    repository = Repository()
    repository.build(hg)
    backbone = Backbone(params["backbone"])
    decomposer = DecompositionAgent(ontology, backbone)
    trace = CausalTrace()

    result = retrieve(query, ontology, hg, repository, decomposer, params)
    preference_vector = np.array(list(result.parsed.preferences.values()))
    scorer = HypergraphScorer(params["hgnn"]["hidden_width"], params["hgnn"]["feature_width"],
                               params["hgnn"]["capability_embedding_width"])
    scorer.pretrain(hg, epochs=params["hgnn"]["pretrain_epochs"], lr=params["hgnn"]["pretrain_lr"])
    ranked, relaxation_log = filter_and_rank(hg, result.candidate_fragments, preference_vector,
                                              repository, scorer, params)
    return _finish_pipeline(hg, result.goal_tree, ranked, relaxation_log, preference_vector,
                             params, trace, result.parsed.level, scorer)


def _finish_pipeline(hg, tree, ranked, relaxation_log, preference_vector, params, trace, level,
                      scorer: HypergraphScorer) -> dict:
    """Shared Algorithm 3 + explanation tail, identical across every ablation.
    scorer must be the SAME instance filter_and_rank used, so the sigma_nn
    values looked up here are consistent with the ranking already produced."""
    if not ranked:
        return {"status": "clarification", "message": "no compatible fragment set was found",
                "confidence": 0.0}

    sigma_nn_lookup = {}
    z2, index = scorer.propagate(hg)
    for rc in ranked[: params["retrieval"]["breadth_k"]]:
        for f in rc.fragments:
            sigma_nn_lookup[f.id] = scorer.score(z2, index, [f], preference_vector)

    application, clarification_reason, conf = assemble(
        ranked, tree, hg.services, preference_vector, params, sigma_nn_lookup, trace,
    )
    if application is None:
        return {"status": "clarification", "message": clarification_reason, "confidence": conf}

    causal_graph = CausalGraph.build(trace, relaxation_log)
    explanation = verbalise_explanation(
        causal_graph, application, register="advanced" if level >= 3 else "student",
        min_coverage=params["assembly"]["min_faithfulness_coverage"],
        max_retries=params["feedback"]["max_explanation_retries"],
    )

    return {
        "status": "clarification" if clarification_reason else "delivered",
        "message": clarification_reason,
        "confidence": conf,
        "services": sorted(application.services.keys()),
        "fragments_used": sorted(application.fragment_ids),
        "quality": {"cost": application.quality[0], "duration": application.quality[1],
                    "reliability": application.quality[2], "complexity": application.quality[3]},
        "explanation": explanation,
        "relaxation_rounds": len(relaxation_log),
    }
