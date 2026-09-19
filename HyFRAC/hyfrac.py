"""
hyfrac.py -- main entry point. Implements Psi: (P, H, Phi, L, M) -> (W*, E, L', M'),
as stated in sec:formulation, by running Algorithms 1 -> 2 -> 3 -> 4 in sequence.

Run as:
    python hyfrac.py --hypergraph <path.json> --ontology <path.json> --query "..."
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from hypergraph import Hypergraph
from fragments import Repository
from goal_tree import Ontology
from agents import Backbone, DecompositionAgent
from retrieval import retrieve
from neurosymbolic import HypergraphScorer, filter_and_rank
from optimisation import assemble
from trace import CausalTrace, CausalGraph
from feedback import EpisodicMemory, verbalise_explanation, update_scorer, update_hyperedge_weights


def load_params(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_hyfrac(hypergraph_path: str, ontology_path: str, params_path: str, query: str,
               memory: EpisodicMemory | None = None) -> dict:
    """One full pass of Psi for a single request. Returns a JSON-serialisable result."""
    params = load_params(params_path)
    hg = Hypergraph.from_json(hypergraph_path)
    ontology = Ontology.from_json(ontology_path)
    repository = Repository()
    repository.build(hg)
    memory = memory or EpisodicMemory()

    backbone = Backbone(params["backbone"])
    decomposer = DecompositionAgent(ontology, backbone)
    trace = CausalTrace()

    # Algorithm 1
    retrieval_result = retrieve(query, ontology, hg, repository, decomposer, params)
    trace.log("plan", {"services": []})  # marks the start of the retrieved-candidate stage

    # Algorithm 2
    preference_vector = np.array(list(retrieval_result.parsed.preferences.values()))
    scorer = HypergraphScorer(
        hidden_width=params["hgnn"]["hidden_width"],
        feature_width=params["hgnn"]["feature_width"],
        cap_embed_width=params["hgnn"]["capability_embedding_width"],
    )
    scorer.pretrain(hg, epochs=params["hgnn"]["pretrain_epochs"], lr=params["hgnn"]["pretrain_lr"])
    ranked, relaxation_log = filter_and_rank(hg, retrieval_result.candidate_fragments,
                                              preference_vector, repository, scorer, params)

    if not ranked:
        return {"status": "clarification", "message": "no compatible fragment set was found",
                "goal_tree_leaves": [leaf.capability for leaf in retrieval_result.goal_tree.leaves]}

    sigma_nn_lookup = {}
    z2, index = scorer.propagate(hg)
    for rc in ranked[: params["retrieval"]["breadth_k"]]:
        for f in rc.fragments:
            sigma_nn_lookup[f.id] = scorer.score(z2, index, [f], preference_vector)

    # Algorithm 3
    application, clarification_reason, conf = assemble(
        ranked, retrieval_result.goal_tree, hg.services, preference_vector, params,
        sigma_nn_lookup, trace,
    )

    if application is None:
        return {"status": "clarification", "message": clarification_reason, "confidence": conf}

    # Algorithm 4 (explanation half; the online-learning half runs after real execution feedback)
    causal_graph = CausalGraph.build(trace, relaxation_log)
    explanation = verbalise_explanation(
        causal_graph, application, register="student" if retrieval_result.parsed.level <= 1 else "advanced",
        min_coverage=params["assembly"]["min_faithfulness_coverage"],
        max_retries=params["feedback"]["max_explanation_retries"],
    )

    result = {
        "status": "clarification" if clarification_reason else "delivered",
        "message": clarification_reason,
        "confidence": conf,
        "services": sorted(application.services.keys()),
        "fragments_used": sorted(application.fragment_ids),
        "quality": {
            "cost": application.quality[0], "duration": application.quality[1],
            "reliability": application.quality[2], "complexity": application.quality[3],
        },
        "explanation": explanation,
        "relaxation_rounds": len(relaxation_log),
    }

    if not clarification_reason:
        reward = float(application.quality[2])  # reliability stands in for the realised success signal
        update_scorer(scorer, hg, list(application.fragment_ids), reward, params, replay_batch=[])
        flagged = update_hyperedge_weights(hg, list(application.fragment_ids), reward,
                                            params["feedback"]["learning_rate_eta"],
                                            params["feedback"]["min_weight_omega"])
        memory.add_episode(list(application.fragment_ids), reward, retrieval_result.parsed.level)
        result["flagged_hyperedges"] = flagged

    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the HyFRAC prototype on one request.")
    parser.add_argument("--hypergraph", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--params", default=str(Path(__file__).resolve().parents[1] / "Config" / "parameters.json"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", default=None, help="optional path to save the JSON result")
    args = parser.parse_args()

    result = run_hyfrac(args.hypergraph, args.ontology, args.params, args.query)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    _cli()
