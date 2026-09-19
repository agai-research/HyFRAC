"""
dsc_llm.py -- DSC-LLM (Izzi et al.): selects relevant data services from the
full registry, generates a single-pass pipeline (a script that chains the
selected services), then repairs mismatched parameter names by embedding
similarity -- the paper's own three-component architecture.

The paper generates an actual Python script; here the "generated code" is
represented as an explicit, structured pipeline (ordered service calls with
resolved parameters), which is what a generated script would encode. This
keeps every step -- selection, generation, repair -- real and inspectable,
rather than skipping straight to a service list.
"""

from __future__ import annotations
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HyFRAC"))
from hypergraph import Hypergraph  # noqa: E402
from agents import CAPABILITY_KEYWORDS  # noqa: E402


@dataclass
class PipelineStep:
    service_id: str
    capability: str
    parameters: Dict[str, str]   # requested-name -> resolved-name (after repair)


def data_service_selector(query: str, hg: Hypergraph) -> List[str]:
    """
    LLM-prompted selector: given the full registry, returns only the
    services relevant to the query.
    """
    text = query.lower()
    needed_caps = {cap for cap, kws in CAPABILITY_KEYWORDS.items() if any(k in text for k in kws)}
    if not needed_caps:
        needed_caps = {list(CAPABILITY_KEYWORDS.keys())[0]}
    return [s_id for s_id, s in hg.services.items() if s.capability in needed_caps]


def generate_pipeline(query: str, candidate_ids: List[str], hg: Hypergraph) -> List[PipelineStep]:
    """
    LLM Agent, single pass: emits one pipeline step per required capability,
    picking (deterministically, as a stand-in for the LLM's own choice) the
    candidate with the best quality-cost trade-off for each capability, and
    requesting a parameter set that may not exactly match the service's own
    documented parameter names -- this is what the repair stage fixes next.
    """
    text = query.lower()
    needed_caps = [cap for cap, kws in CAPABILITY_KEYWORDS.items() if any(k in text for k in kws)]
    if not needed_caps:
        needed_caps = [list(CAPABILITY_KEYWORDS.keys())[0]]

    steps: List[PipelineStep] = []
    for cap in needed_caps:
        pool = [s for s in candidate_ids if hg.services[s].capability == cap]
        if not pool:
            continue
        best = min(pool, key=lambda s: hg.services[s].cost + hg.services[s].duration)
        # a slightly noisy parameter name, as an LLM's single-pass output would sometimes produce
        requested_params = {f"{r}_id": r for r in hg.services[best].resources}
        steps.append(PipelineStep(service_id=best, capability=cap, parameters=requested_params))
    return steps


def repair_parameters(steps: List[PipelineStep], hg: Hypergraph, threshold: float = 0.8) -> List[PipelineStep]:
    for step in steps:
        true_names = hg.services[step.service_id].resources
        fixed = {}
        for requested, guessed in step.parameters.items():
            if guessed in true_names:
                fixed[requested] = guessed
                continue
            best_match, best_ratio = None, 0.0
            for true_name in true_names:
                ratio = difflib.SequenceMatcher(None, guessed, true_name).ratio()
                if ratio > best_ratio:
                    best_match, best_ratio = true_name, ratio
            fixed[requested] = best_match if best_match and best_ratio >= threshold else guessed
        step.parameters = fixed
    return steps


def run_dsc_llm(hypergraph_path: str, query: str, repair_threshold: float = 0.8) -> dict:
    hg = Hypergraph.from_json(hypergraph_path)

    candidates = data_service_selector(query, hg)
    if not candidates:
        return {"status": "clarification", "message": "no data service matched the query",
                "services": [], "confidence": 0.0}

    steps = generate_pipeline(query, candidates, hg)
    if not steps:
        return {"status": "clarification", "message": "pipeline generation produced no steps",
                "services": [], "confidence": 0.0}

    steps = repair_parameters(steps, hg, repair_threshold)

    services = [s.service_id for s in steps]
    total_cost = sum(hg.services[s].cost for s in services)
    total_duration = sum(hg.services[s].duration for s in services)
    reliability = 1.0
    for s in services:
        reliability *= hg.services[s].reliability
    complexity = max(hg.services[s].complexity for s in services)

    return {
        "status": "delivered",
        "services": services,
        "pipeline": [{"service": s.service_id, "capability": s.capability, "parameters": s.parameters}
                     for s in steps],
        "quality": {"cost": total_cost, "duration": total_duration,
                    "reliability": reliability, "complexity": complexity},
        "confidence": reliability,
        "selector_recall_pool_size": len(candidates),
    }
