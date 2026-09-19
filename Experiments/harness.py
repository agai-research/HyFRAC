"""
harness.py -- shared infrastructure every experiment script builds on:
a uniform way to call any of the nine compared methods, a uniform set of
recorded measures, and multi-seed aggregation (mean, std).

Kept separate from both the prototype (HyFRAC/) and the compared methods
(Baselines/) so no experiment script needs to know any method's internal
call signature.
"""

from __future__ import annotations
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HyFRAC"))
sys.path.insert(0, str(ROOT / "Baselines"))
sys.path.insert(0, str(ROOT / "Data"))

from hyfrac import run_hyfrac  # noqa: E402
from hyfrac_ablations import run_hyfrac_bg, run_hyfrac_ns, run_hyfrac_nf, run_hyfrac_wfg  # noqa: E402
from llm4workflow import run_llm4workflow  # noqa: E402
from dsc_llm import run_dsc_llm  # noqa: E402
from trust_mpgnn import run_trust_mpgnn  # noqa: E402
from swdg import run_swdg  # noqa: E402
from goal_tree import GoalTree, GoalNode  # noqa: E402
from feedback import EpisodicMemory  # noqa: E402

METHODS = ["HyFRAC", "HyFRAC-BG", "HyFRAC-NS", "HyFRAC-NF", "HyFRAC-WfG",
           "LLM4Workflow", "DSC-LLM", "Trust-MPGNN", "SWDG"]

LLM_BASED = {"HyFRAC", "HyFRAC-BG", "HyFRAC-NS", "HyFRAC-NF", "HyFRAC-WfG",
             "LLM4Workflow", "DSC-LLM", "SWDG"}  # every method except Trust-MPGNN calls the backbone


def oracle_goal_tree(objectives: List[str], level: int = 1) -> GoalTree:
    tree = GoalTree()
    tree.root.children = [GoalNode(kind="leaf", capability=c, target_complexity=level) for c in objectives]
    return tree


def run_method(name: str, hypergraph_path: str, ontology_path: str, params: dict,
               query: str, objectives: List[str], level: int = 1) -> dict:
    """Dispatches to the right method, returning a result dict with at least
    {status, services, quality, confidence}. Each call is timed here so
    latency is measured uniformly, not by each method's own code."""
    t0 = time.time()
    oracle = oracle_goal_tree(objectives, level)

    if name == "HyFRAC":
        result = run_hyfrac(hypergraph_path, ontology_path, str(ROOT / "Config" / "parameters.json"),
                             query, memory=EpisodicMemory())
    elif name == "HyFRAC-BG":
        result = run_hyfrac_bg(hypergraph_path, ontology_path, params, query)
    elif name == "HyFRAC-NS":
        result = run_hyfrac_ns(hypergraph_path, ontology_path, params, query)
    elif name == "HyFRAC-NF":
        result = run_hyfrac_nf(hypergraph_path, ontology_path, params, query)
    elif name == "HyFRAC-WfG":
        result = run_hyfrac_wfg(hypergraph_path, ontology_path, params, query, oracle)
    elif name == "LLM4Workflow":
        result = run_llm4workflow(hypergraph_path, query)
    elif name == "DSC-LLM":
        result = run_dsc_llm(hypergraph_path, query)
    elif name == "Trust-MPGNN":
        result = run_trust_mpgnn(hypergraph_path, oracle)
    elif name == "SWDG":
        result = run_swdg(hypergraph_path, objectives)
    else:
        raise ValueError(f"unknown method: {name}")

    result["_latency"] = time.time() - t0
    result["_method"] = name
    return result


def compute_metrics(result: dict, required_capabilities: List[str], hg) -> Dict[str, float]:
    """
    A uniform metric set every experiment records:
      validity      -- 1.0 if delivered, else 0.0
      goal_coverage -- share of required capabilities actually covered
      quality       -- mean of the four normalised QoS dimensions
      latency       -- wall-clock seconds for this call
      confidence    -- the method's own reported confidence, if any
    """
    validity = 1.0 if result.get("status") == "delivered" else 0.0
    services = result.get("services", [])
    covered = {hg.services[s].capability for s in services if s in hg.services}
    goal_coverage = len(covered & set(required_capabilities)) / max(len(required_capabilities), 1)

    q = result.get("quality", {})
    if q:
        norm_cost = 1.0 / (1.0 + q.get("cost", 0.0))
        norm_duration = 1.0 / (1.0 + q.get("duration", 0.0))
        reliability = q.get("reliability", 0.0)
        norm_complexity = 1.0 / (1.0 + q.get("complexity", 1))
        quality = float(np.mean([norm_cost, norm_duration, reliability, norm_complexity]))
    else:
        quality = 0.0

    return {
        "validity": validity,
        "goal_coverage": goal_coverage,
        "quality": quality,
        "latency": result.get("_latency", 0.0),
        "confidence": result.get("confidence", 0.0) or 0.0,
    }


def sample_requests(objectives_pool: List[str], num_objectives: int, num_requests: int,
                     level: int = 1, seed: int = 0) -> List[Dict]:
    """
    Draws num_requests distinct (query, objectives) pairs, each with exactly
    num_objectives capabilities, from the given pool. Since this
    implementation's internal randomness (HGNN init, community detection)
    is fixed-seeded for reproducibility, repeating one identical query
    would yield zero run-to-run variance; genuine variance instead comes
    from sampling a distribution of different, comparable requests under
    the same controlled condition -- the same interpretation
    sec:res-stats itself uses ("paired requests ... across seeds").
    """
    rng = np.random.default_rng(seed)
    requests = []
    for i in range(num_requests):
        objectives = list(rng.choice(objectives_pool, size=min(num_objectives, len(objectives_pool)),
                                      replace=False))
        query = f"I need help with {' and '.join(objectives)} for a level {level} session."
        requests.append({"query": query, "objectives": objectives, "level": level})
    return requests


def run_many_seeds(name: str, hypergraph_path: str, ontology_path: str, params: dict,
                    objectives_pool: List[str], num_objectives: int, hg, level: int = 1,
                    num_seeds: int = 10) -> Dict[str, Dict[str, float]]:
    """Runs one method over num_seeds distinct sampled requests and reports mean/std per metric."""
    requests = sample_requests(objectives_pool, num_objectives, num_seeds, level)
    per_run: List[Dict[str, float]] = []
    for req in requests:
        result = run_method(name, hypergraph_path, ontology_path, copy.deepcopy(params),
                             req["query"], req["objectives"], req["level"])
        per_run.append(compute_metrics(result, req["objectives"], hg))

    aggregated: Dict[str, Dict[str, float]] = {}
    for metric in per_run[0].keys():
        values = [r[metric] for r in per_run]
        aggregated[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return aggregated
