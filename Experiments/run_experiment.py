"""
run_experiment.py -- generic runner for the parameter-sweep experiments of
exp_definitions.py. Handles three kinds of varied parameter uniformly:
  (a) a Config/parameters.json override (e.g. fusion_weight_alpha)
  (b) a corpus-construction parameter (num_services, eta_g)
  (c) a request-shape parameter (num_objectives)
Saves one JSON file per experiment under Results/, with mean/std per
method per value of the varied parameter.
"""

from __future__ import annotations
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Experiments"))
sys.path.insert(0, str(ROOT / "HyFRAC"))
sys.path.insert(0, str(ROOT / "Data"))

from harness import run_method, compute_metrics, sample_requests  # noqa: E402
from hypergraph import Hypergraph  # noqa: E402
from build_corpus import build_corpus  # noqa: E402
from build_ontology import build_ontology_from_corpus  # noqa: E402
from exp_definitions import EXPERIMENTS, OBJECTIVE_POOL  # noqa: E402

DEFAULT_HG = str(ROOT / "Data" / "corpus" / "master_hypergraph.json")
DEFAULT_ONT = str(ROOT / "Data" / "corpus" / "master_ontology.json")
CONFIG_PATH = ROOT / "Config" / "parameters.json"

# maps a varied_param name to (section, key) in Config/parameters.json, for
# experiments that override a fixed hyperparameter rather than the corpus
# or the request shape
CONFIG_OVERRIDE = {
    "fusion_weight_alpha": ("filtering", "fusion_weight_alpha"),
    "max_fragments_per_combination": ("filtering", "max_fragments_per_combination"),
    "community_threshold": ("retrieval", "community_threshold"),
    "breadth_k": ("retrieval", "breadth_k"),
    "confidence_gate_theta_L": ("assembly", "confidence_gate_theta_L"),
}

CORPUS_PARAM = {"num_services", "eta_g"}
REQUEST_SHAPE_PARAM = {"num_objectives", "workload_stratum"}


def _build_instance_for_value(varied_param: str, value, base_params: dict):
    """Returns (hypergraph_path, ontology_path) for corpus-varying experiments,
    building and caching a fresh instance only when the corpus itself must change."""
    cache_dir = ROOT / "Results" / "_corpus_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if varied_param == "num_services":
        num_services = value
        num_resources = 500  # sec:res-scale fixes |R|=500 regardless of |S|
        eta_g = base_params["service_space"]["group_capability_share"]
    elif varied_param == "eta_g":
        num_services = 500  # a smaller instance keeps this sweep fast to run repeatedly
        num_resources = 250
        eta_g = value
    else:
        raise ValueError(varied_param)

    tag = f"{varied_param}_{value}".replace(".", "p")
    hg_path = cache_dir / f"hg_{tag}.json"
    ont_path = cache_dir / f"ont_{tag}.json"
    if not hg_path.exists():
        hg = build_corpus(num_services=num_services, num_resources=num_resources,
                           eta_g=eta_g, pi_T=base_params["service_space"]["prereq_multi_tail_share"])
        hg.to_json(str(hg_path))
        ontology = build_ontology_from_corpus(hg)
        ont_path.write_text(json.dumps(ontology))
    return str(hg_path), str(ont_path)


def run_experiment(name: str, num_seeds: int = 3, scale_note: str = "") -> dict:
    """Runs one experiment from exp_definitions.EXPERIMENTS, saving results as JSON."""
    spec = EXPERIMENTS[name]
    base_params = json.loads(CONFIG_PATH.read_text())
    varied_param = spec["varied_param"]
    values = spec["values"]
    methods = spec["methods"]
    fixed = spec.get("fixed", {})

    results: Dict[str, Dict] = {"experiment": name, "title": spec["title"],
                                 "varied_param": varied_param, "values": values,
                                 "methods": methods, "num_seeds": num_seeds,
                                 "scale_note": scale_note, "per_method": {}}

    for method in methods:
        results["per_method"][method] = {}
        for value in values:
            params = copy.deepcopy(base_params)
            hg_path, ont_path = DEFAULT_HG, DEFAULT_ONT

            if varied_param in CONFIG_OVERRIDE:
                section, key = CONFIG_OVERRIDE[varied_param]
                params[section][key] = value
            elif varied_param in CORPUS_PARAM:
                hg_path, ont_path = _build_instance_for_value(varied_param, value, base_params)
            elif varied_param not in REQUEST_SHAPE_PARAM:
                raise ValueError(f"unhandled varied_param: {varied_param}")

            hg = Hypergraph.from_json(hg_path)
            num_objectives = value if varied_param == "num_objectives" else fixed.get("num_objectives", 2)
            level = fixed.get("level", 1)

            t0 = time.time()
            agg = _run_one_condition(method, hg_path, ont_path, params, num_objectives, level,
                                      hg, num_seeds)
            print(f"  {name} | {method} | {varied_param}={value} -> "
                  f"validity={agg['validity']['mean']:.2f} quality={agg['quality']['mean']:.3f} "
                  f"({time.time()-t0:.1f}s)")
            results["per_method"][method][str(value)] = agg

    out_path = ROOT / "Results" / f"{name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    return results


def _run_one_condition(method: str, hg_path: str, ont_path: str, params: dict,
                        num_objectives: int, level: int, hg, num_seeds: int) -> Dict:
    requests = sample_requests(OBJECTIVE_POOL, num_objectives, num_seeds, level, seed=hash(method) % 1000)
    per_run: List[Dict] = []
    for req in requests:
        result = run_method(method, hg_path, ont_path, params, req["query"], req["objectives"], req["level"])
        per_run.append(compute_metrics(result, req["objectives"], hg))
    aggregated = {}
    for metric in per_run[0].keys():
        vals = [r[metric] for r in per_run]
        aggregated[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return aggregated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    run_experiment(args.experiment, num_seeds=args.seeds)
