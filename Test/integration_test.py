"""
integration_test.py -- runs HyFRAC and all eight compared methods (four
ablations, four external baselines) on the same request, side by side.
Not a formal experiment (see Experiments/ for that): this exists only to
confirm every method actually runs end to end without errors before the
real experiment scripts depend on them.
"""

import sys
import json
import time
from pathlib import Path

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

HG = str(ROOT / "Test" / "data" / "hypergraph_seed.json")
ONT = str(ROOT / "Test" / "data" / "ontology_seed.json")
PARAMS = json.loads((ROOT / "Config" / "parameters.json").read_text())
QUERY = "I need to stream my slides and run a quick quiz for a beginner class."
OBJECTIVES = ["broadcasting", "quizzing"]


def oracle_goal_tree() -> GoalTree:
    tree = GoalTree()
    tree.root.children = [GoalNode(kind="leaf", capability=cap, target_complexity=1) for cap in OBJECTIVES]
    return tree


def main() -> None:
    results = {}

    def run_and_record(name, fn):
        t0 = time.time()
        try:
            result = fn()
            elapsed = time.time() - t0
            services = result.get("services", [])
            status = result.get("status", "n/a")
            print(f"{name:16s} status={status:14s} services={services}  ({elapsed:.2f}s)")
            results[name] = result
        except Exception as exc:  # noqa: BLE001 -- this is a smoke test, report every failure
            print(f"{name:16s} FAILED: {exc}")
            results[name] = {"status": "error", "message": str(exc)}

    run_and_record("HyFRAC", lambda: run_hyfrac(HG, ONT, str(ROOT / "Config" / "parameters.json"), QUERY,
                                                 memory=EpisodicMemory()))
    run_and_record("HyFRAC-BG", lambda: run_hyfrac_bg(HG, ONT, PARAMS, QUERY))
    run_and_record("HyFRAC-NS", lambda: run_hyfrac_ns(HG, ONT, PARAMS, QUERY))
    run_and_record("HyFRAC-NF", lambda: run_hyfrac_nf(HG, ONT, PARAMS, QUERY))
    run_and_record("HyFRAC-WfG", lambda: run_hyfrac_wfg(HG, ONT, PARAMS, QUERY, oracle_goal_tree()))
    run_and_record("LLM4Workflow", lambda: run_llm4workflow(HG, QUERY))
    run_and_record("DSC-LLM", lambda: run_dsc_llm(HG, QUERY))
    run_and_record("Trust-MPGNN", lambda: run_trust_mpgnn(HG, oracle_goal_tree()))
    run_and_record("SWDG", lambda: run_swdg(HG, OBJECTIVES))

    out_path = ROOT / "Test" / "integration_test_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved full results to {out_path}")

    failed = [name for name, r in results.items() if r.get("status") == "error"]
    if failed:
        print(f"\n{len(failed)} method(s) FAILED: {failed}")
        sys.exit(1)
    print(f"\nAll {len(results)} methods ran successfully.")


if __name__ == "__main__":
    main()
