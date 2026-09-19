"""
first_test.py -- a dedicated smoke test for the HyFRAC prototype.

Loads a small worked-example hypergraph and a set of saved queries, runs the
full pipeline on each, and prints a readable summary.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HyFRAC"))

from hyfrac import run_hyfrac  # noqa: E402
from feedback import EpisodicMemory  # noqa: E402

HYPERGRAPH = ROOT / "Test" / "data" / "hypergraph_seed.json"
ONTOLOGY = ROOT / "Test" / "data" / "ontology_seed.json"
PARAMS = ROOT / "Config" / "parameters.json"
QUERIES = ROOT / "Test" / "queries.json"
OUTPUT = ROOT / "Test" / "first_test_results.json"


def main() -> None:
    queries = json.loads(QUERIES.read_text())
    memory = EpisodicMemory()
    all_results = []

    for i, query in enumerate(queries, start=1):
        print(f"\n{'=' * 70}\nQuery {i}: {query}\n{'=' * 70}")
        result = run_hyfrac(str(HYPERGRAPH), str(ONTOLOGY), str(PARAMS), query, memory=memory)
        all_results.append({"query": query, "result": result})

        print(f"status:      {result['status']}")
        if result["status"] == "delivered":
            print(f"services:    {result['services']}")
            print(f"fragments:   {result['fragments_used']}")
            print(f"confidence:  {result['confidence']:.3f}")
            print(f"quality:     {result['quality']}")
            print("explanation:")
            print(result["explanation"])
        else:
            print(f"message:     {result.get('message')}")

    OUTPUT.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {len(all_results)} result(s) to {OUTPUT}")


if __name__ == "__main__":
    main()
