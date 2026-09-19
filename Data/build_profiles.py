"""
build_profiles.py -- constructs user profiles and their free-text requests.

MOOCCube's own interaction traces are not reachable from this environment
(only package registries are), so profiles are constructed from the same
capability catalogue the corpus itself uses, and each profile's structured
objectives are turned into a free-text request through a template-based
paraphraser -- a rule-based substitute for the paraphrasing step an LLM
would otherwise perform, kept deliberately simple and stated once here
rather than hidden inside the request-parsing code that later reads it.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capability_catalog import build_capabilities  # noqa: E402

TEMPLATES = [
    "I need help with {objectives} for a level {level} {domain} session.",
    "Please set up {objectives} suited to a level {level} audience.",
    "Can you arrange {objectives}? This is for a {domain} context, level {level}.",
    "We would like {objectives} configured for level {level} learners.",
]


def paraphrase(objectives: List[str], level: int, domain: str, rng: np.random.Generator) -> str:
    joined = " and ".join(objectives)
    template = rng.choice(TEMPLATES)
    return template.format(objectives=joined, level=level, domain=domain)


def build_profiles(num_profiles: int, seed: int = 7) -> List[Dict]:
    rng = np.random.default_rng(seed)
    capabilities = build_capabilities()
    names = [c.name for c in capabilities]

    profiles = []
    for i in range(num_profiles):
        num_objectives = int(rng.integers(1, 4))
        objectives = list(rng.choice(names, size=num_objectives, replace=False))
        level = int(rng.integers(1, 4))
        domain = capabilities[names.index(objectives[0])].domain
        preferences = rng.dirichlet(np.ones(4))  # (cost, duration, reliability, complexity) weights
        budget = {"cost": float(rng.uniform(50, 300)), "duration": float(rng.uniform(30, 180)),
                  "complexity": level + 1}
        request = paraphrase(objectives, level, domain, rng)

        profiles.append({
            "profile_id": f"P{i:04d}",
            "objectives": objectives,
            "level": level,
            "domain": domain,
            "preferences": {"cost": preferences[0], "duration": preferences[1],
                             "reliability": preferences[2], "complexity": preferences[3]},
            "budget": budget,
            "request": request,
        })
    return profiles


def main() -> None:
    profiles = build_profiles(num_profiles=200)
    out_path = Path(__file__).resolve().parent / "corpus" / "profiles.json"
    out_path.write_text(json.dumps(profiles, indent=2))
    print(f"built {len(profiles)} profiles -> {out_path}")
    print("example:", profiles[0]["request"])


if __name__ == "__main__":
    main()
