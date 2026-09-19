"""
build_corpus.py -- constructs the hybrid service hypergraph corpus.

Stage 1: assign a capability, quality vector, and profile to each service.
Stage 2: assign each resource a profile and link it to rho services under a
         Zipf popularity law -- fixes the res-share hyperedges.
Stage 3: sample prereq hyperedges over capability pairs, a share pi_T of
         which get a multi-service tail.
Stage 4: sample co-exec hyperedges so a share eta_g of capabilities need a
         group of 2-5 services.
Stage 5: cluster substitutable same-capability providers into cap-cover
         hyperedges of at most z services each.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HyFRAC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hypergraph import Hypergraph, Service, Hyperedge  # noqa: E402
from fragments import Repository  # noqa: E402
from capability_catalog import build_capabilities  # noqa: E402


def build_services(num_services: int, capabilities, rng: np.random.Generator) -> Dict[str, Service]:
    """Stage 1: one capability, one quality vector, one profile per service."""
    # education-specific capabilities are drawn more often, in the spirit of
    # the paper's smart-education running example
    weights = np.array([3.0 if c.education_specific else 1.0 for c in capabilities])
    weights /= weights.sum()

    services: Dict[str, Service] = {}
    for i in range(num_services):
        cap = rng.choice(capabilities, p=weights)
        cost = float(rng.gamma(shape=2.0, scale=8.0))
        duration = float(rng.gamma(shape=2.0, scale=6.0))
        reliability = float(np.clip(rng.normal(0.92, 0.05), 0.6, 0.999))
        complexity = int(rng.integers(1, 4))
        s_id = f"S{i:04d}"
        services[s_id] = Service(id=s_id, capability=cap.name, cost=cost, duration=duration,
                                  reliability=reliability, complexity=complexity)
    return services


def build_resources(num_resources: int, services: Dict[str, Service], rho_range,
                     alpha_z: float, max_hyperedge_size: int, target_count: int,
                     rng: np.random.Generator) -> Tuple[List[str], List[Hyperedge]]:
    """
    Stage 2: Zipf-linked resources -> the res-share hyperedges. A popular
    resource's users are chunked into groups of 2 (plus the resource
    itself), since a single hyperedge cannot legally exceed z members and
    small groups keep the joint-usage declaration meaningful. If chunking
    yields more candidate edges than the corpus's target density, a random
    subset is kept -- not every incidental co-location is formalised as a
    joint declaration, matching how only some resource overlaps are
    provider-declared rather than accidental.
    """
    resources = [f"R{i:04d}" for i in range(num_resources)]
    # Zipf popularity: resource i is chosen with probability ~ 1 / rank(i)^alpha_z
    ranks = np.arange(1, num_resources + 1)
    popularity = 1.0 / np.power(ranks, alpha_z)
    popularity /= popularity.sum()

    service_ids = list(services.keys())
    resource_users: Dict[str, List[str]] = {r: [] for r in resources}

    for s_id in service_ids:
        rho = int(rng.integers(rho_range[0], rho_range[1] + 1))
        chosen = rng.choice(resources, size=min(rho, num_resources), replace=False, p=popularity)
        services[s_id].resources = list(chosen)
        for r in chosen:
            resource_users[r].append(s_id)

    candidates: List[Hyperedge] = []
    edge_idx = 0
    res_share_chunk = 2
    for r, users in resource_users.items():
        if len(users) < 2:
            continue
        rng.shuffle(users)
        for i in range(0, len(users) - 1, res_share_chunk):
            chunk = users[i: i + res_share_chunk]
            if len(chunk) >= 2:
                edge_idx += 1
                candidates.append(Hyperedge(
                    id=f"rs{edge_idx:05d}", members=set(chunk) | {r}, edge_type="res-share",
                    confidence=float(np.clip(rng.normal(0.95, 0.05), 0.5, 1.0)),
                ))

    if len(candidates) > target_count:
        keep_idx = rng.choice(len(candidates), size=target_count, replace=False)
        candidates = [candidates[i] for i in sorted(keep_idx)]
    return resources, candidates


def build_prereq_hyperedges(services: Dict[str, Service], capabilities, pi_T: float,
                             target_count: int, rng: np.random.Generator) -> List[Hyperedge]:
    """
    Stage 3: prerequisite relations, in the spirit of MOOCCube's prerequisite
    graph. Rather than one edge per consecutive pair of education
    capabilities (which caps the count at 24, far below the observed
    corpus's density), a target number of prerequisite instances is sampled
    over all ordered pairs of education capabilities, each instantiated with
    its own tail/head service subset -- several courses can share the same
    conceptual prerequisite. A share pi_T of instances get a multi-service
    tail (|T(e)| > 1).
    """
    edu_caps = [c.name for c in capabilities if c.education_specific]
    by_cap: Dict[str, List[str]] = {}
    for s_id, s in services.items():
        by_cap.setdefault(s.capability, []).append(s_id)

    edges: List[Hyperedge] = []
    attempts, edge_idx = 0, 0
    max_attempts = target_count * 5
    while edge_idx < target_count and attempts < max_attempts:
        attempts += 1
        tail_cap, head_cap = rng.choice(edu_caps, size=2, replace=False)
        tail_pool, head_pool = by_cap.get(tail_cap, []), by_cap.get(head_cap, [])
        if not tail_pool or not head_pool:
            continue
        multi_tail = rng.random() < pi_T
        tail_size = int(rng.integers(2, 4)) if multi_tail else 1
        tail = list(rng.choice(tail_pool, size=min(tail_size, len(tail_pool)), replace=False))
        head = [rng.choice(head_pool)]
        if set(tail) & set(head):
            continue
        edge_idx += 1
        edges.append(Hyperedge(
            id=f"pr{edge_idx:05d}", members=set(tail) | set(head), edge_type="prereq",
            confidence=float(np.clip(rng.normal(0.90, 0.06), 0.5, 1.0)),
            tail=set(tail), head=set(head),
        ))
    return edges


def build_co_exec_hyperedges(services: Dict[str, Service], eta_g: float,
                              target_count: int, rng: np.random.Generator) -> List[Hyperedge]:
    """
    Stage 4: co-exec hyperedges: repeatedly sample a group of 2-5 services
    sharing one of the eta_g-selected capabilities and jointly declare them.
    A service may appear in more than one such group (different joint
    contexts), matching how a popular provider can be requested alongside
    different partners across distinct compositions.
    """
    by_cap: Dict[str, List[str]] = {}
    for s_id, s in services.items():
        by_cap.setdefault(s.capability, []).append(s_id)

    cap_names = list(by_cap.keys())
    num_grouped = max(1, int(len(cap_names) * eta_g))
    grouped_caps = list(rng.choice(cap_names, size=num_grouped, replace=False))

    edges: List[Hyperedge] = []
    edge_idx, attempts = 0, 0
    max_attempts = target_count * 5
    while edge_idx < target_count and attempts < max_attempts:
        attempts += 1
        cap = rng.choice(grouped_caps)
        pool = by_cap[cap]
        if len(pool) < 2:
            continue
        size = int(rng.integers(2, min(6, len(pool) + 1)))
        group = list(rng.choice(pool, size=size, replace=False))
        edge_idx += 1
        edges.append(Hyperedge(
            id=f"ce{edge_idx:05d}", members=set(group), edge_type="co-exec",
            confidence=float(np.clip(rng.normal(0.93, 0.05), 0.5, 1.0)),
        ))
    return edges


def build_cap_cover_hyperedges(services: Dict[str, Service], max_size: int, target_count: int,
                                rng: np.random.Generator) -> List[Hyperedge]:
    """
    Stage 5: cluster substitutable same-capability providers, capped at
    max_size (z). Non-overlapping clusters are built first; if the target
    count is not yet reached, additional overlapping clusters are sampled
    from capabilities with enough providers, matching how a capability with
    many substitutable services can be summarised by more than one cluster.
    """
    by_cap: Dict[str, List[str]] = {}
    for s_id, s in services.items():
        by_cap.setdefault(s.capability, []).append(s_id)

    edges: List[Hyperedge] = []
    edge_idx = 0
    for cap, pool in by_cap.items():
        pool = list(pool)
        if len(pool) < 2:
            continue
        rng.shuffle(pool)
        for i in range(0, len(pool) - 1, max_size):
            cluster = pool[i: i + max_size]
            if len(cluster) >= 2:
                edge_idx += 1
                edges.append(Hyperedge(
                    id=f"cc{edge_idx:05d}", members=set(cluster), edge_type="cap-cover",
                    confidence=float(np.clip(rng.normal(0.85, 0.06), 0.5, 1.0)),
                ))

    eligible_caps = [cap for cap, pool in by_cap.items() if len(pool) >= 2]
    attempts = 0
    while len(edges) < target_count and eligible_caps and attempts < target_count * 5:
        attempts += 1
        cap = rng.choice(eligible_caps)
        pool = by_cap[cap]
        size = int(rng.integers(2, min(max_size, len(pool)) + 1))
        cluster = list(rng.choice(pool, size=size, replace=False))
        edge_idx += 1
        edges.append(Hyperedge(
            id=f"cc{edge_idx:05d}", members=set(cluster), edge_type="cap-cover",
            confidence=float(np.clip(rng.normal(0.85, 0.06), 0.5, 1.0)),
        ))
    return edges[:target_count] if len(edges) > target_count else edges


def build_corpus(num_services: int, num_resources: int, eta_g: float, pi_T: float,
                  rho_range=(1, 5), alpha_z: float = 0.3, max_hyperedge_size: int = 6,
                  seed: int = 42) -> Hypergraph:
    """
    Runs all five stages under one master seed and returns the assembled
    hypergraph. The prereq, co-exec, and cap-cover target counts scale
    proportionally with num_services, calibrated so the default instance
    (num_services=1000) reproduces sec:dataset's reported density
    (390 prereq, 640 co-exec, 240 cap-cover out of 2,450 hyperedges).
    """
    rng = np.random.default_rng(seed)
    capabilities = build_capabilities()
    scale = num_services / 1000.0

    services = build_services(num_services, capabilities, rng)
    resources, res_share_edges = build_resources(num_resources, services, rho_range, alpha_z,
                                                   max_hyperedge_size, target_count=round(1180 * scale),
                                                   rng=rng)
    prereq_edges = build_prereq_hyperedges(services, capabilities, pi_T,
                                            target_count=round(390 * scale), rng=rng)
    co_exec_edges = build_co_exec_hyperedges(services, eta_g,
                                              target_count=round(640 * scale), rng=rng)
    cap_cover_edges = build_cap_cover_hyperedges(services, max_hyperedge_size,
                                                  target_count=round(240 * scale), rng=rng)

    hg = Hypergraph()
    for s in services.values():
        hg.add_service(s)
    for r in resources:
        hg.resources.add(r)
    for c in capabilities:
        hg.add_concept(c.name)
    for edge in res_share_edges + prereq_edges + co_exec_edges + cap_cover_edges:
        hg.add_hyperedge(edge)
    hg.build_incidence_matrix()
    return hg


def report_counts(hg: Hypergraph, repository: Repository) -> dict:
    counts = {"services": len(hg.services), "resources": len(hg.resources)}
    by_type: Dict[str, int] = {}
    for e in hg.hyperedges.values():
        by_type[e.edge_type] = by_type.get(e.edge_type, 0) + 1
    counts["hyperedges_by_type"] = by_type
    counts["hyperedges_total"] = sum(by_type.values())
    counts["fragments"] = len(repository.fragments)
    sizes = [len(f.services) for f in repository.fragments.values()]
    counts["avg_fragment_size"] = round(float(np.mean(sizes)), 2) if sizes else 0.0
    return counts


def main() -> None:
    hg = build_corpus(num_services=1000, num_resources=500, eta_g=0.30, pi_T=0.25, seed=42)
    repository = Repository()
    repository.build(hg)

    counts = report_counts(hg, repository)
    print(json.dumps(counts, indent=2))

    out_dir = Path(__file__).resolve().parent / "corpus"
    out_dir.mkdir(exist_ok=True)
    hg.to_json(str(out_dir / "master_hypergraph.json"))
    with open(out_dir / "master_counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    print(f"saved corpus to {out_dir}")


if __name__ == "__main__":
    main()
