"""
instance_by_size.py -- builds a corpus instance at a chosen service-space
size |S|, for the scalability experiments of sec:results (res-scale,
res-complexity). Reuses build_corpus with the same master seed so a smaller
instance is a genuine sub-corpus, not an unrelated random draw.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import build_corpus, report_counts  # noqa: E402
from build_ontology import build_ontology_from_corpus  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HyFRAC"))
from fragments import Repository  # noqa: E402


def build_instance(num_services: int, num_resources: int = None, eta_g: float = 0.30,
                    pi_T: float = 0.25, seed: int = 42) -> dict:
    """
    num_resources defaults to 0.5 * num_services (the corpus's default
    ratio), except where an experiment states otherwise -- e.g.
    sec:res-scale fixes |R|=500 regardless of |S|, which the caller passes
    explicitly to override this default.
    """
    num_resources = num_resources if num_resources is not None else round(num_services * 0.5)
    hg = build_corpus(num_services=num_services, num_resources=num_resources,
                       eta_g=eta_g, pi_T=pi_T, seed=seed)
    repository = Repository()
    repository.build(hg)
    ontology = build_ontology_from_corpus(hg)
    counts = report_counts(hg, repository)
    return {"hypergraph": hg, "repository": repository, "ontology": ontology, "counts": counts}


def save_instance(instance: dict, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    instance["hypergraph"].to_json(str(out_dir / f"hypergraph_{tag}.json"))
    (out_dir / f"ontology_{tag}.json").write_text(json.dumps(instance["ontology"], indent=2))
    (out_dir / f"counts_{tag}.json").write_text(json.dumps(instance["counts"], indent=2))


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "corpus" / "instances"
    for size in [100, 250, 500, 1000, 1500, 2000, 2500]:
        instance = build_instance(num_services=size, num_resources=500)  # |R| fixed, per sec:res-scale
        tag = f"S{size}"
        save_instance(instance, out_dir, tag)
        print(tag, instance["counts"])


if __name__ == "__main__":
    main()
