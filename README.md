# HyFRAC — Prototype Implementation

A complete, runnable implementation of the HyFRAC approach: a hypergraph-based,
fragment-reusing, neuro-symbolically filtered, agentic pipeline for composing
smart-service applications from free-text requests.

## Running it

```bash
pip install numpy scipy networkx pydantic --break-system-packages

# a small worked example, no real corpus needed
python Test/first_test.py

# all compared methods on the same request
python Test/integration_test.py

# build the default corpus (candidate services already built under Data/corpus/)
python Data/build_corpus.py
python Data/build_ontology.py

# run one experiment
python Experiments/run_experiment.py res_scale --seeds 5
python Experiments/make_figures.py
```
