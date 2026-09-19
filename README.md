# HyFRAC — Prototype Implementation

A complete, runnable implementation of the HyFRAC approach: a hypergraph-based,
fragment-reusing, neuro-symbolically filtered, agentic pipeline for composing
smart-service applications from free-text requests.

## Structure

```
HyFRAC/
  HyFRAC/            the prototype package: hypergraph, retrieval, neuro-symbolic
                      filter, agents, optimisation, trace, feedback, and the
                      main entry point (hyfrac.py)
  Data/               the dataset-construction pipeline (build_corpus.py,
                      build_ontology.py, build_profiles.py, instance_by_size.py)
                      and the built default corpus (Data/corpus/)
  Baselines/          the four ablations (HyFRAC-BG/NS/NF/WfG) and the four
                      external baselines (LLM4Workflow, DSC-LLM, Trust-MPGNN, SWDG)
  Experiments/        the shared experiment harness, all 14 experiment
                      definitions, the generic runner, and figure generation
  Test/               first_test.py (a small worked example) and
                      integration_test.py (all nine methods, one request)
  Config/             parameters.json -- every tunable hyperparameter
  Results/            experiment output (JSON) and figures
  Results.tex         the write-up of the experiments actually executed
  Implementation.tex  the adjusted Section sec:prototype, stating exactly
                      where this implementation departs from the original
                      specification and why
  dataset_and_results_summary.tex / .docx
                      the corpus composition and every executed experiment's
                      results as one uniformly-structured table each
                      (regenerate with gen_summary.py + gen_summary_docx.js)
  implementation_report.docx
                      the full issue log: every bug found and fixed (or
                      documented as a limitation) across all four phases
```

## Running it

```bash
pip install numpy scipy networkx pydantic --break-system-packages

# a small worked example, no real corpus needed
python Test/first_test.py

# all nine methods on the same request
python Test/integration_test.py

# build the default corpus (1,000 services; already built under Data/corpus/)
python Data/build_corpus.py
python Data/build_ontology.py

# run one experiment
python Experiments/run_experiment.py res_scale --seeds 5
python Experiments/make_figures.py
```

## What is real vs. a documented substitution

No commercial or locally-hosted LLM API is reachable from this environment.
Every LlmXxx step in the algorithms (decomposition, mediation, planning,
critique, ranking, explanation) is realised by a deterministic, rule-based
stand-in behind a single Backbone interface (HyFRAC/agents.py); a real
model can be plugged in there without touching any algorithm. All graph,
retrieval, and neural-scoring logic (the hypergraph, the HGNN, the trust-GNN,
the Bi-LSTM aggregator) is real, not stubbed.

Two baselines have documented reproduction-fidelity limitations, found and
recorded during testing rather than hidden: Trust-MPGNN's decoder is trained
(verified via a controlled synthetic test) but its global discrimination is
weak given simplified features, though within-pool ranking still correlates
positively with true service quality; SWDG's GNN forward pass is real but
its weights are randomly initialised, not supervised-trained, since no
labeled edge-existence data is available here.

## Known limitations of this reduced-scale run

Results.tex documents five experiments executed at reduced scale (1-3
seeds instead of the specified 10-20) and one at a reduced method subset.
The remaining nine experiments are fully implemented in
Experiments/exp_definitions.py and runnable at full scale via
run_experiment.py <name> --seeds N.
