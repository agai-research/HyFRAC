"""
exp_definitions.py -- the 14 experiments of sec:results, as data. Each
entry names its varied parameter(s), the methods compared, and the fixed
settings held constant, exactly as sec:results specifies. Experiments/
run_experiment.py consumes this to build, run, and save each experiment
without per-experiment boilerplate.
"""

ALL_METHODS = ["HyFRAC", "HyFRAC-BG", "HyFRAC-NS", "HyFRAC-NF", "HyFRAC-WfG",
               "LLM4Workflow", "DSC-LLM", "Trust-MPGNN", "SWDG"]

OBJECTIVE_POOL = ["broadcasting", "quizzing", "whiteboarding", "ambient sensing",
                  "ambient regulation", "acoustics", "polling"]

EXPERIMENTS = {

    "res_e2e": {
        "title": "End-to-End Comparison Against Baselines",
        "varied_param": "num_objectives",
        "values": [1, 2, 3, 4],
        "methods": ["HyFRAC", "LLM4Workflow", "DSC-LLM", "Trust-MPGNN", "SWDG"],
        "fixed": {"level": 1},
    },

    "res_scale": {
        "title": "Scalability Across the Service Space",
        "varied_param": "num_services",
        "values": [100, 250, 500, 1000, 1500],
        "methods": ["HyFRAC", "HyFRAC-BG", "LLM4Workflow", "Trust-MPGNN"],
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_complexity": {
        "title": "Impact of Workflow Complexity",
        "varied_param": "num_objectives",
        "values": [1, 2, 3, 4, 5],
        "methods": ALL_METHODS,
        "fixed": {"level": 1},
    },

    "res_fusion": {
        "title": "Sensitivity to the Neuro-Symbolic Fusion Weight",
        "varied_param": "fusion_weight_alpha",
        "values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "methods": ["HyFRAC"],
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_density": {
        "title": "Impact of N-ary Dependency Density",
        "varied_param": "eta_g",
        "values": [0.0, 0.15, 0.30, 0.45, 0.60],
        "methods": ["HyFRAC", "HyFRAC-BG", "HyFRAC-NF", "Trust-MPGNN"],
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_budget": {
        "title": "Sensitivity to the Assembly-Stage Computational Budget",
        "varied_param": "max_fragments_per_combination",
        "values": [2, 3, 4],
        "methods": ["HyFRAC", "HyFRAC-BG", "HyFRAC-NF"],
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_thresholds": {
        "title": "Sensitivity to Community and Node Relevance Thresholds",
        "varied_param": "community_threshold",
        "values": [0.2, 0.3, 0.4, 0.5, 0.6],
        "methods": ["HyFRAC"],
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_confidence": {
        "title": "Confidence Gating: Threshold Effect and Calibration",
        "varied_param": "confidence_gate_theta_L",
        "values": [0.35, 0.45, 0.55, 0.65, 0.75],
        "methods": ["HyFRAC", "HyFRAC-NS"],
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_retrieval_sens": {
        "title": "Sensitivity to Retrieval Depth and Breadth",
        "varied_param": "breadth_k",
        "values": [5, 10, 15, 20],
        "methods": ["HyFRAC", "HyFRAC-BG"],
        "fixed": {"num_objectives": 2, "level": 1, "walk_depth": 3},
    },

    "res_coldstart": {
        "title": "Sensitivity to Fragment-Catalogue Completeness",
        "varied_param": "catalogue_fraction",
        "values": [0.10, 0.25, 0.50, 0.75, 1.00],
        "methods": ["HyFRAC", "HyFRAC-NF"],
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_backbone": {
        "title": "Sensitivity to the LLM Backbone Choice",
        "varied_param": "backbone",
        "values": ["rule_based"],  # only one backbone is actually reachable in this environment
        "methods": ["HyFRAC", "LLM4Workflow", "DSC-LLM"],
        "fixed": {"num_objectives": 2, "level": 1},
        "note": "no commercial or locally-served LLM API is reachable in this environment; "
                "this experiment is defined and executable exactly as specified once one is "
                "available (see README), but only the rule-based backbone is actually run here.",
    },

    "res_reuse": {
        "title": "Benefit of Fragment Reuse Across Repeated Requests",
        "varied_param": "repeated_request_count",
        "values": [1, 5, 10, 20, 30],
        "methods": ["HyFRAC", "HyFRAC-NF", "HyFRAC-BG"],
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_stats": {
        "title": "Statistical Significance and Effect-Size Analysis",
        "varied_param": "workload_stratum",
        "values": [1, 2, 3, 4, 5],
        "methods": ALL_METHODS,
        "fixed": {"num_objectives": 2, "level": 1},
    },

    "res_interaction": {
        "title": "Combined Ablation Interaction Study",
        "varied_param": "ablation_combination",
        "values": ["full", "BG_only", "NS_only", "BG_and_NS"],
        "methods": ["HyFRAC", "HyFRAC-BG", "HyFRAC-NS"],
        "fixed": {"num_objectives": 2, "level": 1},
    },
}
