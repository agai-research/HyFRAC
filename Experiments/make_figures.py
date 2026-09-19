"""
make_figures.py -- builds the figures for the executed experiments, reading
each experiment's saved JSON under Results/. One multi-series line chart
per experiment (validity, quality, and latency where meaningful), matching
the "no captions in the image itself" and "readable text size" requirements.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9.5})

COLORS = {
    "HyFRAC": "#1f77b4", "HyFRAC-BG": "#ff7f0e", "HyFRAC-NS": "#2ca02c",
    "HyFRAC-NF": "#d62728", "HyFRAC-WfG": "#9467bd", "LLM4Workflow": "#8c564b",
    "DSC-LLM": "#e377c2", "Trust-MPGNN": "#17becf", "SWDG": "#7f7f7f",
}
MARKERS = {"HyFRAC": "o", "HyFRAC-BG": "s", "HyFRAC-NS": "^", "HyFRAC-NF": "D",
           "HyFRAC-WfG": "v", "LLM4Workflow": "P", "DSC-LLM": "X", "Trust-MPGNN": "*", "SWDG": "h"}


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def line_chart(data, metrics, xlabel, filename, ylims=None):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    values = [str(v) for v in data["values"]]
    x = np.arange(len(values))

    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in data["methods"]:
            means = [data["per_method"][method][v][metric]["mean"] for v in values]
            stds = [data["per_method"][method][v][metric]["std"] for v in values]
            ax.errorbar(x, means, yerr=stds, label=method, color=COLORS.get(method, "black"),
                        marker=MARKERS.get(method, "o"), capsize=3, linewidth=1.6, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if ylims:
            ax.set_ylim(*ylims)
    axes[-1].legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGDIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {filename}")


def bar_chart(data, metric, ylabel, xlabel, filename):
    values = [str(v) for v in data["values"]]
    x = np.arange(len(values))
    width = 0.8 / len(data["methods"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, method in enumerate(data["methods"]):
        means = [data["per_method"][method][v][metric]["mean"] for v in values]
        ax.bar(x + i * width, means, width=width, label=method, color=COLORS.get(method, "black"))
    ax.set_xticks(x + width * (len(data["methods"]) - 1) / 2)
    ax.set_xticklabels(values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {filename}")


# --- res_e2e: grouped bar chart, validity and quality ---
d = load("res_e2e")
bar_chart(d, "validity", "Validity", "Number of goal-tree leaves", "res_e2e_validity.png")
bar_chart(d, "quality", "Quality score", "Number of goal-tree leaves", "res_e2e_quality.png")

# --- res_scale: multi-series line chart, quality and latency ---
d = load("res_scale")
line_chart(d, [("quality", "Quality score"), ("latency", "Latency (s)")],
           "Number of services |S|", "res_scale.png")

# --- res_complexity: multi-series line chart, validity and quality ---
d = load("res_complexity")
line_chart(d, [("validity", "Validity"), ("quality", "Quality score")],
           "Number of goal-tree leaves", "res_complexity.png")

# --- res_fusion: single line chart, quality vs alpha ---
d = load("res_fusion")
line_chart(d, [("quality", "Quality score"), ("validity", "Validity")],
           r"Fusion weight $\alpha$", "res_fusion.png")

# --- res_density: multi-series line chart, validity vs eta_g ---
d = load("res_density")
line_chart(d, [("validity", "Validity"), ("quality", "Quality score")],
           r"Group-capability share $\eta_g$", "res_density.png")

print("all figures built")
