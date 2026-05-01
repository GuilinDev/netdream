#!/usr/bin/env python3
"""Generate publication-quality figures for NeurIPS submission."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
})

COLORS = {
    "Random": "#999999",
    "HPA": "#000000",
    "NetDream-H5": "#0072B2",
    "NetDream-H5-Safe": "#D55E00",
    "Do-Nothing": "#999999",
}

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig_prediction_accuracy():
    """Figure 1: Multi-step prediction MSE for both domains."""
    horizons = [1, 5, 10]
    k8s_mse = [0.000250, 0.002086, 0.004700]
    grid_mse = [0.000045, 0.000865, 0.005066]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(horizons, k8s_mse, "o-", color="#0072B2", label="K8s Mesh (11 nodes)", markersize=5, linewidth=1.5)
    ax.plot(horizons, grid_mse, "s--", color="#D55E00", label="Power Grid (14 nodes)", markersize=5, linewidth=1.5)
    ax.set_xlabel("Rollout Horizon (steps)")
    ax.set_ylabel("Prediction MSE")
    ax.set_xticks(horizons)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("World Model Prediction Accuracy", fontweight="bold")

    fig.savefig(OUT / "fig_prediction_accuracy.pdf")
    fig.savefig(OUT / "fig_prediction_accuracy.png")
    plt.close(fig)
    print("  Saved: fig_prediction_accuracy")


def fig_planning_comparison():
    """Figure 2: Planning performance — violations by agent × workload."""
    workloads = ["Constant", "Variable", "Bursty", "Flash"]
    agents = ["Random", "HPA", "NetDream-H5", "NetDream-H5-Safe"]

    # Data from experiments
    violations = {
        "Random":          [32.2, 21.4, 7.6, 47.0],
        "HPA":             [5.0, 6.2, 6.6, 15.2],
        "NetDream-H5":     [0.2, 0.2, 8.2, 27.4],
        "NetDream-H5-Safe":[0.0, 0.0, 9.6, 30.8],
    }
    violation_std = {
        "Random":          [5.8, 5.9, 2.7, 24.0],
        "HPA":             [0.9, 1.0, 6.6, 1.0],
        "NetDream-H5":     [0.4, 0.4, 4.4, 11.5],
        "NetDream-H5-Safe":[0.0, 0.0, 4.4, 3.7],
    }

    fig, ax = plt.subplots(figsize=(5, 2.8))
    x = np.arange(len(workloads))
    width = 0.18

    for i, agent in enumerate(agents):
        offset = (i - len(agents)/2 + 0.5) * width
        bars = ax.bar(
            x + offset, violations[agent], width,
            yerr=violation_std[agent], capsize=2,
            color=COLORS[agent], edgecolor="black", linewidth=0.3,
            error_kw={"linewidth": 0.5, "capthick": 0.5},
            label=agent,
        )
        # Annotate zeros
        for j, v in enumerate(violations[agent]):
            if v == 0:
                ax.text(x[j] + offset, 0.5, "0", ha="center", va="bottom", fontsize=6, fontweight="bold", color=COLORS[agent])

    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylabel("Constraint Violations")
    ax.set_title("K8s Mesh: Safety Filter Eliminates Violations on Steady-State Workloads", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(OUT / "fig_planning_violations.pdf")
    fig.savefig(OUT / "fig_planning_violations.png")
    plt.close(fig)
    print("  Saved: fig_planning_violations")


def fig_cost_comparison():
    """Figure 3: Cost comparison."""
    workloads = ["Constant", "Variable", "Bursty", "Flash"]
    agents = ["HPA", "NetDream-H5", "NetDream-H5-Safe"]

    costs = {
        "HPA":             [9.42, 9.49, 9.27, 11.06],
        "NetDream-H5":     [10.44, 9.96, 8.30, 9.90],
        "NetDream-H5-Safe":[9.97, 9.96, 8.36, 9.16],
    }

    fig, ax = plt.subplots(figsize=(4, 2.5))
    x = np.arange(len(workloads))
    width = 0.22

    for i, agent in enumerate(agents):
        offset = (i - len(agents)/2 + 0.5) * width
        ax.bar(x + offset, costs[agent], width,
               color=COLORS[agent], edgecolor="black", linewidth=0.3,
               label=agent)

    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Cost Comparison: NetDream Matches HPA", fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(OUT / "fig_cost_comparison.pdf")
    fig.savefig(OUT / "fig_cost_comparison.png")
    plt.close(fig)
    print("  Saved: fig_cost_comparison")


def fig_horizon_ablation():
    """Figure 4: Planning horizon ablation."""
    horizons = [1, 3, 5]
    violations = [8.4, 9.2, 8.0]
    costs = [8.46, 7.62, 8.32]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.2))

    ax1.bar(range(len(horizons)), violations, color="#E89999", edgecolor="black", linewidth=0.3)
    ax1.set_xticks(range(len(horizons)))
    ax1.set_xticklabels([f"H={h}" for h in horizons])
    ax1.set_ylabel("Violations")
    ax1.set_title("(a) Violations", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(range(len(horizons)), costs, color="#FFD9A8", edgecolor="black", linewidth=0.3)
    ax2.set_xticks(range(len(horizons)))
    ax2.set_xticklabels([f"H={h}" for h in horizons])
    ax2.set_ylabel("Cost (USD)")
    ax2.set_title("(b) Cost", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Planning Horizon Ablation (Bursty Workload)", fontweight="bold", y=1.05)
    fig.tight_layout()

    fig.savefig(OUT / "fig_horizon_ablation.pdf")
    fig.savefig(OUT / "fig_horizon_ablation.png")
    plt.close(fig)
    print("  Saved: fig_horizon_ablation")


def fig_gnn_ablation():
    """Figure 5: GNN type ablation."""
    types = ["GAT", "GCN"]
    mse_1 = [0.000279, 0.001557]
    mse_5 = [0.003702, 0.004617]
    mse_10 = [0.006210, 0.016544]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(types))
    width = 0.25

    ax.bar(x - width, mse_1, width, label="1-step", color="#0072B2", edgecolor="black", linewidth=0.3)
    ax.bar(x, mse_5, width, label="5-step", color="#009E73", edgecolor="black", linewidth=0.3)
    ax.bar(x + width, mse_10, width, label="10-step", color="#D55E00", edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("Prediction MSE")
    ax.set_title("GNN Architecture Ablation (K8s)", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(OUT / "fig_gnn_ablation.pdf")
    fig.savefig(OUT / "fig_gnn_ablation.png")
    plt.close(fig)
    print("  Saved: fig_gnn_ablation")


def fig_architecture_diagram():
    """Figure 6: NetDream architecture overview (text-based for now)."""
    fig, ax = plt.subplots(figsize=(6.75, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Boxes
    boxes = [
        (0.5, 1.5, "Graph\nState\n$G_t$", "#E8E8E8"),
        (2.5, 1.5, "GNN\nEncoder", "#B3D9FF"),
        (4.5, 1.5, "Dynamics\nHead\n$\\Delta \\hat{x}$", "#B3FFB3"),
        (6.5, 2.0, "Reward\nHead\n$\\hat{r}$", "#FFD9B3"),
        (6.5, 1.0, "Constraint\nHead\n$\\hat{c}$", "#FFB3B3"),
        (8.5, 1.5, "MPC\nPlanner", "#D9B3FF"),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x-0.6, y-0.4), 1.2, 0.8, facecolor=color, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=7)

    # Arrows
    arrows = [(1.1, 1.5, 1.3, 0), (3.1, 1.5, 1.3, 0), (5.1, 1.8, 0.8, 0.2),
              (5.1, 1.2, 0.8, -0.2), (7.1, 1.5, 0.8, 0)]
    for x, y, dx, dy in arrows:
        ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Safety filter annotation
    ax.annotate("Safety\nFilter", xy=(7.5, 0.4), fontsize=7, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFB3B3", alpha=0.5))
    ax.annotate("", xy=(8.5, 0.7), xytext=(7.5, 0.6),
                arrowprops=dict(arrowstyle="->", color="red", lw=1, linestyle="--"))

    ax.set_title("NetDream Architecture: GNN Dynamics + Imagination-Based Safe Planning", fontweight="bold")

    fig.savefig(OUT / "fig_architecture.pdf")
    fig.savefig(OUT / "fig_architecture.png")
    plt.close(fig)
    print("  Saved: fig_architecture")


if __name__ == "__main__":
    print("Generating NeurIPS figures...")
    fig_prediction_accuracy()
    fig_planning_comparison()
    fig_cost_comparison()
    fig_horizon_ablation()
    fig_gnn_ablation()
    fig_architecture_diagram()
    print(f"\nAll figures saved to {OUT}/")
