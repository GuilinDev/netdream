#!/usr/bin/env python3
"""Generate updated publication figures with all new results."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.linewidth": 0.6,
})

COLORS = {"Random": "#999999", "HPA": "#000000", "PPO": "#B5D8B0",
           "NetDream-H5": "#A8C9DD", "NetDream-Safe": "#FFD9A8", "Do-Nothing": "#999999"}
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig_prediction():
    horizons = [1, 5, 10]
    k8s = [0.000250, 0.002086, 0.004700]
    grid = [0.000045, 0.000865, 0.005066]
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    ax.plot(horizons, k8s, "o-", color="#5B8DBE", label="K8s Mesh (11 nodes)", markersize=5, lw=1.5)
    ax.plot(horizons, grid, "s--", color="#E89999", label="Power Grid (14 nodes)", markersize=5, lw=1.5)
    ax.set_xlabel("Rollout Horizon (steps)")
    ax.set_ylabel("Prediction MSE")
    ax.set_xticks(horizons)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("World Model Prediction Accuracy", fontweight="bold")
    fig.savefig(OUT / "fig_prediction_accuracy.pdf"); fig.savefig(OUT / "fig_prediction_accuracy.png")
    plt.close(fig)
    print("  fig_prediction_accuracy")


def fig_ablation_full():
    """Full 4-variant GNN ablation."""
    types = ["GAT\n(ours)", "GCN", "SAGE", "MLP\n(no graph)"]
    mse_1 = [0.000279, 0.001557, 0.005950, 0.003170]
    mse_5 = [0.003702, 0.004617, 0.003025, 0.001308]
    mse_10 = [0.006210, 0.016544, 0.005225, 0.009204]

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    x = np.arange(len(types))
    w = 0.22
    ax.bar(x - w, mse_1, w, label="1-step", color="#A8C9DD", edgecolor="black", lw=0.3)
    ax.bar(x, mse_5, w, label="5-step", color="#B5D8B0", edgecolor="black", lw=0.3)
    ax.bar(x + w, mse_10, w, label="10-step", color="#FFD9A8", edgecolor="black", lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels(types)
    ax.set_ylabel("Prediction MSE")
    ax.set_title("GNN Architecture Ablation (K8s Mesh)", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "fig_gnn_ablation.pdf"); fig.savefig(OUT / "fig_gnn_ablation.png")
    plt.close(fig)
    print("  fig_gnn_ablation")


def fig_planning_with_ppo():
    """Planning comparison including PPO baseline."""
    workloads = ["Constant", "Variable", "Bursty", "Flash"]
    agents = ["Random", "HPA", "PPO", "NetDream-Safe"]
    violations = {
        "Random":       [32.2, 21.4, 7.6, 47.0],
        "HPA":          [5.0, 6.2, 6.6, 15.2],
        "PPO":          [0.0, 0.0, 5.2, 0.0],
        "NetDream-Safe": [0.0, 0.0, 9.6, 30.8],
    }
    violation_std = {
        "Random":       [5.8, 5.9, 2.7, 24.0],
        "HPA":          [0.9, 1.0, 6.6, 1.0],
        "PPO":          [0.0, 0.0, 2.6, 0.0],
        "NetDream-Safe": [0.0, 0.0, 4.4, 3.7],
    }

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(workloads))
    w = 0.18
    for i, agent in enumerate(agents):
        offset = (i - len(agents)/2 + 0.5) * w
        ax.bar(x + offset, violations[agent], w, yerr=violation_std[agent], capsize=2,
               color=COLORS[agent], edgecolor="black", lw=0.3,
               error_kw={"lw": 0.5, "capthick": 0.5}, label=agent)
        for j, v in enumerate(violations[agent]):
            if v == 0:
                ax.text(x[j]+offset, 0.5, "0", ha="center", va="bottom", fontsize=5,
                        fontweight="bold", color=COLORS[agent])
    ax.set_xticks(x); ax.set_xticklabels(workloads)
    ax.set_ylabel("Constraint Violations")
    ax.set_title("K8s Mesh: Constraint Violations by Agent and Workload", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "fig_planning_violations.pdf"); fig.savefig(OUT / "fig_planning_violations.png")
    plt.close(fig)
    print("  fig_planning_violations")


def fig_grid2op_planning():
    """Grid2Op planning results."""
    agents = ["Do-Nothing", "NetDream-Safe"]
    survival = [81, 100]
    overloads = [0.4, 0.0]
    game_over = [20, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.2))
    colors = ["#999999", "#FFD9A8"]

    ax1.bar(agents, survival, color=colors, edgecolor="black", lw=0.3)
    ax1.set_ylabel("Survival Steps (max 100)")
    ax1.set_title("(a) Grid Survival", fontweight="bold")
    ax1.set_ylim(0, 110)
    for i, v in enumerate(survival):
        ax1.text(i, v+2, str(v), ha="center", fontsize=8, fontweight="bold")

    ax2.bar(agents, game_over, color=colors, edgecolor="black", lw=0.3)
    ax2.set_ylabel("Game Over Rate (%)")
    ax2.set_title("(b) Blackout Prevention", fontweight="bold")
    for i, v in enumerate(game_over):
        ax2.text(i, v+1, f"{v}%", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Power Grid: NetDream Safety Filter Prevents Blackouts", fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grid2op_planning.pdf"); fig.savefig(OUT / "fig_grid2op_planning.png")
    plt.close(fig)
    print("  fig_grid2op_planning")


def fig_per_feature():
    """Per-feature prediction error."""
    features = ["CPU", "Memory", "QPS", "Latency", "Error\nRate", "Replicas"]
    mse = [0.003527, 0.000109, 0.002075, 0.010186, 0.001216, 0.000055]
    colors = ["#A8C9DD"] * 6
    colors[3] = "#E89999"  # highlight latency as hardest (coral)

    fig, ax = plt.subplots(figsize=(4, 2.2))
    bars = ax.bar(features, mse, color=colors, edgecolor="black", lw=0.3)
    ax.set_ylabel("1-Step Prediction MSE")
    ax.set_title("Per-Feature Prediction Error (K8s)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("Hardest to\npredict", xy=(3, 0.010186), xytext=(4.2, 0.009),
                arrowprops=dict(arrowstyle="->", color="#C0392B"), fontsize=7, color="#C0392B")
    fig.savefig(OUT / "fig_per_feature.pdf"); fig.savefig(OUT / "fig_per_feature.png")
    plt.close(fig)
    print("  fig_per_feature")


def fig_cost():
    agents = ["HPA", "PPO", "NetDream-Safe"]
    costs = {
        "HPA":          [9.42, 9.49, 9.27, 11.06],
        "PPO":          [7.20, 7.20, 7.20, 7.20],
        "NetDream-Safe": [9.97, 9.96, 8.36, 9.16],
    }
    workloads = ["Constant", "Variable", "Bursty", "Flash"]
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    x = np.arange(len(workloads))
    w = 0.22
    for i, agent in enumerate(agents):
        offset = (i - len(agents)/2 + 0.5) * w
        ax.bar(x+offset, costs[agent], w, color=COLORS[agent], edgecolor="black", lw=0.3, label=agent)
    ax.set_xticks(x); ax.set_xticklabels(workloads)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Cost Comparison Across Agents", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "fig_cost_comparison.pdf"); fig.savefig(OUT / "fig_cost_comparison.png")
    plt.close(fig)
    print("  fig_cost_comparison")


if __name__ == "__main__":
    print("Generating updated figures...")
    fig_prediction()
    fig_ablation_full()
    fig_planning_with_ppo()
    fig_grid2op_planning()
    fig_per_feature()
    fig_cost()
    print(f"All saved to {OUT}/")
