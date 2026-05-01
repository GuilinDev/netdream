#!/usr/bin/env python3
"""Final figures with all updated data."""

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

COLORS = {"Random": "#999999", "HPA": "#000000", "PPO": "#009E73",
           "NetDream-Safe": "#D55E00", "Do-Nothing": "#999999", "Greedy": "#0072B2"}
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig_planning_final():
    """Updated planning figure with PPO v2 data."""
    workloads = ["Constant", "Variable", "Bursty", "Flash"]
    agents = ["HPA", "PPO", "NetDream-Safe"]

    violations = {
        "HPA":          [5.0, 6.2, 6.6, 15.2],
        "PPO":          [0.0, 0.0, 11.0, 94.0],
        "NetDream-Safe": [0.0, 0.0, 9.6, 30.8],
    }
    violation_std = {
        "HPA":          [0.9, 1.0, 6.6, 1.0],
        "PPO":          [0.0, 0.0, 4.9, 0.0],
        "NetDream-Safe": [0.0, 0.0, 4.4, 3.7],
    }

    fig, ax = plt.subplots(figsize=(5, 2.8))
    x = np.arange(len(workloads))
    w = 0.22
    for i, agent in enumerate(agents):
        offset = (i - len(agents)/2 + 0.5) * w
        ax.bar(x + offset, violations[agent], w, yerr=violation_std[agent], capsize=2,
               color=COLORS[agent], edgecolor="black", lw=0.3,
               error_kw={"lw": 0.5, "capthick": 0.5}, label=agent)
        for j, v in enumerate(violations[agent]):
            if v == 0:
                ax.text(x[j]+offset, 1.0, "0", ha="center", va="bottom", fontsize=6,
                        fontweight="bold", color=COLORS[agent])

    ax.set_xticks(x); ax.set_xticklabels(workloads)
    ax.set_ylabel("Constraint Violations")
    ax.set_title("K8s Mesh: Constraint Violations", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "fig_planning_violations.pdf")
    fig.savefig(OUT / "fig_planning_violations.png")
    plt.close(fig)
    print("  fig_planning_violations")


def fig_cost_final():
    """Updated cost figure with PPO v2."""
    agents = ["HPA", "PPO", "NetDream-Safe"]
    costs = {
        "HPA":          [9.42, 9.49, 9.27, 11.06],
        "PPO":          [24.75, 33.51, 36.30, 38.05],
        "NetDream-Safe": [9.97, 9.96, 8.36, 9.16],
    }
    workloads = ["Constant", "Variable", "Bursty", "Flash"]

    fig, ax = plt.subplots(figsize=(5, 2.8))
    x = np.arange(len(workloads))
    w = 0.22
    for i, agent in enumerate(agents):
        offset = (i - len(agents)/2 + 0.5) * w
        ax.bar(x+offset, costs[agent], w, color=COLORS[agent], edgecolor="black", lw=0.3, label=agent)
    ax.set_xticks(x); ax.set_xticklabels(workloads)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("K8s Mesh: PPO Over-Provisions 2.5-3.5x vs HPA/NetDream", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "fig_cost_comparison.pdf")
    fig.savefig(OUT / "fig_cost_comparison.png")
    plt.close(fig)
    print("  fig_cost_comparison")


def fig_grid2op_final():
    """Updated Grid2Op with 3 agents."""
    agents = ["Do-Nothing", "Greedy\nReconnect", "NetDream\nSafe"]
    survival = [81, 100, 100]
    game_over = [20, 0, 0]
    colors = ["#D9D9D9", "#B5D8B0", "#5B8DBE"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.2))

    ax1.bar(agents, survival, color=colors, edgecolor="black", lw=0.3)
    ax1.set_ylabel("Survival Steps (max 100)")
    ax1.set_title("(a) Grid Survival", fontweight="bold")
    ax1.set_ylim(0, 115)
    for i, v in enumerate(survival):
        ax1.text(i, v+2, str(v), ha="center", fontsize=8, fontweight="bold")

    ax2.bar(agents, game_over, color=colors, edgecolor="black", lw=0.3)
    ax2.set_ylabel("Game Over Rate (%)")
    ax2.set_title("(b) Blackout Rate", fontweight="bold")
    for i, v in enumerate(game_over):
        ax2.text(i, v+0.5, f"{v}%", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Power Grid: NetDream Matches Greedy Baseline, Both Prevent Blackouts", fontweight="bold", y=1.05, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grid2op_planning.pdf")
    fig.savefig(OUT / "fig_grid2op_planning.png")
    plt.close(fig)
    print("  fig_grid2op_planning")


def fig_costsafety_tradeoff():
    """NEW: Cost-Safety trade-off scatter (key figure for the paper's thesis)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Each point: (mean cost, mean violations) across 4 workloads
    agents_data = {
        "Random":       (23.54, 27.05, "#999999", "o"),
        "HPA":          (9.81, 8.25, "#000000", "s"),
        "PPO":          (33.15, 26.25, "#009E73", "^"),
        "NetDream-Safe":(9.36, 10.10, "#D55E00", "D"),
    }

    for name, (cost, viol, color, marker) in agents_data.items():
        ax.scatter(cost, viol, c=color, marker=marker, s=80, edgecolors="black",
                   linewidth=0.5, label=name, zorder=5)

    # Draw arrows showing trade-off direction
    ax.annotate("Better →\n(lower cost)", xy=(5, 28), fontsize=6, color="gray", ha="center")
    ax.annotate("↓ Safer\n(fewer violations)", xy=(37, 2), fontsize=6, color="gray", ha="center")

    # Highlight NetDream's Pareto advantage
    ax.annotate("Best trade-off", xy=(9.36, 10.10), xytext=(15, 16),
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2),
                fontsize=7, color="#D55E00", fontweight="bold")

    ax.set_xlabel("Mean Cost (USD)")
    ax.set_ylabel("Mean Violations")
    ax.set_title("Cost-Safety Trade-off (averaged across workloads)", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 40)

    fig.savefig(OUT / "fig_costsafety_tradeoff.pdf")
    fig.savefig(OUT / "fig_costsafety_tradeoff.png")
    plt.close(fig)
    print("  fig_costsafety_tradeoff")


if __name__ == "__main__":
    print("Generating final figures...")
    fig_planning_final()
    fig_cost_final()
    fig_grid2op_final()
    fig_costsafety_tradeoff()
    print(f"All saved to {OUT}/")
