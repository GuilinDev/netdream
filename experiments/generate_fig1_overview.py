#!/usr/bin/env python3
"""Generate Figure 1: Architecture overview — the most important figure.

Every NeurIPS best paper has a multi-panel Figure 1 that visually summarizes
the entire contribution. Ours should show:
  Left: Two infrastructure domains as graphs
  Center: GNN dynamics model architecture
  Right: Planning + Safety filter pipeline with key result
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

OUT = Path("paper/figures")


def draw_graph(ax, nodes, edges, node_colors, title, node_labels=None):
    """Draw a small graph with labeled nodes."""
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y2], "k-", lw=0.8, alpha=0.4, zorder=1)
    for i, ((x, y), color) in enumerate(zip(nodes, node_colors)):
        circle = plt.Circle((x, y), 0.12, facecolor=color, edgecolor="black",
                            lw=0.8, zorder=2)
        ax.add_patch(circle)
        if node_labels:
            ax.text(x, y, node_labels[i], ha="center", va="center", fontsize=5,
                    fontweight="bold", zorder=3)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.3, 1.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.axis("off")


def fig1_overview():
    """Create the multi-panel overview figure."""
    fig = plt.figure(figsize=(6.75, 2.8))

    # Layout: 3 panels
    # [Left: Domains] [Center: Architecture] [Right: Results]
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.6, 1.2], wspace=0.15)

    # ─── LEFT: Two infrastructure domains ───
    gs_left = gs[0].subgridspec(2, 1, hspace=0.4)

    # K8s mesh
    ax_k8s = fig.add_subplot(gs_left[0])
    k8s_nodes = [(0.3, 1.5), (0.8, 0.8), (1.5, 1.2), (1.8, 0.5), (0.5, 0.2),
                 (1.2, 0.0)]
    k8s_edges = [
        (k8s_nodes[0], k8s_nodes[1]), (k8s_nodes[0], k8s_nodes[2]),
        (k8s_nodes[0], k8s_nodes[3]), (k8s_nodes[1], k8s_nodes[4]),
        (k8s_nodes[2], k8s_nodes[3]), (k8s_nodes[2], k8s_nodes[5]),
        (k8s_nodes[3], k8s_nodes[5]),
    ]
    k8s_colors = ["#4CAF50", "#2196F3", "#2196F3", "#FF9800", "#2196F3", "#FF9800"]
    draw_graph(ax_k8s, k8s_nodes, k8s_edges, k8s_colors,
               "K8s Mesh (11 svc)", ["F", "C", "P", "S", "R", "D"])

    # Power grid
    ax_grid = fig.add_subplot(gs_left[1])
    grid_nodes = [(0.2, 1.3), (0.8, 1.5), (1.5, 1.3), (0.5, 0.6),
                  (1.2, 0.6), (1.8, 0.8), (0.3, 0.0), (1.0, 0.0)]
    grid_edges = [
        (grid_nodes[0], grid_nodes[1]), (grid_nodes[1], grid_nodes[2]),
        (grid_nodes[0], grid_nodes[3]), (grid_nodes[1], grid_nodes[4]),
        (grid_nodes[2], grid_nodes[5]), (grid_nodes[3], grid_nodes[4]),
        (grid_nodes[3], grid_nodes[6]), (grid_nodes[4], grid_nodes[7]),
        (grid_nodes[5], grid_nodes[4]),
    ]
    grid_colors = ["#D32F2F", "#D32F2F", "#FFC107", "#FFC107",
                   "#D32F2F", "#FFC107", "#FFC107", "#D32F2F"]
    draw_graph(ax_grid, grid_nodes, grid_edges, grid_colors,
               "Power Grid (14 bus)", ["G", "G", "L", "L", "G", "L", "L", "G"])

    # ─── CENTER: GNN Architecture ───
    ax_arch = fig.add_subplot(gs[1])
    ax_arch.axis("off")
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 5)

    # Pipeline boxes
    boxes = [
        (1, 4.0, 1.6, 0.7, "Node\nEncoder", "#E3F2FD"),
        (3.5, 4.0, 1.8, 0.7, "GAT Message\nPassing (×3)", "#BBDEFB"),
        (6.2, 4.5, 1.5, 0.5, "Dynamics\nΔx̂", "#C8E6C9"),
        (6.2, 3.5, 1.5, 0.5, "Reward\nr̂", "#FFF9C4"),
        (6.2, 2.5, 1.5, 0.5, "Constraint\nĉ", "#FFCDD2"),
        (1.5, 1.5, 2.5, 0.7, "Imagination\nRollout (H steps)", "#E1BEE7"),
        (5.5, 1.5, 2.5, 0.7, "Safety Filter\n+ MPC Select", "#FFCDD2"),
    ]

    for x, y, w, h, text, color in boxes:
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor="black", lw=0.6)
        ax_arch.add_patch(rect)
        ax_arch.text(x, y, text, ha="center", va="center", fontsize=5.5,
                     fontweight="bold")

    # Arrows
    arrows = [
        (1.8, 4.0, 2.4, 4.0),   # encoder → GAT
        (4.4, 4.0, 5.3, 4.5),   # GAT → dynamics
        (4.4, 4.0, 5.3, 3.5),   # GAT → reward
        (4.4, 4.0, 5.3, 2.5),   # GAT → constraint
        (6.2, 2.1, 6.2, 1.9),   # constraint → safety
        (2.75, 2.3, 2.75, 3.3), # rollout → GAT (feedback)
        (3.8, 1.5, 4.2, 1.5),   # rollout → safety
    ]
    for x1, y1, x2, y2 in arrows:
        ax_arch.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # Labels
    ax_arch.text(0.2, 4.0, "x_t, a_t", fontsize=6, fontstyle="italic")
    ax_arch.text(8.5, 1.5, "a*", fontsize=7, fontstyle="italic", fontweight="bold",
                 color="#D32F2F")
    ax_arch.annotate("", xy=(8.8, 1.5), xytext=(8.0, 1.5),
                     arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2))

    ax_arch.set_title("NetDream Architecture", fontsize=9, fontweight="bold")

    # ─── RIGHT: Key Results ───
    ax_res = fig.add_subplot(gs[2])

    # Cost-safety scatter (simplified)
    agents = {
        "HPA":          (9.81, 8.25, "#000000", "s"),
        "PPO":          (33.15, 26.25, "#009E73", "^"),
        "NetDream\nSafe": (9.36, 10.10, "#D55E00", "D"),
    }
    for name, (cost, viol, color, marker) in agents.items():
        ax_res.scatter(cost, viol, c=color, marker=marker, s=60,
                       edgecolors="black", lw=0.5, zorder=5)
        offset = (1, -3) if name != "PPO" else (-8, 2)
        ax_res.annotate(name, (cost, viol), textcoords="offset points",
                        xytext=offset, fontsize=6, fontweight="bold", color=color)

    # Highlight Pareto position
    ax_res.annotate("Pareto\noptimal", xy=(9.36, 10.10), xytext=(18, 2),
                    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1),
                    fontsize=6, color="#D55E00", fontstyle="italic")

    ax_res.set_xlabel("Cost (USD)", fontsize=7)
    ax_res.set_ylabel("Violations", fontsize=7)
    ax_res.set_title("Cost-Safety Trade-off", fontsize=9, fontweight="bold")
    ax_res.set_xlim(0, 40)
    ax_res.set_ylim(0, 32)
    ax_res.grid(True, alpha=0.2)

    fig.suptitle("", y=1.02)
    fig.savefig(OUT / "fig_overview.pdf")
    fig.savefig(OUT / "fig_overview.png")
    plt.close(fig)
    print("  Saved: fig_overview")


if __name__ == "__main__":
    fig1_overview()
