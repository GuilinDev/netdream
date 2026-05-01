#!/usr/bin/env python3
"""Generate paper-ready figures from EKS cluster experiment logs.

Reads `results/cluster_summary.json` (produced by aggregate_cluster_results.py)
and emits:
    paper/figures/fig_planning_violations.pdf   — violations × (agent, workload)
    paper/figures/fig_cost_comparison.pdf       — cost × (agent, workload)
    paper/figures/fig_costsafety_tradeoff.pdf   — 2D Pareto scatter (avg)
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.linewidth": 0.6,
})

COLORS = {
    "hpa":              "#A8C9DD",  # light blue (baseline)
    "hpa-min3":         "#FFD9A8",  # light peach
    "hpa-min5":         "#F5C6C6",  # light coral
    "random":           "#D9D9D9",  # light gray
    "ppo":              "#B5D8B0",  # light mint
    "netdream-unsafe":  "#C9B4D8",  # light lavender
    "netdream":         "#5B8DBE",  # accent blue (Ours)
}
LABELS = {
    "hpa": "HPA", "hpa-min3": "HPA-min3", "hpa-min5": "HPA-min5",
    "random": "Random", "ppo": "PPO-500K",
    "netdream": "NetDream-Safe", "netdream-unsafe": "NetDream-Unsafe",
}
AGENTS = ["hpa", "hpa-min3", "hpa-min5", "random", "ppo",
          "netdream-unsafe", "netdream"]
WORKLOADS = ["constant", "variable", "bursty", "flash"]


def _load_summary(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    data = json.loads(path.read_text())
    idx: dict[tuple[str, str], dict] = {}
    for row in data["rows"]:
        idx[(row["agent"], row["workload"])] = row
    return idx


def fig_violations(idx, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(WORKLOADS))
    w = 0.11
    for i, agent in enumerate(AGENTS):
        offset = (i - (len(AGENTS) - 1) / 2) * w
        means = [idx.get((agent, wl), {}).get("violations_mean", float("nan"))
                 for wl in WORKLOADS]
        stds = [idx.get((agent, wl), {}).get("violations_std", 0.0)
                for wl in WORKLOADS]
        ax.bar(x + offset, means, w, yerr=stds, capsize=1.5,
               color=COLORS.get(agent, "#555555"), edgecolor="black", lw=0.3,
               error_kw={"lw": 0.5, "capthick": 0.5}, label=LABELS.get(agent, agent))
    ax.set_xticks(x); ax.set_xticklabels([w.capitalize() for w in WORKLOADS])
    ax.set_ylabel("SLO violations / episode")
    ax.set_title("EKS cluster: SLO violations", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, ncol=4, fontsize=6)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out); plt.close(fig)
    print(f"  {out.name}")


def fig_cost(idx, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(WORKLOADS))
    w = 0.11
    for i, agent in enumerate(AGENTS):
        offset = (i - (len(AGENTS) - 1) / 2) * w
        means = [idx.get((agent, wl), {}).get("cost_mean", float("nan"))
                 for wl in WORKLOADS]
        stds = [idx.get((agent, wl), {}).get("cost_std", 0.0)
                for wl in WORKLOADS]
        ax.bar(x + offset, means, w, yerr=stds, capsize=1.5,
               color=COLORS.get(agent, "#555555"), edgecolor="black", lw=0.3,
               error_kw={"lw": 0.5, "capthick": 0.5}, label=LABELS.get(agent, agent))
    ax.set_xticks(x); ax.set_xticklabels([w.capitalize() for w in WORKLOADS])
    ax.set_ylabel(r"Cost (replica-steps $\times$ \$0.01)")
    ax.set_title("EKS cluster: cost", fontweight="bold")
    ax.legend(framealpha=0.9, ncol=4, fontsize=6)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out); plt.close(fig)
    print(f"  {out.name}")


def fig_tradeoff(idx, out: pathlib.Path) -> None:
    """2D Pareto scatter: x=cost, y=violations (both averaged across workloads)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    for agent in AGENTS:
        costs = [idx[(agent, wl)]["cost_mean"] for wl in WORKLOADS
                 if (agent, wl) in idx]
        viols = [idx[(agent, wl)]["violations_mean"] for wl in WORKLOADS
                 if (agent, wl) in idx]
        if not costs:
            continue
        ax.scatter(np.mean(costs), np.mean(viols), s=80,
                   color=COLORS.get(agent, "#555555"),
                   edgecolor="black", lw=0.5,
                   label=LABELS.get(agent, agent), zorder=3)
    ax.set_xlabel("Avg. cost per episode")
    ax.set_ylabel("Avg. SLO violations per episode")
    ax.set_title("Pareto frontier: cost vs.\\ SLO violations",
                 fontweight="bold", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(out); plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    summary_path = pathlib.Path("results/cluster_summary.json")
    out_dir = pathlib.Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        print(f"error: {summary_path} not found; run aggregate_cluster_results.py first")
        return

    idx = _load_summary(summary_path)
    print(f"loaded {len(idx)} (agent, workload) cells")

    fig_violations(idx, out_dir / "fig_planning_violations.pdf")
    fig_cost(idx, out_dir / "fig_cost_comparison.pdf")
    fig_tradeoff(idx, out_dir / "fig_costsafety_tradeoff.pdf")


if __name__ == "__main__":
    main()
