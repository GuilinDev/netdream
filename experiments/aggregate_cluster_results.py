#!/usr/bin/env python3
"""Aggregate per-episode cluster JSON logs into paper-ready statistics.

Reads all `results/cluster_runs/<cluster>_<agent>_<workload>_seed<N>.json` files
and emits:
  - A summary table (mean ± std) per (agent, workload) cell
  - A Pareto-ready scatter of (cost, violations) per episode
  - Time-series snapshots for the first seed of each cell

Cost and SLO-violations are derived from the logged episode rather than
simulator internals:
  cost       = sum over steps of (sum of replicas_after) * STEP_PRICE_PER_REP
  violations = sum over steps of (frontend_error_rate > 0.05 OR p95 > 500ms)
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

# Same accounting as the simulator's evaluate_planning.py:
STEP_PRICE_PER_REP = 0.01   # $ per replica-step (dimensionless; just for ranking)
SLO_LATENCY_MS = 500.0
ERROR_TOLERANCE = 0.05      # > 5% 5xx rate counts as a violation


def score_episode(log: dict) -> dict:
    """Return {cost, violations, steps, final_replicas_sum}."""
    cost = 0.0
    violations = 0
    for step in log["steps"]:
        reps = sum(step["replicas_after"])
        cost += reps * STEP_PRICE_PER_REP
        if step["frontend_latency_ms"] > SLO_LATENCY_MS or \
           step["frontend_error_rate"] > ERROR_TOLERANCE:
            violations += 1
    return {
        "cost": cost,
        "violations": violations,
        "steps": len(log["steps"]),
        "final_replicas_sum": (sum(log["steps"][-1]["replicas_after"])
                               if log["steps"] else 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/cluster_runs")
    ap.add_argument("--out", default="results/cluster_summary.json")
    ap.add_argument("--cluster", default="eks",
                    help="which cluster's logs to aggregate (prefix filter)")
    args = ap.parse_args()

    log_dir = pathlib.Path(args.dir)
    paths = sorted(log_dir.glob(f"{args.cluster}_*.json"))
    if not paths:
        print(f"No {args.cluster}_*.json under {log_dir}")
        return

    # Group by (agent, workload) → list of per-seed scores
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in paths:
        try:
            log = json.loads(p.read_text())
        except Exception as e:
            print(f"[warn] failed to parse {p.name}: {e}")
            continue
        agent = log.get("agent", "unknown")
        workload = log.get("workload", "unknown")
        score = score_episode(log)
        score["seed"] = log.get("seed")
        score["file"] = p.name
        cells[(agent, workload)].append(score)

    # Summary table
    rows = []
    print(f"{'agent':20} {'workload':10} {'seeds':5}  "
          f"{'cost(mean±std)':>18}  {'viol(mean±std)':>18}")
    print("-" * 80)
    for (agent, workload), seeds in sorted(cells.items()):
        costs = np.array([s["cost"] for s in seeds])
        viols = np.array([s["violations"] for s in seeds])
        row = {
            "agent": agent, "workload": workload, "n_seeds": len(seeds),
            "cost_mean": float(costs.mean()), "cost_std": float(costs.std()),
            "violations_mean": float(viols.mean()), "violations_std": float(viols.std()),
            "per_seed": seeds,
        }
        rows.append(row)
        print(f"{agent:20} {workload:10} {len(seeds):5d}  "
              f"{costs.mean():7.2f} ± {costs.std():4.2f}   "
              f"{viols.mean():7.2f} ± {viols.std():4.2f}")

    pathlib.Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
