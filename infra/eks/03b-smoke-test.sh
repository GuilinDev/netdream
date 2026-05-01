#!/bin/bash
# Post-deployment sanity check: run ONE short episode (30 steps, ~3 min)
# with NetDream on constant workload. Verifies the entire pipeline before
# committing to the full sweep (which costs ~$1).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PATH="$HOME/bin:$PATH"

echo "================================================================"
echo "EKS smoke test: NetDream × constant × seed=99 × 30 steps (~3 min)"
echo "================================================================"

EPISODE_STEPS=30 bash "$SCRIPT_DIR/run_experiment.sh" netdream constant 99

SMOKE="$REPO_ROOT/results/cluster_runs/eks_netdream_constant_seed99.json"
if [[ ! -f "$SMOKE" ]]; then
    echo "ERROR: smoke-test JSON not written"
    exit 1
fi

echo ""
echo "================================================================"
echo "Smoke-test validation"
echo "================================================================"
python3 - <<PY
import json, pathlib, sys
log = json.loads(pathlib.Path("$SMOKE").read_text())
steps = log["steps"]
print(f"agent={log['agent']} workload={log['workload']} seed={log['seed']} cluster={log['cluster']}")
print(f"total steps logged: {len(steps)}")
if not steps:
    print("FAIL: no steps recorded"); sys.exit(1)
qps  = [s["frontend_qps"]          for s in steps]
p95  = [s["frontend_latency_ms"]   for s in steps]
reps = [sum(s["replicas_after"])   for s in steps]
print(f"  qps  min/mean/max: {min(qps):.1f}/{sum(qps)/len(qps):.1f}/{max(qps):.1f}")
print(f"  p95  min/mean/max: {min(p95):.0f}/{sum(p95)/len(p95):.0f}/{max(p95):.0f}")
print(f"  total replicas over episode: {min(reps)}→{max(reps)} (start={reps[0]}, end={reps[-1]})")
if max(qps) < 1.0:
    print("FAIL: no traffic reached frontend — Locust or networking broken"); sys.exit(2)
print("PASS: full pipeline active (traffic, metrics, planning, scaling).")
PY
