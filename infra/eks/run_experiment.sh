#!/bin/bash
# EKS version of run_experiment.sh: run a single (agent × workload × seed)
# episode. Assumes kubectl points at the EKS cluster.
set -euo pipefail

AGENT="${1:-netdream}"
WORKLOAD="${2:-constant}"
SEED="${3:-42}"
EPISODE_STEPS="${EPISODE_STEPS:-60}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PATH="$HOME/bin:$PATH"

# Activate venv so both locust and python are on PATH
# (must come before the Locust launch block below)
# shellcheck disable=SC1091
source "$REPO_ROOT/venv/bin/activate"

mkdir -p "$REPO_ROOT/results/cluster_runs"

# Kill any stale port-forwards from prior runs (avoids EADDRINUSE)
pkill -f "kubectl.*port-forward" 2>/dev/null || true
sleep 2

# --- Prometheus port-forward ---
kubectl -n observability port-forward svc/kube-prom-stack-kube-prome-prometheus 9090:9090 \
    > /tmp/prom-pf.log 2>&1 &
PROM_PID=$!
sleep 3

# --- Frontend target: prefer ELB hostname; fall back to port-forward ---
FRONTEND_HOST=$(cat /tmp/netdream-frontend-host 2>/dev/null || echo "")
FE_PID=""
if [[ -n "$FRONTEND_HOST" ]]; then
    FRONTEND_URL="http://$FRONTEND_HOST"
else
    kubectl -n boutique port-forward svc/frontend 8080:80 > /tmp/frontend-pf.log 2>&1 &
    FE_PID=$!
    sleep 3
    FRONTEND_URL="http://localhost:8080"
fi
echo "[run] frontend: $FRONTEND_URL"

# --- Locust (autostart: begins test immediately but keeps web UI for stats) ---
cd "$SCRIPT_DIR/../microk8s"  # reuse the locustfile
LOCUST_WORKLOAD="$WORKLOAD" \
    locust -f locustfile.py -H "$FRONTEND_URL" \
    --autostart --users 100 --spawn-rate 20 \
    --run-time "$((EPISODE_STEPS * 5 + 30))s" \
    --web-host 127.0.0.1 --web-port 8089 \
    --autoquit 10 \
    > /tmp/locust-$WORKLOAD.log 2>&1 &
LOCUST_PID=$!
sleep 15  # Locust warmup + first stats publish

cleanup() {
    kill "$PROM_PID" "$LOCUST_PID" 2>/dev/null || true
    [[ -n "$FE_PID" ]] && kill "$FE_PID" 2>/dev/null || true
    pkill -f "kubectl.*port-forward" 2>/dev/null || true
}
trap cleanup EXIT

# --- run the controller ---
cd "$REPO_ROOT"
source venv/bin/activate
python -m netdream.k8s_controller \
    --agent "$AGENT" \
    --workload "$WORKLOAD" \
    --seed "$SEED" \
    --namespace boutique \
    --prometheus-url http://localhost:9090 \
    --locust-url http://localhost:8089 \
    --episode-steps "$EPISODE_STEPS" \
    --cluster eks \
    --out results/cluster_runs

echo "[run] done: $AGENT $WORKLOAD seed=$SEED (EKS)"
