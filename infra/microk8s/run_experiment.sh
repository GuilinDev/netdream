#!/bin/bash
# Run a single (agent × workload × seed) experiment on the currently-active
# Kubernetes cluster (MicroK8s or EKS). Handles: Prometheus port-forward,
# Locust foreground + Locust stats endpoint, NetDream controller, and
# teardown of all three on exit.
set -euo pipefail

AGENT="${1:-netdream}"
WORKLOAD="${2:-constant}"
SEED="${3:-42}"
CLUSTER="${4:-microk8s}"
EPISODE_STEPS="${EPISODE_STEPS:-60}"

# Kubectl wrapper (MicroK8s vs real kubectl)
if [[ "$CLUSTER" == "microk8s" ]]; then
    KCTL="sudo -n /snap/bin/microk8s kubectl"
else
    KCTL="kubectl"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Activate venv (locust + python + controller all live here)
# shellcheck disable=SC1091
source "$REPO_ROOT/venv/bin/activate"

mkdir -p "$REPO_ROOT/results/cluster_runs"

# --- port-forward Prometheus ---
$KCTL -n observability port-forward svc/kube-prom-stack-kube-prome-prometheus 9090:9090 \
    > /tmp/prom-pf.log 2>&1 &
PROM_PID=$!
sleep 3

# --- port-forward Online Boutique frontend ---
$KCTL -n boutique port-forward svc/frontend-external 8080:80 \
    > /tmp/frontend-pf.log 2>&1 &
FE_PID=$!
sleep 2

# --- launch Locust headless ---
cd "$SCRIPT_DIR"
LOCUST_WORKLOAD="$WORKLOAD" \
    locust -f locustfile.py -H http://localhost:8080 \
    --autostart --users 100 --spawn-rate 20 \
    --run-time "$((EPISODE_STEPS * 5 + 30))s" \
    --web-host 127.0.0.1 --web-port 8089 \
    --autoquit 10 \
    > /tmp/locust-$WORKLOAD.log 2>&1 &
LOCUST_PID=$!
sleep 10  # give Locust time to warm up and publish stats

cleanup() {
    echo "[run_experiment] cleaning up..."
    kill "$PROM_PID" "$FE_PID" "$LOCUST_PID" 2>/dev/null || true
    wait "$PROM_PID" "$FE_PID" "$LOCUST_PID" 2>/dev/null || true
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
    --cluster "$CLUSTER" \
    --out results/cluster_runs

echo "[run_experiment] done: $AGENT $WORKLOAD seed=$SEED cluster=$CLUSTER"
