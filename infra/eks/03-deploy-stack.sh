#!/bin/bash
# Deploy Online Boutique + kube-prometheus-stack on the active EKS cluster.
# Assumes eksctl just created the cluster and kubeconfig is pointing at it.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PATH="$HOME/bin:$PATH"

echo "================================================================"
echo "Phase A: deploy Online Boutique (11 services, x86 native)"
echo "================================================================"

kubectl create namespace boutique 2>&1 | grep -v "already exists" || true
# Upstream manifest is x86-only; EKS t3.medium nodes are x86 so it works natively.
kubectl -n boutique apply -f \
    https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml

# Remove the in-cluster loadgenerator — we use external Locust
kubectl -n boutique delete deployment loadgenerator --ignore-not-found
kubectl -n boutique delete service loadgenerator --ignore-not-found

echo "Waiting for all 11 services to be ready (up to 6 min)..."
kubectl -n boutique wait --for=condition=available --timeout=360s deployment --all
kubectl -n boutique get deployments

echo ""
echo "================================================================"
echo "Phase B: deploy kube-prometheus-stack via Helm"
echo "================================================================"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update prometheus-community 2>&1 | tail -3

kubectl create namespace observability 2>&1 | grep -v "already exists" || true
helm upgrade --install kube-prom-stack prometheus-community/kube-prometheus-stack \
    -n observability \
    --set prometheus.prometheusSpec.retention=1d \
    --set prometheus.prometheusSpec.resources.requests.memory=512Mi \
    --set prometheus.prometheusSpec.resources.limits.memory=1Gi \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --set grafana.enabled=false \
    --set alertmanager.enabled=false \
    --wait --timeout 5m

# Expose Prometheus service under the name our controller expects
kubectl -n observability get svc kube-prom-stack-kube-prome-prometheus

echo ""
echo "================================================================"
echo "Phase C: expose frontend to external traffic"
echo "================================================================"

# Online Boutique's frontend-external is type LoadBalancer — on EKS this
# provisions an AWS ELB automatically.
kubectl -n boutique get svc frontend-external
echo "Waiting for ELB to get an external hostname..."
for i in $(seq 1 60); do
    HOST=$(kubectl -n boutique get svc frontend-external -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
    if [[ -n "$HOST" ]]; then
        echo "ELB hostname: $HOST"
        echo "$HOST" > /tmp/netdream-frontend-host
        break
    fi
    sleep 5
done
if [[ -z "${HOST:-}" ]]; then
    echo "WARNING: ELB did not materialize; will use port-forward instead."
fi

echo ""
echo "================================================================"
echo "EKS stack deployed. Summary:"
echo "================================================================"
kubectl -n boutique get deployments
echo ""
kubectl -n observability get pods | head -5
