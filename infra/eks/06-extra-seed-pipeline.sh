#!/bin/bash
# Extra-seed pipeline: recreate EKS + deploy + run 1 more seed (seed=456)
# for all 7 agents × 4 workloads = 28 episodes + destroy.
# Purpose: provide a 3rd seed to firm up Pareto-dominance claim.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH="$HOME/bin:$PATH"

DESTROY() {
    echo "=== DESTROY PHASE ==="
    bash "$SCRIPT_DIR/99-destroy.sh" 2>&1 | tee /tmp/eks-destroy3.log | tail -5
}
trap DESTROY EXIT

echo "=== PHASE 1 / 4: CREATE CLUSTER ==="
echo yes | bash "$SCRIPT_DIR/02-create-cluster.sh" 2>&1 | tee /tmp/eks-create3.log | tail -3
if ! eksctl get cluster --name netdream-experiment --region us-west-2 &>/dev/null; then
    echo "!! create cluster failed — aborting"; exit 1
fi
echo "=== Phase 1 complete ==="

echo ""
echo "=== PHASE 2 / 4: DEPLOY STACK ==="
bash "$SCRIPT_DIR/03-deploy-stack.sh" 2>&1 | tee /tmp/eks-deploy3.log | tail -3
if ! kubectl -n boutique get deployments frontend &>/dev/null; then
    echo "!! deploy failed"; exit 1
fi
echo "=== Phase 2 complete ==="

echo ""
echo "=== PHASE 3 / 4: EXTRA-SEED SWEEP (seed=456, 7 agents × 4 workloads = 28 episodes) ==="
export SEEDS="456"
export AGENTS="hpa random ppo netdream netdream-unsafe hpa-min3 hpa-min5"
echo yes | bash "$SCRIPT_DIR/04-run-sweep.sh" 2>&1 | tee /tmp/eks-sweep3.log | grep -E "^=== \[|Sweep|complete"
echo "=== Phase 3 complete ==="

echo ""
echo "=== PHASE 4 / 4: DESTROY (via trap) ==="
