#!/bin/bash
# Chain: create -> deploy -> HPA-variant sweep -> destroy.
# Each phase logs to its own file; fails loudly but still runs destroy on the way out.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH="$HOME/bin:$PATH"

DESTROY() {
    echo "=== DESTROY PHASE ==="
    bash "$SCRIPT_DIR/99-destroy.sh" 2>&1 | tee /tmp/eks-destroy2.log | tail -10
}
trap DESTROY EXIT

echo "=== PHASE 1 / 4: CREATE CLUSTER ==="
echo yes | bash "$SCRIPT_DIR/02-create-cluster.sh" 2>&1 | tee /tmp/eks-create2.log | tail -3
if ! eksctl get cluster --name netdream-experiment --region us-west-2 &>/dev/null; then
    echo "!! create cluster failed — aborting"
    exit 1
fi
echo "=== Phase 1 complete ==="

echo ""
echo "=== PHASE 2 / 4: DEPLOY STACK ==="
bash "$SCRIPT_DIR/03-deploy-stack.sh" 2>&1 | tee /tmp/eks-deploy2.log | tail -3
if ! kubectl -n boutique get deployments frontend &>/dev/null; then
    echo "!! deploy failed — aborting before sweep"
    exit 1
fi
echo "=== Phase 2 complete ==="

echo ""
echo "=== PHASE 3 / 4: HPA-VARIANT SWEEP ==="
echo yes | bash "$SCRIPT_DIR/04b-run-sweep-hpa-variants.sh" 2>&1 | tee /tmp/eks-sweep2.log | grep -E "^=== \[|Sweep|complete"
echo "=== Phase 3 complete ==="

echo ""
echo "=== PHASE 4 / 4: DESTROY (handled by trap) ==="
# trap runs DESTROY on EXIT
