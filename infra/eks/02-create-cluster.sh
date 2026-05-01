#!/bin/bash
# Create the EKS cluster via eksctl. Expected duration: 15-20 minutes.
# COST: starts billing the moment this succeeds (~$0.20/hour).
# Always run 99-destroy.sh when finished.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="$SCRIPT_DIR/cluster-config.yaml"

echo "================================================================"
echo "Creating EKS cluster: netdream-experiment"
echo "Region: us-east-1"
echo "Config: $CONFIG"
echo "Expected cost: ~\$0.20/hour while running"
echo "================================================================"
echo ""
read -p "Type 'yes' to proceed: " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 1
fi

START=$(date +%s)
eksctl create cluster -f "$CONFIG"
END=$(date +%s)

echo ""
echo "================================================================"
echo "Cluster created in $((END-START)) seconds"
echo "================================================================"

# Verify kubectl access
kubectl get nodes
kubectl get ns

# Record creation time for autoteardown cron
date -u +%s > /tmp/netdream-cluster-created-at
echo ""
echo "Cluster creation timestamp recorded at /tmp/netdream-cluster-created-at"
echo "Remember: run 99-destroy.sh when finished. Cost alarm at \$10."
