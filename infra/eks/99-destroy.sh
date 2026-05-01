#!/bin/bash
# Destroy the EKS cluster and all associated AWS resources.
# Idempotent: safe to run multiple times.
# Run this IMMEDIATELY after experiments finish.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="$SCRIPT_DIR/cluster-config.yaml"

echo "================================================================"
echo "Destroying EKS cluster: netdream-experiment"
echo "================================================================"

if ! eksctl get cluster --name netdream-experiment --region us-west-2 &>/dev/null; then
    echo "Cluster does not exist. Nothing to destroy."
    exit 0
fi

eksctl delete cluster -f "$CONFIG" --disable-nodegroup-eviction --wait

echo ""
echo "================================================================"
echo "Verification: listing any remaining netdream-tagged resources..."
echo "================================================================"

# Check for orphaned EC2 instances
aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=netdream-paper2" "Name=instance-state-name,Values=running,pending,stopping" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' \
    --output table --region us-west-2

# Check for orphaned EBS volumes
aws ec2 describe-volumes \
    --filters "Name=tag:Project,Values=netdream-paper2" \
    --query 'Volumes[].[VolumeId,State]' \
    --output table --region us-west-2

# Check for orphaned load balancers
aws elbv2 describe-load-balancers --region us-west-2 \
    --query "LoadBalancers[?contains(LoadBalancerName, 'netdream')].[LoadBalancerName,State.Code]" \
    --output table

echo ""
echo "If any rows above are non-empty, manually clean them up to stop billing."
echo "Current estimated charges (next update in 6-12 hours):"
aws cloudwatch get-metric-statistics \
    --namespace AWS/Billing --metric-name EstimatedCharges \
    --dimensions Name=Currency,Value=USD \
    --start-time "$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S)" \
    --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
    --period 21600 --statistics Maximum \
    --region us-west-2 --output text --query 'Datapoints[-1].Maximum' 2>/dev/null || echo "  (not yet available)"
