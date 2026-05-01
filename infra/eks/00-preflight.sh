#!/bin/bash
# Preflight: verify AWS credentials, IAM permissions, cost alarm setup
# Run this BEFORE any resource-creating command.
set -euo pipefail

echo "================================================================"
echo "NetDream EKS Preflight Check"
echo "================================================================"

echo -n "[1/6] aws CLI installed... "
if ! command -v aws &>/dev/null; then
    echo "FAIL"
    echo "  Install aws CLI first"
    exit 1
fi
echo "OK ($(aws --version 2>&1 | head -1))"

echo -n "[2/6] eksctl installed... "
if ! command -v eksctl &>/dev/null; then
    echo "FAIL"
    exit 1
fi
echo "OK ($(eksctl version))"

echo -n "[3/6] AWS credentials configured... "
if ! aws sts get-caller-identity &>/dev/null; then
    echo "FAIL"
    echo "  Run: aws configure"
    echo "  Need: Access Key ID, Secret Access Key, region=us-west-2"
    exit 1
fi
CALLER=$(aws sts get-caller-identity --output json)
ACCOUNT=$(echo "$CALLER" | grep -oP '"Account": "\K[^"]+')
ARN=$(echo "$CALLER" | grep -oP '"Arn": "\K[^"]+')
echo "OK"
echo "  Account: $ACCOUNT"
echo "  Identity: $ARN"

# Safety: refuse to run as root user (prevent accidental root usage)
if echo "$ARN" | grep -q ":root$"; then
    echo "  ERROR: You are using AWS root credentials."
    echo "  Create a dedicated IAM user first. Do not run experiments as root."
    exit 1
fi

echo -n "[4/6] Region is us-west-2... "
REGION=$(aws configure get region 2>/dev/null || echo "")
if [[ "$REGION" != "us-west-2" ]]; then
    echo "WARN: current region is '$REGION', expected us-west-2"
    echo "  Set with: aws configure set region us-west-2"
fi
echo "OK"

echo -n "[5/6] No existing netdream-experiment cluster... "
if eksctl get cluster --name netdream-experiment --region us-west-2 &>/dev/null; then
    echo "WARN: cluster already exists!"
    echo "  Either use the existing cluster or destroy it first with:"
    echo "  bash 99-destroy.sh"
else
    echo "OK"
fi

echo -n "[6/6] Billing alarm configured... "
if aws cloudwatch describe-alarms --alarm-names "netdream-billing-alarm" --region us-west-2 \
   --query 'MetricAlarms[0].AlarmName' --output text 2>/dev/null | grep -q "netdream-billing-alarm"; then
    echo "OK"
else
    echo "MISSING"
    echo "  Set up with: bash 01-billing-alarm.sh"
fi

echo "================================================================"
echo "Preflight PASSED. You may proceed to resource creation."
echo "================================================================"
