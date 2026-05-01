#!/bin/bash
# Create CloudWatch billing alarm that triggers at $10 total estimated charges.
# Must be in us-east-1 (billing metrics are only published there).
# Run this BEFORE creating the cluster.
set -euo pipefail

ALARM_NAME="netdream-billing-alarm"
THRESHOLD=10
EMAIL="${BILLING_ALARM_EMAIL:-}"

if [[ -z "$EMAIL" ]]; then
    echo "ERROR: set BILLING_ALARM_EMAIL env var to your email address"
    echo "  Example: BILLING_ALARM_EMAIL=you@example.com bash 01-billing-alarm.sh"
    exit 1
fi

echo "Setting up billing alarm: \$${THRESHOLD} threshold → ${EMAIL}"

# Enable billing metrics (once per account; idempotent)
echo "Enabling billing alerts on account..."
aws ce update-cost-allocation-tags-status \
    --cost-allocation-tags-status '[{"TagKey":"Project","Status":"Active"}]' \
    --region us-east-1 2>/dev/null || true

# Create SNS topic
TOPIC_ARN=$(aws sns create-topic \
    --name netdream-billing-alerts \
    --region us-east-1 \
    --query TopicArn --output text)
echo "  SNS topic: $TOPIC_ARN"

# Subscribe email to SNS topic
aws sns subscribe \
    --topic-arn "$TOPIC_ARN" \
    --protocol email \
    --notification-endpoint "$EMAIL" \
    --region us-east-1 > /dev/null
echo "  Email subscribed: $EMAIL"
echo "  >> CHECK YOUR EMAIL and confirm subscription (required)"

# Create the alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$ALARM_NAME" \
    --alarm-description "NetDream experiment cost guard" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 21600 \
    --threshold "$THRESHOLD" \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --dimensions Name=Currency,Value=USD \
    --alarm-actions "$TOPIC_ARN" \
    --region us-east-1

echo ""
echo "Billing alarm installed. You will receive an email if total AWS charges"
echo "exceed \$${THRESHOLD}. Confirm the SNS subscription email before proceeding."
