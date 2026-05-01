# NetDream EKS Infrastructure

One-shot AWS EKS cluster for paper 2 validation. Designed to spin up, run
experiments, and tear down in a single 4-8 hour window. Estimated cost: **~$2**.

## Prerequisites

1. AWS account with billing enabled
2. Dedicated IAM user `netdream-experiment` with `AdministratorAccess`
   (do not use root credentials)
3. `aws configure` completed with that user's access keys, region `us-east-1`
4. `eksctl`, `aws`, `kubectl` installed (see `~/bin`)
5. Email address for the billing alarm

## Run order

```bash
# Step 0: verify environment
bash 00-preflight.sh

# Step 1: set up $10 billing alarm (once per account)
BILLING_ALARM_EMAIL=you@example.com bash 01-billing-alarm.sh
# >> confirm SNS subscription email before proceeding

# Step 2: create cluster (15-20 min, asks for confirmation)
bash 02-create-cluster.sh

# Step 3: deploy Online Boutique + run experiments
# (handled by separate experiment scripts — not in this dir)

# Step 99: destroy EVERYTHING (idempotent)
bash 99-destroy.sh
```

## Cost breakdown (estimated)

| Resource | Rate | 6-hour run | 24-hour run |
|---|---|---|---|
| EKS control plane | $0.10/hr | $0.60 | $2.40 |
| 3× t3.medium workers | $0.125/hr combined | $0.75 | $3.00 |
| 3× 30GB gp3 EBS | negligible | ~$0.04 | ~$0.15 |
| Data transfer | minimal | ~$0.10 | ~$0.20 |
| **Total** | | **~$1.50** | **~$5.75** |

## Safety features

- `00-preflight.sh` refuses to run as AWS root user
- `01-billing-alarm.sh` sets a $10 threshold alarm with email notification
- `02-create-cluster.sh` requires typed "yes" confirmation
- `99-destroy.sh` is idempotent and verifies orphaned resources
- All resources tagged `Project=netdream-paper2` for identification
- `MaxCostUSD=10` tag documents the intended budget
