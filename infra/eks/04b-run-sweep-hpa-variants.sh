#!/bin/bash
# Focused sweep: HPA-over-provisioned variants only.
# Complements the main 40-episode sweep with 16 extra episodes so that
# reviewers cannot dismiss our Pareto claim with "just raise HPA's floor".
#
# Grid: 2 agents × 4 workloads × 2 seeds = 16 episodes × 5 min ≈ 80 min.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PATH="$HOME/bin:$PATH"

AGENTS=(hpa-min3 hpa-min5)
WORKLOADS=(constant variable bursty flash)
SEEDS=(42 123)
EPISODE_STEPS="${EPISODE_STEPS:-60}"

TOTAL=$((${#AGENTS[@]} * ${#WORKLOADS[@]} * ${#SEEDS[@]}))
echo "================================================================"
echo "HPA over-provisioned sweep: $TOTAL episodes"
echo "  Agents:    ${AGENTS[*]}"
echo "  Workloads: ${WORKLOADS[*]}"
echo "  Seeds:     ${SEEDS[*]}"
echo "  ETA: $((TOTAL * (EPISODE_STEPS * 5 + 30) / 60)) min"
echo "================================================================"
read -p "Type 'yes' to proceed: " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 1
fi

START=$(date +%s)
I=0
for agent in "${AGENTS[@]}"; do
    for wl in "${WORKLOADS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            I=$((I + 1))
            echo ""
            echo "=== [$I/$TOTAL] agent=$agent workload=$wl seed=$seed ==="
            bash "$SCRIPT_DIR/run_experiment.sh" "$agent" "$wl" "$seed" \
                || echo "  (episode failed; continuing)"
            sleep 20
        done
    done
done
END=$(date +%s)
echo ""
echo "================================================================"
echo "HPA-variant sweep complete in $((END-START)) seconds"
echo ">> IMMEDIATELY run bash $SCRIPT_DIR/99-destroy.sh <<"
echo "================================================================"
