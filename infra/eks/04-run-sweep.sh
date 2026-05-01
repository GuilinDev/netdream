#!/bin/bash
# Full experiment sweep on the EKS cluster.
# Grid: AGENTS × WORKLOADS × SEEDS.
# Default: 5 agents × 4 workloads × 2 seeds = 40 episodes × 5 min ≈ 3.3 hours.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PATH="$HOME/bin:$PATH"

AGENTS=("${AGENTS:-hpa random ppo netdream netdream-unsafe}")
# Single-word split
AGENTS=($AGENTS)
WORKLOADS=(constant variable bursty flash)
SEEDS_STR="${SEEDS:-42 123}"
SEEDS=($SEEDS_STR)
EPISODE_STEPS="${EPISODE_STEPS:-60}"

TOTAL=$((${#AGENTS[@]} * ${#WORKLOADS[@]} * ${#SEEDS[@]}))
echo "================================================================"
echo "EKS experiment sweep: $TOTAL episodes, ${EPISODE_STEPS} steps each"
echo "  Agents:    ${AGENTS[*]}"
echo "  Workloads: ${WORKLOADS[*]}"
echo "  Seeds:     ${SEEDS[*]}"
echo "  Estimated time: $((TOTAL * (EPISODE_STEPS * 5 + 30) / 60)) min"
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
            # Brief cooldown: let cluster settle
            sleep 20
        done
    done
done
END=$(date +%s)
echo ""
echo "================================================================"
echo "Sweep complete in $((END-START)) seconds ($((I)) episodes)"
echo "Results written under: $REPO_ROOT/results/cluster_runs/"
echo ""
echo ">> IMMEDIATELY run bash $SCRIPT_DIR/99-destroy.sh <<"
echo "================================================================"
