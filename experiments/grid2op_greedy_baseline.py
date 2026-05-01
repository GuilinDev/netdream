#!/usr/bin/env python3
"""Grid2Op greedy baseline: reconnect tripped lines when grid stress is high."""

import sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from envs.grid2op_env import Grid2OpGraphEnv


def greedy_reconnect_agent(env, obs):
    """Greedy agent that reconnects any disconnected line.

    Strategy: scan all lines, if any is disconnected, reconnect it.
    Otherwise do nothing. This is a simple but competitive baseline
    for Grid2Op that doesn't require a world model.
    """
    action = np.ones(env.num_nodes, dtype=int)  # default: do nothing

    g2op_obs = env._obs
    for line_idx in range(env._n_lines):
        if not g2op_obs.line_status[line_idx]:
            # Find the substation this line connects to and set action to reconnect
            or_sub = g2op_obs.line_or_to_subid[line_idx]
            action[or_sub] = 2  # reconnect
            break  # one reconnection per step

    return action


def evaluate():
    seeds = [42, 123, 456, 789, 1024]
    env = Grid2OpGraphEnv(max_steps=100)

    for name, agent_fn in [
        ("Do-Nothing", lambda e, o: np.ones(e.num_nodes, dtype=int)),
        ("Greedy-Reconnect", greedy_reconnect_agent),
    ]:
        all_steps, all_overloads, all_game_over = [], [], []
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            steps, overloads = 0, 0
            for _ in range(100):
                action = agent_fn(env, obs)
                obs, r, term, trunc, info = env.step(action)
                steps += 1
                if info.get("max_rho", 0) > 1.0:
                    overloads += 1
                if term:
                    break
            all_steps.append(steps)
            all_overloads.append(overloads)
            all_game_over.append(1 if steps < 100 else 0)

        print(f"{name:20s} steps={np.mean(all_steps):.0f}±{np.std(all_steps):.0f}  "
              f"overloads={np.mean(all_overloads):.1f}±{np.std(all_overloads):.1f}  "
              f"game_over={np.mean(all_game_over)*100:.0f}%")


if __name__ == "__main__":
    print("Grid2Op Baseline Comparison:")
    evaluate()
