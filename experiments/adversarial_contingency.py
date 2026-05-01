#!/usr/bin/env python3
"""N-2 adversarial contingency: disconnect 2 lines simultaneously.

On the 14-bus grid, greedy-reconnect handles single line trips (N-1).
But with N-2 (2 simultaneous trips), greedy can only reconnect one per step,
potentially choosing the wrong one. The world model can predict which
reconnection is more critical.
"""

import sys, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from netdream.dynamics_model import GNNDynamicsModel
from netdream.fast_planner import FastRandomShootingPlanner
from envs.grid2op_env import Grid2OpGraphEnv


def simulate_n2_contingency(env, agent_fn, n_disconnects=2, num_episodes=20, max_steps=50):
    """Simulate N-2 contingency: force-disconnect n_disconnects lines at step 5."""
    all_survival = []
    all_overloads = []

    for seed in range(num_episodes):
        obs, _ = env.reset(seed=seed + 100)
        steps = 0
        overloads = 0
        game_over = False

        for t in range(max_steps):
            # At step 5, force-disconnect n_disconnects random lines
            if t == 5 and not game_over:
                g2op_obs = env._obs
                connected_lines = np.where(g2op_obs.line_status)[0]
                if len(connected_lines) >= n_disconnects:
                    rng = np.random.default_rng(seed + 1000)
                    lines_to_trip = rng.choice(connected_lines, size=n_disconnects, replace=False)
                    # Force disconnect via Grid2Op action
                    for line_idx in lines_to_trip:
                        disc_action = env._env.action_space({})
                        disc_action.line_set_status = [(int(line_idx), -1)]
                        new_obs, _, done_disc, _ = env._env.step(disc_action)
                        env._obs = new_obs
                        if done_disc:
                            game_over = True
                            break
                    if not game_over:
                        obs = env._obs_to_graph(env._obs)

            # Agent acts
            action = agent_fn(env, obs)
            obs, r, term, trunc, info = env.step(action)
            steps += 1

            if info.get("max_rho", 0) > 1.0:
                overloads += 1
            if term:
                game_over = True
                break

        all_survival.append(steps)
        all_overloads.append(overloads)

    return {
        "survival_mean": float(np.mean(all_survival)),
        "survival_std": float(np.std(all_survival)),
        "overloads_mean": float(np.mean(all_overloads)),
        "game_over_pct": float(np.mean([s < max_steps for s in all_survival]) * 100),
    }


def do_nothing_agent(env, obs):
    return np.ones(env.num_nodes, dtype=int)


def greedy_reconnect_agent(env, obs):
    action = np.ones(env.num_nodes, dtype=int)
    g2op_obs = env._obs
    for line_idx in range(env._n_lines):
        if not g2op_obs.line_status[line_idx]:
            or_sub = g2op_obs.line_or_to_subid[line_idx]
            action[or_sub] = 2
            break  # only reconnect ONE line per step
    return action


def make_netdream_agent(model, env):
    planner = FastRandomShootingPlanner(
        model=model, horizon=3, num_candidates=64,
        action_dim=3, num_nodes=env.num_nodes, use_safety=True,
    )
    def agent_fn(env_inner, obs):
        return planner.plan(obs.data.x.numpy(), obs.data.edge_index.numpy())
    return agent_fn


if __name__ == "__main__":
    print("=" * 60)
    print("  N-2 Adversarial Contingency Experiment")
    print("=" * 60)

    model = GNNDynamicsModel(
        node_feat_dim=7, action_dim=3, hidden_dim=128,
        num_gnn_layers=3, num_constraints=1, gnn_type="gat",
    )
    model.load_state_dict(torch.load("models/grid2op_best.pt", weights_only=True))
    model.eval()

    env = Grid2OpGraphEnv(max_steps=50)

    for n_disc in [1, 2, 3]:
        print(f"\n  N-{n_disc} Contingency (force-disconnect {n_disc} lines at step 5):")

        for name, agent_fn in [
            ("Do-Nothing", do_nothing_agent),
            ("Greedy-Reconnect", greedy_reconnect_agent),
            ("NetDream-Safe", make_netdream_agent(model, env)),
        ]:
            r = simulate_n2_contingency(env, agent_fn, n_disconnects=n_disc, num_episodes=20, max_steps=50)
            print(f"    {name:<20} survival={r['survival_mean']:.0f}±{r['survival_std']:.0f}  "
                  f"overloads={r['overloads_mean']:.1f}  game_over={r['game_over_pct']:.0f}%")
