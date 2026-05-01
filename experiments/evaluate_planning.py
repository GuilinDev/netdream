#!/usr/bin/env python3
"""Evaluate NetDream planning vs model-free RL vs rule-based baselines.

Experiments:
  1. Planning performance (cost + SLO/constraint violations)
  2. Safety filter effectiveness
  3. Cascade prevention (flash-crowd / N-1 contingency)
  4. Planning horizon ablation
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from netdream.dynamics_model import GNNDynamicsModel
from netdream.fast_planner import FastRandomShootingPlanner
from envs.k8s_mesh_env import K8sMeshEnv
from envs.grid2op_env import Grid2OpGraphEnv


def evaluate_agent(
    env,
    agent_fn,
    num_episodes: int = 5,
    seeds: list[int] = [42, 123, 456, 789, 1024],
) -> dict:
    """Evaluate an agent on an environment across multiple seeds.

    Args:
        env: GraphEnv instance
        agent_fn: callable(obs, env) -> action
        num_episodes: number of seeds
        seeds: random seeds

    Returns:
        dict with mean/std of cost, violations, replicas
    """
    all_costs = []
    all_violations = []
    all_steps = []

    for seed in seeds[:num_episodes]:
        obs, _ = env.reset(seed=seed)
        total_cost = 0.0
        total_violations = 0
        steps = 0

        while True:
            action = agent_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_cost += info.get("total_cost", 0)
            total_violations += info.get("slo_violations", 0)
            # For grid2op
            if "game_over" in info and info["game_over"]:
                total_violations += 10  # heavy penalty for game over
            steps += 1

            if terminated or truncated:
                break

        all_costs.append(total_cost)
        all_violations.append(total_violations)
        all_steps.append(steps)

    return {
        "cost_mean": float(np.mean(all_costs)),
        "cost_std": float(np.std(all_costs)),
        "violations_mean": float(np.mean(all_violations)),
        "violations_std": float(np.std(all_violations)),
        "steps_mean": float(np.mean(all_steps)),
    }


# ─── Agent functions ─────────────────────────────────────────────

def random_agent(obs, env):
    return np.random.randint(0, env.action_dim, size=env.num_nodes)


def noop_agent(obs, env):
    """Always do nothing (action=1 for K8s, action=1 for grid)."""
    return np.ones(env.num_nodes, dtype=int)


def hpa_agent_k8s(obs, env):
    """Rule-based HPA: scale based on CPU utilization per service."""
    x = obs.data.x.numpy()
    cpu = x[:, 0] * 100  # denormalize
    replicas = x[:, 5] * 10  # denormalize
    target = 70.0

    actions = np.ones(env.num_nodes, dtype=int)  # default no-op
    for i in range(env.num_nodes):
        desired = np.ceil(replicas[i] * (cpu[i] / target))
        delta = desired - replicas[i]
        if delta >= 1:
            actions[i] = 2  # scale up
        elif delta <= -1:
            actions[i] = 0  # scale down
    return actions


def make_netdream_agent(
    model: GNNDynamicsModel,
    horizon: int = 5,
    num_candidates: int = 64,
    use_safety: bool = True,
    action_dim: int = 3,
    num_nodes: int = 11,
):
    """Create a NetDream agent using fast batched random-shooting planner."""
    planner = FastRandomShootingPlanner(
        model=model,
        horizon=horizon,
        num_candidates=num_candidates,
        action_dim=action_dim,
        num_nodes=num_nodes,
        use_safety=use_safety,
    )

    def agent_fn(obs, env):
        x = obs.data.x.numpy()
        edge_index = obs.data.edge_index.numpy()
        return planner.plan(x, edge_index)

    return agent_fn


# ─── Main evaluation ─────────────────────────────────────────────

def evaluate_k8s(model_path: str = "models/k8s_best.pt") -> dict:
    """Evaluate all agents on K8s mesh environment."""
    print("\n" + "=" * 60)
    print("  K8s Mesh: Planning Evaluation")
    print("=" * 60)

    results = {}

    # Load model
    model = GNNDynamicsModel(
        node_feat_dim=6, action_dim=3, hidden_dim=128,
        num_gnn_layers=3, num_constraints=1, gnn_type="gat",
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    workloads = ["constant", "variable", "bursty", "flash"]

    for wl in workloads:
        print(f"\n  Workload: {wl}")
        env = K8sMeshEnv(workload_type=wl, max_steps=60)

        agents = {
            "Random": random_agent,
            "HPA": hpa_agent_k8s,
            "NetDream-H5": make_netdream_agent(model, horizon=5, use_safety=False, num_nodes=11, num_candidates=32),
            "NetDream-H5-Safe": make_netdream_agent(model, horizon=5, use_safety=True, num_nodes=11, num_candidates=32),
        }

        for name, agent_fn in agents.items():
            r = evaluate_agent(env, agent_fn, num_episodes=5)
            results[f"k8s_{wl}_{name}"] = r
            print(f"    {name:<20} cost={r['cost_mean']:.4f}±{r['cost_std']:.4f}  "
                  f"viol={r['violations_mean']:.1f}±{r['violations_std']:.1f}")

    # Planning horizon ablation
    print(f"\n  Horizon ablation (bursty workload):")
    env = K8sMeshEnv(workload_type="bursty", max_steps=60)
    for h in [1, 3, 5]:
        agent = make_netdream_agent(model, horizon=h, use_safety=True, num_nodes=11, num_candidates=32)
        r = evaluate_agent(env, agent, num_episodes=5)
        results[f"k8s_bursty_H{h}"] = r
        print(f"    H={h:<3} cost={r['cost_mean']:.4f}  viol={r['violations_mean']:.1f}")

    return results


def evaluate_grid2op(model_path: str = "models/grid2op_best.pt") -> dict:
    """Evaluate agents on Grid2Op power grid."""
    print("\n" + "=" * 60)
    print("  Grid2Op: Planning Evaluation")
    print("=" * 60)

    results = {}

    model = GNNDynamicsModel(
        node_feat_dim=7, action_dim=3, hidden_dim=128,
        num_gnn_layers=3, num_constraints=1, gnn_type="gat",
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    env = Grid2OpGraphEnv(max_steps=100)

    agents = {
        "Do-Nothing": noop_agent,
        "NetDream-H5": make_netdream_agent(
            model, horizon=5, use_safety=False,
            action_dim=3, num_nodes=env.num_nodes,
            num_candidates=64,
        ),
        "NetDream-H5-Safe": make_netdream_agent(
            model, horizon=5, use_safety=True,
            action_dim=3, num_nodes=env.num_nodes,
            num_candidates=64,
        ),
    }

    for name, agent_fn in agents.items():
        r = evaluate_agent(env, agent_fn, num_episodes=5)
        results[f"grid2op_{name}"] = r
        print(f"  {name:<20} cost={r['cost_mean']:.4f}±{r['cost_std']:.4f}  "
              f"viol={r['violations_mean']:.1f}±{r['violations_std']:.1f}  "
              f"steps={r['steps_mean']:.0f}")

    return results


if __name__ == "__main__":
    all_results = {}

    k8s_results = evaluate_k8s()
    all_results.update(k8s_results)

    grid2op_results = evaluate_grid2op()
    all_results.update(grid2op_results)

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/planning_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/planning_results.json")
