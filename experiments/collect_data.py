#!/usr/bin/env python3
"""Collect transition data from both environments for dynamics model training."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.k8s_mesh_env import K8sMeshEnv
from envs.grid2op_env import Grid2OpGraphEnv


def collect_k8s_data(
    num_episodes: int = 200,
    max_steps: int = 240,
    seed: int = 42,
    output_path: str = "data/k8s_transitions.pkl",
) -> None:
    """Collect transitions from K8s mesh environment."""
    rng = np.random.default_rng(seed)
    all_transitions = []
    workloads = ["constant", "periodic", "variable", "bursty", "ramp", "flash"]

    for ep in range(num_episodes):
        wl = workloads[ep % len(workloads)]
        env = K8sMeshEnv(workload_type=wl, max_steps=max_steps, seed=int(rng.integers(0, 100000)))
        obs, _ = env.reset()

        for step in range(max_steps):
            # Mix of random and biased-toward-no-op actions
            if rng.random() < 0.6:
                action = np.ones(env.num_nodes, dtype=int)  # no-op
            else:
                action = rng.integers(0, env.action_dim, size=env.num_nodes)

            next_obs, reward, terminated, truncated, info = env.step(action)

            all_transitions.append({
                "x": obs.data.x.numpy().copy(),
                "edge_index": obs.data.edge_index.numpy().copy(),
                "action": action.copy(),
                "next_x": next_obs.data.x.numpy().copy(),
                "reward": float(reward),
                "constraint": obs.constraint_values.copy(),
                "done": terminated or truncated,
            })

            obs = next_obs
            if terminated or truncated:
                break

        if (ep + 1) % 50 == 0:
            print(f"  K8s: {ep + 1}/{num_episodes} episodes, {len(all_transitions)} transitions")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_transitions, f)
    print(f"  K8s: saved {len(all_transitions)} transitions to {output_path}")


def collect_grid2op_data(
    num_episodes: int = 200,
    max_steps: int = 288,
    seed: int = 42,
    output_path: str = "data/grid2op_transitions.pkl",
) -> None:
    """Collect transitions from Grid2Op environment."""
    rng = np.random.default_rng(seed)
    all_transitions = []

    env = Grid2OpGraphEnv(env_name="l2rpn_case14_sandbox", max_steps=max_steps, seed=seed)

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100000)))

        for step in range(max_steps):
            # Mostly maintain lines, occasionally toggle
            action = np.ones(env._n_lines, dtype=int)  # maintain all
            if rng.random() < 0.1:
                line_idx = rng.integers(0, env._n_lines)
                action[line_idx] = rng.choice([0, 2])  # disconnect or reconnect

            next_obs, reward, terminated, truncated, info = env.step(action)

            all_transitions.append({
                "x": obs.data.x.numpy().copy(),
                "edge_index": obs.data.edge_index.numpy().copy(),
                "action": action[:env.num_nodes].copy(),  # truncate to num_nodes for consistency
                "next_x": next_obs.data.x.numpy().copy(),
                "reward": float(reward),
                "constraint": obs.constraint_values.copy(),
                "done": terminated or truncated,
            })

            obs = next_obs
            if terminated or truncated:
                break

        if (ep + 1) % 50 == 0:
            print(f"  Grid2Op: {ep + 1}/{num_episodes} episodes, {len(all_transitions)} transitions")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_transitions, f)
    print(f"  Grid2Op: saved {len(all_transitions)} transitions to {output_path}")


def collect_water_data(
    num_episodes: int = 200,
    max_steps: int = 96,
    seed: int = 42,
    output_path: str = "data/water_transitions.pkl",
) -> None:
    """Collect transitions from WaterNet environment."""
    from envs.water_net_env import WaterNetEnv
    rng = np.random.default_rng(seed)
    all_transitions = []

    for ep in range(num_episodes):
        demand_pattern = "daily" if ep % 3 != 0 else "random"
        env = WaterNetEnv(max_steps=max_steps, demand_pattern=demand_pattern,
                          seed=int(rng.integers(0, 100000)))
        obs, _ = env.reset()

        for step in range(max_steps):
            # Mix: mostly run pumps at normal, sometimes vary
            if rng.random() < 0.5:
                action = np.ones(env.num_nodes, dtype=int)   # normal speed
            elif rng.random() < 0.5:
                action = np.ones(env.num_nodes, dtype=int) * 2  # high speed
            else:
                action = rng.integers(0, env.action_dim, size=env.num_nodes)

            next_obs, reward, terminated, truncated, info = env.step(action)
            all_transitions.append({
                "x": obs.data.x.numpy().copy(),
                "edge_index": obs.data.edge_index.numpy().copy(),
                "action": action.copy(),
                "next_x": next_obs.data.x.numpy().copy(),
                "reward": float(reward),
                "constraint": obs.constraint_values.copy(),
                "done": terminated or truncated,
            })
            obs = next_obs
            if terminated or truncated:
                break

        if (ep + 1) % 50 == 0:
            print(f"  Water: {ep + 1}/{num_episodes} episodes, {len(all_transitions)} transitions")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_transitions, f)
    print(f"  Water: saved {len(all_transitions)} transitions to {output_path}")


def collect_traffic_data(
    num_episodes: int = 200,
    max_steps: int = 120,
    seed: int = 42,
    output_path: str = "data/traffic_transitions.pkl",
) -> None:
    """Collect transitions from TrafficSignal environment."""
    from envs.traffic_env import TrafficSignalEnv
    rng = np.random.default_rng(seed)
    all_transitions = []

    for ep in range(num_episodes):
        demand_type = ["peak", "uniform", "random"][ep % 3]
        env = TrafficSignalEnv(max_steps=max_steps, demand_type=demand_type,
                               seed=int(rng.integers(0, 100000)))
        obs, _ = env.reset()

        for step in range(max_steps):
            # Mix of coordinated (alternating NS/EW) and random
            if rng.random() < 0.4:
                action = (step // 4 % 2) * np.ones(env.num_nodes, dtype=int)  # cycle
            else:
                action = rng.integers(0, env.action_dim, size=env.num_nodes)

            next_obs, reward, terminated, truncated, info = env.step(action)
            all_transitions.append({
                "x": obs.data.x.numpy().copy(),
                "edge_index": obs.data.edge_index.numpy().copy(),
                "action": action.copy(),
                "next_x": next_obs.data.x.numpy().copy(),
                "reward": float(reward),
                "constraint": obs.constraint_values.copy(),
                "done": terminated or truncated,
            })
            obs = next_obs
            if terminated or truncated:
                break

        if (ep + 1) % 50 == 0:
            print(f"  Traffic: {ep + 1}/{num_episodes} episodes, {len(all_transitions)} transitions")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_transitions, f)
    print(f"  Traffic: saved {len(all_transitions)} transitions to {output_path}")


def collect_epinet_data(
    num_episodes: int = 200,
    max_steps: int = 60,
    seed: int = 42,
    output_path: str = "data/epinet_transitions.pkl",
) -> None:
    """Collect transitions from EpiNet-20 epidemic control environment."""
    from envs.epinet_env import EpiNetEnv, HUB_NODES
    rng = np.random.default_rng(seed)
    all_transitions = []

    for ep in range(num_episodes):
        env = EpiNetEnv(max_steps=max_steps, seed=int(rng.integers(0, 100000)))
        obs, _ = env.reset()

        for step in range(max_steps):
            # Training mix: 40% greedy-reactive, 40% random, 20% always-strong
            r = rng.random()
            if r < 0.40:
                # Greedy-reactive: intervene at nodes with I > 0.05
                I_vals = obs.data.x[:, 1].numpy()
                action = np.where(I_vals > 0.05, 1, 0).astype(int)
                for h in HUB_NODES:
                    if I_vals[h] > 0.08:
                        action[h] = 2  # strong at outbreak hubs
            elif r < 0.80:
                action = rng.integers(0, env.action_dim, size=env.num_nodes)
            else:
                action = np.full(env.num_nodes, 2, dtype=int)

            next_obs, reward, terminated, truncated, info = env.step(action)
            all_transitions.append({
                "x": obs.data.x.numpy().copy(),
                "edge_index": obs.data.edge_index.numpy().copy(),
                "action": action.copy(),
                "next_x": next_obs.data.x.numpy().copy(),
                "reward": float(reward),
                "constraint": obs.constraint_values.copy(),
                "done": terminated or truncated,
            })
            obs = next_obs
            if terminated or truncated:
                break

        if (ep + 1) % 50 == 0:
            print(f"  EpiNet: {ep + 1}/{num_episodes} episodes, {len(all_transitions)} transitions")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_transitions, f)
    print(f"  EpiNet: saved {len(all_transitions)} transitions to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "k8s", "grid2op", "water", "traffic", "epinet"])
    args = parser.parse_args()

    if args.mode in ("all", "k8s"):
        print("=" * 60)
        print("  Collecting K8s mesh transitions")
        print("=" * 60)
        collect_k8s_data()

    if args.mode in ("all", "grid2op"):
        print()
        print("=" * 60)
        print("  Collecting Grid2Op transitions")
        print("=" * 60)
        collect_grid2op_data()

    if args.mode in ("all", "water"):
        print()
        print("=" * 60)
        print("  Collecting WaterNet transitions")
        print("=" * 60)
        collect_water_data()

    if args.mode in ("all", "traffic"):
        print()
        print("=" * 60)
        print("  Collecting Traffic Signal transitions")
        print("=" * 60)
        collect_traffic_data()

    if args.mode in ("all", "epinet"):
        print()
        print("=" * 60)
        print("  Collecting EpiNet-20 transitions")
        print("=" * 60)
        collect_epinet_data()
