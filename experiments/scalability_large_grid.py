#!/usr/bin/env python3
"""Scalability experiment: train and evaluate world model on a larger grid.

Uses l2rpn_neurips_2020_track1_large (36 substations, 59 lines) to show
that GNN prediction accuracy scales to larger graphs.
"""

import sys, warnings, pickle
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import grid2op
from grid2op.Parameters import Parameters
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

from netdream.dynamics_model import GNNDynamicsModel, DynamicsModelTrainer


def collect_large_grid_data(env_name="l2rpn_neurips_2020_track1_large", num_episodes=100, max_steps=100):
    """Collect data from a larger Grid2Op environment."""
    print(f"Setting up {env_name}...")
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = False
    env = grid2op.make(env_name, param=params)

    n_sub = env.n_sub
    n_line = env.n_line
    print(f"  Substations: {n_sub}, Lines: {n_line}")

    # Build edge index
    obs = env.reset()
    or_sub = obs.line_or_to_subid
    ex_sub = obs.line_ex_to_subid
    src = np.concatenate([or_sub, ex_sub])
    dst = np.concatenate([ex_sub, or_sub])
    edge_index = np.array([src, dst], dtype=np.int64)

    transitions = []
    rng = np.random.default_rng(42)

    for ep in range(num_episodes):
        obs = env.reset()
        for step in range(max_steps):
            # Extract node features (same as grid2op_env.py)
            x = np.zeros((n_sub, 7), dtype=np.float32)
            for i in range(n_line):
                if obs.line_status[i]:
                    or_s = obs.line_or_to_subid[i]
                    ex_s = obs.line_ex_to_subid[i]
                    x[or_s, 0] = obs.v_or[i] / 1000.0 if obs.v_or[i] > 0 else 1.0
                    x[ex_s, 0] = obs.v_ex[i] / 1000.0 if obs.v_ex[i] > 0 else 1.0
            for i in range(obs.n_gen):
                sub = obs.gen_to_subid[i]
                x[sub, 1] += obs.gen_p[i] / 100.0
                x[sub, 2] += obs.gen_q[i] / 100.0
                x[sub, 5] = 1.0
            for i in range(obs.n_load):
                sub = obs.load_to_subid[i]
                x[sub, 3] += obs.load_p[i] / 100.0
                x[sub, 4] += obs.load_q[i] / 100.0
                x[sub, 6] = 1.0
            x = np.nan_to_num(x, nan=0.0)

            # Do-nothing action
            action = env.action_space({})
            next_obs, reward, done, info = env.step(action)

            # Extract next_x
            next_x = np.zeros((n_sub, 7), dtype=np.float32)
            for i in range(n_line):
                if next_obs.line_status[i]:
                    or_s = next_obs.line_or_to_subid[i]
                    ex_s = next_obs.line_ex_to_subid[i]
                    next_x[or_s, 0] = next_obs.v_or[i] / 1000.0 if next_obs.v_or[i] > 0 else 1.0
                    next_x[ex_s, 0] = next_obs.v_ex[i] / 1000.0 if next_obs.v_ex[i] > 0 else 1.0
            for i in range(next_obs.n_gen):
                sub = next_obs.gen_to_subid[i]
                next_x[sub, 1] += next_obs.gen_p[i] / 100.0
                next_x[sub, 2] += next_obs.gen_q[i] / 100.0
                next_x[sub, 5] = 1.0
            for i in range(next_obs.n_load):
                sub = next_obs.load_to_subid[i]
                next_x[sub, 3] += next_obs.load_p[i] / 100.0
                next_x[sub, 4] += next_obs.load_q[i] / 100.0
                next_x[sub, 6] = 1.0
            next_x = np.nan_to_num(next_x, nan=0.0)

            # No-op action encoding
            node_action = np.ones(n_sub, dtype=int)

            transitions.append({
                "x": x, "edge_index": edge_index,
                "action": node_action, "next_x": next_x,
                "reward": float(reward),
                "constraint": np.array([float(np.any(obs.rho > 1.0))]),
                "done": done,
            })

            obs = next_obs
            if done:
                break

        if (ep + 1) % 25 == 0:
            print(f"  Collected {ep+1}/{num_episodes} episodes, {len(transitions)} transitions")

    return transitions, n_sub, edge_index


def train_and_evaluate(transitions, n_sub, edge_index):
    """Train GAT and MLP on the larger grid, report prediction MSE."""
    from experiments.train_dynamics import transitions_to_pyg_dataset, evaluate_multistep

    dataset = transitions_to_pyg_dataset(transitions, action_dim=3, node_feat_dim=7)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(list(train_set), batch_size=64, shuffle=True)
    val_loader = DataLoader(list(val_set), batch_size=64)

    results = {}

    for gnn_type, n_layers, label in [("gat", 3, "GAT"), ("gcn", 0, "MLP")]:
        print(f"\n  Training {label} on {n_sub}-node grid...")
        model = GNNDynamicsModel(
            node_feat_dim=7, action_dim=3, hidden_dim=128,
            num_gnn_layers=n_layers, num_constraints=1, gnn_type=gnn_type,
        )
        trainer = DynamicsModelTrainer(model, lr=3e-4)

        best_val = float("inf")
        for epoch in range(60):
            for batch in train_loader:
                trainer.train_step(batch)
            val_losses = []
            for batch in val_loader:
                loss = trainer.evaluate(batch)
                val_losses.append(loss["eval_loss_dynamics"])
            vl = np.mean(val_losses)
            if vl < best_val:
                best_val = vl
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}: val_loss={vl:.6f}, best={best_val:.6f}")

        mse = evaluate_multistep(model, transitions, action_dim=3, horizons=[1, 5, 10], device=str(trainer.device))
        results[label] = mse
        print(f"  {label}: 1-step={mse[1]:.6f}, 5-step={mse[5]:.6f}, 10-step={mse[10]:.6f}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  Scalability: Large Grid Prediction")
    print("=" * 60)

    transitions, n_sub, edge_index = collect_large_grid_data(num_episodes=100, max_steps=100)
    print(f"\nCollected {len(transitions)} transitions from {n_sub}-node grid")

    results = train_and_evaluate(transitions, n_sub, edge_index)

    # Compare with 14-bus results
    print("\n" + "=" * 60)
    print("  SCALABILITY COMPARISON")
    print("=" * 60)
    print(f"  {'Grid':<15} {'Arch':<6} {'1-step':<12} {'5-step':<12} {'10-step':<12}")
    print(f"  {'14-bus':<15} {'GAT':<6} {'4.50e-5':<12} {'8.65e-4':<12} {'5.07e-3':<12}")
    print(f"  {'14-bus':<15} {'MLP':<6} {'—':<12} {'—':<12} {'—':<12}")
    for label, mse in results.items():
        print(f"  {f'{n_sub}-bus':<15} {label:<6} {mse[1]:<12.6f} {mse[5]:<12.6f} {mse[10]:<12.6f}")
