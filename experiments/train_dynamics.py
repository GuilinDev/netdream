#!/usr/bin/env python3
"""Train GNN dynamics model on collected transition data.

Supports training on K8s, Grid2Op, or both (multi-domain).
Reports 1/5/10-step prediction MSE for the paper.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from netdream.dynamics_model import GNNDynamicsModel, DynamicsModelTrainer


def load_transitions(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def transitions_to_pyg_dataset(
    transitions: list[dict],
    action_dim: int,
    node_feat_dim: int,
) -> list[Data]:
    """Convert transition dicts to PyG Data objects for training."""
    dataset = []
    for t in transitions:
        x = torch.tensor(t["x"], dtype=torch.float32)
        next_x = torch.tensor(t["next_x"], dtype=torch.float32)
        edge_index = torch.tensor(t["edge_index"], dtype=torch.long)
        num_nodes = x.size(0)

        # Action encoding: one-hot per node
        raw_action = t["action"]
        if len(raw_action) < num_nodes:
            raw_action = np.pad(raw_action, (0, num_nodes - len(raw_action)), constant_values=1)
        elif len(raw_action) > num_nodes:
            raw_action = raw_action[:num_nodes]
        action_idx = torch.tensor(raw_action, dtype=torch.long).clamp(0, action_dim - 1)
        action_onehot = F.one_hot(action_idx, action_dim).float()

        # Targets
        delta_x = next_x - x
        reward = torch.tensor([t["reward"]], dtype=torch.float32)

        # Constraint target (binary)
        constraint = t["constraint"]
        if len(constraint.shape) == 1 and len(constraint) > 1:
            constraint_target = torch.tensor(
                [float(np.any(constraint > 0.5))], dtype=torch.float32
            )
        else:
            constraint_target = torch.tensor(
                [float(np.any(constraint > 0.5))], dtype=torch.float32
            )

        data = Data(
            x=x,
            edge_index=edge_index,
            action=action_onehot,
            target_delta_x=delta_x,
            target_reward=reward,
            target_constraint=constraint_target,
            num_nodes=num_nodes,
        )
        dataset.append(data)

    return dataset


def evaluate_multistep(
    model: GNNDynamicsModel,
    transitions: list[dict],
    action_dim: int,
    horizons: list[int] = [1, 5, 10],
    num_eval_episodes: int = 20,
    episode_len: int = 50,
    device: str = "cuda",
) -> dict[int, float]:
    """Evaluate multi-step prediction accuracy.

    Rolls out the model for H steps and compares against ground truth.
    Returns MSE per horizon.
    """
    model.eval()
    device = torch.device(device)
    results = {h: [] for h in horizons}

    # Group transitions into episodes
    episodes = []
    current_ep = []
    for t in transitions:
        current_ep.append(t)
        if t["done"] or len(current_ep) >= episode_len:
            if len(current_ep) >= max(horizons) + 1:
                episodes.append(current_ep)
            current_ep = []
    episodes = episodes[:num_eval_episodes]

    with torch.no_grad():
        for ep in episodes:
            for h in horizons:
                if len(ep) < h + 1:
                    continue
                # Start from random point in episode
                start = np.random.randint(0, len(ep) - h)

                x = torch.tensor(ep[start]["x"], dtype=torch.float32, device=device)
                edge_index = torch.tensor(ep[start]["edge_index"], dtype=torch.long, device=device)

                # Rollout h steps
                current_x = x
                for step in range(h):
                    t_data = ep[start + step]
                    raw_action = t_data["action"]
                    num_nodes = current_x.size(0)
                    if len(raw_action) < num_nodes:
                        raw_action = np.pad(raw_action, (0, num_nodes - len(raw_action)), constant_values=1)
                    elif len(raw_action) > num_nodes:
                        raw_action = raw_action[:num_nodes]
                    action_idx = torch.tensor(raw_action, dtype=torch.long, device=device).clamp(0, action_dim - 1)
                    action_onehot = F.one_hot(action_idx, action_dim).float()

                    preds = model.predict_next_state(current_x, action_onehot, edge_index)
                    current_x = preds["next_x"]

                # Compare against ground truth at step h
                true_x = torch.tensor(ep[start + h]["x"], dtype=torch.float32, device=device)
                mse = F.mse_loss(current_x, true_x).item()
                results[h].append(mse)

    return {h: float(np.mean(v)) if v else float("nan") for h, v in results.items()}


def train_domain(
    domain: str,
    data_path: str,
    node_feat_dim: int,
    action_dim: int,
    num_constraints: int,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    gnn_type: str = "gat",
    hidden_dim: int = 128,
    num_gnn_layers: int = 3,
    save_path: str = "models/",
) -> GNNDynamicsModel:
    """Train dynamics model on one domain."""
    print(f"\n{'='*60}")
    print(f"  Training on {domain}")
    print(f"{'='*60}")

    transitions = load_transitions(data_path)
    print(f"  Loaded {len(transitions)} transitions")

    dataset = transitions_to_pyg_dataset(transitions, action_dim, node_feat_dim)

    # Split 80/20
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(list(train_set), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(val_set), batch_size=batch_size, shuffle=False)

    model = GNNDynamicsModel(
        node_feat_dim=node_feat_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        num_constraints=num_constraints,
        gnn_type=gnn_type,
    )

    trainer = DynamicsModelTrainer(model, lr=lr)
    device = trainer.device
    print(f"  Device: {device}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        # Train
        train_losses = []
        for batch in train_loader:
            loss = trainer.train_step(batch)
            train_losses.append(loss["loss_dynamics"])

        # Validate
        val_losses = []
        for batch in val_loader:
            loss = trainer.evaluate(batch)
            val_losses.append(loss["eval_loss_dynamics"])

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # Save best model
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"{domain}_best.pt")
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(Path(save_path) / f"{domain}_best.pt", weights_only=True))
    model.to(device)

    # Multi-step evaluation
    print(f"\n  Multi-step prediction MSE:")
    mse_results = evaluate_multistep(
        model, transitions, action_dim,
        horizons=[1, 5, 10],
        device=str(device),
    )
    for h, mse in mse_results.items():
        print(f"    {h}-step MSE: {mse:.6f}")

    return model


def train_ablation(
    data_path: str,
    domain: str,
    node_feat_dim: int,
    action_dim: int,
    num_constraints: int,
) -> dict:
    """Run GNN type ablation: GCN vs GAT vs SAGE vs MLP-only."""
    print(f"\n{'='*60}")
    print(f"  Ablation: GNN type comparison on {domain}")
    print(f"{'='*60}")

    results = {}
    for gnn_type in ["gat", "gcn", "sage"]:
        print(f"\n  --- {gnn_type.upper()} ---")
        model = train_domain(
            domain=f"{domain}_{gnn_type}",
            data_path=data_path,
            node_feat_dim=node_feat_dim,
            action_dim=action_dim,
            num_constraints=num_constraints,
            epochs=50,
            gnn_type=gnn_type,
            save_path=f"models/ablation/",
        )

        transitions = load_transitions(data_path)
        mse = evaluate_multistep(
            model, transitions, action_dim,
            horizons=[1, 5, 10],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        results[gnn_type] = mse

    # MLP baseline (no graph structure — treat each node independently)
    print(f"\n  --- MLP (no graph) ---")
    model_mlp = train_domain(
        domain=f"{domain}_mlp",
        data_path=data_path,
        node_feat_dim=node_feat_dim,
        action_dim=action_dim,
        num_constraints=num_constraints,
        epochs=50,
        gnn_type="gcn",  # use GCN but with 0 layers effectively
        num_gnn_layers=0,
        save_path=f"models/ablation/",
    )
    transitions = load_transitions(data_path)
    mse_mlp = evaluate_multistep(
        model_mlp, transitions, action_dim,
        horizons=[1, 5, 10],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    results["mlp"] = mse_mlp

    print(f"\n  Ablation summary ({domain}):")
    print(f"  {'Type':<8} {'1-step':<12} {'5-step':<12} {'10-step':<12}")
    for gnn_type, mse in results.items():
        print(f"  {gnn_type:<8} {mse[1]:<12.6f} {mse[5]:<12.6f} {mse[10]:<12.6f}")

    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["k8s", "grid2op", "both", "ablation",
                                       "water", "traffic", "epinet", "all"], default="both")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--gnn-type", default="gat")
    args = p.parse_args()

    if args.mode in ("k8s", "both"):
        train_domain(
            domain="k8s",
            data_path="data/k8s_transitions.pkl",
            node_feat_dim=6,
            action_dim=3,
            num_constraints=1,
            epochs=args.epochs,
            gnn_type=args.gnn_type,
        )

    if args.mode in ("grid2op", "both"):
        train_domain(
            domain="grid2op",
            data_path="data/grid2op_transitions.pkl",
            node_feat_dim=7,
            action_dim=3,
            num_constraints=1,
            epochs=args.epochs,
            gnn_type=args.gnn_type,
        )

    if args.mode in ("water", "all"):
        train_domain(
            domain="water",
            data_path="data/water_transitions.pkl",
            node_feat_dim=5,   # [pressure, demand, tank_level, pump_speed, elevation]
            action_dim=3,
            num_constraints=1,
            epochs=args.epochs,
            gnn_type=args.gnn_type,
        )

    if args.mode in ("traffic", "all"):
        train_domain(
            domain="traffic",
            data_path="data/traffic_transitions.pkl",
            node_feat_dim=6,   # [q_NS, q_EW, throughput, speed, phase_0, phase_1]
            action_dim=3,
            num_constraints=1,
            epochs=args.epochs,
            gnn_type=args.gnn_type,
        )

    if args.mode in ("epinet", "all"):
        train_domain(
            domain="epinet",
            data_path="data/epinet_transitions.pkl",
            node_feat_dim=5,   # [S, I, R, intervention_level, mobility]
            action_dim=3,
            num_constraints=1,
            epochs=args.epochs,
            gnn_type=args.gnn_type,
        )

    if args.mode == "ablation":
        print("\n" + "=" * 60)
        print("  Running full GNN type ablation")
        print("=" * 60)

        k8s_results = train_ablation(
            "data/k8s_transitions.pkl", "k8s",
            node_feat_dim=6, action_dim=3, num_constraints=1,
        )

        grid2op_results = train_ablation(
            "data/grid2op_transitions.pkl", "grid2op",
            node_feat_dim=7, action_dim=3, num_constraints=1,
        )
