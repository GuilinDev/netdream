#!/usr/bin/env python3
"""Run SAGE and MLP ablations that were missing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_dynamics import train_domain, evaluate_multistep, load_transitions
from netdream.dynamics_model import GNNDynamicsModel
import torch
import numpy as np

print("=" * 60)
print("  Running missing ablations: SAGE + MLP")
print("=" * 60)

# SAGE
print("\n--- SAGE ---")
model_sage = train_domain(
    domain="k8s_sage",
    data_path="data/k8s_transitions.pkl",
    node_feat_dim=6, action_dim=3, num_constraints=1,
    epochs=80, gnn_type="sage",
    save_path="models/ablation/",
)

# MLP (0 GNN layers — no message passing)
print("\n--- MLP (no graph structure) ---")
model_mlp = GNNDynamicsModel(
    node_feat_dim=6, action_dim=3, hidden_dim=128,
    num_gnn_layers=0, num_constraints=1, gnn_type="gcn",
)

from netdream.dynamics_model import DynamicsModelTrainer
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from experiments.train_dynamics import transitions_to_pyg_dataset

transitions = load_transitions("data/k8s_transitions.pkl")
dataset = transitions_to_pyg_dataset(transitions, action_dim=3, node_feat_dim=6)
n_train = int(0.8 * len(dataset))
train_set, val_set = random_split(dataset, [n_train, len(dataset) - n_train])
train_loader = DataLoader(list(train_set), batch_size=128, shuffle=True)
val_loader = DataLoader(list(val_set), batch_size=128)

trainer = DynamicsModelTrainer(model_mlp, lr=3e-4)

best_val = float("inf")
for epoch in range(80):
    train_losses = []
    for batch in train_loader:
        loss = trainer.train_step(batch)
        train_losses.append(loss["loss_dynamics"])
    val_losses = []
    for batch in val_loader:
        loss = trainer.evaluate(batch)
        val_losses.append(loss["eval_loss_dynamics"])

    vl = np.mean(val_losses)
    if vl < best_val:
        best_val = vl
        torch.save(model_mlp.state_dict(), "models/ablation/k8s_mlp_best.pt")

    if (epoch + 1) % 10 == 0:
        print(f"  MLP Epoch {epoch+1}: train={np.mean(train_losses):.6f} val={vl:.6f} best={best_val:.6f}")

model_mlp.load_state_dict(torch.load("models/ablation/k8s_mlp_best.pt", weights_only=True))
model_mlp.to(trainer.device)

mse_mlp = evaluate_multistep(model_mlp, transitions, action_dim=3, device=str(trainer.device))
print(f"\n  MLP Multi-step MSE:")
for h, v in mse_mlp.items():
    print(f"    {h}-step: {v:.6f}")

# Summary
print("\n" + "=" * 60)
print("  FULL ABLATION SUMMARY (K8s)")
print("=" * 60)
print(f"  {'Type':<8} {'1-step':<12} {'5-step':<12} {'10-step':<12}")

# Load GAT and GCN results from earlier runs
gat_mse = {1: 0.000279, 5: 0.003702, 10: 0.006210}
gcn_mse = {1: 0.001557, 5: 0.004617, 10: 0.016544}

# Get SAGE results
sage_mse = evaluate_multistep(model_sage, transitions, action_dim=3, device=str(trainer.device))

for name, mse in [("GAT", gat_mse), ("GCN", gcn_mse), ("SAGE", sage_mse), ("MLP", mse_mlp)]:
    print(f"  {name:<8} {mse[1]:<12.6f} {mse[5]:<12.6f} {mse[10]:<12.6f}")
