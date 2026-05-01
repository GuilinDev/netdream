"""GNN-based dynamics model for graph-structured infrastructure systems.

Predicts next-state residuals: Δx = f_θ(G, x_t, a_t)
so that x̂_{t+1} = x_t + Δx

Architecture:
  1. Node encoder: project [node_features || action] to hidden dim
  2. GNN message passing (K layers): propagate information across graph
  3. Dynamics head: predict per-node Δx (residual)
  4. Reward head: predict scalar reward from graph-level readout
  5. Constraint head: predict per-constraint violation probability
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    SAGEConv,
    global_mean_pool,
)


class GNNDynamicsModel(nn.Module):
    """Graph Neural Network dynamics model for networked infrastructure.

    Learns f_θ such that x̂_{t+1} = x_t + f_θ(G, x_t, a_t).
    """

    def __init__(
        self,
        node_feat_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_constraints: int = 1,
        gnn_type: str = "gat",  # "gcn", "gat", "sage"
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Node encoder: project [features || action] → hidden
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # GNN message passing layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            if gnn_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif gnn_type == "gat":
                conv = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
            elif gnn_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            self.gnn_layers.append(conv)
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Dynamics head: predict per-node Δx (residual next-state)
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim),
        )

        # Reward head: predict scalar reward from graph readout
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Constraint head: predict violation probability per constraint
        self.constraint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_constraints),
            nn.Sigmoid(),
        )

    def encode(
        self, x: torch.Tensor, action: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Encode node features + actions through GNN, return node embeddings.

        Args:
            x: [num_nodes, node_feat_dim]
            action: [num_nodes, action_dim] (one-hot or scalar per node)
            edge_index: [2, num_edges]

        Returns:
            h: [num_nodes, hidden_dim] node embeddings after message passing
        """
        # Concatenate features and actions
        h = self.node_encoder(torch.cat([x, action], dim=-1))

        # Message passing with residual connections
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual connection

        return h

    def predict_next_state(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict next state, reward, and constraint violations.

        Args:
            x: [total_nodes, node_feat_dim] node features
            action: [total_nodes, action_dim] per-node actions
            edge_index: [2, total_edges] edge connectivity
            batch: [total_nodes] batch assignment (for batched graphs)

        Returns:
            dict with keys: 'delta_x', 'next_x', 'reward', 'constraint_prob'
        """
        h = self.encode(x, action, edge_index)

        # Per-node dynamics (residual prediction)
        delta_x = self.dynamics_head(h)
        next_x = x + delta_x

        # Graph-level reward
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_emb = global_mean_pool(h, batch)
        reward = self.reward_head(graph_emb).squeeze(-1)

        # Per-graph constraint violation probability
        constraint_prob = self.constraint_head(graph_emb).squeeze(-1)

        return {
            "delta_x": delta_x,
            "next_x": next_x,
            "reward": reward,
            "constraint_prob": constraint_prob,
        }

    def forward(self, data: Data | Batch) -> dict[str, torch.Tensor]:
        """Forward pass on a PyG Data or Batch object.

        Expects data to have: x, edge_index, action, batch (if batched).
        """
        batch = data.batch if hasattr(data, "batch") else None
        return self.predict_next_state(data.x, data.action, data.edge_index, batch)

    def rollout(
        self,
        x: torch.Tensor,
        action_sequence: torch.Tensor,
        edge_index: torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Multi-step imagination rollout.

        Args:
            x: [num_nodes, node_feat_dim] initial state
            action_sequence: [H, num_nodes, action_dim] actions for each step
            edge_index: [2, num_edges] fixed topology
            horizon: number of steps (inferred from action_sequence if None)

        Returns:
            dict with:
                'states': [H+1, num_nodes, node_feat_dim] predicted trajectory
                'rewards': [H] predicted rewards
                'constraint_probs': [H] predicted constraint violation probs
        """
        if horizon is None:
            horizon = action_sequence.size(0)

        states = [x]
        rewards = []
        constraint_probs = []
        current_x = x

        for t in range(horizon):
            preds = self.predict_next_state(
                current_x, action_sequence[t], edge_index
            )
            current_x = preds["next_x"]
            states.append(current_x)
            rewards.append(preds["reward"])
            constraint_probs.append(preds["constraint_prob"])

        return {
            "states": torch.stack(states),
            "rewards": torch.stack(rewards),
            "constraint_probs": torch.stack(constraint_probs),
        }


class DynamicsModelTrainer:
    """Trainer for the GNN dynamics model."""

    def __init__(
        self,
        model: GNNDynamicsModel,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.dynamics_loss_fn = nn.MSELoss()
        self.reward_loss_fn = nn.MSELoss()
        self.constraint_loss_fn = nn.BCELoss()

    def train_step(self, batch: Batch) -> dict[str, float]:
        """Single training step on a batch of transitions.

        Batch must have: x, action, edge_index, target_delta_x, target_reward,
                         target_constraint, batch.
        """
        self.model.train()
        batch = batch.to(self.device)

        preds = self.model(batch)

        # Dynamics loss (residual prediction)
        loss_dynamics = self.dynamics_loss_fn(preds["delta_x"], batch.target_delta_x)

        # Reward loss
        loss_reward = self.reward_loss_fn(preds["reward"], batch.target_reward)

        # Constraint loss (binary cross-entropy)
        loss_constraint = self.constraint_loss_fn(
            preds["constraint_prob"], batch.target_constraint
        )

        # Combined loss
        loss = loss_dynamics + 0.1 * loss_reward + 0.1 * loss_constraint

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss_total": loss.item(),
            "loss_dynamics": loss_dynamics.item(),
            "loss_reward": loss_reward.item(),
            "loss_constraint": loss_constraint.item(),
        }

    @torch.no_grad()
    def evaluate(self, batch: Batch) -> dict[str, float]:
        """Evaluate model on a batch."""
        self.model.eval()
        batch = batch.to(self.device)
        preds = self.model(batch)

        loss_dynamics = self.dynamics_loss_fn(preds["delta_x"], batch.target_delta_x)
        loss_reward = self.reward_loss_fn(preds["reward"], batch.target_reward)

        return {
            "eval_loss_dynamics": loss_dynamics.item(),
            "eval_loss_reward": loss_reward.item(),
        }
