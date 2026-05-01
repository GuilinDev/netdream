"""Fast batched planner using random shooting with GPU-accelerated rollouts.

Instead of evaluating candidates one-by-one (slow CEM), this planner:
1. Samples K action sequences
2. Replicates the initial state K times
3. Batch-rollouts all K candidates through the GNN in parallel
4. Picks the best by cumulative reward

This is orders of magnitude faster than sequential CEM.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch, Data
from .dynamics_model import GNNDynamicsModel


class FastRandomShootingPlanner:
    """Fast random-shooting MPC with batched GPU rollouts + safety filter."""

    def __init__(
        self,
        model: GNNDynamicsModel,
        horizon: int = 5,
        num_candidates: int = 64,
        action_dim: int = 3,
        num_nodes: int = 11,
        safety_threshold: float = 0.5,
        use_safety: bool = True,
        device: str = "auto",
        noop_action: int = 1,
        noop_prob: float = 0.7,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.horizon = horizon
        self.K = num_candidates
        self.action_dim = action_dim
        self.num_nodes = num_nodes
        self.safety_threshold = safety_threshold
        self.use_safety = use_safety
        self.noop_action = noop_action
        self.noop_prob = noop_prob

    @torch.no_grad()
    def plan(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """Plan best action via batched random shooting.

        Args:
            x: [N, F] current node features
            edge_index: [2, E] edge connectivity

        Returns:
            action: [N] best first-step action
        """
        N = self.num_nodes
        K = self.K
        H = self.horizon
        A = self.action_dim

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        ei = torch.tensor(edge_index, dtype=torch.long, device=self.device)

        # Sample K random action sequences: [K, H, N]
        action_seqs = torch.randint(0, A, (K, H, N), device=self.device)

        # Optionally bias toward a default action (e.g., no-op for K8s)
        if self.noop_prob > 0.0:
            noop_mask = torch.rand(K, H, N, device=self.device) < self.noop_prob
            action_seqs[noop_mask] = self.noop_action

        # Replicate initial state K times
        # Create a batch of K identical graphs
        x_batch = x_t.unsqueeze(0).expand(K, -1, -1).reshape(K * N, -1)  # [K*N, F]

        # Replicate edge_index for each graph in the batch
        batch_ei = []
        batch_vec = []
        for k in range(K):
            batch_ei.append(ei + k * N)
            batch_vec.append(torch.full((N,), k, dtype=torch.long, device=self.device))
        batch_edge_index = torch.cat(batch_ei, dim=1)  # [2, K*E]
        batch_assignment = torch.cat(batch_vec)  # [K*N]

        # Rollout H steps in parallel
        cumulative_reward = torch.zeros(K, device=self.device)
        is_safe = torch.ones(K, dtype=torch.bool, device=self.device)
        current_x = x_batch

        for t in range(H):
            # One-hot encode actions for this timestep
            actions_t = action_seqs[:, t, :]  # [K, N]
            actions_flat = actions_t.reshape(K * N)  # [K*N]
            action_onehot = F.one_hot(actions_flat, A).float()  # [K*N, A]

            # Batch forward pass through GNN
            preds = self.model.predict_next_state(
                current_x, action_onehot, batch_edge_index, batch_assignment
            )

            current_x = preds["next_x"]
            cumulative_reward += preds["reward"]  # [K]

            # Safety check
            if self.use_safety:
                unsafe = preds["constraint_prob"] > self.safety_threshold  # [K]
                is_safe &= ~unsafe

        # Select best candidate
        if self.use_safety and is_safe.any():
            # Among safe candidates, pick highest reward
            safe_rewards = cumulative_reward.clone()
            safe_rewards[~is_safe] = -1e10
            best_idx = safe_rewards.argmax()
        else:
            # Fallback: pick highest reward regardless
            best_idx = cumulative_reward.argmax()

        # Return first action of best sequence
        return action_seqs[best_idx, 0].cpu().numpy()  # [N]
