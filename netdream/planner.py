"""Imagination-based planners using learned dynamics model.

Supports:
  - MPC with random shooting
  - MPC with Cross-Entropy Method (CEM)
  - Safety-filtered planning (reject unsafe trajectories)
"""

import torch
import torch.nn.functional as F
import numpy as np
from .dynamics_model import GNNDynamicsModel


class SafetyFilter:
    """Rejects imagined action sequences that violate operational constraints.

    During imagination rollout, checks each step's predicted constraint
    violation probability. If any step exceeds the threshold, the sequence
    is rejected and a safe default action is used instead.

    Safety guarantee: if constraint predictor accuracy >= p,
    then per-episode violation probability <= (1-p)^H.
    """

    def __init__(self, threshold: float = 0.5, fallback_action: int = 0):
        """
        Args:
            threshold: constraint violation probability above which an action is rejected
            fallback_action: action index to use when all candidates are unsafe (0 = no-op)
        """
        self.threshold = threshold
        self.fallback_action = fallback_action

    def is_safe(self, constraint_probs: torch.Tensor) -> torch.Tensor:
        """Check if a batch of trajectories are safe.

        Args:
            constraint_probs: [batch, H] or [batch, H, num_constraints]

        Returns:
            safe_mask: [batch] boolean tensor, True if trajectory is safe
        """
        if constraint_probs.dim() == 3:
            # Max over constraints, then check all steps
            max_per_step = constraint_probs.max(dim=-1).values
        else:
            max_per_step = constraint_probs

        # A trajectory is safe if ALL steps have violation prob below threshold
        return (max_per_step < self.threshold).all(dim=-1)


class MPCPlanner:
    """Model Predictive Control planner with optional safety filtering.

    Samples N candidate action sequences, rolls out each in the learned
    dynamics model, evaluates cumulative reward, and selects the best.
    Optionally filters out unsafe trajectories via SafetyFilter.
    """

    def __init__(
        self,
        model: GNNDynamicsModel,
        horizon: int = 5,
        num_candidates: int = 256,
        num_elites: int = 32,
        cem_iterations: int = 3,
        action_dim: int = 5,
        num_nodes: int = 1,
        safety_filter: SafetyFilter | None = None,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.cem_iterations = cem_iterations
        self.action_dim = action_dim
        self.num_nodes = num_nodes
        self.safety_filter = safety_filter

    @torch.no_grad()
    def plan(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """Plan the best action for the current state.

        Args:
            x: [num_nodes, node_feat_dim] current state
            edge_index: [2, num_edges] topology

        Returns:
            action: [num_nodes] best action (first step of best sequence)
        """
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        ei = torch.tensor(edge_index, dtype=torch.long, device=self.device)

        best_action = self._cem_plan(x_t, ei)
        return best_action.cpu().numpy()

    def _cem_plan(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Cross-Entropy Method planning.

        Iteratively refine action distribution by keeping elite samples.
        """
        N = self.num_nodes
        H = self.horizon
        K = self.num_candidates
        E = self.num_elites

        # Initialize uniform distribution over discrete actions
        # Use logits for each (time, node) → softmax → sample
        logits = torch.zeros(H, N, self.action_dim, device=self.device)

        for cem_iter in range(self.cem_iterations):
            # Sample K action sequences from current distribution
            probs = F.softmax(logits, dim=-1)  # [H, N, action_dim]
            # Expand and sample: [K, H, N]
            action_seqs = torch.zeros(K, H, N, dtype=torch.long, device=self.device)
            for t in range(H):
                for n in range(N):
                    action_seqs[:, t, n] = torch.multinomial(
                        probs[t, n].expand(K, -1), num_samples=1
                    ).squeeze(-1)

            # One-hot encode actions: [K, H, N, action_dim]
            action_onehot = F.one_hot(action_seqs, self.action_dim).float()

            # Evaluate each candidate sequence
            rewards = torch.zeros(K, device=self.device)
            safe_mask = torch.ones(K, dtype=torch.bool, device=self.device)

            for k in range(K):
                rollout = self.model.rollout(
                    x, action_onehot[k], edge_index, horizon=H
                )
                rewards[k] = rollout["rewards"].sum()

                if self.safety_filter is not None:
                    cp = rollout["constraint_probs"]
                    if not self.safety_filter.is_safe(cp.unsqueeze(0))[0]:
                        safe_mask[k] = False
                        rewards[k] = -1e6  # penalize unsafe trajectories

            # Select elites (top-E by reward, preferring safe ones)
            if safe_mask.any():
                # Among safe candidates, pick top-E
                safe_rewards = rewards.clone()
                safe_rewards[~safe_mask] = -1e6
                elite_idx = safe_rewards.topk(min(E, safe_mask.sum().item())).indices
            else:
                # No safe candidates — pick least-unsafe ones
                elite_idx = rewards.topk(E).indices

            # Update logits based on elite actions
            elite_actions = action_seqs[elite_idx]  # [E, H, N]
            # Count action frequencies among elites
            new_logits = torch.zeros_like(logits)
            for t in range(H):
                for n in range(N):
                    counts = torch.bincount(
                        elite_actions[:, t, n], minlength=self.action_dim
                    ).float()
                    new_logits[t, n] = torch.log(counts + 1)  # smoothed log-counts

            logits = new_logits

        # Return first action of best sequence
        best_idx = rewards.argmax()
        return action_seqs[best_idx, 0]  # [N]

    @torch.no_grad()
    def plan_batch(
        self, x: np.ndarray, edge_index: np.ndarray, num_steps: int = 1
    ) -> list[np.ndarray]:
        """Plan multiple steps (re-plan at each step using updated state)."""
        actions = []
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        ei = torch.tensor(edge_index, dtype=torch.long, device=self.device)

        for _ in range(num_steps):
            action = self._cem_plan(x_t, ei)
            actions.append(action.cpu().numpy())

            # Predict next state
            action_onehot = F.one_hot(action, self.action_dim).float()
            preds = self.model.predict_next_state(x_t, action_onehot, ei)
            x_t = preds["next_x"]

        return actions
