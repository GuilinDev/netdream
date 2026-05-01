"""Unified GraphEnv interface for graph-structured infrastructure environments.

All environments expose observations as PyG Data objects with:
  - x: node features [num_nodes, node_feat_dim]
  - edge_index: [2, num_edges]
  - edge_attr: edge features [num_edges, edge_feat_dim] (optional)
  - action_mask: valid actions per node (optional)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphObs:
    """Graph-structured observation returned by GraphEnv."""

    data: Data  # PyG Data with x, edge_index, edge_attr
    global_features: np.ndarray  # graph-level features (e.g., total cost)
    constraint_values: np.ndarray  # per-constraint values for safety checking


class GraphEnv(ABC):
    """Abstract base class for graph-structured infrastructure environments."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[GraphObs, dict]:
        """Reset environment, return (graph_obs, info)."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[GraphObs, float, bool, bool, dict]:
        """Execute action, return (graph_obs, reward, terminated, truncated, info)."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes in the current graph."""

    @property
    @abstractmethod
    def node_feature_dim(self) -> int:
        """Dimension of per-node feature vector."""

    @property
    @abstractmethod
    def edge_feature_dim(self) -> int:
        """Dimension of per-edge feature vector (0 if no edge features)."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of per-node action space."""

    @property
    @abstractmethod
    def constraint_names(self) -> list[str]:
        """Names of operational constraints (e.g., ['slo_violation', 'voltage_bound'])."""

    @abstractmethod
    def get_topology(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (edge_index [2, E], edge_attr [E, D]) for the current topology."""

    def collect_transitions(
        self, policy: str = "random", num_episodes: int = 100, seed: int = 42
    ) -> list[dict]:
        """Collect (s, a, s', r, constraint, done) transitions for dynamics model training.

        Args:
            policy: "random" for uniform random, or path to a trained SB3 model
            num_episodes: number of episodes to collect
            seed: random seed

        Returns:
            List of transition dicts with graph-structured states.
        """
        rng = np.random.default_rng(seed)
        transitions = []

        for ep in range(num_episodes):
            obs, _ = self.reset(seed=int(rng.integers(0, 100000)))
            done = False
            step_count = 0

            while not done:
                if policy == "random":
                    action = rng.integers(0, self.action_dim, size=self.num_nodes)
                else:
                    raise NotImplementedError("Trained policy loading not yet supported")

                next_obs, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated

                transitions.append({
                    "x": obs.data.x.numpy().copy(),
                    "edge_index": obs.data.edge_index.numpy().copy(),
                    "action": action.copy(),
                    "next_x": next_obs.data.x.numpy().copy(),
                    "reward": reward,
                    "constraint_values": obs.constraint_values.copy(),
                    "done": done,
                })

                obs = next_obs
                step_count += 1

        return transitions
