"""Epidemic control environment on a community mobility graph (EpiNet-20).

20-node community network; dynamics follow a discrete SIR model with
inter-community transmission along mobility edges.

Network layout (EpiNet-20):
  - 4 metropolitan hubs (nodes 0, 5, 10, 15): high connectivity
  - 16 regional communities (1-4, 6-9, 11-14, 16-19): cluster around hubs

SIR dynamics (per step = 1 day):
  new_I_i = beta * S_i * (local_I_i + alpha * sum_j A_ij * I_j) * intervention_factor
  new_R_i = gamma * I_i
  S_i, I_i, R_i updated via finite-difference Euler

Control actions (per node, ACTION_DIM = 3):
  0 = no intervention    (beta_factor = 1.0, cost = 0.0)
  1 = moderate (NPIs)    (beta_factor = 0.5, cost = 0.3)
  2 = strong (lockdown)  (beta_factor = 0.1, cost = 1.0)

Node features (NODE_FEAT_DIM = 5):
  [S_norm, I_norm, R_norm, intervention_level, mobility_index]

Constraint: I_i > OUTBREAK_THRESHOLD at any node = violation.
"""

import numpy as np
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(__file__).replace("envs/epinet_env.py", ""))
from netdream.graph_env import GraphEnv, GraphObs

# ─── Network constants ──────────────────────────────────────────────────────────

NUM_NODES = 20
NODE_FEAT_DIM = 5
ACTION_DIM = 3

OUTBREAK_THRESHOLD = 0.10   # I fraction > 10% → violation

# SIR parameters
BETA = 0.35          # within-node transmission rate
GAMMA = 0.08         # recovery rate  (R0 ≈ 4.4)
ALPHA = 0.12         # inter-community spread factor (scaled per edge)

# Intervention beta multipliers
BETA_FACTOR = {0: 1.0, 1: 0.50, 2: 0.10}
# Intervention costs (per node per step)
INTER_COST  = {0: 0.0, 1: 0.30, 2: 1.00}

# Hub nodes (metro areas — higher degree, higher mobility)
HUB_NODES = [0, 5, 10, 15]

# Node mobility index (hub=1.0, regional=0.5-0.8)
MOBILITY = np.array([
    1.0, 0.6, 0.6, 0.7, 0.7,   # cluster 0: hub0 + 4 regional
    1.0, 0.6, 0.6, 0.7, 0.7,   # cluster 1: hub5 + 4 regional
    1.0, 0.6, 0.6, 0.7, 0.7,   # cluster 2: hub10 + 4 regional
    1.0, 0.6, 0.6, 0.7, 0.7,   # cluster 3: hub15 + 4 regional
], dtype=np.float32)

# Fixed mobility network
EDGE_LIST = [
    # Hub-hub long-range connections
    (0, 5), (0, 10), (5, 15), (10, 15), (0, 15), (5, 10),
    # Hub-to-regional (each hub connects its 4 locals)
    (0, 1), (0, 2), (0, 3), (0, 4),
    (5, 6), (5, 7), (5, 8), (5, 9),
    (10, 11), (10, 12), (10, 13), (10, 14),
    (15, 16), (15, 17), (15, 18), (15, 19),
    # Ring within clusters
    (1, 2), (3, 4),
    (6, 7), (8, 9),
    (11, 12), (13, 14),
    (16, 17), (18, 19),
    # Cross-cluster commuter links
    (2, 6), (4, 11), (7, 16), (9, 14),
]

_src = [e[0] for e in EDGE_LIST] + [e[1] for e in EDGE_LIST]
_dst = [e[1] for e in EDGE_LIST] + [e[0] for e in EDGE_LIST]
EDGE_INDEX = np.array([_src, _dst], dtype=np.int64)

# Precompute adjacency list for fast neighbor lookup
_adj = [[] for _ in range(NUM_NODES)]
for i, j in EDGE_LIST:
    _adj[i].append(j)
    _adj[j].append(i)
NEIGHBORS = _adj
NODE_DEGREE = np.array([len(nb) for nb in _adj], dtype=np.float32)


class EpiNetEnv(GraphEnv):
    """20-node epidemic control environment (SIR + community mobility graph).

    Objective: minimise total outbreak burden while keeping intervention costs low.
    The agent must proactively intervene at high-risk nodes before infection peaks.
    """

    def __init__(
        self,
        max_steps: int = 60,       # 60-day episode
        seed_nodes: int = 2,       # number of initially infected communities
        seed_fraction: float = 0.05,
        seed: int = 42,
    ):
        self.max_steps = max_steps
        self.seed_nodes_count = seed_nodes
        self.seed_fraction = seed_fraction
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # State: S, I, R fractions (sum to 1 per node)
        self._S = np.ones(NUM_NODES, dtype=np.float64)
        self._I = np.zeros(NUM_NODES, dtype=np.float64)
        self._R = np.zeros(NUM_NODES, dtype=np.float64)
        self._intervention = np.zeros(NUM_NODES, dtype=np.int32)

    # ─── SIR dynamics ──────────────────────────────────────────────────────────

    def _sir_step(self, intervention: np.ndarray) -> np.ndarray:
        """Advance SIR by one day. Returns new_infections per node."""
        S, I, R = self._S.copy(), self._I.copy(), self._R.copy()
        new_I = np.zeros(NUM_NODES, dtype=np.float64)

        for i in range(NUM_NODES):
            bf = BETA_FACTOR[int(intervention[i])]

            # Within-community force of infection
            local_foi = BETA * bf * S[i] * I[i]

            # Inter-community transmission from neighbours
            inter_foi = 0.0
            for j in NEIGHBORS[i]:
                mob_ij = (MOBILITY[i] + MOBILITY[j]) * 0.5
                inter_foi += ALPHA * mob_ij * BETA * bf * S[i] * I[j] / max(NODE_DEGREE[i], 1.0)

            delta_I = min(local_foi + inter_foi, S[i])   # can't infect more than susceptibles
            delta_R = GAMMA * I[i]

            self._S[i] = S[i] - delta_I
            self._I[i] = I[i] + delta_I - delta_R
            self._R[i] = R[i] + delta_R
            self._I[i] = max(self._I[i], 0.0)
            self._S[i] = max(self._S[i], 0.0)
            new_I[i] = delta_I

        return new_I

    # ─── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> GraphObs:
        inter_norm = self._intervention.astype(np.float32) / 2.0
        x = np.stack([
            self._S.astype(np.float32),
            self._I.astype(np.float32),
            self._R.astype(np.float32),
            inter_norm,
            MOBILITY,
        ], axis=-1)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(EDGE_INDEX, dtype=torch.long),
        )

        # Constraint: any node in active outbreak
        outbreak = (self._I > OUTBREAK_THRESHOLD).astype(np.float32)
        constraint_values = np.array([float(np.any(outbreak))], dtype=np.float32)

        global_features = np.array([
            float(np.mean(self._I)),   # population-average infection rate
            float(np.max(self._I)),    # worst-case node
        ], dtype=np.float32)

        return GraphObs(
            data=data,
            global_features=global_features,
            constraint_values=constraint_values,
        )

    def _compute_reward(self, new_I: np.ndarray, intervention: np.ndarray) -> float:
        infection_burden = float(np.sum(new_I)) * 10.0
        cost = float(np.sum([INTER_COST[int(a)] for a in intervention]))
        return -(infection_burden + cost)

    # ─── GraphEnv interface ────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return NUM_NODES

    @property
    def node_feature_dim(self) -> int:
        return NODE_FEAT_DIM

    @property
    def edge_feature_dim(self) -> int:
        return 0

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def constraint_names(self) -> list[str]:
        return ["outbreak_violation"]

    def get_topology(self) -> tuple[np.ndarray, np.ndarray]:
        return EDGE_INDEX, np.zeros((EDGE_INDEX.shape[1], 0))

    def reset(self, seed: int | None = None) -> tuple[GraphObs, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._S = np.ones(NUM_NODES, dtype=np.float64)
        self._I = np.zeros(NUM_NODES, dtype=np.float64)
        self._R = np.zeros(NUM_NODES, dtype=np.float64)
        self._intervention = np.zeros(NUM_NODES, dtype=np.int32)

        # Seed infection in random nodes (biased toward hubs for realism)
        seed_pool = HUB_NODES + list(range(NUM_NODES))  # hubs appear twice → 2× more likely
        chosen = self._rng.choice(seed_pool, size=self.seed_nodes_count, replace=False)
        for node in chosen:
            frac = self._rng.uniform(0.03, self.seed_fraction)
            self._I[node] = frac
            self._S[node] = 1.0 - frac

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[GraphObs, float, bool, bool, dict]:
        self._step_count += 1
        intervention = np.array([int(a) for a in action], dtype=np.int32)
        self._intervention = intervention

        new_I = self._sir_step(intervention)
        reward = self._compute_reward(new_I, intervention)
        terminated = self._step_count >= self.max_steps

        outbreak_nodes = int(np.sum(self._I > OUTBREAK_THRESHOLD))
        info = {
            "step": self._step_count,
            "total_infected_fraction": float(np.sum(self._I)),
            "outbreak_nodes": outbreak_nodes,
            "new_infections": float(np.sum(new_I)),
            "intervention_cost": float(np.sum([INTER_COST[int(a)] for a in intervention])),
        }

        return self._get_obs(), reward, terminated, False, info
