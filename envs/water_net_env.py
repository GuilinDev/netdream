"""Water distribution network environment wrapped as GraphEnv.

Custom 10-node pump-controlled water network (WaterNet-10).
No external dependencies — pure NumPy hydraulic simulation.

Physics:
  - Simplified Hazen-Williams head-loss: dH = R * Q^1.852
    (linearized around nominal flow for step-wise simulation)
  - Mass balance at each junction: sum(inflows) - demand = d(tank_level)/dt
  - Pump head gain: H_pump = pump_speed * H_rated * (1 - 0.3*(Q/Q_rated)^2)

Network layout (WaterNet-10):
  - 1 reservoir  (node 0):  fixed head = 100 m
  - 2 pump nodes (nodes 1, 5): controllable pump stations
  - 1 storage tank (node 4): head tracks tank level
  - 6 demand junctions (nodes 2, 3, 6, 7, 8, 9)

Pipe topology (bidirectional, 11 pipes):
  0→1, 1→2, 2→3, 3→4, 4→6, 1→5, 5→6, 6→7, 7→8, 8→9, 3→7

Control actions (per node, ACTION_DIM = 3):
  Pump nodes (1, 5): 0=off, 1=normal (speed=1.0), 2=high (speed=1.5)
  Other nodes: action ignored (mapped to maintain)

Node features (NODE_FEAT_DIM = 5):
  [pressure_head_norm, demand_norm, tank_level_norm, pump_speed, elevation_norm]

Constraint: minimum pressure ≥ MIN_PRESSURE_M at all demand nodes.
"""

import numpy as np
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(__file__).replace("envs/water_net_env.py", ""))
from netdream.graph_env import GraphEnv, GraphObs

# ─── Network constants ─────────────────────────────────────────────────────────

NUM_NODES = 10
NODE_FEAT_DIM = 5
ACTION_DIM = 3

MIN_PRESSURE_M = 20.0    # m — minimum service pressure constraint
MAX_TANK_LEVEL = 1.0     # normalized
MIN_TANK_LEVEL = 0.05

# Node types
RESERVOIR = 0
PUMP = 1
DEMAND = 2
TANK = 3

NODE_TYPES = [RESERVOIR, PUMP, DEMAND, DEMAND, TANK, PUMP, DEMAND, DEMAND, DEMAND, DEMAND]

# Elevation profile (m)
ELEVATION = np.array([0., 5., 20., 30., 40., 15., 35., 50., 45., 40.], dtype=np.float32)

# Reservoir fixed head (m above datum)
RESERVOIR_HEAD = 100.0

# Base demands (L/s), 0 for non-demand nodes
BASE_DEMAND = np.array([0., 0., 8., 10., 0., 0., 12., 15., 10., 8.], dtype=np.float32)

# Pump parameters: rated head (m) and rated flow (L/s)
PUMP_H_RATED = {1: 30.0, 5: 25.0}
PUMP_Q_RATED = {1: 40.0, 5: 35.0}

# Tank volume (L) and initial level (normalized 0–1)
TANK_NODE = 4
TANK_VOLUME = 5000.0   # litres
TANK_INIT_LEVEL = 0.6

# Pipe topology: list of (from, to) undirected
PIPE_LIST = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (4, 6), (1, 5), (5, 6), (6, 7),
    (7, 8), (8, 9), (3, 7),
]
# Pipe resistance coefficients (pressure-drop per unit flow^1.852, simplified)
PIPE_R = np.array([0.01, 0.05, 0.04, 0.06, 0.07, 0.04, 0.05, 0.06, 0.05, 0.04, 0.08],
                  dtype=np.float32)

# Build edge_index (bidirectional)
_src = [p[0] for p in PIPE_LIST] + [p[1] for p in PIPE_LIST]
_dst = [p[1] for p in PIPE_LIST] + [p[0] for p in PIPE_LIST]
EDGE_INDEX = np.array([_src, _dst], dtype=np.int64)

# Demand nodes for constraint checking
DEMAND_NODES = [i for i, t in enumerate(NODE_TYPES) if t == DEMAND]
PUMP_NODES = [i for i, t in enumerate(NODE_TYPES) if t == PUMP]


class WaterNetEnv(GraphEnv):
    """10-node pump-controlled water distribution network.

    Control objective: maintain minimum pressure at all demand nodes
    while minimizing pumping energy cost.
    """

    def __init__(
        self,
        max_steps: int = 96,           # 24h at 15-min intervals
        dt: float = 900.0,             # 15 min in seconds
        demand_pattern: str = "daily", # 'daily' or 'random'
        seed: int = 42,
    ):
        self.max_steps = max_steps
        self.dt = dt
        self.demand_pattern = demand_pattern
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # State variables
        self._head = np.zeros(NUM_NODES, dtype=np.float32)      # hydraulic head (m)
        self._pump_speed = np.zeros(NUM_NODES, dtype=np.float32)  # 0/1/1.5
        self._tank_level = TANK_INIT_LEVEL                        # normalized
        self._demand_factor = 1.0                                 # time-varying multiplier

    # ─── Hydraulic simulation ──────────────────────────────────────────────────

    def _daily_demand_factor(self, step: int) -> float:
        """Sinusoidal daily demand pattern with two peaks (morning + evening)."""
        t = (step % 96) / 96.0 * 2 * np.pi
        return 0.7 + 0.4 * np.sin(t) + 0.2 * np.sin(2 * t + 0.5)

    def _compute_heads(self, pump_speeds: np.ndarray, demand_factor: float) -> np.ndarray:
        """Compute steady-state hydraulic heads via linear network analysis (graph Laplacian).

        Models the water network as a resistance network: conductance C = 1/R per pipe.
        Pumps are modelled as ideal head sources (pressure booster = pump_head added
        to the hydraulic head at the outlet node).
        Solves the sparse linear system: G * H = sources (known heads + pump boosts).

        Returns head array [NUM_NODES] in metres above datum.
        """
        n = NUM_NODES
        # Pipe conductances (C = 1/R; higher C = easier flow = less head loss)
        pipe_C = 1.0 / np.maximum(PIPE_R, 1e-6)   # shape [num_pipes]

        # Build graph Laplacian conductance matrix
        G = np.zeros((n, n), dtype=np.float64)
        for k, (i, j) in enumerate(PIPE_LIST):
            G[i, i] += pipe_C[k]
            G[j, j] += pipe_C[k]
            G[i, j] -= pipe_C[k]
            G[j, i] -= pipe_C[k]

        # Right-hand side: demand injections (negative = demand, positive = supply)
        demand = BASE_DEMAND * demand_factor
        b = -demand.astype(np.float64)  # withdrawals are negative injections

        # Pump head boosts: add to b for the outlet node of each pump
        # Pump 1 (node 1) boosts head from reservoir side (node 0) to zone side (node 2)
        # Pump 5 (node 5) boosts head from node 1 side to zone side (node 6)
        pump_outlet_neighbors = {1: 2, 5: 6}   # pump node → downstream junction
        for pump_node, outlet in pump_outlet_neighbors.items():
            spd = float(pump_speeds[pump_node])
            if spd > 0:
                h_boost = PUMP_H_RATED[pump_node] * spd   # head added (m)
                b[outlet] += pipe_C[PIPE_LIST.index((1, 2) if pump_node == 1
                                                     else (5, 6))] * h_boost

        # Fixed head boundary conditions: reservoir (node 0) and tank (node 4)
        h_fixed = {0: float(RESERVOIR_HEAD)}
        tank_height = 15.0  # m
        h_fixed[TANK_NODE] = float(ELEVATION[TANK_NODE] + self._tank_level * tank_height)

        free_nodes = [i for i in range(n) if i not in h_fixed]
        fixed_nodes = list(h_fixed.keys())

        # Partition system: G_ff * h_free = b_free - G_fc * h_known
        G_ff = G[np.ix_(free_nodes, free_nodes)]
        G_fc = G[np.ix_(free_nodes, fixed_nodes)]
        h_known = np.array([h_fixed[i] for i in fixed_nodes], dtype=np.float64)
        b_free = b[free_nodes] - G_fc @ h_known

        # Add small regularisation to make system non-singular
        G_ff += np.eye(len(free_nodes)) * 1e-6

        try:
            h_free = np.linalg.solve(G_ff, b_free)
        except np.linalg.LinAlgError:
            h_free = np.linalg.lstsq(G_ff, b_free, rcond=None)[0]

        head = np.zeros(n, dtype=np.float32)
        for k, i in enumerate(free_nodes):
            head[i] = float(h_free[k])
        for i, h in h_fixed.items():
            head[i] = float(h)

        # Clip to physically plausible range
        head = np.clip(head, ELEVATION, RESERVOIR_HEAD + 50.0)
        return head

    def _update_tank(self, pump_speeds: np.ndarray, demand_factor: float) -> float:
        """Update tank level based on net inflow. Returns new level."""
        # Net inflow from pump 5 (which feeds into the tank zone)
        inflow_rate = pump_speeds[5] * PUMP_Q_RATED[5] * 0.5  # L/s
        outflow_rate = BASE_DEMAND[TANK_NODE + 2:TANK_NODE + 4].sum() * demand_factor
        net_flow = (inflow_rate - outflow_rate) * self.dt  # litres
        new_level = self._tank_level + net_flow / TANK_VOLUME
        return float(np.clip(new_level, 0.0, MAX_TANK_LEVEL))

    # ─── Observation building ───────────────────────────────────────────────────

    def _get_obs(self) -> GraphObs:
        demand = BASE_DEMAND * self._demand_factor

        # Normalize features
        head_norm = (self._head - ELEVATION) / 60.0   # available pressure head / 60m
        demand_norm = demand / 20.0
        tank_norm = np.zeros(NUM_NODES, dtype=np.float32)
        tank_norm[TANK_NODE] = self._tank_level
        pump_feat = self._pump_speed / 1.5  # normalize to [0,1]
        elev_norm = ELEVATION / 60.0

        x = np.stack([
            head_norm.astype(np.float32),
            demand_norm.astype(np.float32),
            tank_norm,
            pump_feat.astype(np.float32),
            elev_norm,
        ], axis=-1)

        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(EDGE_INDEX, dtype=torch.long),
        )

        # Constraint: pressure deficit at demand nodes
        pressure = self._head - ELEVATION  # available pressure (m)
        pressure_violation = float(np.any(pressure[DEMAND_NODES] < MIN_PRESSURE_M))
        tank_violation = float(
            self._tank_level < MIN_TANK_LEVEL or self._tank_level > MAX_TANK_LEVEL
        )
        constraint_values = np.array([pressure_violation, tank_violation], dtype=np.float32)

        global_features = np.array([
            float(np.min(pressure[DEMAND_NODES])),  # worst-case pressure margin
            self._tank_level,
        ], dtype=np.float32)

        return GraphObs(
            data=data,
            global_features=global_features,
            constraint_values=constraint_values,
        )

    def _compute_reward(self, pressure: np.ndarray) -> float:
        """Penalize pressure violations and pumping energy cost."""
        pressure_deficit = np.sum(np.maximum(MIN_PRESSURE_M - pressure[DEMAND_NODES], 0.0))
        energy_cost = float(np.sum(self._pump_speed[PUMP_NODES]) * 0.5)
        return -(pressure_deficit * 5.0 + energy_cost)

    # ─── GraphEnv interface ─────────────────────────────────────────────────────

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
        return ["pressure_violation", "tank_violation"]

    def get_topology(self) -> tuple[np.ndarray, np.ndarray]:
        return EDGE_INDEX, np.zeros((EDGE_INDEX.shape[1], 0))

    def reset(self, seed: int | None = None) -> tuple[GraphObs, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._tank_level = TANK_INIT_LEVEL + self._rng.uniform(-0.1, 0.1)
        self._pump_speed = np.zeros(NUM_NODES, dtype=np.float32)
        self._pump_speed[1] = 1.0  # pump 1 starts on normal
        self._pump_speed[5] = 1.0  # pump 5 starts on normal
        self._demand_factor = self._daily_demand_factor(0)
        self._head = self._compute_heads(self._pump_speed, self._demand_factor)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[GraphObs, float, bool, bool, dict]:
        self._step_count += 1

        # Map per-node action to pump speeds
        speed_map = {0: 0.0, 1: 1.0, 2: 1.5}
        new_pump_speed = self._pump_speed.copy()
        for pump_node in PUMP_NODES:
            new_pump_speed[pump_node] = speed_map[int(action[pump_node])]

        self._pump_speed = new_pump_speed

        # Advance demand pattern
        self._demand_factor = self._daily_demand_factor(self._step_count)
        if self.demand_pattern == "random":
            self._demand_factor *= self._rng.uniform(0.8, 1.3)

        # Update tank level
        self._tank_level = self._update_tank(self._pump_speed, self._demand_factor)

        # Compute new heads
        self._head = self._compute_heads(self._pump_speed, self._demand_factor)

        pressure = self._head - ELEVATION
        reward = self._compute_reward(pressure)
        terminated = self._step_count >= self.max_steps
        truncated = False

        pressure_violation = bool(np.any(pressure[DEMAND_NODES] < MIN_PRESSURE_M))
        info = {
            "step": self._step_count,
            "min_pressure_m": float(np.min(pressure[DEMAND_NODES])),
            "tank_level": self._tank_level,
            "pump_speeds": self._pump_speed[PUMP_NODES].tolist(),
            "pressure_violation": pressure_violation,
            "demand_factor": self._demand_factor,
        }

        return self._get_obs(), reward, terminated, truncated, info
