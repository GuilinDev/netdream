"""Traffic signal control environment wrapped as GraphEnv.

4×4 grid road network (16 intersections) with the Cell Transmission Model (CTM)
for macroscopic traffic flow. Pure NumPy — no SUMO dependency.

Network:
  - 16 intersections (nodes) in a 4×4 grid
  - 24 internal road segments (edges) — horizontal + vertical bidirectional
  - Each intersection has up to 4 incoming roads (N/S/E/W)

Node features (NODE_FEAT_DIM = 6):
  [queue_NS_norm, queue_EW_norm, throughput_norm, avg_speed_norm,
   phase_onehot_1, phase_onehot_2]
  (Phases: 0=NS_green/EW_red, 1=EW_green/NS_red, 2=all_red)

Actions (ACTION_DIM = 3):
  0 = switch to NS-green (allow N↔S traffic)
  1 = switch to EW-green (allow E↔W traffic)
  2 = hold current phase

Constraint: max queue length ≤ QUEUE_THRESHOLD vehicles (congestion constraint).
Reward: negative total delay = -Σ queue_lengths across all intersections.

CTM simplified: queue updates per step (30s), inflow limited by green phase,
outflow limited by saturation flow rate and demand.
"""

import numpy as np
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(__file__).replace("envs/traffic_env.py", ""))
from netdream.graph_env import GraphEnv, GraphObs

# ─── Network constants ─────────────────────────────────────────────────────────

GRID_N = 4              # 4×4 grid
NUM_NODES = GRID_N * GRID_N   # 16 intersections
NODE_FEAT_DIM = 6
ACTION_DIM = 3

# Traffic parameters
SAT_FLOW = 0.5          # vehicles per second (saturation flow rate per lane)
QUEUE_THRESHOLD = 25.0  # max queue length before constraint violation
DT = 30.0               # seconds per step
MIN_GREEN_STEPS = 2     # minimum steps before phase change allowed

# Arrival rates: base demand (vehicles/step entering each intersection)
BASE_ARRIVAL_NS = 3.0   # vehicles per step from N/S approaches
BASE_ARRIVAL_EW = 3.0   # vehicles per step from E/W approaches


def _build_grid_edges(n: int) -> np.ndarray:
    """Build bidirectional edge_index for n×n grid intersections."""
    src, dst = [], []
    for r in range(n):
        for c in range(n):
            node = r * n + c
            # East neighbor
            if c + 1 < n:
                east = r * n + (c + 1)
                src += [node, east]
                dst += [east, node]
            # South neighbor
            if r + 1 < n:
                south = (r + 1) * n + c
                src += [node, south]
                dst += [south, node]
    return np.array([src, dst], dtype=np.int64)


EDGE_INDEX = _build_grid_edges(GRID_N)


class TrafficSignalEnv(GraphEnv):
    """4×4 grid traffic signal control using simplified CTM.

    Each intersection controls its signal phase to minimize queues.
    Constraint: no intersection exceeds QUEUE_THRESHOLD vehicles waiting.
    """

    def __init__(
        self,
        max_steps: int = 120,          # 60 minutes at 30s steps
        demand_type: str = "peak",     # 'peak', 'uniform', or 'random'
        seed: int = 42,
    ):
        self.max_steps = max_steps
        self.demand_type = demand_type
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # State: per-intersection
        self._queue_ns = np.zeros(NUM_NODES, dtype=np.float32)   # NS queue (vehicles)
        self._queue_ew = np.zeros(NUM_NODES, dtype=np.float32)   # EW queue (vehicles)
        self._phase = np.zeros(NUM_NODES, dtype=np.int32)         # 0=NS_green, 1=EW_green, 2=all_red
        self._phase_steps = np.zeros(NUM_NODES, dtype=np.int32)  # steps in current phase
        self._throughput = np.zeros(NUM_NODES, dtype=np.float32)

    def _arrival_rate(self, step: int, direction: str) -> float:
        """Time-varying demand with morning/evening peaks."""
        if self.demand_type == "uniform":
            return BASE_ARRIVAL_NS

        # Peak pattern: morning rush at step 20, evening at step 80
        t = step / self.max_steps
        peak = 1.5 + 0.8 * np.exp(-((t - 0.3) ** 2) / 0.01) + \
               0.6 * np.exp(-((t - 0.7) ** 2) / 0.01)

        if self.demand_type == "random":
            peak *= self._rng.uniform(0.7, 1.3)

        base = BASE_ARRIVAL_NS if direction == "NS" else BASE_ARRIVAL_EW
        return base * peak

    def _step_queues(self, action: np.ndarray) -> float:
        """Update queue lengths using simplified CTM dynamics."""
        total_throughput = 0.0

        for i in range(NUM_NODES):
            # Respect minimum green duration
            if self._phase_steps[i] >= MIN_GREEN_STEPS:
                self._phase[i] = int(action[i]) if action[i] in [0, 1, 2] else self._phase[i]

            self._phase_steps[i] += 1
            if self._phase[i] != int(action[i]) and self._phase_steps[i] >= MIN_GREEN_STEPS:
                self._phase[i] = int(action[i])
                self._phase_steps[i] = 0

            # Arrivals
            arr_ns = self._arrival_rate(self._step_count, "NS") + \
                     self._rng.poisson(0.5)  # stochastic component
            arr_ew = self._arrival_rate(self._step_count, "EW") + \
                     self._rng.poisson(0.5)

            # Departures depend on phase
            if self._phase[i] == 0:  # NS green
                depart_ns = min(self._queue_ns[i] + arr_ns, SAT_FLOW * DT)
                depart_ew = 0.0
            elif self._phase[i] == 1:  # EW green
                depart_ns = 0.0
                depart_ew = min(self._queue_ew[i] + arr_ew, SAT_FLOW * DT)
            else:  # all-red (clearance)
                depart_ns = min(self._queue_ns[i], SAT_FLOW * DT * 0.1)
                depart_ew = min(self._queue_ew[i], SAT_FLOW * DT * 0.1)

            # Update queues
            self._queue_ns[i] = max(self._queue_ns[i] + arr_ns - depart_ns, 0.0)
            self._queue_ew[i] = max(self._queue_ew[i] + arr_ew - depart_ew, 0.0)

            # Throughput (vehicles served this step)
            step_tp = depart_ns + depart_ew
            self._throughput[i] = step_tp
            total_throughput += step_tp

        return total_throughput

    def _get_obs(self) -> GraphObs:
        # Normalize features
        q_ns_norm = self._queue_ns / (QUEUE_THRESHOLD * 2)
        q_ew_norm = self._queue_ew / (QUEUE_THRESHOLD * 2)
        tp_norm = self._throughput / (SAT_FLOW * DT * 2)
        speed_norm = np.clip(1.0 - (self._queue_ns + self._queue_ew) /
                             (QUEUE_THRESHOLD * 4), 0.0, 1.0).astype(np.float32)
        phase_0 = (self._phase == 0).astype(np.float32)
        phase_1 = (self._phase == 1).astype(np.float32)

        x = np.stack([
            q_ns_norm.astype(np.float32),
            q_ew_norm.astype(np.float32),
            tp_norm.astype(np.float32),
            speed_norm,
            phase_0,
            phase_1,
        ], axis=-1)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(EDGE_INDEX, dtype=torch.long),
        )

        # Constraint: any intersection exceeds queue threshold
        total_queue = self._queue_ns + self._queue_ew
        congestion_violation = float(np.any(total_queue > QUEUE_THRESHOLD))
        severe_violation = float(np.any(total_queue > QUEUE_THRESHOLD * 2))
        constraint_values = np.array([congestion_violation, severe_violation],
                                     dtype=np.float32)

        global_features = np.array([
            float(np.mean(total_queue)),
            float(np.max(total_queue)),
        ], dtype=np.float32)

        return GraphObs(
            data=data,
            global_features=global_features,
            constraint_values=constraint_values,
        )

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
        return ["congestion_violation", "severe_congestion"]

    def get_topology(self) -> tuple[np.ndarray, np.ndarray]:
        return EDGE_INDEX, np.zeros((EDGE_INDEX.shape[1], 0))

    def reset(self, seed: int | None = None) -> tuple[GraphObs, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        # Initialize with small random queues
        self._queue_ns = self._rng.uniform(0, 5, size=NUM_NODES).astype(np.float32)
        self._queue_ew = self._rng.uniform(0, 5, size=NUM_NODES).astype(np.float32)
        self._phase = np.zeros(NUM_NODES, dtype=np.int32)
        self._phase_steps = np.zeros(NUM_NODES, dtype=np.int32)
        self._throughput = np.zeros(NUM_NODES, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[GraphObs, float, bool, bool, dict]:
        self._step_count += 1
        total_tp = self._step_queues(action)

        total_queue = self._queue_ns + self._queue_ew
        total_delay = float(np.sum(total_queue))
        reward = -(total_delay * 0.01 + float(np.sum(total_queue > QUEUE_THRESHOLD)) * 2.0)

        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "step": self._step_count,
            "mean_queue": float(np.mean(total_queue)),
            "max_queue": float(np.max(total_queue)),
            "congested_intersections": int(np.sum(total_queue > QUEUE_THRESHOLD)),
            "total_throughput": total_tp,
        }

        return self._get_obs(), reward, terminated, truncated, info
