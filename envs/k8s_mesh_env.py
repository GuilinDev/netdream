"""Multi-service Kubernetes microservice mesh environment.

Models the Online Boutique application with 11 services connected via API calls.
Each service has independent CPU/memory/QPS/latency metrics, and scaling one service
affects downstream services through request propagation and resource contention.

Cascade dynamics:
  - frontend QPS increase → downstream services QPS increase (fan-out)
  - If any service latency exceeds SLO → callers get timeouts → error rate increases
  - Pod startup delay: new replicas take 3-5 steps to become ready
  - Scaling causes brief network IO spike (load balancer redistribution)
"""

import numpy as np
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(__file__).replace("envs/k8s_mesh_env.py", ""))
from netdream.graph_env import GraphEnv, GraphObs


# Online Boutique service topology (11 services)
# Edges represent API call dependencies (caller → callee)
SERVICE_NAMES = [
    "frontend",             # 0
    "cartservice",          # 1
    "productcatalogservice",# 2
    "currencyservice",      # 3
    "shippingservice",      # 4
    "adservice",            # 5
    "checkoutservice",      # 6
    "recommendationservice",# 7
    "emailservice",         # 8
    "paymentservice",       # 9
    "redis-cart",           # 10
]

# Directed edges: (caller, callee)
SERVICE_EDGES = [
    (0, 1),   # frontend → cartservice
    (0, 2),   # frontend → productcatalogservice
    (0, 3),   # frontend → currencyservice
    (0, 4),   # frontend → shippingservice
    (0, 5),   # frontend → adservice
    (0, 6),   # frontend → checkoutservice
    (0, 7),   # frontend → recommendationservice
    (6, 1),   # checkoutservice → cartservice
    (6, 4),   # checkoutservice → shippingservice
    (6, 3),   # checkoutservice → currencyservice
    (6, 8),   # checkoutservice → emailservice
    (6, 9),   # checkoutservice → paymentservice
    (7, 2),   # recommendationservice → productcatalogservice
    (1, 10),  # cartservice → redis-cart
]

# Per-node feature: [cpu, memory, qps, p95_latency, error_rate, replicas]
NODE_FEAT_DIM = 6
# Action: per-service scaling delta {-1, 0, +1}
ACTION_DIM = 3


class K8sMeshEnv(GraphEnv):
    """Multi-service Kubernetes mesh with cascade dynamics."""

    def __init__(
        self,
        max_steps: int = 240,
        decision_interval: float = 15.0,  # seconds
        slo_latency_ms: float = 500.0,
        slo_error_rate: float = 0.05,
        external_qps_range: tuple[float, float] = (50.0, 300.0),
        workload_type: str = "variable",
        seed: int = 42,
    ):
        self.max_steps = max_steps
        self.decision_interval = decision_interval
        self.slo_latency_ms = slo_latency_ms
        self.slo_error_rate = slo_error_rate
        self.external_qps_range = external_qps_range
        self.workload_type = workload_type
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._num_services = len(SERVICE_NAMES)
        self._edge_index = self._build_edge_index()

        # State arrays
        self._replicas = np.ones(self._num_services, dtype=np.float64)
        self._cpu = np.zeros(self._num_services)
        self._memory = np.zeros(self._num_services)
        self._qps = np.zeros(self._num_services)
        self._latency = np.zeros(self._num_services)
        self._error_rate = np.zeros(self._num_services)
        self._step_count = 0
        self._startup_timers = np.zeros(self._num_services)  # pod startup delay

    def _build_edge_index(self) -> np.ndarray:
        """Build bidirectional edge index from service edges."""
        edges = SERVICE_EDGES.copy()
        # Add reverse edges for message passing (GNN needs bidirectional)
        reverse = [(e[1], e[0]) for e in edges]
        all_edges = edges + reverse
        src = [e[0] for e in all_edges]
        dst = [e[1] for e in all_edges]
        return np.array([src, dst], dtype=np.int64)

    def _generate_workload(self, step: int) -> float:
        """Generate external QPS hitting the frontend."""
        lo, hi = self.external_qps_range
        mid = (lo + hi) / 2
        amp = (hi - lo) / 2
        t = step / self.max_steps

        noise = self._rng.normal(0, 5)

        if self.workload_type == "constant":
            return mid + noise
        elif self.workload_type == "periodic":
            return mid + amp * np.sin(2 * np.pi * t * 4) + noise
        elif self.workload_type == "variable":
            self._walk = getattr(self, "_walk", mid)
            self._walk += self._rng.normal(0, 8)
            self._walk = np.clip(self._walk, lo, hi)
            return self._walk + noise
        elif self.workload_type == "bursty":
            base = mid * 0.5
            if self._rng.random() < 0.08:
                return hi + self._rng.exponential(50)
            return base + noise
        elif self.workload_type == "ramp":
            return lo + (hi - lo) * t + noise
        elif self.workload_type == "flash":
            if 0.3 < t < 0.5:
                return hi * 1.5 + noise
            return mid * 0.6 + noise
        return mid + noise

    def _propagate_traffic(self, external_qps: float) -> None:
        """Propagate request traffic through the service graph."""
        # Frontend receives all external traffic
        self._qps[0] = external_qps

        # Fan-out ratios (fraction of frontend QPS forwarded to each service)
        fanout = {
            (0, 1): 0.3,   # 30% of requests touch cart
            (0, 2): 0.8,   # 80% browse products
            (0, 3): 0.8,   # 80% need currency conversion
            (0, 4): 0.1,   # 10% check shipping
            (0, 5): 0.5,   # 50% see ads
            (0, 6): 0.05,  # 5% checkout
            (0, 7): 0.4,   # 40% get recommendations
            (6, 1): 1.0,   # all checkouts touch cart
            (6, 4): 1.0,   # all checkouts need shipping
            (6, 3): 1.0,   # all checkouts need currency
            (6, 8): 1.0,   # all checkouts send email
            (6, 9): 1.0,   # all checkouts process payment
            (7, 2): 1.0,   # recommendations query catalog
            (1, 10): 1.0,  # cart always hits redis
        }

        # Topological-order propagation (BFS from frontend)
        visited = {0}
        queue = [0]
        while queue:
            src = queue.pop(0)
            for edge in SERVICE_EDGES:
                if edge[0] == src:
                    dst = edge[1]
                    ratio = fanout.get(edge, 0.5)
                    self._qps[dst] += self._qps[src] * ratio
                    if dst not in visited:
                        visited.add(dst)
                        queue.append(dst)

    def _compute_metrics(self) -> None:
        """Compute CPU, memory, latency, error_rate from QPS and replicas."""
        eff_replicas = np.maximum(self._replicas - self._startup_timers, 0.5)

        per_replica_load = self._qps / eff_replicas

        # CPU: proportional to per-replica load
        self._cpu = np.clip(5.0 + per_replica_load * 0.7, 0, 100)

        # Memory: base + load-dependent
        self._memory = np.clip(80.0 + per_replica_load * 0.25, 40, 512)

        # Latency: base + load-dependent with exponential degradation
        base_latency = 20.0  # ms
        self._latency = base_latency + (per_replica_load ** 1.3) * 0.05
        # Under heavy load, latency spikes exponentially
        overload_mask = per_replica_load > 150
        self._latency[overload_mask] *= 1.0 + (per_replica_load[overload_mask] - 150) * 0.02

        # Cascade: if upstream latency is high, downstream callers see timeouts
        for src, dst in SERVICE_EDGES:
            if self._latency[src] > self.slo_latency_ms * 0.8:
                # Caller experiences timeout-induced latency increase
                caller_penalty = (self._latency[src] / self.slo_latency_ms) * 50
                self._latency[dst] += caller_penalty

        # Error rate: spikes when latency exceeds SLO
        self._error_rate = np.zeros(self._num_services)
        overloaded = self._latency > self.slo_latency_ms
        self._error_rate[overloaded] = np.clip(
            (self._latency[overloaded] - self.slo_latency_ms) / self.slo_latency_ms * 0.5,
            0, 1,
        )

        # Startup delay: new pods take 3 steps to become ready
        self._startup_timers = np.maximum(self._startup_timers - 1, 0)

    def _get_obs(self) -> GraphObs:
        """Build graph-structured observation."""
        # Normalize features to [0, 1] range
        x = np.stack([
            self._cpu / 100.0,
            self._memory / 512.0,
            self._qps / 500.0,
            self._latency / 1000.0,
            self._error_rate,
            self._replicas / 10.0,
        ], axis=-1).astype(np.float32)

        data = Data(
            x=torch.tensor(x),
            edge_index=torch.tensor(self._edge_index, dtype=torch.long),
        )

        # Constraint: SLO violation per service
        slo_violated = (
            (self._latency > self.slo_latency_ms) |
            (self._error_rate > self.slo_error_rate)
        ).astype(np.float32)

        return GraphObs(
            data=data,
            global_features=np.array([
                np.sum(self._replicas) * 0.01,  # total cost
                np.mean(self._latency),
            ], dtype=np.float32),
            constraint_values=slo_violated,
        )

    def _compute_reward(self) -> float:
        """Reward: negative cost + SLO penalty."""
        cost = np.sum(self._replicas) * 0.01
        slo_violations = np.sum(
            (self._latency > self.slo_latency_ms) |
            (self._error_rate > self.slo_error_rate)
        )
        return -(cost + slo_violations * 0.5)

    # ─── GraphEnv interface ──────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self._num_services

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
        return ["slo_violation"]

    def get_topology(self) -> tuple[np.ndarray, np.ndarray]:
        return self._edge_index, np.zeros((self._edge_index.shape[1], 0))

    def reset(self, seed: int | None = None) -> tuple[GraphObs, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._replicas = np.ones(self._num_services, dtype=np.float64)
        self._replicas[0] = 2  # frontend starts with 2 replicas
        self._qps = np.zeros(self._num_services)
        self._cpu = np.full(self._num_services, 5.0)
        self._memory = np.full(self._num_services, 80.0)
        self._latency = np.full(self._num_services, 20.0)
        self._error_rate = np.zeros(self._num_services)
        self._startup_timers = np.zeros(self._num_services)
        self._step_count = 0
        self._walk = (self.external_qps_range[0] + self.external_qps_range[1]) / 2

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[GraphObs, float, bool, bool, dict]:
        """Execute per-service scaling actions.

        Args:
            action: [num_services] array of {0: scale-down, 1: no-op, 2: scale-up}
        """
        self._step_count += 1

        # Apply scaling actions
        deltas = action.astype(np.float64) - 1  # map {0,1,2} → {-1,0,+1}
        old_replicas = self._replicas.copy()
        self._replicas = np.clip(self._replicas + deltas, 1, 10)

        # Pod startup delay for scale-up
        scaled_up = self._replicas > old_replicas
        self._startup_timers[scaled_up] = 3  # 3 steps to become ready

        # Generate external traffic and propagate
        external_qps = self._generate_workload(self._step_count)
        self._qps = np.zeros(self._num_services)
        self._propagate_traffic(external_qps)

        # Compute metrics (CPU, latency, etc.)
        self._compute_metrics()

        # Compute reward
        reward = self._compute_reward()

        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "step": self._step_count,
            "total_cost": float(np.sum(self._replicas) * 0.01),
            "slo_violations": int(np.sum(
                (self._latency > self.slo_latency_ms) |
                (self._error_rate > self.slo_error_rate)
            )),
            "mean_replicas": float(np.mean(self._replicas)),
            "external_qps": float(external_qps),
        }

        return self._get_obs(), reward, terminated, truncated, info
