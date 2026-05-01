"""Grid2Op power grid environment wrapped as GraphEnv.

Maps Grid2Op observations to graph-structured format:
  - Nodes = buses (substations)
  - Edges = power lines (transmission lines)
  - Node features = [voltage_magnitude, active_power_gen, reactive_power_gen,
                      active_power_load, reactive_power_load, ...]
  - Actions = line status toggle (connect/disconnect) or bus reconfiguration

Constraints:
  - Voltage magnitude within [0.95, 1.05] p.u.
  - Line thermal limits (flow < capacity)
"""

import numpy as np
import torch
from torch_geometric.data import Data

import grid2op
from grid2op.Parameters import Parameters

import sys
sys.path.insert(0, str(__file__).replace("envs/grid2op_env.py", ""))
from netdream.graph_env import GraphEnv, GraphObs

# Per-bus features
# [v_mag, p_gen, q_gen, p_load, q_load, is_gen, is_load]
NODE_FEAT_DIM = 7
# Action: per-line {0: disconnect, 1: maintain, 2: reconnect}
ACTION_DIM = 3


class Grid2OpGraphEnv(GraphEnv):
    """Grid2Op power grid wrapped as a GraphEnv with graph-structured observations."""

    def __init__(
        self,
        env_name: str = "l2rpn_case14_sandbox",
        max_steps: int = 288,  # 24 hours at 5-min intervals
        voltage_low: float = 0.95,
        voltage_high: float = 1.05,
        thermal_limit_fraction: float = 1.0,
        seed: int = 42,
    ):
        self.env_name = env_name
        self.max_steps = max_steps
        self.voltage_low = voltage_low
        self.voltage_high = voltage_high
        self.thermal_limit_fraction = thermal_limit_fraction
        self._seed = seed

        # Create Grid2Op environment
        params = Parameters()
        params.MAX_LINE_STATUS_CHANGED = 999
        params.MAX_SUB_CHANGED = 999
        params.NO_OVERFLOW_DISCONNECTION = False

        self._env = grid2op.make(
            env_name,
            param=params,
        )
        self._obs = None
        self._step_count = 0

        # Cache topology info
        self._n_buses = self._env.n_sub  # number of substations (nodes)
        self._n_lines = self._env.n_line  # number of lines (edges)

        # Build edge index from line connectivity
        self._edge_index = self._build_edge_index()

    def _build_edge_index(self) -> np.ndarray:
        """Build bidirectional edge index from power line connectivity."""
        obs = self._env.reset()
        or_sub = obs.line_or_to_subid  # origin substation per line
        ex_sub = obs.line_ex_to_subid  # extremity substation per line

        # Bidirectional edges
        src = np.concatenate([or_sub, ex_sub])
        dst = np.concatenate([ex_sub, or_sub])
        return np.array([src, dst], dtype=np.int64)

    def _obs_to_graph(self, obs) -> GraphObs:
        """Convert Grid2Op observation to graph-structured format."""
        n = self._n_buses

        # Aggregate per-bus features from generators and loads
        v_mag = np.zeros(n, dtype=np.float32)
        p_gen = np.zeros(n, dtype=np.float32)
        q_gen = np.zeros(n, dtype=np.float32)
        p_load = np.zeros(n, dtype=np.float32)
        q_load = np.zeros(n, dtype=np.float32)
        is_gen = np.zeros(n, dtype=np.float32)
        is_load = np.zeros(n, dtype=np.float32)

        # Voltage from substations (approximate from line or/ex voltages)
        for i in range(self._n_lines):
            if obs.line_status[i]:
                or_sub = obs.line_or_to_subid[i]
                ex_sub = obs.line_ex_to_subid[i]
                v_mag[or_sub] = obs.v_or[i] / 1000.0 if obs.v_or[i] > 0 else 1.0
                v_mag[ex_sub] = obs.v_ex[i] / 1000.0 if obs.v_ex[i] > 0 else 1.0

        # Generator power
        for i in range(obs.n_gen):
            sub = obs.gen_to_subid[i]
            p_gen[sub] += obs.gen_p[i] / 100.0  # normalize
            q_gen[sub] += obs.gen_q[i] / 100.0
            is_gen[sub] = 1.0

        # Load power
        for i in range(obs.n_load):
            sub = obs.load_to_subid[i]
            p_load[sub] += obs.load_p[i] / 100.0
            q_load[sub] += obs.load_q[i] / 100.0
            is_load[sub] = 1.0

        # Stack node features
        x = np.stack([v_mag, p_gen, q_gen, p_load, q_load, is_gen, is_load], axis=-1)

        # Handle any NaN values
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(self._edge_index, dtype=torch.long),
        )

        # Constraints: voltage out of bounds + line overload
        voltage_violation = np.zeros(1, dtype=np.float32)
        if np.any(v_mag > 0):  # only check if we have voltage data
            valid_v = v_mag[v_mag > 0]
            if len(valid_v) > 0:
                voltage_violation[0] = float(
                    np.any(valid_v < self.voltage_low) or
                    np.any(valid_v > self.voltage_high)
                )

        # Line overload
        rho = obs.rho  # line loading ratio (flow / thermal_limit)
        thermal_violation = float(np.any(rho > self.thermal_limit_fraction))

        constraint_values = np.array([
            voltage_violation[0],
            thermal_violation,
        ], dtype=np.float32)

        global_features = np.array([
            float(np.sum(obs.gen_p)),   # total generation
            float(np.max(rho)) if len(rho) > 0 else 0.0,  # max line loading
        ], dtype=np.float32)

        return GraphObs(
            data=data,
            global_features=global_features,
            constraint_values=constraint_values,
        )

    def _compute_reward(self, obs) -> float:
        """Reward: penalize line overloads and voltage violations."""
        rho = obs.rho
        overload_penalty = np.sum(np.maximum(rho - 1.0, 0.0)) * 10.0
        cost = np.sum(obs.gen_p) * 0.001  # generation cost

        return -(cost + overload_penalty)

    # ─── GraphEnv interface ──────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self._n_buses

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
        return ["voltage_violation", "thermal_violation"]

    def get_topology(self) -> tuple[np.ndarray, np.ndarray]:
        return self._edge_index, np.zeros((self._edge_index.shape[1], 0))

    def reset(self, seed: int | None = None) -> tuple[GraphObs, dict]:
        if seed is not None:
            self._env.seed(seed)

        obs = self._env.reset()
        self._obs = obs
        self._step_count = 0

        return self._obs_to_graph(obs), {}

    def get_safe_action_mask(self) -> np.ndarray:
        """Return mask of safe actions per node.

        For power grids, disconnecting lines with high loading (rho > 0.5) is unsafe.
        Only reconnecting already-disconnected lines or doing nothing is safe.
        """
        obs = self._obs
        mask = np.ones((self._n_buses, ACTION_DIM), dtype=bool)

        # For now, only allow: 0=do-nothing, 1=do-nothing, 2=reconnect-if-disconnected
        # Disable "disconnect" (action=0) for all nodes by default
        # This prevents the catastrophic line disconnection
        mask[:, 0] = False  # no disconnection allowed

        return mask

    def step(self, action: np.ndarray) -> tuple[GraphObs, float, bool, bool, dict]:
        """Execute per-node actions on the grid with safety masking.

        Args:
            action: [n_buses] array of {0: disconnect nearby line, 1: do-nothing, 2: reconnect nearby line}
                    Action 0 is masked to prevent dangerous disconnections.
        """
        self._step_count += 1

        # Safety: override disconnect actions to do-nothing
        safe_action = action.copy()
        safe_action[safe_action == 0] = 1  # force disconnect → do-nothing

        # Convert per-node actions to per-line Grid2Op actions
        g2op_action = self._env.action_space({})

        # For each node that wants to reconnect (action=2),
        # find disconnected lines connected to that substation and reconnect them
        obs = self._obs
        for node_idx in range(min(len(safe_action), self._n_buses)):
            if safe_action[node_idx] == 2:  # reconnect
                # Find lines connected to this substation that are disconnected
                for line_idx in range(self._n_lines):
                    if not obs.line_status[line_idx]:
                        if (obs.line_or_to_subid[line_idx] == node_idx or
                                obs.line_ex_to_subid[line_idx] == node_idx):
                            g2op_action.line_set_status = [(line_idx, 1)]
                            break  # reconnect one line per step

        obs, reward_g2op, done, info = self._env.step(g2op_action)
        self._obs = obs

        reward = self._compute_reward(obs)
        terminated = done or self._step_count >= self.max_steps
        truncated = False

        step_info = {
            "step": self._step_count,
            "max_rho": float(np.max(obs.rho)) if len(obs.rho) > 0 else 0.0,
            "total_gen": float(np.sum(obs.gen_p)),
            "lines_disconnected": int(np.sum(~obs.line_status)),
            "game_over": done,
        }

        return self._obs_to_graph(obs), reward, terminated, truncated, step_info
